"""A small GPT-style language model in plain Python + NumPy (no deep-learning library).

Same architecture as GPT-2 and most modern LLMs, only much smaller:

    text -> characters -> token ids
         -> token embedding + position embedding
         -> N x [ LayerNorm -> causal self-attention -> + residual
                  LayerNorm -> MLP (feed-forward)    -> + residual ]
         -> LayerNorm -> multiply by the embedding matrix -> probability of each next character

Training needs gradients, so every layer has a forward function (compute the output
and remember what it needs) and a backward function (given how the loss changes with
the output, work out how it changes with the inputs and weights). That is all
backpropagation is: the chain rule, applied layer by layer in reverse.

forward(..., record=True) also keeps a trace of every intermediate result so that
visualize.py can draw the network while it answers a prompt.
"""

import math
import pickle
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class GPTConfig:
    vocab_size: int = 96
    block_size: int = 64    # how many characters the model can look back at
    n_layer: int = 3        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 96        # width of the vector that represents each character


class CharTokenizer:
    """Maps each distinct character in the training text to an integer."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        # Characters never seen in training are dropped rather than crashing.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


# ----------------------------------------------------------------------------
# Building blocks: each has a forward and a backward.
# ----------------------------------------------------------------------------

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    """Rescale each vector to mean 0 and variance 1, then apply a learned scale and shift."""
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dout, cache):
    xhat, rstd, g = cache
    dg = (dout * xhat).sum(axis=(0, 1))
    db = dout.sum(axis=(0, 1))
    dxhat = dout * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


GELU_C = math.sqrt(2 / math.pi)


def gelu(x):
    """Smooth version of ReLU, used by GPT. Also returns the tanh, which the backward pass reuses."""
    t = np.tanh(GELU_C * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1 + t), t


def gelu_grad(x, t):
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x)


def mm(x, W):
    """x @ W for x of shape (B, T, n). Flattening to one 2-D product lets NumPy hand it
    to the fast BLAS routine in one go, which is ~40x faster than a batched 3-D matmul."""
    return (x.reshape(-1, x.shape[-1]) @ W).reshape(*x.shape[:-1], W.shape[-1])


def matmul_grads(x, dout):
    """Weight gradient of y = x @ W + b, summing over every batch and position."""
    return x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1]), dout.sum(axis=(0, 1))


class GPT:
    def __init__(self, cfg, seed=0, dtype=np.float32):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        C, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer

        def w(*shape, std=0.02):
            return (rng.standard_normal(shape) * std).astype(dtype)

        def ones(n):
            return np.ones(n, dtype)

        def zeros(n):
            return np.zeros(n, dtype)

        resid_std = 0.02 / math.sqrt(2 * L)  # keeps the residual stream from growing with depth
        p = {"wte": w(V, C), "wpe": w(cfg.block_size, C), "lnf_g": ones(C), "lnf_b": zeros(C)}
        for l in range(L):
            p.update({
                f"{l}.ln1_g": ones(C), f"{l}.ln1_b": zeros(C),
                f"{l}.w_qkv": w(C, 3 * C), f"{l}.b_qkv": zeros(3 * C),
                f"{l}.w_o": w(C, C, std=resid_std), f"{l}.b_o": zeros(C),
                f"{l}.ln2_g": ones(C), f"{l}.ln2_b": zeros(C),
                f"{l}.w_fc": w(C, 4 * C), f"{l}.b_fc": zeros(4 * C),
                f"{l}.w_proj": w(4 * C, C, std=resid_std), f"{l}.b_proj": zeros(C),
            })
        self.params = p
        self.mask = np.tril(np.ones((cfg.block_size, cfg.block_size), dtype=bool))
        self.trace = None
        self._cache = None

    def num_params(self):
        return sum(v.size for v in self.params.values())

    # --- attention -----------------------------------------------------------

    def _attention(self, h, l):
        """Each position compares its query with the keys of earlier positions and
        takes a weighted average of their values."""
        p = self.params
        B, T, C = h.shape
        H = self.cfg.n_head
        hs = C // H
        qkv = mm(h, p[f"{l}.w_qkv"]) + p[f"{l}.b_qkv"]
        # (B, T, C) -> (B, heads, T, head_size) for each of q, k, v
        q, k, v = (t.reshape(B, T, H, hs).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, axis=-1))
        scale = 1.0 / math.sqrt(hs)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        # Causal mask: a character may not look at characters that come after it.
        scores = np.where(self.mask[:T, :T], scores, -1e9)
        att = softmax(scores)                                 # (B, H, T, T)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)  # heads side by side again
        out = mm(y, p[f"{l}.w_o"]) + p[f"{l}.b_o"]
        return out, (h, q, k, v, att, y, scale)

    def _attention_backward(self, dout, cache, l, grads):
        h, q, k, v, att, y, scale = cache
        B, H, T, hs = q.shape
        C = H * hs
        grads[f"{l}.w_o"], grads[f"{l}.b_o"] = matmul_grads(y, dout)
        dy = mm(dout, self.params[f"{l}.w_o"].T).reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True)) * scale  # softmax backward
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, C) for t in (dq, dk, dv)], axis=-1)
        grads[f"{l}.w_qkv"], grads[f"{l}.b_qkv"] = matmul_grads(h, dqkv)
        return mm(dqkv, self.params[f"{l}.w_qkv"].T)

    # --- MLP -----------------------------------------------------------------

    def _mlp(self, h, l):
        """Per-position two-layer network: where most of the 'knowledge' is stored."""
        p = self.params
        pre = mm(h, p[f"{l}.w_fc"]) + p[f"{l}.b_fc"]
        act, t = gelu(pre)
        return mm(act, p[f"{l}.w_proj"]) + p[f"{l}.b_proj"], (h, pre, act, t)

    def _mlp_backward(self, dout, cache, l, grads):
        h, pre, act, t = cache
        grads[f"{l}.w_proj"], grads[f"{l}.b_proj"] = matmul_grads(act, dout)
        dpre = mm(dout, self.params[f"{l}.w_proj"].T) * gelu_grad(pre, t)
        grads[f"{l}.w_fc"], grads[f"{l}.b_fc"] = matmul_grads(h, dpre)
        return mm(dpre, self.params[f"{l}.w_fc"].T)

    # --- whole network -------------------------------------------------------

    def forward(self, idx, targets=None, record=False):
        """idx: (B, T) array of token ids. Returns (logits, loss).

        With targets, the intermediate values are kept so backward() can run.
        With record=True, self.trace is filled for visualization.
        """
        p = self.params
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} exceeds block_size {self.cfg.block_size}"
        x = p["wte"][idx] + p["wpe"][:T]
        residuals, block_caches = [x], []
        for l in range(self.cfg.n_layer):
            h, ln1 = layernorm(x, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
            a, att = self._attention(h, l)
            x = x + a
            h, ln2 = layernorm(x, p[f"{l}.ln2_g"], p[f"{l}.ln2_b"])
            m, mlp = self._mlp(h, l)
            x = x + m
            residuals.append(x)
            block_caches.append((ln1, att, ln2, mlp))
        xf, lnf = layernorm(x, p["lnf_g"], p["lnf_b"])
        logits = mm(xf, p["wte"].T)  # reuse the embedding matrix as the output layer ("weight tying")

        if record:
            self.trace = {
                "residuals": residuals,                           # n_layer + 1 arrays of (B, T, C)
                "attn": [c[1][4] for c in block_caches],          # (B, heads, T, T)
                "mlp": [c[3][2] for c in block_caches],           # (B, T, 4C)
                # "Logit lens": what would the model predict if it stopped after each layer?
                "layer_logits": [mm(layernorm(r, p["lnf_g"], p["lnf_b"])[0], p["wte"].T) for r in residuals],
            }

        loss = None
        if targets is not None:
            probs = softmax(logits)
            b, t = np.indices(targets.shape)
            loss = float(-np.log(probs[b, t, targets] + 1e-12).mean())
            self._cache = (idx, targets, probs, xf, lnf, block_caches)
        return logits, loss

    def backward(self):
        """Gradients of the last forward() loss with respect to every parameter."""
        idx, targets, probs, xf, lnf, block_caches = self._cache
        p = self.params
        B, T = idx.shape
        grads = {}

        # Cross-entropy + softmax: gradient is (predicted probability - 1 for the right answer).
        dlogits = probs.copy()
        b, t = np.indices(targets.shape)
        dlogits[b, t, targets] -= 1
        dlogits /= B * T

        grads["wte"] = dlogits.reshape(-1, dlogits.shape[-1]).T @ xf.reshape(-1, xf.shape[-1])
        dx = mm(dlogits, p["wte"])
        dx, grads["lnf_g"], grads["lnf_b"] = layernorm_backward(dx, lnf)

        for l in reversed(range(self.cfg.n_layer)):
            ln1, att, ln2, mlp = block_caches[l]
            # x_out = x_mid + mlp(ln2(x_mid)): gradient flows both through the MLP and straight past it.
            dh, grads[f"{l}.ln2_g"], grads[f"{l}.ln2_b"] = layernorm_backward(
                self._mlp_backward(dx, mlp, l, grads), ln2)
            dx = dx + dh
            # x_mid = x_in + attention(ln1(x_in))
            dh, grads[f"{l}.ln1_g"], grads[f"{l}.ln1_b"] = layernorm_backward(
                self._attention_backward(dx, att, l, grads), ln1)
            dx = dx + dh

        grads["wpe"] = np.zeros_like(p["wpe"])
        grads["wpe"][:T] = dx.sum(axis=0)
        np.add.at(grads["wte"], idx, dx)  # embedding lookup: add each position's gradient to its row
        return grads

    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=None, rng=None, on_step=None):
        """Append max_new_tokens predicted token ids to the list ids, one at a time.

        on_step(context_ids, probs, chosen_id) is called after each token, with
        self.trace filled, so a caller can print or draw every step.
        """
        rng = rng or np.random.default_rng()
        ids = list(ids)
        for _ in range(max_new_tokens):
            ctx = ids[-self.cfg.block_size:]
            logits, _ = self.forward(np.array([ctx]), record=on_step is not None)
            logits = logits[0, -1].astype(np.float64) / max(temperature, 1e-6)
            if top_k:
                kth = np.sort(logits)[-min(top_k, logits.size)]
                logits[logits < kth] = -np.inf
            probs = softmax(logits)
            nxt = int(rng.choice(len(probs), p=probs))
            if on_step is not None:
                on_step(ctx, probs, nxt)
            ids.append(nxt)
        return ids


class Adam:
    """The standard optimizer for transformers: per-weight adaptive step sizes,
    plus decoupled weight decay on the weight matrices (AdamW)."""

    def __init__(self, params, betas=(0.9, 0.99), weight_decay=0.1, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.b1, self.b2 = betas
        self.wd, self.eps, self.t = weight_decay, eps, 0

    def step(self, params, grads, lr, clip=1.0):
        # Gradient clipping: if one batch produces a huge gradient, scale it down.
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-6))
        self.t += 1
        for k, w in params.items():
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            if w.ndim == 2:
                w -= lr * self.wd * w
            w -= (lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(w.dtype)
        return norm


def save_checkpoint(path, model, tokenizer, extra=None):
    with open(path, "wb") as f:
        pickle.dump({"config": asdict(model.cfg), "chars": tokenizer.chars,
                     "params": model.params, **(extra or {})}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.params = ckpt["params"]
    return model, CharTokenizer(ckpt["chars"]), ckpt
