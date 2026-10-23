"""A small GPT-style language model that works on characters.

This is the same architecture as GPT-2 and most modern LLMs, only much smaller:

    text -> characters -> token ids
         -> token embedding + position embedding
         -> N x [ LayerNorm -> causal self-attention -> + residual
                  LayerNorm -> MLP (feed-forward)    -> + residual ]
         -> LayerNorm -> linear "unembedding" -> probability of each next character

Every module can record what it computed (`record=True`) so that visualize.py can
draw the network while it answers a prompt.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 96
    block_size: int = 128   # how many characters the model can look back at
    n_layer: int = 4        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # width of the vector that represents each character
    dropout: float = 0.1


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


class CausalSelfAttention(nn.Module):
    """Each position looks at earlier positions and pulls in information from them."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))
        self.last_attn = None

    def forward(self, x, record=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))

        # How well does each query match each key? Scaled dot product.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # Causal mask: a character may not look at characters that come after it.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if record:
            self.last_attn = att.detach()
        y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Per-position feed-forward network: where most of the 'knowledge' is stored."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.last_hidden = None

    def forward(self, x, record=False):
        h = F.gelu(self.fc(x))
        if record:
            self.last_hidden = h.detach()
        return self.drop(self.proj(h))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, record=False):
        x = x + self.attn(self.ln1(x), record)
        x = x + self.mlp(self.ln2(x), record)
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: fewer parameters, trains faster
        self.apply(self._init_weights)
        self.trace = None

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, record=False):
        """Returns (logits, loss). With record=True, fills self.trace for visualization."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        residuals = [x.detach()]
        for block in self.blocks:
            x = block(x, record)
            residuals.append(x.detach())
        logits = self.head(self.ln_f(x))

        if record:
            self.trace = {
                "residuals": residuals,  # n_layer + 1 tensors of (B, T, C)
                "attn": [b.attn.last_attn for b in self.blocks],  # (B, heads, T, T)
                "mlp": [b.mlp.last_hidden for b in self.blocks],  # (B, T, 4C)
                # "Logit lens": what would the model predict if it stopped after each layer?
                "layer_logits": [self.head(self.ln_f(r)).detach() for r in residuals],
                "logits": logits.detach(),
            }

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, on_step=None):
        """Append max_new_tokens predicted tokens to idx, one at a time.

        on_step(context_ids, probs, chosen_id) is called after each token, with
        self.trace filled, so a caller can draw every step.
        """
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx, record=on_step is not None)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if on_step is not None:
                on_step(ctx[0].tolist(), probs[0], nxt.item())
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def save_checkpoint(path, model, tokenizer, extra=None):
    torch.save(
        {
            "config": asdict(model.cfg),
            "chars": tokenizer.chars,
            "state_dict": model.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, CharTokenizer(ckpt["chars"]), ckpt


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
