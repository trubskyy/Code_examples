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

# ============================================================================
# PYTHON NOTES FOR THIS FILE
# ----------------------------------------------------------------------------
# * A string in triple quotes ("""...""") as the first statement of a file,
#   function or class is its "docstring": built-in documentation, not code.
#   Lines starting with # are ordinary comments.
# * Python has no braces: indentation (4 spaces) marks where a block
#   (function body, loop body, if-branch) begins and ends.
# * Variables are not declared with types. `x = 5` creates x.
# * `import numpy as np` loads the NumPy library and lets us call it `np`.
#   NumPy provides fast n-dimensional arrays ("ndarray"): think of a C array of
#   floats with a shape such as (32, 64, 96), plus whole-array operations
#   implemented in C. `a + b`, `a * b`, `np.exp(a)` work element by element on
#   entire arrays at once, so we almost never write loops over numbers.
# * `a @ b` is matrix multiplication (for arrays). `a.T` is the transpose.
# * Broadcasting: when shapes differ, NumPy stretches size-1 (or missing)
#   dimensions to match. E.g. (32, 64, 96) + (96,) adds the same 96-vector to
#   every one of the 32*64 rows. This is used everywhere for biases.
# * `axis=-1` means "the last dimension". Negative indices count from the end
#   in Python: a[-1] is the last element, a[-3:] the last three.
# * Slicing: a[start:stop] takes elements start..stop-1. Either end can be
#   omitted: a[:T] = first T elements, a[n:] = from n to the end.
# ============================================================================

import math                             # standard-library maths: sqrt, pi, cos...
import pickle                           # standard-library object serialisation (save/load any Python object)
from dataclasses import asdict, dataclass  # helper to make simple "struct"-like classes

import numpy as np


# `@dataclass` is a decorator: a function that modifies the class defined right
# below it. Here it auto-generates a constructor, so
#     GPTConfig(vocab_size=80, n_layer=4)
# works and fills in the defaults for anything not given. The `name: type = value`
# lines declare fields; the type annotations are documentation only — Python
# does not enforce them.
@dataclass
class GPTConfig:
    vocab_size: int = 96
    block_size: int = 64    # how many characters the model can look back at
    n_layer: int = 3        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 96        # width of the vector that represents each character


class CharTokenizer:
    """Maps each distinct character in the training text to an integer."""

    # `__init__` is the constructor. Every method receives the object itself as
    # its first parameter, conventionally named `self` (like `this` in C++/Java,
    # but explicit). `self.chars = ...` creates/sets a field on the object.
    def __init__(self, chars):
        self.chars = list(chars)   # index -> character, e.g. chars[5] == 'a'
        # A "dict comprehension": builds a hash map in one expression.
        # `enumerate(xs)` yields (index, value) pairs, so this produces
        # {'\n': 0, ' ': 1, '!': 2, ...}: character -> index.
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    # `@classmethod` makes this a factory method called on the class itself:
    # `CharTokenizer.from_text("hello")`. `cls` is the class (like a static
    # method that knows which class it belongs to).
    @classmethod
    def from_text(cls, text):
        # set(text) = the distinct characters; sorted() = a sorted list of them.
        return cls(sorted(set(text)))

    # `@property` lets this method be read like a field: `tok.vocab_size`
    # (no parentheses).
    @property
    def vocab_size(self):
        return len(self.chars)   # len() is the built-in "length of" function

    def encode(self, text):
        # A "list comprehension": [expression for item in sequence if condition]
        # builds a new list. Iterating over a string yields its characters.
        # Characters never seen in training are dropped rather than crashing.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        # "".join(list_of_strings) concatenates them with "" between each.
        # The argument here is a "generator expression" — a list comprehension
        # without the brackets, which produces items lazily.
        return "".join(self.chars[i] for i in ids)


# ----------------------------------------------------------------------------
# Building blocks: each has a forward and a backward.
#
# Shape conventions used throughout (all are NumPy array dimensions):
#   B = batch size (how many text windows we process at once)
#   T = time / sequence length (characters per window)
#   C = channels = n_embd (length of the vector representing one character)
#   H = number of attention heads,  hs = C // H = size of each head
#   V = vocabulary size (number of distinct characters)
# ----------------------------------------------------------------------------

def softmax(x):
    """Turn a vector of arbitrary scores into probabilities that sum to 1.

    softmax(x)_i = exp(x_i) / sum_j exp(x_j). Subtracting the max first doesn't
    change the result but prevents exp() overflowing for large scores.
    """
    # keepdims=True keeps the reduced axis as size 1, e.g. (B, T, V) -> (B, T, 1),
    # so the result broadcasts back against x correctly.
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    """Rescale each vector to mean 0 and variance 1, then apply a learned scale and shift.

    Keeps the numbers flowing through a deep network in a sensible range.
    `eps=1e-5` is a default argument: callers may omit it.
    Returns two things: the output and a "cache" of values backward() will need.
    """
    mu = x.mean(-1, keepdims=True)                     # mean of each vector
    rstd = 1.0 / np.sqrt(x.var(-1, keepdims=True) + eps)  # 1 / standard deviation
    xhat = (x - mu) * rstd                              # normalised: mean 0, variance 1
    # Python can return several values at once; they come back as a "tuple"
    # (an immutable list), which the caller unpacks: `out, cache = layernorm(...)`.
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dout, cache):
    """Given dLoss/d(output), return dLoss/d(input), dLoss/dg, dLoss/db.

    Naming convention for all backward functions: `dfoo` means
    "the gradient of the loss with respect to foo" — how much the loss would
    change if foo changed slightly. It always has the same shape as foo.
    """
    xhat, rstd, g = cache                     # tuple unpacking
    dg = (dout * xhat).sum(axis=(0, 1))       # sum over batch and time: g is shared by all positions
    db = dout.sum(axis=(0, 1))
    dxhat = dout * g
    # The derivative of normalisation (standard result; worked out with the chain
    # rule through the mean and the variance).
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


GELU_C = math.sqrt(2 / math.pi)   # module-level constant (ALL_CAPS is just a naming convention)


def gelu(x):
    """Smooth version of ReLU, used by GPT. Also returns the tanh, which the backward pass reuses.

    This is the non-linearity inside the MLP. Without some non-linear function,
    stacking layers of matrix multiplications would collapse into a single
    matrix multiplication and the network couldn't learn complex patterns.
    (x * x * x is used instead of x ** 3 because NumPy computes it much faster.)
    """
    t = np.tanh(GELU_C * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1 + t), t


def gelu_grad(x, t):
    """Derivative of gelu at x (t is the tanh value computed in the forward pass)."""
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x)


def mm(x, W):
    """x @ W for x of shape (B, T, n). Flattening to one 2-D product lets NumPy hand it
    to the fast BLAS routine in one go, which is ~40x faster than a batched 3-D matmul."""
    # reshape(-1, n): -1 means "whatever size makes the total element count work",
    # so (B, T, n) becomes (B*T, n). No data is copied; it's a different view.
    # `*x.shape[:-1]` unpacks the tuple (B, T) into separate arguments, so the
    # final reshape is reshape(B, T, W.shape[-1]).
    return (x.reshape(-1, x.shape[-1]) @ W).reshape(*x.shape[:-1], W.shape[-1])


def matmul_grads(x, dout):
    """Weight gradient of y = x @ W + b, summing over every batch and position.

    If y = x @ W + b, then dW = x^T @ dy and db = sum of dy over all rows.
    """
    return x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1]), dout.sum(axis=(0, 1))


class GPT:
    """The whole language model: holds the weights and runs forward/backward."""

    # `seed=0, dtype=np.float32` are default argument values. float32 halves the
    # memory and doubles the speed compared with float64, and is precise enough.
    def __init__(self, cfg, seed=0, dtype=np.float32):
        self.cfg = cfg
        # A seeded random number generator: same seed -> same initial weights.
        rng = np.random.default_rng(seed)
        C, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer   # multiple assignment

        # Functions can be defined inside functions. These small helpers create
        # weight arrays. `*shape` collects any number of positional arguments
        # into a tuple, so w(V, C) gets shape == (V, C).
        def w(*shape, std=0.02):
            # Small random normal values: start near zero but not exactly zero
            # (identical weights would all learn the same thing).
            return (rng.standard_normal(shape) * std).astype(dtype)

        def ones(n):
            return np.ones(n, dtype)

        def zeros(n):
            return np.zeros(n, dtype)

        resid_std = 0.02 / math.sqrt(2 * L)  # keeps the residual stream from growing with depth

        # All weights live in one dict (hash map) from name to array. That makes
        # the optimizer, saving and loading trivial: just loop over the dict.
        p = {
            "wte": w(V, C),                 # token embedding: one C-vector per character
            "wpe": w(cfg.block_size, C),    # position embedding: one C-vector per position
            "lnf_g": ones(C), "lnf_b": zeros(C),  # final LayerNorm scale and shift
        }
        for l in range(L):   # range(L) yields 0, 1, ..., L-1
            # f-strings: f"{l}.ln1_g" substitutes the value of l, giving "0.ln1_g",
            # "1.ln1_g", ... so each block gets its own uniquely named weights.
            # dict.update() adds all these entries to p.
            p.update({
                f"{l}.ln1_g": ones(C), f"{l}.ln1_b": zeros(C),
                # Attention: one matrix produces query, key and value at once (3*C wide).
                f"{l}.w_qkv": w(C, 3 * C), f"{l}.b_qkv": zeros(3 * C),
                f"{l}.w_o": w(C, C, std=resid_std), f"{l}.b_o": zeros(C),
                f"{l}.ln2_g": ones(C), f"{l}.ln2_b": zeros(C),
                # MLP: expand to 4*C neurons, then project back down to C.
                f"{l}.w_fc": w(C, 4 * C), f"{l}.b_fc": zeros(4 * C),
                f"{l}.w_proj": w(4 * C, C, std=resid_std), f"{l}.b_proj": zeros(C),
            })
        self.params = p
        # Lower-triangular True/False matrix: mask[i, j] is True when j <= i,
        # i.e. position i is allowed to look at position j.
        self.mask = np.tril(np.ones((cfg.block_size, cfg.block_size), dtype=bool))
        self.trace = None    # `None` is Python's null
        self._cache = None   # a leading underscore means "internal, not for outside use" (convention only)

    def num_params(self):
        # .values() iterates over a dict's values; .size is an array's element count.
        return sum(v.size for v in self.params.values())

    # --- attention -----------------------------------------------------------

    def _attention(self, h, l):
        """Each position compares its query with the keys of earlier positions and
        takes a weighted average of their values.

        Intuition: every character asks a question (query), advertises what it
        contains (key), and offers some information (value). A character gathers
        information from the earlier characters whose keys best match its query.
        Several "heads" do this in parallel, each free to look for something different.
        """
        p = self.params
        B, T, C = h.shape    # unpack the shape tuple into three variables
        H = self.cfg.n_head
        hs = C // H          # `//` is integer division
        qkv = mm(h, p[f"{l}.w_qkv"]) + p[f"{l}.b_qkv"]   # (B, T, 3C)
        # np.split cuts the last axis into 3 equal parts: q, k, v, each (B, T, C).
        # Then each is reshaped to (B, T, H, hs) and transposed to (B, H, T, hs):
        # transpose(0, 2, 1, 3) means "new axis order = old axes 0, 2, 1, 3".
        # The generator expression returns three results that are unpacked into q, k, v.
        q, k, v = (t.reshape(B, T, H, hs).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, axis=-1))
        scale = 1.0 / math.sqrt(hs)   # keeps dot products from getting too large
        # @ on 4-D arrays multiplies the last two dims for every (batch, head) pair:
        # (B, H, T, hs) @ (B, H, hs, T) -> (B, H, T, T). scores[b, h, i, j] = how much
        # position i's query matches position j's key.
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        # Causal mask: a character may not look at characters that come after it.
        # np.where(cond, a, b) picks a where cond is True, else b. A score of -1e9
        # becomes probability 0 after softmax.
        scores = np.where(self.mask[:T, :T], scores, -1e9)
        att = softmax(scores)                                 # (B, H, T, T); each row sums to 1
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)  # weighted sum of values; heads side by side again
        out = mm(y, p[f"{l}.w_o"]) + p[f"{l}.b_o"]           # mix the heads' results together
        return out, (h, q, k, v, att, y, scale)

    def _attention_backward(self, dout, cache, l, grads):
        """Reverse of _attention. Writes weight gradients into `grads` (a dict,
        passed by reference, so changes are visible to the caller) and returns
        the gradient with respect to the input h."""
        h, q, k, v, att, y, scale = cache
        B, H, T, hs = q.shape
        C = H * hs
        # Each step below undoes one line of the forward pass, last line first.
        grads[f"{l}.w_o"], grads[f"{l}.b_o"] = matmul_grads(y, dout)
        dy = mm(dout, self.params[f"{l}.w_o"].T).reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        # y = att @ v  ->  datt = dy @ v^T,  dv = att^T @ dy
        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        # Softmax backward: d(scores) = att * (datt - sum(datt * att)) per row.
        # Masked positions have att == 0, so they automatically get zero gradient.
        dscores = att * (datt - (datt * att).sum(-1, keepdims=True)) * scale
        # scores = q @ k^T  ->  dq = dscores @ k,  dk = dscores^T @ q
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        # Undo the head split: back to (B, T, C) each, then glue q, k, v
        # gradients side by side to match the (B, T, 3C) qkv array.
        dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, C) for t in (dq, dk, dv)], axis=-1)
        grads[f"{l}.w_qkv"], grads[f"{l}.b_qkv"] = matmul_grads(h, dqkv)
        return mm(dqkv, self.params[f"{l}.w_qkv"].T)

    # --- MLP -----------------------------------------------------------------

    def _mlp(self, h, l):
        """Per-position two-layer network: where most of the 'knowledge' is stored.

        Unlike attention, the MLP looks at one position at a time. It turns the
        information gathered by attention into a better guess (e.g. "after 'th'
        and a vowel, an 'e' or 'r' is likely").
        """
        p = self.params
        pre = mm(h, p[f"{l}.w_fc"]) + p[f"{l}.b_fc"]   # (B, T, 4C): expand
        act, t = gelu(pre)                               # non-linearity: the "neurons firing"
        return mm(act, p[f"{l}.w_proj"]) + p[f"{l}.b_proj"], (h, pre, act, t)   # (B, T, C): project back

    def _mlp_backward(self, dout, cache, l, grads):
        h, pre, act, t = cache
        grads[f"{l}.w_proj"], grads[f"{l}.b_proj"] = matmul_grads(act, dout)
        # Chain rule through gelu: multiply by its derivative element by element.
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
        # `assert cond, message` raises an error if cond is false. Used for sanity checks.
        assert T <= self.cfg.block_size, f"sequence of {T} exceeds block_size {self.cfg.block_size}"
        # Fancy indexing: indexing an array with an array of integers.
        # p["wte"][idx] replaces every id in the (B, T) array with its row of the
        # embedding table, giving (B, T, C). Position vectors (T, C) are added by
        # broadcasting to every sequence in the batch.
        x = p["wte"][idx] + p["wpe"][:T]
        residuals, block_caches = [x], []    # two lists; [] is an empty list
        for l in range(self.cfg.n_layer):
            # Each block READS from x and ADDS its result back to x (the
            # "residual stream"). Adding rather than replacing lets information
            # and gradients flow easily through many layers.
            h, ln1 = layernorm(x, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
            a, att = self._attention(h, l)
            x = x + a
            h, ln2 = layernorm(x, p[f"{l}.ln2_g"], p[f"{l}.ln2_b"])
            m, mlp = self._mlp(h, l)
            x = x + m
            residuals.append(x)                        # list.append adds to the end
            block_caches.append((ln1, att, ln2, mlp))  # a tuple of four caches
        xf, lnf = layernorm(x, p["lnf_g"], p["lnf_b"])
        # Logits: one score per possible next character, at every position.
        # Reuse the embedding matrix as the output layer ("weight tying"): the
        # score for character c is the dot product of the final vector with
        # c's embedding — "how much does this look like c?"
        logits = mm(xf, p["wte"].T)  # (B, T, V)

        if record:
            # Save everything interesting for the visualizer. c[1][4] picks the
            # 5th item (att) from the attention cache inside each block cache.
            self.trace = {
                "residuals": residuals,                           # n_layer + 1 arrays of (B, T, C)
                "attn": [c[1][4] for c in block_caches],          # (B, heads, T, T)
                "mlp": [c[3][2] for c in block_caches],           # (B, T, 4C)
                # "Logit lens": what would the model predict if it stopped after each layer?
                "layer_logits": [mm(layernorm(r, p["lnf_g"], p["lnf_b"])[0], p["wte"].T) for r in residuals],
            }

        loss = None
        if targets is not None:   # `is not None` is the idiomatic null check
            probs = softmax(logits)
            # np.indices(shape) returns arrays of row and column indices, so
            # probs[b, t, targets] picks, for every (batch, position), the
            # probability the model gave to the CORRECT next character.
            b, t = np.indices(targets.shape)
            # Cross-entropy loss: average of -log(probability of the right answer).
            # 0 if the model is certain and right; large if it was confident and wrong.
            # The 1e-12 avoids log(0).
            loss = float(-np.log(probs[b, t, targets] + 1e-12).mean())
            self._cache = (idx, targets, probs, xf, lnf, block_caches)
        return logits, loss

    def backward(self):
        """Gradients of the last forward() loss with respect to every parameter.

        Walks the network from the loss back to the input, undoing each step of
        forward() in reverse order and applying the chain rule.
        Returns a dict with the same keys and shapes as self.params.
        """
        idx, targets, probs, xf, lnf, block_caches = self._cache
        p = self.params
        B, T = idx.shape
        grads = {}   # empty dict

        # Cross-entropy + softmax: gradient is (predicted probability - 1 for the right answer).
        # .copy() is needed: arrays are references, and we must not modify `probs` in place.
        dlogits = probs.copy()
        b, t = np.indices(targets.shape)
        dlogits[b, t, targets] -= 1
        dlogits /= B * T     # because the loss was an average over B*T predictions

        # logits = xf @ wte^T, so wte gets a gradient from here (and again from the
        # embedding lookup at the end, since the matrix is used twice).
        grads["wte"] = dlogits.reshape(-1, dlogits.shape[-1]).T @ xf.reshape(-1, xf.shape[-1])
        dx = mm(dlogits, p["wte"])
        dx, grads["lnf_g"], grads["lnf_b"] = layernorm_backward(dx, lnf)

        # reversed(range(n)) yields n-1, ..., 1, 0: last block first.
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

        # x = wte[idx] + wpe[:T]: the position embedding gets the gradient summed over the batch.
        grads["wpe"] = np.zeros_like(p["wpe"])   # zeros with the same shape and type
        grads["wpe"][:T] = dx.sum(axis=0)
        # Embedding lookup: add each position's gradient to its row. np.add.at is
        # used instead of `grads["wte"][idx] += dx` because the same character can
        # appear many times, and plain `+=` would only count it once.
        np.add.at(grads["wte"], idx, dx)
        return grads

    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=None, rng=None, on_step=None):
        """Append max_new_tokens predicted token ids to the list ids, one at a time.

        on_step(context_ids, probs, chosen_id) is called after each token, with
        self.trace filled, so a caller can print or draw every step.
        (Functions are values in Python and can be passed around like any other.)
        """
        # `a or b` returns a if a is "truthy" (not None/0/empty), otherwise b.
        rng = rng or np.random.default_rng()
        ids = list(ids)   # copy, so the caller's list isn't modified
        for _ in range(max_new_tokens):    # `_` is the convention for an unused loop variable
            ctx = ids[-self.cfg.block_size:]   # only the last block_size characters fit
            # np.array([ctx]) makes a (1, T) array: a batch containing one sequence.
            logits, _ = self.forward(np.array([ctx]), record=on_step is not None)
            # [0, -1] = first (only) sequence, last position: the prediction for
            # the character that comes next. Use float64 for the sampling maths.
            # Temperature: dividing scores by <1 sharpens the distribution (safer
            # choices), by >1 flattens it (more surprising choices).
            logits = logits[0, -1].astype(np.float64) / max(temperature, 1e-6)
            if top_k:
                # Keep only the k highest scores; set the rest to -infinity so
                # they get probability 0. np.sort is ascending, so [-k] is the k-th largest.
                kth = np.sort(logits)[-min(top_k, logits.size)]
                logits[logits < kth] = -np.inf   # boolean-mask assignment: only where the condition holds
            probs = softmax(logits)
            # Pick a random character, weighted by its probability.
            nxt = int(rng.choice(len(probs), p=probs))
            if on_step is not None:
                on_step(ctx, probs, nxt)
            ids.append(nxt)
        return ids


class Adam:
    """The standard optimizer for transformers: per-weight adaptive step sizes,
    plus decoupled weight decay on the weight matrices (AdamW).

    Plain gradient descent does w -= lr * gradient. Adam additionally keeps,
    for every weight, a running average of its gradient (m, "momentum": smooths
    out noise between batches) and of its squared gradient (v: lets weights with
    consistently small gradients take relatively bigger steps).
    """

    # betas=(0.9, 0.99) is a tuple default value.
    def __init__(self, params, betas=(0.9, 0.99), weight_decay=0.1, eps=1e-8):
        # One zero-filled array per parameter, same shape. `.items()` yields (key, value) pairs.
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
            # Bias correction: m and v start at 0, so early averages are too small.
            m_hat = self.m[k] / (1 - self.b1 ** self.t)   # ** is exponentiation
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            # `w -= ...` modifies the array in place, so the model's params dict
            # (which holds the same array object) sees the update.
            if w.ndim == 2:
                # Weight decay: shrink weight matrices slightly every step, which
                # discourages the model from relying on a few huge weights.
                w -= lr * self.wd * w
            w -= (lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(w.dtype)
        return norm


def save_checkpoint(path, model, tokenizer, extra=None):
    """Write the model's config, vocabulary and weights to one file."""
    # `with` opens a resource and guarantees it is closed when the block ends,
    # even if an error happens (like try/finally). "wb" = write, binary mode.
    with open(path, "wb") as f:
        # `**(extra or {})` unpacks the extra dict's entries into this dict literal.
        pickle.dump({"config": asdict(model.cfg), "chars": tokenizer.chars,
                     "params": model.params, **(extra or {})}, f)


def load_checkpoint(path):
    """Read a file written by save_checkpoint. Returns (model, tokenizer, raw dict)."""
    # Note: pickle can run code while loading, so only load checkpoints you trust.
    with open(path, "rb") as f:   # "rb" = read, binary mode
        ckpt = pickle.load(f)
    # GPTConfig(**d) passes every key of the dict d as a named argument.
    model = GPT(GPTConfig(**ckpt["config"]))
    model.params = ckpt["params"]
    return model, CharTokenizer(ckpt["chars"]), ckpt
