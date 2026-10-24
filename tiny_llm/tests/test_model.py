"""Checks that the hand-written backward pass matches numerical gradients, and that training learns.

Run with:  python -m pytest -q tests
pytest finds every function whose name starts with `test_` and runs it. A test
passes if it finishes; it fails if an `assert` is false or an error is raised.
"""

import sys
from pathlib import Path

import numpy as np

# Make `import model` work when pytest runs from another folder: add this test
# file's parent's parent folder (tiny_llm/) to the list of places Python searches
# for modules. __file__ is the path of this file; .parents[1] goes up two levels.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, Adam, CharTokenizer, GPTConfig  # noqa: E402  (tells the linter this late import is intended)


def test_gradients_match_finite_differences():
    """The classic way to check backpropagation code.

    For a weight w, the gradient says how fast the loss changes when w changes.
    We can measure that directly: nudge w up by a tiny eps, nudge it down, and
    compare the losses: (loss(w + eps) - loss(w - eps)) / (2 * eps). If that
    matches what backward() computed, the hand-written calculus is right.
    """
    # A very small model so the check is fast. float64 because finite differences
    # need more precision than float32 offers.
    cfg = GPTConfig(vocab_size=7, block_size=6, n_layer=2, n_head=2, n_embd=8)
    model = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(0)
    # Random (non-default) LayerNorm parameters so their gradients are tested properly.
    for k, v in model.params.items():
        if "ln" in k:                              # `in` on strings tests for a substring
            v[:] = rng.standard_normal(v.shape)   # v[:] = ... overwrites the array's contents in place
    x = rng.integers(0, 7, (3, 6))   # random input ids, shape (3, 6)
    y = rng.integers(0, 7, (3, 6))   # random target ids
    model.forward(x, y)
    grads = model.backward()
    eps = 1e-6
    for name, w in model.params.items():
        # Check 5 randomly chosen entries of every parameter array.
        # w.flat views the array as 1-D, so w.flat[i] works for any shape.
        for i in rng.choice(w.size, size=min(5, w.size), replace=False):
            old = w.flat[i]
            w.flat[i] = old + eps
            up = model.forward(x, y)[1]     # [1] = the loss
            w.flat[i] = old - eps
            down = model.forward(x, y)[1]
            w.flat[i] = old                 # restore
            numeric = (up - down) / (2 * eps)
            # `assert condition, message`: fail with the parameter name if they differ.
            assert abs(numeric - grads[name].flat[i]) < 1e-6 + 1e-4 * abs(numeric), name


def test_training_memorizes_a_short_text():
    """Train briefly on a repetitive sentence: the loss must drop and the model must reproduce it."""
    text = "the cat sat on the mat. " * 20   # * on a string repeats it
    tok = CharTokenizer.from_text(text)
    data = np.array(tok.encode(text))
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32))
    opt = Adam(model.params)
    rng = np.random.default_rng(0)
    for step in range(200):
        # The same batching as train.py's get_batch: random 16-character windows,
        # targets shifted by one.
        ix = rng.integers(0, len(data) - 17, 16)
        x = np.stack([data[i:i + 16] for i in ix])
        y = np.stack([data[i + 1:i + 17] for i in ix])
        _, loss = model.forward(x, y)
        if step == 0:
            first = loss
        opt.step(model.params, model.backward(), lr=3e-3)
    assert loss < 0.3 * first   # loss fell by more than 70%
    # Low temperature = nearly always pick the most likely character.
    out = tok.decode(model.generate(tok.encode("the c"), 10, temperature=0.1, rng=rng))
    assert out.startswith("the cat sat")
