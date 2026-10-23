"""Checks that the hand-written backward pass matches numerical gradients, and that training learns."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, Adam, CharTokenizer, GPTConfig  # noqa: E402


def test_gradients_match_finite_differences():
    cfg = GPTConfig(vocab_size=7, block_size=6, n_layer=2, n_head=2, n_embd=8)
    model = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(0)
    # Random (non-default) LayerNorm parameters so their gradients are tested properly.
    for k, v in model.params.items():
        if "ln" in k:
            v[:] = rng.standard_normal(v.shape)
    x = rng.integers(0, 7, (3, 6))
    y = rng.integers(0, 7, (3, 6))
    model.forward(x, y)
    grads = model.backward()
    eps = 1e-6
    for name, w in model.params.items():
        for i in rng.choice(w.size, size=min(5, w.size), replace=False):
            old = w.flat[i]
            w.flat[i] = old + eps
            up = model.forward(x, y)[1]
            w.flat[i] = old - eps
            down = model.forward(x, y)[1]
            w.flat[i] = old
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - grads[name].flat[i]) < 1e-6 + 1e-4 * abs(numeric), name


def test_training_memorizes_a_short_text():
    text = "the cat sat on the mat. " * 20
    tok = CharTokenizer.from_text(text)
    data = np.array(tok.encode(text))
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32))
    opt = Adam(model.params)
    rng = np.random.default_rng(0)
    for step in range(200):
        ix = rng.integers(0, len(data) - 17, 16)
        x = np.stack([data[i:i + 16] for i in ix])
        y = np.stack([data[i + 1:i + 17] for i in ix])
        _, loss = model.forward(x, y)
        if step == 0:
            first = loss
        opt.step(model.params, model.backward(), lr=3e-3)
    assert loss < 0.3 * first
    out = tok.decode(model.generate(tok.encode("the c"), 10, temperature=0.1, rng=rng))
    assert out.startswith("the cat sat")
