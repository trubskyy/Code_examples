"""Train the tiny GPT on one or more books (plain .txt files).

    python train.py books/*.txt                    # ~5-10 min on a laptop
    python train.py books/*.txt --preset tiny      # ~1 min, to see it working
    python train.py books/*.txt --preset large     # better text, ~1 hour

Press Ctrl+C at any time: the best model so far is already saved.
"""

# ============================================================================
# HOW TRAINING WORKS
# ----------------------------------------------------------------------------
# 1. Read the books into one long string and turn every character into an
#    integer id.
# 2. Repeat thousands of times:
#      - pick 32 random windows of text (a "batch"),
#      - ask the model to predict every next character in them (forward),
#      - measure how wrong it was (the "loss"),
#      - compute how each weight should change to be less wrong (backward),
#      - nudge every weight a tiny bit in that direction (optimizer step).
# 3. Every few hundred steps, measure the loss on text the model never trains
#    on (the "validation" set) and save the model if it is the best so far.
#    If train loss keeps falling but validation loss rises, the model is
#    memorising the book instead of learning general patterns ("overfitting").
#
# PYTHON NOTES: see the top of model.py for the basics (indentation, lists,
# dicts, NumPy arrays, slicing, f-strings).
# ============================================================================

import argparse            # standard library: parses command-line options like --preset tiny
import math
import time
from pathlib import Path   # standard library: object-oriented file paths

import numpy as np

# Import names from our own model.py file (it's in the same folder, so Python finds it).
from model import GPT, Adam, CharTokenizer, GPTConfig, save_checkpoint

# A dict of dicts. dict(a=1, b=2) is another way to write {"a": 1, "b": 2}.
#   n_layer    - number of transformer blocks (depth)
#   n_head     - attention heads per block
#   n_embd     - size of each character's vector (width)
#   block_size - how many previous characters the model can see
#   max_iters  - number of training steps
#   batch_size - text windows per step
PRESETS = {
    "tiny":  dict(n_layer=2, n_head=2, n_embd=64,  block_size=32,  max_iters=1500, batch_size=32),
    "small": dict(n_layer=3, n_head=4, n_embd=96,  block_size=64,  max_iters=4000, batch_size=32),
    "large": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, max_iters=8000, batch_size=32),
}


def load_text(paths):
    """Read every file in `paths` and join them into one string."""
    parts = []
    for p in paths:
        # errors="replace" turns any undecodable bytes into a placeholder
        # instead of crashing on a slightly broken file.
        parts.append(Path(p).read_text(encoding="utf-8", errors="replace"))
        # {:,} formats a number with thousands separators: 567,038.
        print(f"  {p}: {len(parts[-1]):,} characters")
    # Join the books with a blank line between them.
    return "\n\n".join(parts)


def get_batch(data, block_size, batch_size, rng):
    """Random windows of text; the target is the same window shifted one character right.

    Example with block_size 4 and the text "hello":
        x = "hell"   y = "ello"
    so at every position the model learns to predict the following character:
    after "h" comes "e", after "he" comes "l", after "hel" comes "l", ...
    """
    # batch_size random start positions, each leaving room for a full window.
    ix = rng.integers(0, len(data) - block_size - 1, batch_size)
    # np.stack turns a list of 1-D arrays into one 2-D array: (batch_size, block_size).
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def estimate_loss(model, splits, args, rng, iters=20):
    """Average the loss over a few random batches from each split (train and val).

    A single batch is noisy, so we average 20 of them. No backward pass: we only
    measure, we don't learn here.
    """
    # A dict comprehension that loops over splits = {"train": ..., "val": ...}.
    # model.forward(...) returns (logits, loss); [1] takes the loss.
    # `*get_batch(...)` unpacks the returned (x, y) tuple into two arguments.
    return {name: float(np.mean([model.forward(*get_batch(data, args.block_size, args.batch_size, rng))[1]
                                 for _ in range(iters)]))
            for name, data in splits.items()}


def lr_at(step, args):
    """Linear warmup, then cosine decay to 10% of the peak learning rate.

    The learning rate is the size of each nudge. Start small (the random initial
    model produces wild gradients), ramp up, then gently reduce it so the model
    settles into a good solution instead of bouncing around it.
    """
    warmup = min(200, args.max_iters // 10)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.max_iters - warmup)   # goes 0 -> 1
    # cos goes 1 -> -1 as progress goes 0 -> 1, so this goes from lr down to 0.1 * lr.
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    # --- command-line options ------------------------------------------------
    # argparse builds a parser from these declarations and generates --help
    # automatically. __doc__ is this file's docstring (the text at the top).
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # A name without "--" is a positional argument; nargs="+" means "one or more".
    ap.add_argument("books", nargs="+", help="one or more .txt files to learn from")
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--out", default="checkpoint.pt")
    # These have no default, so they are None unless given; the preset fills them in below.
    ap.add_argument("--n_layer", type=int)
    ap.add_argument("--n_head", type=int)
    ap.add_argument("--n_embd", type=int)
    ap.add_argument("--block_size", type=int)
    ap.add_argument("--max_iters", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float, default=3e-3)   # 3e-3 = 0.003 (scientific notation)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    # parse_args() reads the command line and returns an object with one field
    # per option: args.books, args.preset, args.n_layer, ...
    args = ap.parse_args()
    # Fill any option the user didn't give from the chosen preset.
    # getattr(obj, "name") / setattr(obj, "name", value) read/write a field whose
    # name is held in a string variable.
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    # --- data ----------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    print("Reading books:")
    text = load_text(args.books)
    tok = CharTokenizer.from_text(text)
    # The whole book as one long array of integer character ids.
    data = np.array(tok.encode(text), dtype=np.int64)
    n = int(0.9 * len(data))  # last 10% is held out to check the model generalizes
    splits = {"train": data[:n], "val": data[n:]}
    print(f"{len(text):,} characters, vocabulary of {tok.vocab_size} distinct characters")

    # --- model and optimizer ---------------------------------------------------
    # Arguments can be passed by name (keyword arguments), in any order.
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPT(cfg, seed=args.seed)
    # {x:.2f} formats a number with 2 decimal places.
    print(f"Model: {model.num_params() / 1e6:.2f}M parameters  ({cfg})")
    opt = Adam(model.params)

    # --- training loop ---------------------------------------------------------
    best_val = float("inf")   # positive infinity: any real loss will be lower
    start = time.time()       # seconds since 1970, used to show elapsed time
    # try/except: if the user presses Ctrl+C, Python raises KeyboardInterrupt;
    # we catch it and finish cleanly instead of crashing with a stack trace.
    try:
        for step in range(args.max_iters + 1):
            # % is modulo: evaluate every eval_every steps, and at the very end.
            if step % args.eval_every == 0 or step == args.max_iters:
                losses = estimate_loss(model, splits, args, rng)
                mark = ""
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(args.out, model, tok, {"step": step, "val_loss": best_val})
                    mark = "  (saved)"
                # Two f-strings next to each other are joined into one string.
                # {step:5d} pads the integer to 5 characters; {x:6.0f} = 6 wide, 0 decimals.
                print(f"step {step:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
                      f" | {time.time() - start:6.0f}s{mark}")
                # Show what the model writes right now, starting from a newline.
                # Watching this sample improve is the most fun part of training.
                sample = tok.decode(model.generate(tok.encode("\n"), 100, rng=rng))
                print("   sample: " + sample.strip().replace("\n", " "))
            if step == args.max_iters:
                break   # leave the loop

            x, y = get_batch(splits["train"], args.block_size, args.batch_size, rng)
            model.forward(x, y)                # 1. predict every next character, measure the loss
            grads = model.backward()           # 2. how should each weight change to lower the loss?
            opt.step(model.params, grads, lr_at(step, args))  # 3. nudge the weights that way
    except KeyboardInterrupt:
        print("\nStopped early.")
    print(f"Best model (val loss {best_val:.3f}) is in {args.out}")
    print(f'Try it:  python chat.py --prompt "Alice was" --visualize')


# Every Python file can be run as a program or imported as a library.
# __name__ is "__main__" only when the file is run directly
# (`python train.py`), so main() doesn't run when another file imports this one.
if __name__ == "__main__":
    main()
