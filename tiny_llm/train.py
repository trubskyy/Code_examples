"""Train the tiny GPT on one or more books (plain .txt files).

    python train.py books/*.txt                    # ~10-20 min on a laptop CPU
    python train.py books/*.txt --preset tiny      # ~2 min, to see it working
    python train.py books/*.txt --preset large     # better text; wants a GPU or patience

Press Ctrl+C at any time: the best model so far is already saved.
"""

import argparse
import math
import time
from pathlib import Path

import torch

from model import GPT, CharTokenizer, GPTConfig, pick_device, save_checkpoint

PRESETS = {
    #         layers heads width context  steps   batch
    "tiny":   dict(n_layer=2, n_head=2, n_embd=64,  block_size=64,  max_iters=1500, batch_size=32),
    "small":  dict(n_layer=4, n_head=4, n_embd=128, block_size=128, max_iters=5000, batch_size=32),
    "large":  dict(n_layer=6, n_head=6, n_embd=384, block_size=256, max_iters=8000, batch_size=64),
}


def load_text(paths):
    parts = []
    for p in paths:
        parts.append(Path(p).read_text(encoding="utf-8", errors="replace"))
        print(f"  {p}: {len(parts[-1]):,} characters")
    return "\n\n".join(parts)


def get_batch(data, block_size, batch_size, device):
    """Random windows of text; the target is the same window shifted one character right."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, args, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, args.block_size, args.batch_size, device))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(step, args):
    """Linear warmup, then cosine decay to 10% of the peak learning rate."""
    warmup = min(200, args.max_iters // 10)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.max_iters - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("books", nargs="+", help="one or more .txt files to learn from")
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--out", default="checkpoint.pt")
    ap.add_argument("--n_layer", type=int)
    ap.add_argument("--n_head", type=int)
    ap.add_argument("--n_embd", type=int)
    ap.add_argument("--block_size", type=int)
    ap.add_argument("--max_iters", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--device", default=pick_device())
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    print("Reading books:")
    text = load_text(args.books)
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # last 10% is held out to check the model generalizes
    splits = {"train": data[:n], "val": data[n:]}
    print(f"{len(text):,} characters, vocabulary of {tok.vocab_size} distinct characters")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(cfg).to(args.device)
    print(f"Model: {model.num_params() / 1e6:.2f}M parameters on {args.device}  ({cfg})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    best_val = float("inf")
    start = time.time()
    try:
        for step in range(args.max_iters + 1):
            if step % args.eval_every == 0 or step == args.max_iters:
                losses = estimate_loss(model, splits, args, args.device)
                mark = ""
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(args.out, model, tok, {"step": step, "val_loss": best_val})
                    mark = "  (saved)"
                print(f"step {step:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
                      f" | {time.time() - start:6.0f}s{mark}")
                model.eval()
                sample = model.generate(torch.tensor([tok.encode("\n")], device=args.device), 120)
                model.train()
                print("   sample: " + tok.decode(sample[0].tolist()).strip().replace("\n", " ")[:120])
            if step == args.max_iters:
                break

            for g in opt.param_groups:
                g["lr"] = lr_at(step, args)
            x, y = get_batch(splits["train"], args.block_size, args.batch_size, args.device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    except KeyboardInterrupt:
        print("\nStopped early.")
    print(f"Best model (val loss {best_val:.3f}) is in {args.out}")
    print(f'Try it:  python chat.py --prompt "Alice was" --visualize')


if __name__ == "__main__":
    main()
