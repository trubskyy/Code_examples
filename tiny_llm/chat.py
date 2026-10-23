"""Give the trained model a prompt and let it continue the text.

    python chat.py                                   # interactive: type prompts, Ctrl+C to quit
    python chat.py --prompt "Alice was"              # one answer
    python chat.py --prompt "Alice was" --visualize  # + picture of the network for the first character
    python chat.py --prompt "Alice was" --animate    # + live picture for every generated character
    python chat.py --prompt "Alice was" --gif run.gif --length 40   # save the animation

This is a character-level model trained on a few books: it continues text in
the style of those books, it does not follow instructions like a chatbot.
"""

import argparse
import os
import sys

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")  # no screen (e.g. a server): save images instead of opening windows
import matplotlib.pyplot as plt
import torch

from model import load_checkpoint, pick_device
from visualize import draw_step


def has_screen():
    return matplotlib.get_backend().lower() != "agg"


def answer(model, tok, prompt, args, device):
    ids = tok.encode(prompt) or tok.encode("\n")
    idx = torch.tensor([ids], device=device)
    visual = args.visualize or args.animate or args.gif
    fig = plt.figure(figsize=(18, 10)) if visual else None
    frames, generated = [], []

    def on_step(ctx_ids, probs, chosen):
        step = len(generated)
        if visual and (step == 0 or args.animate or args.gif):
            draw_step(fig, model, tok, ctx_ids, probs, chosen, prompt + tok.decode(generated))
            if step == 0:
                fig.savefig(args.png, dpi=110)
            if args.gif:
                fig.canvas.draw()
                frames.append(_fig_to_image(fig))
            if args.animate and has_screen():
                plt.pause(args.delay)
        generated.append(chosen)
        print(tok.chars[chosen], end="", flush=True)

    print(prompt, end="", flush=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model.generate(idx, args.length, args.temperature, args.top_k, on_step=on_step)
    print()

    if visual:
        print(f"[network picture for the first predicted character saved to {args.png}]")
        if args.gif and frames:
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=int(args.delay * 1000),
                           loop=0)
            print(f"[animation of {len(frames)} steps saved to {args.gif}]")
        if has_screen() and not args.gif:
            plt.show()
        plt.close(fig)


def _fig_to_image(fig):
    from PIL import Image
    buf = fig.canvas.buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), bytes(buf)).convert("P", palette=Image.ADAPTIVE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--prompt", help="text to continue (omit for interactive mode)")
    ap.add_argument("--length", type=int, default=300, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="lower = safer, higher = more creative")
    ap.add_argument("--top_k", type=int, default=20, help="only sample from the k most likely characters")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--visualize", action="store_true", help="draw the network for the first predicted character")
    ap.add_argument("--animate", action="store_true", help="redraw the network for every generated character")
    ap.add_argument("--gif", help="save an animated GIF of every generation step to this file")
    ap.add_argument("--png", default="network.png", help="where to save the picture")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds per frame when animating")
    ap.add_argument("--device", default=pick_device())
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"No {args.checkpoint} found. Train first:  python train.py books/*.txt")
    model, tok, ckpt = load_checkpoint(args.checkpoint, args.device)
    print(f"Loaded {model.num_params() / 1e6:.2f}M-parameter model (val loss {ckpt.get('val_loss', float('nan')):.3f})")

    if args.prompt is not None:
        answer(model, tok, args.prompt, args, args.device)
        return
    print("Type a prompt and press Enter (Ctrl+C to quit).")
    try:
        while True:
            prompt = input("\n> ")
            answer(model, tok, prompt, args, args.device)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
