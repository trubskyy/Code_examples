"""Give the trained model a prompt and let it continue the text.

    python chat.py                                   # interactive: type prompts, Ctrl+C to quit
    python chat.py --prompt "Alice was"              # one answer
    python chat.py --prompt "Alice was" --visualize  # + picture of the network for the first character
    python chat.py --prompt "Alice was" --animate    # + live picture for every generated character
    python chat.py --prompt "Alice was" --gif run.gif --length 40   # save the animation

This is a character-level model trained on a few books: it continues text in
the style of those books, it does not follow instructions like a chatbot.
"""

# ============================================================================
# HOW ANSWERING WORKS
# ----------------------------------------------------------------------------
# The model only ever does one thing: given some text, output a probability
# for every possible next character. To write a whole answer we loop:
#     prompt -> model -> probabilities -> pick one character at random
#            -> append it to the text -> feed the longer text back in -> ...
# All of that happens in model.generate(). This file loads the trained model,
# calls generate(), prints each character as it arrives, and optionally draws
# the network at each step (visualize.py).
#
# PYTHON NOTES: see the top of model.py for the basics.
# ============================================================================

import argparse   # command-line option parsing
import os         # operating-system access: environment variables, file existence
import sys        # interpreter access: platform name, exiting with a message

import matplotlib   # the standard Python plotting library

# Matplotlib needs a "backend" to draw with. On Linux without a graphical
# display (e.g. a remote server), opening a window would fail, so switch to
# "Agg", which only renders to image files. This must happen BEFORE pyplot is
# imported, which is why this `if` sits between two import lines.
# os.environ is a dict of environment variables; .get() returns None if missing.
if not os.environ.get("DISPLAY") and sys.platform.startswith("linux") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")  # no screen (e.g. a server): save images instead of opening windows
import matplotlib.pyplot as plt   # pyplot: the figure/window API, conventionally called plt
import numpy as np

from model import load_checkpoint
from visualize import draw_step


def has_screen():
    """True if matplotlib can open windows (i.e. we're not in image-only mode)."""
    return matplotlib.get_backend().lower() != "agg"


def answer(model, tok, prompt, args):
    """Continue `prompt`, printing characters as they are generated, and draw if asked."""
    # If the prompt is empty (or has no known characters), encode() returns an
    # empty list, which counts as False, so `or` falls back to a newline.
    ids = tok.encode(prompt) or tok.encode("\n")
    # `or` between values returns the first truthy one; used here as a boolean
    # "any of these options set?" (args.gif is a file name or None).
    visual = args.visualize or args.animate or args.gif
    # A figure is a drawing canvas; figsize is in inches (18 x 10).
    fig = plt.figure(figsize=(18, 10)) if visual else None
    frames, generated = [], []   # GIF frames, and the ids generated so far

    # A nested function ("closure"): it can read the variables of the enclosing
    # function (fig, frames, generated, args, ...). model.generate() calls it
    # once per new character. Lists like `generated` can be modified in place
    # from here (append), which is how state survives between calls.
    def on_step(ctx_ids, probs, chosen):
        step = len(generated)   # how many characters we've produced before this one
        # Draw on the first step always; on every step if animating or making a GIF.
        if visual and (step == 0 or args.animate or args.gif):
            draw_step(fig, model, tok, ctx_ids, probs, chosen, tok.decode(generated))
            if step == 0:
                fig.savefig(args.png, dpi=110)   # save the first picture as a PNG file
            if args.gif:
                fig.canvas.draw()                # render the figure into pixels
                frames.append(_fig_to_image(fig))
            if args.animate and has_screen():
                plt.pause(args.delay)            # update the window and wait a moment
        generated.append(chosen)
        # end="" stops print from adding a newline; flush=True shows the character
        # immediately instead of waiting for the output buffer to fill.
        print(tok.chars[chosen], end="", flush=True)

    print(prompt, end="", flush=True)
    rng = np.random.default_rng(args.seed)   # seed None = different text every run
    # Pass the function itself (no parentheses) so generate() can call it.
    model.generate(ids, args.length, args.temperature, args.top_k, rng, on_step=on_step)
    print()   # final newline

    if visual:
        print(f"[network picture for the first predicted character saved to {args.png}]")
        if args.gif and frames:
            # Pillow writes an animated GIF from the first image plus the rest.
            # frames[1:] = every frame except the first. duration is ms per frame.
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=int(args.delay * 1000),
                           loop=0)
            print(f"[animation of {len(frames)} steps saved to {args.gif}]")
        if has_screen() and not args.gif:
            plt.show()      # open the window and wait until the user closes it
        plt.close(fig)      # free the figure's memory


def _fig_to_image(fig):
    """Convert the rendered figure into a Pillow image (one GIF frame)."""
    # Imports can also appear inside a function; this one only runs when making a GIF.
    from PIL import Image
    buf = fig.canvas.buffer_rgba()   # raw pixel bytes: red, green, blue, alpha per pixel
    # "P" mode with an adaptive palette = 256 colours, which is what GIF supports.
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), bytes(buf)).convert("P", palette=Image.ADAPTIVE)


def main():
    # --- command-line options (see train.py for how argparse works) ---------
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--prompt", help="text to continue (omit for interactive mode)")
    ap.add_argument("--length", type=int, default=300, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="lower = safer, higher = more creative")
    ap.add_argument("--top_k", type=int, default=20, help="only sample from the k most likely characters")
    ap.add_argument("--seed", type=int)
    # action="store_true" makes a flag with no value: present = True, absent = False.
    ap.add_argument("--visualize", action="store_true", help="draw the network for the first predicted character")
    ap.add_argument("--animate", action="store_true", help="redraw the network for every generated character")
    ap.add_argument("--gif", help="save an animated GIF of every generation step to this file")
    ap.add_argument("--png", default="network.png", help="where to save the picture")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds per frame when animating")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        # sys.exit with a string prints it and ends the program with an error code.
        sys.exit(f"No {args.checkpoint} found. Train first:  python train.py books/*.txt")
    model, tok, ckpt = load_checkpoint(args.checkpoint)
    # ckpt.get(key, default) returns the default if the key is missing; float('nan') = "not a number".
    print(f"Loaded {model.num_params() / 1e6:.2f}M-parameter model (val loss {ckpt.get('val_loss', float('nan')):.3f})")

    # One-shot mode: answer the single prompt from the command line, then stop.
    if args.prompt is not None:
        answer(model, tok, args.prompt, args)
        return
    # Interactive mode: keep asking for prompts until Ctrl+C (KeyboardInterrupt)
    # or Ctrl+D (EOFError, "end of input").
    print("Type a prompt and press Enter (Ctrl+C to quit).")
    try:
        while True:
            prompt = input("\n> ")   # input() prints the text, then reads one line from the keyboard
            answer(model, tok, prompt, args)
    except (KeyboardInterrupt, EOFError):   # a tuple catches either exception type
        print()


# Run main() only when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()
