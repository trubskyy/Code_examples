"""Draw what the network does while it predicts the next character.

One figure per prediction, with four views of the same forward pass:

  1. Network flow     - every character of the prompt (rows) passing through the
                        embedding and each transformer block (columns). Node colour is
                        how strongly that vector is activated; lines are attention:
                        which earlier characters each position pulled information from.
  2. Next-character   - the final probability distribution, and which one was picked.
  3. Attention maps   - each block's full attention matrix (averaged over heads).
  4. Logit lens       - what the model would guess if it stopped after each layer,
                        showing the prediction forming as it moves up the network.
  5. MLP neurons      - the feed-forward neurons that fired for the last character.
"""

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch

from model import softmax

MAX_SHOWN = 14  # most recent characters drawn in the network diagram
CMAP = matplotlib.colormaps["viridis"]
ATTN_COLOR = "#e4572e"


def show_char(c):
    return {"\n": "↵", " ": "␣", "\t": "→"}.get(c, c)


def _norm(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)


def draw_step(fig, model, tok, ctx_ids, probs, chosen_id, generated_so_far=""):
    """Render the trace that model.forward(record=True) left in model.trace."""
    tr = model.trace
    fig.clear()
    n_layer = model.cfg.n_layer
    T = len(ctx_ids)
    shown = list(range(max(0, T - MAX_SHOWN), T))
    chars = [show_char(c) for c in tok.decode(ctx_ids)]

    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.1, 1], width_ratios=[2.4, 0.05, 1],
                  hspace=0.32, wspace=0.12, left=0.04, right=0.98, top=0.87, bottom=0.06)
    prompt_txt = tok.decode(ctx_ids).replace("\n", " ")
    if len(prompt_txt) > 70:
        prompt_txt = "…" + prompt_txt[-70:]
    fig.suptitle(f'Input: "{prompt_txt}"   →   next character: '
                 f'"{show_char(tok.chars[chosen_id])}"  (p = {probs[chosen_id]:.2f})',
                 fontsize=13, fontweight="bold", y=0.985)
    if generated_so_far:
        fig.text(0.5, 0.935, f'Generated so far: "{generated_so_far[-90:]}"'.replace("\n", " "),
                 ha="center", fontsize=10, color="dimgray")

    _draw_network(fig.add_subplot(gs[0, 0]), tr, chars, shown, n_layer)
    _draw_probs(fig.add_subplot(gs[0, 2]), tok, probs, chosen_id)

    bottom = GridSpecFromSubplotSpec(1, n_layer + 2, subplot_spec=gs[1, :], wspace=0.35,
                                     width_ratios=[1] * n_layer + [1.6, 1.5])
    for l in range(n_layer):
        _draw_attention_map(fig.add_subplot(bottom[0, l]), tr["attn"][l][0], chars, shown, l)
    _draw_mlp(fig.add_subplot(bottom[0, n_layer]), tr["mlp"])
    _draw_logit_lens(fig.add_subplot(bottom[0, n_layer + 1]), tok, tr["layer_logits"])


def _draw_network(ax, tr, chars, shown, n_layer):
    ax.set_title("Network flow: each column is a layer, each row a character of the prompt\n"
                 "node colour = activation strength · red lines = attention (who reads from whom)",
                 fontsize=10)
    cols = ["input", "embed"] + [f"block {l + 1}" for l in range(n_layer)] + ["output"]
    xs = np.arange(len(cols)) * 1.0
    ys = {p: -i for i, p in enumerate(shown)}
    last = shown[-1]

    # Attention edges into block l+1 come from the residual stream after block l.
    for l in range(n_layer):
        att = tr["attn"][l][0].mean(0)  # (T, T) averaged over heads
        x0, x1 = xs[l + 1], xs[l + 2]
        for i in shown:
            for j in shown:
                w = att[i, j]
                if j > i or w < 0.03:
                    continue
                is_last = i == last
                ax.plot([x0, x1], [ys[j], ys[i]], lw=0.5 + (3.5 if is_last else 1.5) * w,
                        color=ATTN_COLOR if is_last else "gray",
                        alpha=min(1.0, 0.15 + w) if is_last else 0.08 + 0.4 * w, zorder=1)

    # Nodes: vector norm of the residual stream at every position and layer.
    for c, res in enumerate(tr["residuals"]):
        strength = _norm(np.linalg.norm(res[0, shown], axis=-1))
        for k, p in enumerate(shown):
            ax.scatter(xs[c + 1], ys[p], s=260, color=CMAP(0.15 + 0.85 * strength[k]),
                       edgecolor="black" if p == last else "white", linewidth=1.5 if p == last else 0.5,
                       zorder=3)

    for p in shown:
        ax.text(xs[0], ys[p], chars[p], ha="center", va="center", fontsize=12, family="monospace",
                fontweight="bold" if p == last else "normal",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff4cc" if p == last else "#f2f2f2", ec="gray"))
        ax.add_patch(FancyArrowPatch((xs[0] + 0.18, ys[p]), (xs[1] - 0.15, ys[p]),
                                     arrowstyle="-|>", mutation_scale=8, color="lightgray"))

    ax.add_patch(FancyArrowPatch((xs[-2] + 0.15, ys[last]), (xs[-1] - 0.2, ys[last]),
                                 arrowstyle="-|>", mutation_scale=16, color="black", lw=1.5))
    ax.text(xs[-1], ys[last], "softmax\n→ probs", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#d9f0d3", ec="green"))

    for x, name in zip(xs, cols):
        ax.text(x, 1.0, name, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, xs[-1] + 0.6)
    ax.set_ylim(-len(shown) + 0.4, 1.6)
    ax.axis("off")


def _draw_probs(ax, tok, probs, chosen_id, k=12):
    p = probs
    top = np.argsort(p)[::-1][:k][::-1]
    colors = [ATTN_COLOR if i == chosen_id else "#4c72b0" for i in top]
    ax.barh(range(len(top)), p[top], color=colors)
    ax.set_yticks(range(len(top)), [f"'{show_char(tok.chars[i])}'" for i in top], family="monospace")
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("Output: probability of each next character\n(red = the one sampled)", fontsize=10)
    for y, i in enumerate(top):
        ax.text(p[i] + 0.01, y, f"{p[i]:.2f}", va="center", fontsize=8)


def _draw_attention_map(ax, att, chars, shown, l):
    a = att.mean(0)[np.ix_(shown, shown)]
    ax.imshow(a, cmap="Reds", vmin=0, vmax=a.max())
    labels = [chars[p] for p in shown]
    ax.set_xticks(range(len(shown)), labels, fontsize=7, family="monospace")
    ax.set_yticks(range(len(shown)), labels, fontsize=7, family="monospace")
    ax.set_title(f"block {l + 1} attention\nrow reads from column", fontsize=9)
    ax.tick_params(length=0)


def _draw_mlp(ax, mlp):
    acts = np.stack([h[0, -1] for h in mlp])  # (n_layer, 4C) for the last character
    acts = acts / (np.abs(acts).max(axis=1, keepdims=True) + 1e-9)  # each block on its own scale
    ax.imshow(acts, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(mlp)), [f"block {l + 1}" for l in range(len(mlp))], fontsize=8)
    ax.set_xlabel(f"{acts.shape[1]} MLP neurons", fontsize=8)
    ax.set_xticks([])
    ax.set_title("MLP neurons firing for\nthe last character", fontsize=9)


def _draw_logit_lens(ax, tok, layer_logits, k=3):
    ax.set_title("Logit lens: best guesses if the\nnetwork stopped after each layer", fontsize=9)
    names = ["embed"] + [f"block {l + 1}" for l in range(len(layer_logits) - 1)]
    rows = []
    for logits in layer_logits:
        p = softmax(logits[0, -1])
        rows.append([f"'{show_char(tok.chars[i])}' {p[i]:.2f}" for i in np.argsort(p)[::-1][:k]])
    table = ax.table(cellText=rows, rowLabels=names, colLabels=["1st", "2nd", "3rd"], loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        if r > 0 and c == 0:
            cell.set_facecolor(cm.Greens(0.2 + 0.6 * r / len(rows)))
    ax.axis("off")
