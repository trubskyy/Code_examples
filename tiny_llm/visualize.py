"""Draw what the network does while it predicts the next character.

One figure per prediction, with five views of the same forward pass:

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

# ============================================================================
# MATPLOTLIB IN ONE PARAGRAPH
# ----------------------------------------------------------------------------
# A "figure" is the whole canvas (one window or image file). It contains
# "axes": individual plot areas, each with its own coordinate system. You
# create axes with fig.add_subplot(...), then call drawing methods on them:
#     ax.plot(xs, ys)        lines            ax.scatter(x, y)   dots
#     ax.text(x, y, "hi")    text             ax.barh(...)       horizontal bars
#     ax.imshow(matrix)      a 2-D array drawn as a coloured grid (heat map)
#     ax.set_title(...)      ax.axis("off")   hide the axis lines and ticks
# A GridSpec divides the figure into a grid of cells so axes can be placed in
# rows and columns of different sizes.
#
# All the numbers drawn come from model.trace, which model.forward(record=True)
# fills with the intermediate results of one forward pass.
#
# PYTHON NOTES: see the top of model.py for the basics. Functions whose names
# start with _ are private helpers used only inside this file.
# ============================================================================

import matplotlib
import numpy as np
from matplotlib import cm                     # colour maps (number -> colour)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch

from model import softmax

MAX_SHOWN = 14  # most recent characters drawn in the network diagram
# A colour map turns a number from 0 to 1 into a colour: viridis goes dark purple -> green -> yellow.
CMAP = matplotlib.colormaps["viridis"]
ATTN_COLOR = "#e4572e"   # hex RGB colour (orange-red) used for attention lines and the chosen bar


def show_char(c):
    """Make invisible characters visible: newline -> ↵, space -> ␣, tab -> →."""
    # dict.get(key, default): look c up; if it's not one of the three, return c unchanged.
    return {"\n": "↵", " ": "␣", "\t": "→"}.get(c, c)


def _norm(a):
    """Rescale an array to the range 0..1 (smallest -> 0, largest -> 1)."""
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)   # + 1e-9 avoids dividing by zero if all values are equal


def draw_step(fig, model, tok, ctx_ids, probs, chosen_id, generated_so_far=""):
    """Render the trace that model.forward(record=True) left in model.trace.

    fig              - the matplotlib figure to draw into (cleared first)
    ctx_ids          - the ids of the text the model just read
    probs            - its probability for every possible next character
    chosen_id        - the character that was actually picked
    generated_so_far - text produced so far, shown under the title
    """
    tr = model.trace
    fig.clear()
    n_layer = model.cfg.n_layer
    T = len(ctx_ids)
    # Indices of the last MAX_SHOWN characters (drawing all 64 would be unreadable).
    # range(a, b) yields a, a+1, ..., b-1; list() turns that into a real list.
    shown = list(range(max(0, T - MAX_SHOWN), T))
    chars = [show_char(c) for c in tok.decode(ctx_ids)]

    # Layout: 2 rows x 3 columns. Top row: network diagram | thin spacer | probabilities.
    # Bottom row: one wide cell, subdivided further below. The ratios set relative
    # sizes; left/right/top/bottom are margins as fractions of the figure.
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.1, 1], width_ratios=[2.4, 0.05, 1],
                  hspace=0.32, wspace=0.12, left=0.04, right=0.98, top=0.87, bottom=0.06)
    prompt_txt = tok.decode(ctx_ids).replace("\n", " ")
    if len(prompt_txt) > 70:
        prompt_txt = "…" + prompt_txt[-70:]   # keep only the last 70 characters
    fig.suptitle(f'Input: "{prompt_txt}"   →   next character: '
                 f'"{show_char(tok.chars[chosen_id])}"  (p = {probs[chosen_id]:.2f})',
                 fontsize=13, fontweight="bold", y=0.985)
    if generated_so_far:   # an empty string counts as False
        # fig.text places text at figure coordinates: (0, 0) bottom-left, (1, 1) top-right.
        fig.text(0.5, 0.935, f'Generated so far: "{generated_so_far[-90:]}"'.replace("\n", " "),
                 ha="center", fontsize=10, color="dimgray")

    # gs[0, 0] = row 0, column 0 of the grid.
    _draw_network(fig.add_subplot(gs[0, 0]), tr, chars, shown, n_layer)
    _draw_probs(fig.add_subplot(gs[0, 2]), tok, probs, chosen_id)

    # Split the bottom row (gs[1, :] = row 1, all columns) into n_layer attention
    # maps + the MLP panel + the logit-lens table. [1] * n_layer makes a list of
    # n_layer ones; + joins lists.
    bottom = GridSpecFromSubplotSpec(1, n_layer + 2, subplot_spec=gs[1, :], wspace=0.35,
                                     width_ratios=[1] * n_layer + [1.6, 1.5])
    for l in range(n_layer):
        # tr["attn"][l] has shape (1, heads, T, T); [0] drops the batch dimension.
        _draw_attention_map(fig.add_subplot(bottom[0, l]), tr["attn"][l][0], chars, shown, l)
    _draw_mlp(fig.add_subplot(bottom[0, n_layer]), tr["mlp"])
    _draw_logit_lens(fig.add_subplot(bottom[0, n_layer + 1]), tok, tr["layer_logits"])


def _draw_network(ax, tr, chars, shown, n_layer):
    """The big diagram: characters (rows) flowing through layers (columns)."""
    # Adjacent string literals are joined; \n is a line break inside the title.
    ax.set_title("Network flow: each column is a layer, each row a character of the prompt\n"
                 "node colour = activation strength · red lines = attention (who reads from whom)",
                 fontsize=10)
    # Column labels: ["input", "embed", "block 1", ..., "block N", "output"].
    cols = ["input", "embed"] + [f"block {l + 1}" for l in range(n_layer)] + ["output"]
    xs = np.arange(len(cols)) * 1.0           # x coordinate of each column: 0.0, 1.0, 2.0, ...
    ys = {p: -i for i, p in enumerate(shown)}  # y coordinate of each character row (top = 0, going down)
    last = shown[-1]                           # the position whose next character is being predicted

    # --- attention lines -------------------------------------------------------
    # Attention edges into block l+1 come from the residual stream after block l.
    for l in range(n_layer):
        att = tr["attn"][l][0].mean(0)  # (T, T) averaged over heads; att[i, j] = how much i reads from j
        x0, x1 = xs[l + 1], xs[l + 2]   # from this column to the next one
        for i in shown:          # i = the character doing the reading
            for j in shown:      # j = the character being read
                w = att[i, j]
                # `continue` skips to the next loop iteration. Skip the future
                # (impossible, always 0) and very weak links (visual clutter).
                if j > i or w < 0.03:
                    continue
                is_last = i == last
                # Thicker and more opaque = more attention. Lines into the last
                # character are red and strong; the rest are faint grey.
                ax.plot([x0, x1], [ys[j], ys[i]], lw=0.5 + (3.5 if is_last else 1.5) * w,
                        color=ATTN_COLOR if is_last else "gray",
                        alpha=min(1.0, 0.15 + w) if is_last else 0.08 + 0.4 * w, zorder=1)

    # --- nodes ---------------------------------------------------------------
    # Nodes: vector norm of the residual stream at every position and layer.
    # The norm (length) of a character's vector is a rough measure of how
    # "active" it is. zorder=3 draws nodes on top of the lines (zorder=1).
    for c, res in enumerate(tr["residuals"]):
        # res[0, shown] = batch 0, the shown positions, all C numbers -> (len(shown), C).
        strength = _norm(np.linalg.norm(res[0, shown], axis=-1))
        for k, p in enumerate(shown):
            ax.scatter(xs[c + 1], ys[p], s=260, color=CMAP(0.15 + 0.85 * strength[k]),
                       edgecolor="black" if p == last else "white", linewidth=1.5 if p == last else 0.5,
                       zorder=3)

    # --- input characters and arrows into the embedding ---------------------
    for p in shown:
        # bbox draws a rounded box behind the text. dict(a=1) == {"a": 1}.
        ax.text(xs[0], ys[p], chars[p], ha="center", va="center", fontsize=12, family="monospace",
                fontweight="bold" if p == last else "normal",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff4cc" if p == last else "#f2f2f2", ec="gray"))
        ax.add_patch(FancyArrowPatch((xs[0] + 0.18, ys[p]), (xs[1] - 0.15, ys[p]),
                                     arrowstyle="-|>", mutation_scale=8, color="lightgray"))

    # --- final arrow from the last block to the output ------------------------
    # xs[-2] = second-to-last column (last block), xs[-1] = the output column.
    ax.add_patch(FancyArrowPatch((xs[-2] + 0.15, ys[last]), (xs[-1] - 0.2, ys[last]),
                                 arrowstyle="-|>", mutation_scale=16, color="black", lw=1.5))
    ax.text(xs[-1], ys[last], "softmax\n→ probs", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#d9f0d3", ec="green"))

    # zip(a, b) walks two sequences in step: (a[0], b[0]), (a[1], b[1]), ...
    for x, name in zip(xs, cols):
        ax.text(x, 1.0, name, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, xs[-1] + 0.6)
    ax.set_ylim(-len(shown) + 0.4, 1.6)
    ax.axis("off")


def _draw_probs(ax, tok, probs, chosen_id, k=12):
    """Bar chart of the k most likely next characters."""
    p = probs
    # np.argsort gives the indices that would sort the array (ascending).
    # [::-1] reverses (step -1), [:k] takes the first k, i.e. the k most likely.
    # The final [::-1] puts the most likely at the TOP of a horizontal bar chart,
    # since barh draws the first bar at the bottom.
    top = np.argsort(p)[::-1][:k][::-1]
    colors = [ATTN_COLOR if i == chosen_id else "#4c72b0" for i in top]
    # p[top]: indexing with an array of indices returns those elements.
    ax.barh(range(len(top)), p[top], color=colors)
    ax.set_yticks(range(len(top)), [f"'{show_char(tok.chars[i])}'" for i in top], family="monospace")
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("Output: probability of each next character\n(red = the one sampled)", fontsize=10)
    for y, i in enumerate(top):
        ax.text(p[i] + 0.01, y, f"{p[i]:.2f}", va="center", fontsize=8)   # number next to each bar


def _draw_attention_map(ax, att, chars, shown, l):
    """Heat map of one block's attention (averaged over heads) for the shown characters."""
    # att.mean(0) averages over the heads axis -> (T, T).
    # np.ix_(rows, cols) selects a sub-grid: only the shown rows AND shown columns.
    a = att.mean(0)[np.ix_(shown, shown)]
    ax.imshow(a, cmap="Reds", vmin=0, vmax=a.max())
    labels = [chars[p] for p in shown]
    ax.set_xticks(range(len(shown)), labels, fontsize=7, family="monospace")
    ax.set_yticks(range(len(shown)), labels, fontsize=7, family="monospace")
    ax.set_title(f"block {l + 1} attention\nrow reads from column", fontsize=9)
    ax.tick_params(length=0)


def _draw_mlp(ax, mlp):
    """One row per block showing every MLP neuron's activation for the last character."""
    # h[0, -1] = batch 0, last position -> a vector of 4C neuron values.
    # np.stack puts one such vector per block into a (n_layer, 4C) matrix.
    acts = np.stack([h[0, -1] for h in mlp])  # (n_layer, 4C) for the last character
    acts = acts / (np.abs(acts).max(axis=1, keepdims=True) + 1e-9)  # each block on its own scale
    # aspect="auto" lets the cells stretch to fill the panel; RdBu_r = blue (negative) to red (positive).
    ax.imshow(acts, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(mlp)), [f"block {l + 1}" for l in range(len(mlp))], fontsize=8)
    ax.set_xlabel(f"{acts.shape[1]} MLP neurons", fontsize=8)
    ax.set_xticks([])
    ax.set_title("MLP neurons firing for\nthe last character", fontsize=9)


def _draw_logit_lens(ax, tok, layer_logits, k=3):
    """Table: the top-k guesses if we read out the prediction after each layer."""
    ax.set_title("Logit lens: best guesses if the\nnetwork stopped after each layer", fontsize=9)
    names = ["embed"] + [f"block {l + 1}" for l in range(len(layer_logits) - 1)]
    rows = []
    for logits in layer_logits:
        p = softmax(logits[0, -1])   # probabilities at the last position
        # One table row: e.g. ["'e' 0.41", "'a' 0.12", "' ' 0.09"].
        rows.append([f"'{show_char(tok.chars[i])}' {p[i]:.2f}" for i in np.argsort(p)[::-1][:k]])
    table = ax.table(cellText=rows, rowLabels=names, colLabels=["1st", "2nd", "3rd"], loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    # get_celld() returns a dict {(row, col): cell}. Shade the first column
    # darker green for deeper layers. (Row 0 is the header.)
    for (r, c), cell in table.get_celld().items():
        if r > 0 and c == 0:
            cell.set_facecolor(cm.Greens(0.2 + 0.6 * r / len(rows)))
    ax.axis("off")
