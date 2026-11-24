"""Draws architecture.png for ../decoder_only_transformer.md. Needs matplotlib.
Usage: python arch_figure.py [OUT_DIR]   (default: this folder)"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, BLUE_DARK, BLUE_FILL = "#2a78d6", "#1c5cab", "#dce9fa"
ORANGE, ORANGE_FILL = "#eb6834", "#fbe1d6"
GRAY_FILL, BAND = "#efeee9", "#f5f4f0"

fig = plt.figure(figsize=(11, 9.1), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 11)
ax.set_ylim(0, 9.1)
ax.axis("off")


def box(x, y, w, h, fc, ec, line1, line2=None, fs=10, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=fc, ec=ec, lw=1.2, zorder=z))
    if line2 is None:
        ax.text(x, y, line1, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)
    else:
        ax.text(x, y + 0.1, line1, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)
        ax.text(x, y - 0.13, line2, ha="center", va="center", fontsize=fs - 1.5, color=INK2, zorder=z + 1)


def arrow(p0, p1, color=INK2, lw=1.2, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def plus(x, y, r=0.15, z=3):
    ax.add_patch(Circle((x, y), r, fc=SURFACE, ec=INK2, lw=1.2, zorder=z))
    ax.text(x, y - 0.01, "+", ha="center", va="center", fontsize=11, color=INK2, zorder=z + 1)


# shared rows (y), identical in both panels
Y_IDS, Y_EMB, Y_PE = 0.5, 1.2, 1.9
Y_ATT, Y_AN1, Y_FFN, Y_AN2 = 3.0, 3.85, 4.75, 5.55
Y_LIN, Y_SM, Y_OUT = 6.6, 7.3, 8.0
BLOCK_LO, BLOCK_HI = 2.35, 6.0
H2, H1 = 0.56, 0.4  # box heights: two-line, one-line

# block band across both panels
ax.add_patch(FancyBboxPatch((0.6, BLOCK_LO), 10.3, BLOCK_HI - BLOCK_LO, boxstyle="round,pad=0,rounding_size=0.12",
                            fc=BAND, ec=AXIS, lw=1, ls=(0, (4, 3)), zorder=0))
ax.text(0.33, (BLOCK_LO + BLOCK_HI) / 2, "N ×", ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
ax.text(0.75, BLOCK_HI - 0.18, "decoder block", ha="left", va="center", fontsize=8.5, color=MUTED)

# ---------------- left panel: components ----------------
cx, w = 2.55, 2.6
ax.text(cx, 8.8, "Components, in order", ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(cx, 8.52, "tensor shapes on the right: B = batch size, n = tokens", ha="center", fontsize=9, color=INK2)

ax.text(cx, Y_IDS, "token ids", ha="center", va="center", fontsize=10, color=INK)
box(cx, Y_EMB, w, H2, GRAY_FILL, AXIS, "Embedding:  E[ids] · √d_model", "E: (n_vocab, d_model)")
plus(cx, Y_PE)
ax.text(cx + 0.3, Y_PE, "PE[:n]", ha="left", va="center", fontsize=8.5, color=INK2)
box(cx, Y_ATT, w, H2, BLUE_FILL, BLUE, "Masked multi-head attention", "h heads, causal mask")
box(cx, Y_AN1, w, H1, GRAY_FILL, AXIS, "Add & LayerNorm")
box(cx, Y_FFN, w, H2, ORANGE_FILL, ORANGE, "Feed-forward network", "d_model → d_ff → d_model")
box(cx, Y_AN2, w, H1, GRAY_FILL, AXIS, "Add & LayerNorm")
box(cx, Y_LIN, w, H2, GRAY_FILL, AXIS, "Linear:  x · Eᵀ", "same E as the embedding")
box(cx, Y_SM, w, H1, GRAY_FILL, AXIS, "Softmax over the vocabulary")
ax.text(cx, Y_OUT, "next-token probabilities", ha="center", va="center", fontsize=10, color=INK)

arrow((cx, Y_IDS + 0.14), (cx, Y_EMB - H2 / 2))
arrow((cx, Y_EMB + H2 / 2), (cx, Y_PE - 0.15))
arrow((cx, Y_PE + 0.15), (cx, Y_ATT - H2 / 2))
arrow((cx, Y_ATT + H2 / 2), (cx, Y_AN1 - H1 / 2))
arrow((cx, Y_AN1 + H1 / 2), (cx, Y_FFN - H2 / 2))
arrow((cx, Y_FFN + H2 / 2), (cx, Y_AN2 - H1 / 2))
arrow((cx, Y_AN2 + H1 / 2), (cx, Y_LIN - H2 / 2))
arrow((cx, Y_LIN + H2 / 2), (cx, Y_SM - H1 / 2))
arrow((cx, Y_SM + H1 / 2), (cx, Y_OUT - 0.14))

# residual paths: branch below each sublayer, rejoin at its Add & LayerNorm
xr = cx - w / 2 - 0.3
for y_branch, y_join in [(Y_ATT - H2 / 2 - 0.2, Y_AN1), (Y_AN1 + H1 / 2 + 0.12, Y_AN2)]:
    ax.plot([cx, xr, xr], [y_branch, y_branch, y_join], color=INK2, lw=1.2, zorder=2)
    arrow((xr, y_join), (cx - w / 2, y_join))
ax.text(xr - 0.08, (Y_ATT + Y_AN1) / 2, "residual", rotation=90, ha="right", va="center", fontsize=8, color=MUTED)
ax.text(xr - 0.08, (Y_FFN + Y_AN2) / 2, "residual", rotation=90, ha="right", va="center", fontsize=8, color=MUTED)

# shapes
xs = cx + w / 2 + 0.3
def bracket(y0, y1, label):
    ax.plot([xs, xs + 0.08, xs + 0.08, xs], [y0, y0, y1, y1], color=MUTED, lw=1)
    ax.text(xs + 0.18, (y0 + y1) / 2, label, ha="left", va="center", fontsize=9, color=INK)
ax.text(xs + 0.18, Y_IDS, "(B, n)", ha="left", va="center", fontsize=9, color=INK)
bracket(Y_EMB + H2 / 2 + 0.05, Y_LIN - H2 / 2 - 0.05, "(B, n, d_model)\nat every step")
bracket(Y_LIN + H2 / 2 + 0.05, Y_OUT, "(B, n, n_vocab)")

# ---------------- right panel: per position ----------------
cols = [6.15, 7.2, 8.25, 9.3, 10.35]
pw = 0.8
inputs = ["The", "cat", "sat", "on", "the"]
targets = ["cat", "sat", "on", "the", "mat"]
rx = sum(cols) / len(cols)
ax.text(rx, 8.8, "The same model, drawn per position", ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(rx, 8.52, "one column per token; each row is the step beside it on the left", ha="center", fontsize=9, color=INK2)

for i, x in enumerate(cols):
    ax.plot([x, x], [Y_IDS + 0.15, Y_OUT - 0.17], color=AXIS, lw=1.4, zorder=1)  # this position's vector, bottom to top
    ax.text(x, Y_IDS, inputs[i], ha="center", va="center", fontsize=10.5, color=INK)
    ax.text(x, Y_IDS - 0.3, f"i = {i + 1}", ha="center", va="center", fontsize=8, color=MUTED)
    box(x, Y_EMB, pw, H1, GRAY_FILL, AXIS, "E[t_i]", fs=9)
    plus(x, Y_PE, r=0.13)
    ax.text(x + 0.17, Y_PE, f"PE_{i + 1}", ha="left", va="center", fontsize=7.5, color=INK2)
    box(x, Y_AN1, pw, H1 * 0.85, GRAY_FILL, AXIS, "add & norm", fs=8)
    box(x, Y_FFN, pw, H2 * 0.8, ORANGE_FILL, ORANGE, "FFN", fs=9.5)
    box(x, Y_AN2, pw, H1 * 0.85, GRAY_FILL, AXIS, "add & norm", fs=8)
    box(x, Y_LIN, pw, H1, GRAY_FILL, AXIS, "· Eᵀ", fs=9)
    box(x, Y_SM, pw, H1, GRAY_FILL, AXIS, "softmax", fs=8.5)
    ax.text(x, Y_OUT, targets[i], ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")

# attention: position i reads positions j <= i (lines into the last position drawn dark)
ya0, ya1 = Y_ATT - 0.33, Y_ATT + 0.33
ax.add_patch(FancyBboxPatch((cols[0] - pw / 2, ya0 - 0.08), cols[-1] - cols[0] + pw, ya1 - ya0 + 0.16,
                            boxstyle="round,pad=0,rounding_size=0.07", fc=BLUE_FILL, ec=BLUE, lw=1.2, zorder=2))
for i in range(5):
    for j in range(i + 1):
        last = i == 4
        ax.plot([cols[j], cols[i]], [ya0, ya1], color=BLUE_DARK if last else BLUE,
                lw=1.8 if last else 0.9, alpha=1 if last else 0.45, zorder=3)
for x in cols:
    for y in (ya0, ya1):
        ax.add_patch(Circle((x, y), 0.055, fc=BLUE_DARK, ec="none", zorder=4))

# right-panel notes
ax.text(rx, BLOCK_LO + 0.12, "lines: position i reads positions j ≤ i  (dark lines: what position 5 reads)",
        ha="center", va="center", fontsize=8, color=INK2, zorder=5, bbox=dict(fc=BAND, ec="none", pad=1.5))
ax.text(rx, Y_FFN + 0.4, "same weights at every position; positions do not interact",
        ha="center", va="center", fontsize=8, color=INK2, zorder=5, bbox=dict(fc=BAND, ec="none", pad=1.5))
ax.text(rx, Y_OUT + 0.3, "prediction at each position = the next input token", ha="center", va="center", fontsize=8, color=INK2)

fig.savefig(os.path.join(OUT, "architecture.png"), dpi=170, facecolor=SURFACE)
print("ok")
