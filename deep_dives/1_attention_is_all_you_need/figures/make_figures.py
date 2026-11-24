"""Draws attention_matrix.png, softmax_scaling.png, positional_encoding.png and lr_schedule.png for
../decoder_only_transformer.md. Needs torch, numpy and matplotlib.
Usage: python make_figures.py [OUT_DIR]   (default: this folder)"""
import math, sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)


# 1. Attention matrix: scores -> causal mask -> softmax weights
torch.manual_seed(3)
tokens = ["The", "cat", "sat", "on", "the", "mat"]
n, d_model, d_k = len(tokens), 16, 8
X = torch.randn(n, d_model)
W_q, W_k = torch.randn(d_model, d_k) / d_model**0.5, torch.randn(d_model, d_k) / d_model**0.5
S = (X @ W_q) @ (X @ W_k).T / math.sqrt(d_k) * 2.0  # x2 so the softmax weights are visibly uneven
mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
A = torch.softmax(S.masked_fill(~mask, float("-inf")), dim=-1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
titles = ["1. Scores  S = QKᵀ / √d_k", "2. Causal mask: S_ij = −∞ for j > i", "3. Weights  A = softmax(S) by row"]
lim = S.abs().max().item()
for p, ax in enumerate(axes):
    if p == 2:
        ax.imshow(A.numpy(), cmap=SEQ, vmin=0, vmax=1)
    else:
        M = S.numpy().copy()
        if p == 1:
            M = np.ma.masked_where(~mask.numpy(), M)
        ax.imshow(M, cmap=DIV, vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if p > 0 and j > i:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#ebeae6", edgecolor=SURFACE, hatch="////", lw=0))
                ax.text(j, i, "−∞" if p == 1 else "0", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            val = A[i, j].item() if p == 2 else S[i, j].item()
            dark = (val > 0.55) if p == 2 else abs(val) > 0.6 * lim
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="white" if dark else INK)
    ax.set_xticks(range(n), tokens)
    ax.set_yticks(range(n), tokens)
    ax.xaxis.tick_top()
    ax.set_xlabel("key position j (attended to)")
    ax.xaxis.set_label_position("top")
    ax.set_title(titles[p], pad=34, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel("query position i (attending)")
axes[2].text(n - .5, n + .1, "each row sums to 1", ha="right", va="top", color=INK2, fontsize=9)
save(fig, "attention_matrix.png")


# 2. Why divide by sqrt(d_k): softmax saturation
g = torch.Generator().manual_seed(7)
d_k, n_keys = 64, 10
q = torch.randn(d_k, generator=g)
K = torch.randn(n_keys, d_k, generator=g)
raw = K @ q
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, scores, title in [
    (axes[0], raw, f"Without scaling: scores q·k (std ≈ √{d_k} = 8)"),
    (axes[1], raw / math.sqrt(d_k), f"With scaling: q·k / √{d_k} (std ≈ 1)"),
]:
    w = torch.softmax(scores, dim=0).numpy()
    ax.bar(range(n_keys), w, color=BLUE, width=0.7)
    ax.set_title(title, loc="left")
    ax.set_xticks(range(n_keys), [f"k{j+1}" for j in range(n_keys)])
    ax.set_xlabel("key")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    top = int(w.argmax())
    ax.text(top, w[top] + 0.02, f"{w[top]:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
axes[0].set_ylabel("softmax weight")
save(fig, "softmax_scaling.png")


# 3. Sinusoidal positional encoding
d_model, n_pos = 128, 100
pos = np.arange(n_pos)[:, None]
two_i = np.arange(0, d_model, 2)[None, :]
angle = pos / 10000 ** (two_i / d_model)
pe = np.zeros((n_pos, d_model))
pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})
im = axes[0].imshow(pe, cmap=DIV, vmin=-1, vmax=1, aspect="auto")
axes[0].set_xlabel("embedding dimension")
axes[0].set_ylabel("position pos")
axes[0].set_title(f"PE matrix (100 positions, d_model = {d_model})", loc="left")
cb = fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02, ticks=[-1, 0, 1])
cb.outline.set_visible(False)
for dim, color in [(8, BLUE), (24, ORANGE), (48, AQUA)]:
    wl = 2 * math.pi * 10000 ** (dim / d_model)
    axes[1].plot(pos[:, 0], pe[:, dim], color=color, lw=2, label=f"dim {dim}: wavelength {wl:.0f}")
axes[1].set_xlabel("position pos")
axes[1].set_ylabel("PE value")
axes[1].set_title("Columns 8, 24 and 48 of the matrix", loc="left")
axes[1].grid(color=GRID, lw=0.8)
axes[1].set_ylim(-1.1, 1.1)
axes[1].legend(frameon=False, loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.16), ncol=3)
save(fig, "positional_encoding.png")


# 4. Learning-rate schedule
d_model, warmup = 512, 4000
steps = np.arange(1, 100_001)
lr = d_model ** -0.5 * np.minimum(steps ** -0.5, steps * warmup ** -1.5)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(steps, lr, color=BLUE, lw=2)
peak = d_model ** -0.5 * warmup ** -0.5
ax.annotate(f"peak {peak:.1e} at step {warmup}", (warmup, peak), (12000, peak * 0.95), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.axvline(warmup, color=AXIS, lw=1, ls="--")
ax.text(warmup + 1500, peak * 0.08, "← linear warmup over the first 4000 steps", ha="left", color=INK2, fontsize=9)
ax.text(55000, lr[55000] * 1.25, "decay ∝ 1/√step", ha="center", color=INK2, fontsize=9)
ax.set_xlabel("training step")
ax.set_ylabel("learning rate")
ax.set_title("lrate = d_model^−0.5 · min(step^−0.5, step · warmup^−1.5),  d_model = 512, warmup = 4000", loc="left", fontsize=10)
ax.grid(color=GRID, lw=0.8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
save(fig, "lr_schedule.png")
print("ok")
