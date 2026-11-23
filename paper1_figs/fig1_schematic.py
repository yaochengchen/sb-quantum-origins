"""MSBQA paper 1, Fig. 1: schematic (no data). (a) ablation chain; (b) classical branches with an energy crossing;
(c) quantum avoided crossing whose gap closes as exp(-S/h) and moves to p_c as h -> 0. Toy curves only.
    python paper1_figs/fig1_schematic.py
"""
import numpy as np

import p1common as pc

plt = pc.setup_mpl()
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(3.4, 3.5))
gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.38, wspace=0.42)
a0 = fig.add_subplot(gs[0, :])
a1 = fig.add_subplot(gs[1, 0])
a2 = fig.add_subplot(gs[1, 1])

# (a) chain
a0.set_axis_off()
a0.set_xlim(0, 1); a0.set_ylim(0, 1)
boxes = [("L0\nquantum", pc.PURPLE), ("L1\nTWA", pc.RED), ("L2\nclass.\nKPO", pc.ORANGE),
         ("L3\naSB", pc.GREEN), ("L4\nbSB-\ntype", pc.BLUE), ("bSB$^*$\ndSB$^*$", "#333333")]
steps = ["no Moyal\nterms", "no vacuum\nnoise", "no Kerr\ncoupling", "walls,\nno $x^3$", "uniform\ndetuning"]
xs = np.linspace(0.075, 0.925, len(boxes))
bw = 0.115
for x, (t, c) in zip(xs, boxes):
    a0.add_patch(FancyBboxPatch((x - bw / 2, 0.42), bw, 0.52, boxstyle="round,pad=0.008", fc=c, ec="none", alpha=0.2))
    a0.text(x, 0.68, t, ha="center", va="center", fontsize=6.2)
for x0, x1, s in zip(xs[:-1], xs[1:], steps):
    a0.annotate("", xy=(x1 - bw / 2 - 0.004, 0.68), xytext=(x0 + bw / 2 + 0.004, 0.68),
                arrowprops=dict(arrowstyle="->", lw=0.7, shrinkA=0, shrinkB=0))
    a0.text(0.5 * (x0 + x1), 0.16, s, ha="center", va="center", fontsize=5.3, color="#444444")

# (b) classical branches (toy): trap branch born first, ground branch born later and crosses below at p_c
p = np.linspace(0, 4, 400)
pb_trap, pb_gs, pc_ = 0.5, 0.8, 2.2
Et = np.where(p >= pb_trap, -0.25 * (p - pb_trap) ** 2, np.nan)
Eg = np.where(p >= pb_gs, Et + 0.3 * (pc_ - p), np.nan)
a1.plot(p, Et, c=pc.RED, lw=1.3, label="trap cfg.")
a1.plot(p, Eg, c="k", lw=1.3, label="ground cfg.")
a1.axvline(pc_, c="k", ls="--", lw=0.6)
a1.text(pc_ + 0.07, 0.25, r"$p_{\rm c}$", fontsize=8)
a1.set_xlabel(r"pump $p$"); a1.set_ylabel(r"classical $E_{\mathbf{s}}(p)$")
a1.set_xticks([]); a1.set_yticks([])
a1.set_xlim(0, 4)
a1.legend(loc="lower left", frameon=False, fontsize=6)

# (c) quantum two lowest levels (toy two-level model around a crossing shifted by zero-point energy)
for h, col in ((1.0, pc.PURPLE), (0.5, pc.BLUE), (0.25, pc.GREEN)):
    ps = pc_ + 0.8 * h                         # p* -> p_c as h -> 0
    gap = 0.5 * np.exp(-0.9 / h)
    e = 0.3 * (p - ps)
    a2.plot(p, -np.sqrt(e ** 2 + gap ** 2), c=col, lw=1.0, label=f"$h={h:g}$")
    a2.plot(p, +np.sqrt(e ** 2 + gap ** 2), c=col, lw=1.0, ls="--")
a2.axvline(pc_, c="k", ls="--", lw=0.6)
a2.text(pc_ + 0.07, 0.62, r"$p_{\rm c}$", fontsize=8)
a2.set_xlabel(r"pump $p$"); a2.set_ylabel(r"quantum $E_{0},E_{1}$")
a2.set_xticks([]); a2.set_yticks([])
a2.set_xlim(0, 4); a2.set_ylim(-0.8, 0.8)
a2.legend(loc="lower left", frameon=False, fontsize=6)
for a_, t in zip((a0, a1, a2), "abc"):
    a_.text(-0.02, 1.02, f"({t})", transform=a_.transAxes, va="bottom", ha="left", fontweight="bold")
pc.save(fig, "fig1_schematic")
