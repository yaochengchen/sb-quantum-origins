"""MSBQA paper 1, Fig. 1 (v17; schematic, no data): the comparisons made and the crossing correspondence.
Same content as Jiapei's version B1 (fig1_B1.py), drawn at the width of one journal column (3.4 in) so that the
lettering is not scaled down in the paper, with the font and panel labels of the other figures.
 (a) the three comparisons on the same Ising instances
 (b) two classical configuration branches exchanging their energy order at p_c
 (c) the two lowest even-parity quantum levels at three values of h (illustrative functions)
Run from msbqa_phase1/:
    python paper1_figs/fig1_schematic_v17.py
Writes paper1_figs/fig1_schematic_v17.pdf / .png.
"""
import numpy as np

import p1common as pc

plt = pc.setup_mpl()
INK, MUTED = "#202833", "#586474"
fig = plt.figure(figsize=(3.4, 3.55))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.42, left=0.085, right=0.985, top=0.955,
                      bottom=0.075)
ax = fig.add_subplot(gs[0, :])
a1 = fig.add_subplot(gs[1, 0])
a2 = fig.add_subplot(gs[1, 1])

# ---------------------------------------------------------------- (a) the comparisons
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
XC = (0.0, 0.54, 0.83)
for x, t, ha in zip(XC, ("what is compared", "reference", "against"), ("left", "center", "center")):
    ax.text(x, 0.93, t, ha=ha, va="center", fontsize=6.5, color=MUTED)
ax.plot([0, 1], [0.85, 0.85], c=INK, lw=0.8)
ROWS = [("initial fluctuations", "L0", "quantum", "L1, L2", "Wigner, uniform noise", pc.BLUE),
        ("equations of motion", "L2", "classical KPO", "L3, L4", "aSB, bSB-type", pc.GREEN),
        ("solver performance", "L0", "quantum", "bSB$^*$, dSB$^*$", "tuned separately", pc.ORANGE)]
for k, (what, ref, refs, oth, oths, col) in enumerate(ROWS):
    y = 0.70 - 0.27 * k
    ax.text(XC[0], y, what, ha="left", va="center", fontsize=7.5, color=col, fontweight="bold")
    for x, big, small in ((XC[1], ref, refs), (XC[2], oth, oths)):
        ax.text(x, y + 0.055, big, ha="center", va="center", fontsize=8, color=INK, fontweight="bold")
        ax.text(x, y - 0.065, small, ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.plot([0, 1], [y - 0.14, y - 0.14], c="#b9bec4", lw=0.5)

# ---------------------------------------------------------------- (b) classical branches (illustrative)
p = np.linspace(0, 4, 400)
PC = 2.2
Et = np.where(p >= 0.5, -0.25 * (p - 0.5) ** 2, np.nan)
Eg = np.where(p >= 0.8, Et + 0.3 * (PC - p), np.nan)
a1.plot(p, Et, c=pc.RED, lw=1.3)
a1.plot(p, Eg, c="k", lw=1.3)
a1.axvline(PC, c=INK, ls=":", lw=0.8)
a1.text(PC + 0.12, 0.0, r"$p_{\rm c}$", fontsize=8, va="center")
a1.text(0.15, -2.45, "configuration", fontsize=6.5, color=MUTED)
a1.text(0.15, -2.90, "competing", fontsize=7, color=pc.RED)
a1.text(0.15, -3.35, "ground", fontsize=7, color="k")
a1.set(xlim=(0, 4), xlabel=r"pump $p$", ylabel="classical branch energy")
a1.set_xticks([]); a1.set_yticks([])

# ---------------------------------------------------------------- (c) quantum levels (illustrative)
for h, col in ((1.0, pc.PURPLE), (0.5, pc.BLUE), (0.25, pc.GREEN)):
    ps = PC + 0.8 * h
    gap = 0.5 * np.exp(-0.9 / h)
    lvl = np.sqrt((0.3 * (p - ps)) ** 2 + gap ** 2)
    a2.plot(p, -lvl, c=col, lw=1.2, label=fr"${h:g}$")
    a2.plot(p, +lvl, c=col, lw=1.2, ls="--")
a2.axvline(PC, c=INK, ls=":", lw=0.8)
a2.text(PC + 0.1, -0.74, r"$p_{\rm c}$", fontsize=8)
a2.text(3.25, -0.66, r"$E_0$", fontsize=7.5)
a2.text(3.25, 0.58, r"$E_1$", fontsize=7.5)
a2.set(xlim=(0, 4), ylim=(-0.8, 0.8), xlabel=r"pump $p$", ylabel="quantum levels")
a2.set_xticks([]); a2.set_yticks([])
a2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=6.5, handlelength=1.0,
          columnspacing=0.6, handletextpad=0.3, title=None)
a2.text(-0.02, 1.07, r"$h=$", transform=a2.transAxes, fontsize=6.5, ha="left", va="bottom")

for a, t, x in ((ax, "a", -0.035), (a1, "b", -0.08), (a2, "c", -0.08)):
    a.text(x, 1.03 if a is ax else 1.16, f"({t})", transform=a.transAxes, va="bottom", ha="left", fontweight="bold")
pc.save(fig, "fig1_schematic_v17")
