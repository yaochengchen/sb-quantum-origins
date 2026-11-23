"""MSBQA paper 1, Fig. 2 (v16 layout): quantum vs classical baselines, per instance, n = 4 test set (G4 0-299).
Replaces the five scatter panels of fig2_who_wins.py (all points piled up at P_Q ~ 1) by one strip plot of the
per-instance advantage P_Q - P_B for the six baselines, plus the class-resolved counts with P_Q - P_B > 0.1.
Same inputs and numbers as fig2_who_wins.py. Run from msbqa_phase1/:
    python paper1_figs/fig2_who_wins_v16.py
Writes paper1_figs/fig2_who_wins_v16.pdf / .png and fig2_who_wins_v16_numbers.txt.
"""
import numpy as np

import p1common as pc

say = pc.Numbers("fig2_who_wins_v16")
df = pc.load_1b()
Q = df["Q_weyl"]
BASE = [(pc.C_L2G_GOTO, "L2\n" + r"$\varepsilon=10^{-6}$"),
        (pc.C_L2G, "L2\n" + r"$\varepsilon^*$"),
        (pc.C_L3, "L3\n(aSB)"),
        (pc.C_L4, "L4\n(bSB-type)"),
        (pc.C_BSB, r"bSB$^*$"),
        (pc.C_DSB, r"dSB$^*$")]
ORDER = ("other", "trap, no crossing", "crossing")          # drawing order: crossings on top

say(f"test instances: {len(df)};  classes: " + ", ".join(f"{k} {v}" for k, v in df.cls.value_counts().items()))
counts = {}
for col, _ in BASE:
    d = Q - df[col]
    counts[col] = {k: int(((d > 0.1) & (df.cls == k)).sum()) for k in ORDER}
    say(f"{col:45s} Q-B>0.1: {int((d > 0.1).sum()):3d} [crossing {counts[col]['crossing']}, trap-no-cross "
        f"{counts[col]['trap, no crossing']}, other {counts[col]['other']}] | B-Q>0.02: {int((d < -0.02).sum()):3d} "
        f"| min(Q-B) {d.min():+.3f}")

plt = pc.setup_mpl()
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[2.3, 1.0]))
rng = np.random.default_rng(2)
x0 = np.arange(len(BASE))
for xi, (col, _) in zip(x0, BASE):
    d = (Q - df[col]).values
    jit = rng.uniform(-0.32, 0.32, len(d))
    for k in ORDER:
        m = (df.cls == k).values
        st = pc.CLS_STYLE[k]
        a.scatter(xi + jit[m], d[m], s=st["s"], c=st["c"], alpha=st["alpha"], lw=0, zorder=st["z"],
                  label=k if xi == 0 else None)
a.axhline(0.0, c="k", lw=0.6, zorder=0)
a.axhline(0.1, c=pc.GREY, ls=":", lw=0.8, zorder=0)
a.set_xticks(x0)
a.set_xticklabels([lab for _, lab in BASE], fontsize=7.5)
a.set_xlim(-0.6, len(BASE) - 0.4)
a.set_ylim(-1.08, 1.05)
a.set_ylabel(r"$P_{\rm Q}-P_{\rm B}$ (per instance)")
for xi in x0[:-1]:
    a.axvline(xi + 0.5, c="0.9", lw=0.6, zorder=0)

labs = ["L2\n" + r"$10^{-6}$", "L2\n" + r"$\varepsilon^*$", "L3", "L4", r"bSB$^*$", r"dSB$^*$"]
bottom = np.zeros(len(BASE))
for k in ("crossing", "trap, no crossing", "other"):
    v = np.array([counts[col][k] for col, _ in BASE])
    b.bar(x0, v, bottom=bottom, color=pc.CLS_STYLE[k]["c"], width=0.7, label=k)
    bottom += v
for xi, tot in zip(x0, bottom):
    b.text(xi, tot + 1.5, f"{int(tot)}", ha="center", va="bottom", fontsize=7)
b.set_xticks(x0)
b.set_xticklabels(labs, fontsize=7)
b.set_ylabel(r"instances with $P_{\rm Q}-P_{\rm B}>0.1$")
b.set_ylim(0, max(bottom) * 1.15 + 2)
h, l = b.get_legend_handles_labels()
b.legend(h[::-1], l[::-1], loc="upper right", frameon=False, fontsize=7, handlelength=1.0)
pc.panel_labels([a, b])
fig.tight_layout(w_pad=1.2)
pc.save(fig, "fig2_who_wins_v16")
say.write()
