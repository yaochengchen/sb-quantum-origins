"""MSBQA paper 1, Fig. 2 (claim C1): quantum vs classical baselines, per instance, n = 4 test set (G4 0-299).
Reads run_v4/phase1b/per_instance_1b.csv and run_v4/branches/branches_weyl.csv. Run from msbqa_phase1/:
    python paper1_figs/fig2_who_wins.py
"""
import numpy as np

import p1common as pc

say = pc.Numbers("fig2_who_wins")
df = pc.load_1b()
Q = df["Q_weyl"]
BASE = [(pc.C_L2G_GOTO, r"L2, $\varepsilon=10^{-6}$ (Goto)"),
        (pc.C_L2G, r"L2, $\varepsilon^*=0.03$"),
        (pc.C_L3, r"L3 (aSB), $\varepsilon^*=10^{-3}$"),
        (pc.C_L4, r"L4 (bSB-type), $\varepsilon^*=10^{-6}$"),
        (pc.C_BSB, r"bSB$^*$")]
EXTRA = [(pc.C_DSB, "dSB*")]

say(f"test instances: {len(df)};  classes: " + ", ".join(f"{k} {v}" for k, v in df.cls.value_counts().items()))
say(f"quantum (weyl, h = 1, T = 700) mean: all {Q.mean():.4f}, crossing {Q[df.cross == 1].mean():.4f}, "
    f"other {Q[df.cross == 0].mean():.4f};  goto family mean {df['Q_goto'].mean():.4f}")
counts = {}
for col, lab in BASE + EXTRA:
    B = df[col]
    d = Q - B
    row = {k: int(((d > 0.1) & (df.cls == k)).sum()) for k in pc.CLS_STYLE}
    counts[col] = row
    say(f"{col:45s} mean B {B.mean():.3f} (crossing {B[df.cross == 1].mean():.3f}, no crossing "
        f"{B[df.cross == 0].mean():.3f}) | Q-B>0.1: {int((d > 0.1).sum()):3d} "
        f"[crossing {row['crossing']}, trap-no-cross {row['trap, no crossing']}, other {row['other']}] "
        f"| Q-B>0.2: {int((d > 0.2).sum()):3d} | B-Q>0.02: {int((d < -0.02).sum()):3d}")
best_kpo = df[[pc.C_L2G, pc.C_L3, pc.C_L4]].max(axis=1)
cr = df.cross == 1
say(f"crossing instances: best KPO-type (max L2g/L3/L4 at eps*) mean {best_kpo[cr].mean():.3f}, quantum "
    f"{Q[cr].mean():.3f}, Q - best KPO > 0.1 on {int(((Q - best_kpo) > 0.1)[cr].sum())} of {int(cr.sum())}")
say(f"goto family: bSB* - Q_goto > 0.02 on {int((df[pc.C_BSB] - df['Q_goto'] > 0.02).sum())}")

plt = pc.setup_mpl()
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.5))
axs = axs.ravel()
for a, (col, lab) in zip(axs[:5], BASE):
    for k, st in pc.CLS_STYLE.items():
        m = df.cls == k
        a.scatter(df.loc[m, col], Q[m], s=st["s"], c=st["c"], alpha=st["alpha"], lw=0, zorder=st["z"], label=k)
    a.plot([0, 1], [0, 1], "k--", lw=0.6)
    a.plot([0, 0.9], [0.1, 1.0], c=pc.GREY, ls=":", lw=0.6)
    a.set_xlim(-0.03, 1.03); a.set_ylim(-0.03, 1.03)
    a.set_xlabel(lab, fontsize=8)
    a.set_ylabel(r"quantum $P_{\rm succ}$")
axs[0].legend(loc="lower right", frameon=False, markerscale=1.5, fontsize=6.5)

a = axs[5]
labs = ["L2\n(Goto)", "L2\n$\\varepsilon^*$", "L3", "L4", "bSB$^*$", "dSB$^*$"]
x = np.arange(len(labs))
bottom = np.zeros(len(labs))
for k in ("crossing", "trap, no crossing", "other"):
    v = np.array([counts[col][k] for col, _ in BASE + EXTRA])
    a.bar(x, v, bottom=bottom, color=pc.CLS_STYLE[k]["c"], width=0.7, label=k)
    bottom += v
for xi, b in zip(x, bottom):
    a.text(xi, b + 1, f"{int(b)}", ha="center", va="bottom", fontsize=7)
a.set_xticks(x); a.set_xticklabels(labs, fontsize=7)
a.set_ylabel(r"# instances, $P_Q-P_B>0.1$")
a.set_ylim(0, max(bottom) * 1.22 + 1)
pc.panel_labels(axs)
fig.tight_layout(w_pad=0.8, h_pad=0.8)
pc.save(fig, "fig2_who_wins")
say.write()
