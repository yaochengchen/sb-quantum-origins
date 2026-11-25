"""MSBQA paper 1, Fig. 7 (v17): single-setting replacements between L4 and bSB*, Ising-window holds, n = 4 and n = 6.
 (a) L4 with one setting replaced by its bSB* value (+), and bSB* with one setting put back to its L4 value (-);
     run_v4/phase1c/ablate_per_instance.csv, G4 0-499, trap / non-trap
 (b) L4 with a hold of 1000/K at p = mean detuning or at p_f (IW columns of run_v4/phase1d/per_instance_1d.csv)
 (c) mean success on the crossing instances for n = 4 (17 of 300) and n = 6 (36 of 300); the n = 6 quantum value
     is the Landau-Zener estimate (hatched)
Same data and numbers as fig7_sb_n6.py / fig7_sb_n6_v15.py. Run from msbqa_phase1/:
    python paper1_figs/fig7_sb_n6_v17.py
Writes paper1_figs/fig7_sb_n6_v17.pdf / .png and fig7_sb_n6_v17_numbers.txt.
"""
import numpy as np
import pandas as pd

import p1common as pc

say = pc.Numbers("fig7_sb_n6_v17")
plt = pc.setup_mpl()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw=dict(width_ratios=[1.5, 0.95, 1.05]))

# ---------------------------------------------------------------- (a) one setting replaced
ab = pd.read_csv(pc.P("run_v4", "phase1c", "ablate_per_instance.csv")).set_index("inst")
CFG = ["L4", "L4 +D", "L4 +af", "L4 +T", "L4 +init", "L4 +c", "bSB* -D", "bSB* -T", "bSB*"]
LAB = ["L4", "$+$D", "$+a_f$", "$+$T", "$+$init", "$+$c", "$-$D", "$-$T", "bSB$^*$"]
tr = ab.trap == 1
say(f"(a) {len(ab)} instances, trap {int(tr.sum())}")
mt = [ab.loc[tr, c].mean() for c in CFG]
mn = [ab.loc[~tr, c].mean() for c in CFG]
for c, a_, b_ in zip(CFG, mt, mn):
    say(f"    {c:10s} trap {a_:.3f}  non-trap {b_:.3f}  all {ab[c].mean():.3f}")
a = ax[0]
x = np.arange(len(CFG), dtype=float)
x[6:8] += 1.0
x[8] += 1.6
a.bar(x - 0.2, mt, 0.4, color=pc.RED, label="trap")
a.bar(x + 0.2, mn, 0.4, color=pc.BLUE, label="non-trap")
a.set_xticks(x); a.set_xticklabels(LAB, fontsize=6.8)
a.set_ylim(0, 1.22); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a.set_ylabel(r"mean $P_{\rm succ}$")
a.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.5, columnspacing=1.0, borderaxespad=0.1)
tx = a.get_xaxis_transform()
for x0, x1, t in ((x[1], x[5], "L4 with one setting\ntaken from bSB$^*$"), (x[6], x[7], "bSB$^*$ with one\nsetting from L4")):
    a.plot([x0 - 0.4, x1 + 0.4], [-0.15, -0.15], c="k", lw=0.6, transform=tx, clip_on=False)
    a.text(0.5 * (x0 + x1), -0.18, t, ha="center", va="top", fontsize=6.3, transform=tx, linespacing=1.1)

# ---------------------------------------------------------------- (b) Ising window
d = pd.read_csv(pc.P("run_v4", "phase1d", "per_instance_1d.csv")).set_index("inst")
iw = d[d["IW kpo base"].notna()]
grp = [("uniform-mean " + r"$\bar\Delta$", "mean", iw.cross_mean.clip(lower=0) == 1),
       ("per-spin " + r"$\Delta_i$", "kpo", iw.cross_kpo.clip(lower=0) == 1)]
CF = [("base", "ramp only"), ("hold@Dbar 1000", r"hold at $p=\bar\Delta$"), ("hold@pf 1000", r"hold at $p=p_f$")]
say(f"(b) Ising window: {len(iw)} instances")
for lab, det, m in grp:
    say(f"    {det:4s} detuning, crossing instances N {int(m.sum()):3d}: " +
        ", ".join(f"{c} {iw.loc[m, f'IW {det} {c}'].mean():.3f}" for c, _ in CF))
bsb_mean = ab.loc[ab.index.isin(iw.index[grp[0][2]]), "bSB*"].mean()
say(f"    bSB* on the uniform-mean crossing instances: {bsb_mean:.3f}")
b = ax[1]
w = 0.26
for j, ((c, lab), col) in enumerate(zip(CF, (pc.BLUE, pc.GREEN, pc.GREY))):
    v = [iw.loc[m, f"IW {det} {c}"].mean() for _, det, m in grp]
    b.bar(np.arange(len(grp)) + (j - 1) * w, v, w, color=col, label=lab)
b.plot([-0.45, 0.45], [bsb_mean] * 2, c="k", ls="--", lw=0.8)
b.text(0.0, bsb_mean + 0.015, "bSB$^*$", ha="center", va="bottom", fontsize=6.5)
b.set_xticks(range(len(grp)))
b.set_xticklabels([f"{lab}\n{int(m.sum())} crossings" for lab, _, m in grp], fontsize=6.5)
b.set_ylim(0, 1.22); b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); b.set_ylabel(r"mean $P_{\rm succ}$ (L4)")
b.legend(loc="upper right", frameon=False, fontsize=6.3, borderaxespad=0.1, labelspacing=0.3)

# ---------------------------------------------------------------- (c) n = 4 and n = 6, crossing instances
b4 = pc.load_1b()
e6 = pd.read_csv(pc.P("run_v4", "phase1e", "per_instance_1e.csv")).set_index("inst")
e6["cross"] = e6["cross"].fillna(0).clip(lower=0).astype(int)
c4, c6 = b4.cross == 1, e6.cross == 1
rows = [("L2 $\\varepsilon^*$", b4.loc[c4, pc.C_L2G].mean(), e6.loc[c6, "L2g_eps*"].mean()),
        ("L4 $\\varepsilon^*$", b4.loc[c4, pc.C_L4].mean(), e6.loc[c6, "L4_eps*"].mean()),
        ("bSB$^*$", b4.loc[c4, pc.C_BSB].mean(), e6.loc[c6, "bSB*"].mean()),
        ("quantum", b4.loc[c4, "Q_weyl"].mean(), e6.loc[c6 & e6.PQ_LZ.notna(), "PQ_LZ"].mean())]
say(f"(c) n=4: crossing {int(c4.sum())}/{len(b4)} ({c4.mean():.3f});  n=6: crossing {int(c6.sum())}/{len(e6)} ({c6.mean():.3f}), "
    f"with gap {int((c6 & e6.Dmin.notna()).sum())}")
for lab, v4, v6 in rows:
    say(f"    {lab:22s} n=4 {v4:.3f}   n=6 {v6:.3f}")
c = ax[2]
xr = np.arange(len(rows))
c.bar(xr - 0.2, [r[1] for r in rows], 0.4, color="#9DB4D3", label=f"$n=4$ ({int(c4.sum())} of {len(b4)})")
c.bar(xr + 0.2, [r[2] for r in rows], 0.4, color=pc.BLUE, label=f"$n=6$ ({int(c6.sum())} of {len(e6)})")
c.bar(xr[-1] + 0.2, rows[-1][2], 0.4, color="none", edgecolor="w", hatch="////", lw=0)
c.text(xr[-1] + 0.2, rows[-1][2] + 0.015, "LZ", ha="center", va="bottom", fontsize=6.3)
c.set_xticks(xr); c.set_xticklabels([r[0] for r in rows], fontsize=6.5)
c.set_ylim(0, 1.42); c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); c.set_ylabel(r"mean $P_{\rm succ}$ on crossings")
c.legend(loc="upper left", frameon=False, fontsize=6.5, borderaxespad=0.1, title="crossing instances", title_fontsize=6.5,
         alignment="left")
pc.panel_labels(ax)
fig.tight_layout(w_pad=0.9)
pc.save(fig, "fig7_sb_n6_v17")
say.write()
