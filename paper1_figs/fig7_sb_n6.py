"""MSBQA paper 1, Fig. 7 (claims C6, C7).
 (a) one-ingredient ablation L4 <-> bSB* (run_v4/phase1c/ablate_per_instance.csv, 500 instances), trap / non-trap
 (b) Ising window (IW columns of run_v4/phase1d/per_instance_1d.csv, instances 0-499)
 (c) n = 4 vs n = 6 (phase1b per-instance + branches_weyl for n = 4; phase1e per-instance for n = 6)
Run from msbqa_phase1/:
    python paper1_figs/fig7_sb_n6.py
"""
import numpy as np
import pandas as pd

import p1common as pc

say = pc.Numbers("fig7_sb_n6")
plt = pc.setup_mpl()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.1]))

# ---------------------------------------------------------------- (a) ablation
ab = pd.read_csv(pc.P("run_v4", "phase1c", "ablate_per_instance.csv")).set_index("inst")
CFG = ["L4", "L4 +D", "L4 +af", "L4 +T", "L4 +init", "L4 +c", "bSB* -D", "bSB* -T", "bSB*"]
LAB = ["L4", "+D", "+af", "+T", "+init", "+c", "$-$D", "$-$T", "bSB$^*$"]
tr = ab.trap == 1
say(f"(a) ablation: {len(ab)} instances, trap {int(tr.sum())}")
mt = [ab.loc[tr, c].mean() for c in CFG]
mn = [ab.loc[~tr, c].mean() for c in CFG]
for c, a_, b_ in zip(CFG, mt, mn):
    say(f"    {c:10s} trap {a_:.3f}  non-trap {b_:.3f}  all {ab[c].mean():.3f}")
x = np.arange(len(CFG))
ax[0].bar(x - 0.2, mt, 0.4, color=pc.RED, label="trap")
ax[0].bar(x + 0.2, mn, 0.4, color=pc.BLUE, label="non-trap")
ax[0].set_xticks(x); ax[0].set_xticklabels(LAB, fontsize=6.2)
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel(r"mean $P_{\rm succ}$")
ax[0].axvline(5.5, c=pc.GREY, lw=0.5, ls=":")
ax[0].set_xlabel("L4 + one ingredient  |  bSB$^*$ $-$ one", fontsize=7)
ax[0].set_ylim(0, 1.18)
ax[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=6.2)

# ---------------------------------------------------------------- (b) Ising window
d = pd.read_csv(pc.P("run_v4", "phase1d", "per_instance_1d.csv")).set_index("inst")
iw = d[d["IW kpo base"].notna()]
# each detuning on the instances that have a crossing under that detuning
grp = [("uniform $\\bar\\Delta$", "mean", iw.cross_mean.clip(lower=0) == 1),
       ("per-spin $\\Delta_i$", "kpo", iw.cross_kpo.clip(lower=0) == 1)]
neither = (iw.trapK == 0) & (iw.trapU == 0) & (iw.cross_kpo.clip(lower=0) == 0) & (iw.cross_mean.clip(lower=0) == 0)
CF = [("base", "ramp only"), ("hold@Dbar 1000", r"hold at $p=\bar\Delta$"), ("hold@pf 1000", r"hold at $p=p_f$")]
say(f"(b) Ising window: {len(iw)} instances")
for lab, det, m in grp:
    say(f"    {det:4s} detuning, crossing instances N {int(m.sum()):3d}: " +
        ", ".join(f"{c} {iw.loc[m, f'IW {det} {c}'].mean():.3f}" for c, _ in CF))
say(f"    control (no trap, no crossing) N {int(neither.sum())}: " +
    ", ".join(f"{det} {c} {iw.loc[neither, f'IW {det} {c}'].mean():.3f}" for det in ("mean", "kpo") for c, _ in CF))
bsb_mean = ab.loc[ab.index.isin(iw.index[grp[0][2]]), "bSB*"].mean()
say(f"    bSB* on the uniform-detuning crossing instances: {bsb_mean:.3f}")
cols = [pc.BLUE, pc.GREEN, pc.GREY]
w = 0.26
for j, ((c, lab), col) in enumerate(zip(CF, cols)):
    v = [iw.loc[m, f"IW {det} {c}"].mean() for _, det, m in grp]
    ax[1].bar(np.arange(len(grp)) + (j - 1) * w, v, w, color=col, label=lab)
ax[1].plot([-0.45, 0.45], [bsb_mean] * 2, c="k", ls="--", lw=0.8)
ax[1].text(0.0, bsb_mean + 0.015, "bSB$^*$", ha="center", va="bottom", fontsize=6.5)
ax[1].set_xticks(range(len(grp)))
ax[1].set_xticklabels([f"{lab}\ncrossing (N={int(m.sum())})" for lab, _, m in grp], fontsize=6.5)
ax[1].set_ylim(0, 1.15); ax[1].set_ylabel(r"mean $P_{\rm succ}$ (L4 dynamics)")
ax[1].legend(loc="upper right", frameon=False, fontsize=6)

# ---------------------------------------------------------------- (c) n = 4 vs n = 6
b4 = pc.load_1b()
e6 = pd.read_csv(pc.P("run_v4", "phase1e", "per_instance_1e.csv")).set_index("inst")
e6["cross"] = e6["cross"].fillna(0).clip(lower=0).astype(int)
c4, c6 = b4.cross == 1, e6.cross == 1
rows = [("cross.\nfrac.", c4.mean(), c6.mean()),
        ("L2 $\\varepsilon^*$", b4.loc[c4, pc.C_L2G].mean(), e6.loc[c6, "L2g_eps*"].mean()),
        ("L4 $\\varepsilon^*$", b4.loc[c4, pc.C_L4].mean(), e6.loc[c6, "L4_eps*"].mean()),
        ("bSB$^*$", b4.loc[c4, pc.C_BSB].mean(), e6.loc[c6, "bSB*"].mean()),
        ("quantum", b4.loc[c4, "Q_weyl"].mean(), e6.loc[c6 & e6.PQ_LZ.notna(), "PQ_LZ"].mean())]
say(f"(c) n=4: trap {b4.trap.mean():.3f}, crossing {int(c4.sum())}/{len(b4)};  n=6: trap {e6.trap.mean():.3f}, "
    f"crossing {int(c6.sum())}/{len(e6)}, crossing with gap {int((c6 & e6.Dmin.notna()).sum())}")
for lab, v4, v6 in rows:
    say(f"    {lab.replace(chr(10), ' '):22s} n=4 {v4:.3f}   n=6 {v6:.3f}")
say(f"    n=6 Dmin median crossing {e6.loc[c6, 'Dmin'].median():.3f}, no crossing {e6.loc[~c6, 'Dmin'].median():.3f};  "
    f"bSB* == 0 on {int((e6['bSB*'] < 0.01).sum())} instances")
x = np.arange(len(rows))
ax[2].bar(x - 0.2, [r[1] for r in rows], 0.4, color="#9DB4D3", label="$n=4$")
ax[2].bar(x + 0.2, [r[2] for r in rows], 0.4, color=pc.BLUE, label="$n=6$")
ax[2].set_xticks(x); ax[2].set_xticklabels([r[0] for r in rows], fontsize=6.2)
ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("fraction / success on crossing")
ax[2].legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False, fontsize=6.5)
ax[2].set_ylim(0, 1.12)
pc.panel_labels(ax)
fig.tight_layout(w_pad=0.6)
pc.save(fig, "fig7_sb_n6")
say.write()
