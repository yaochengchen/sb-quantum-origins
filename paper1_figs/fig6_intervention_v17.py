"""MSBQA paper 1, Fig. 6 (v17): paired per-spin (P) / uniform-max (U) detuning at fixed Ising couplings.
 (a) minimum gap (weyl, h = 1) under the two detunings, 119 instances in four groups
 (b) ln(Delta_min^U / Delta_min^P) per instance, group medians
 (c) classical success of L2, L3, L4 (eps = 1e-6) in the groups KO and MO, before and after
Same data and numbers as fig6_intervention.py / fig6_intervention_v14.py (run_v4/phase1d/per_instance_1d.csv).
Run from msbqa_phase1/:
    python paper1_figs/fig6_intervention_v17.py
Writes paper1_figs/fig6_intervention_v17.pdf / .png and fig6_intervention_v17_numbers.txt.
"""
import numpy as np
import pandas as pd

import p1common as pc

say = pc.Numbers("fig6_intervention_v17")
d = pd.read_csv(pc.P("run_v4", "phase1d", "per_instance_1d.csv")).set_index("inst")
GROUPS = [("KO", pc.RED, "KO: crossing removed"), ("MO", pc.BLUE, "MO: crossing created"),
          ("BOTH", pc.PURPLE, "BOTH: crossing kept"), ("CTRL", pc.GREY, "CTRL: no crossing")]
LAY = [("L2g", "L2"), ("L3", "L3"), ("L4", "L4")]
gp = d[d["group"].notna() & d.Dmin_K.notna() & d.Dmin_U.notna()].copy()
gp["lr"] = np.log(gp.Dmin_U / gp.Dmin_K)
for gname, _, _ in GROUPS + [("NEG", None, None)]:
    s = gp[gp.group == gname]
    if len(s):
        say(f"{gname:5s} N {len(s):3d}  median Dmin_P {s.Dmin_K.median():.3e}  Dmin_U {s.Dmin_U.median():.3e}  "
            f"geo-median ratio {np.exp(s.lr.median()):.3f}  frac U>P {(s.lr > 0).mean():.2f}")
chg = gp[gp.group.isin(["KO", "MO"])]
okdir = ((chg.group == "KO") & (chg.lr > 0)) | ((chg.group == "MO") & (chg.lr < 0))
say(f"direction as expected from the branches: {int(okdir.sum())} of {len(chg)}")
for gname in ("KO", "MO"):
    s = d[d.group == gname]
    say(f"classical {gname}: " + ", ".join(f"{lab} {s[f'{c}_K'].mean():.3f} -> {s[f'{c}_U'].mean():.3f}" for c, lab in LAY))
say(f"#514 (MO): Dmin_P {gp.Dmin_K[514]:.3e}, Dmin_U {gp.Dmin_U[514]:.3e} at p*_U {gp.pstar_U[514]:.2f} (end of the scan)")

plt = pc.setup_mpl()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.05]))
a = ax[0]
for gname, col, lab in GROUPS:
    s = gp[gp.group == gname]
    a.loglog(s.Dmin_K, s.Dmin_U, "o", ms=3, c=col, alpha=0.8, lw=0, label=lab)
lo = min(gp.Dmin_K.min(), gp.Dmin_U.min()) * 0.7
hi = max(gp.Dmin_K.max(), gp.Dmin_U.max()) * 1.4
a.plot([lo, hi], [lo, hi], "k--", lw=0.6)
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel(r"$\Delta_{\min}^{P}/K$ (per-spin detuning)")
a.set_ylabel(r"$\Delta_{\min}^{U}/K$ (uniform-max)")
a.legend(loc="lower right", frameon=False, fontsize=6.3, handletextpad=0.1, borderaxespad=0.1)

b = ax[1]
rng = np.random.default_rng(0)
xy514 = None
for k, (gname, col, lab) in enumerate(GROUPS):
    s = gp[gp.group == gname]
    xj = k + rng.uniform(-0.18, 0.18, len(s))
    b.scatter(xj, s.lr, s=6, c=col, lw=0, alpha=0.8)
    b.plot([k - 0.3, k + 0.3], [s.lr.median()] * 2, c="k", lw=1.2)
    if 514 in s.index:
        xy514 = (xj[list(s.index).index(514)], s.lr[514])
for t in (np.log(3), -np.log(3)):
    b.axhline(t, c=pc.GREY, ls="--", lw=0.6)
b.axhline(0, c="k", lw=0.4)
b.text(3.45, np.log(3) + 0.12, r"$\times3$", fontsize=6.3, color=pc.GREY, ha="right", va="bottom")
b.text(3.45, -np.log(3) - 0.12, r"$\times1/3$", fontsize=6.3, color=pc.GREY, ha="right", va="top")
b.set_xticks(range(len(GROUPS))); b.set_xticklabels([g_[0] for g_ in GROUPS], fontsize=7.5)
b.set_ylabel(r"$\ln(\Delta_{\min}^{U}/\Delta_{\min}^{P})$")
for aa, point, off in ((a, (gp.Dmin_K[514], gp.Dmin_U[514]), (-22, -14)), (b, xy514, (14, -9))):
    aa.scatter([point[0]], [point[1]], s=37, facecolors="none", edgecolors="k", linewidths=0.7, zorder=6)
    aa.annotate("#514", point, xytext=off, textcoords="offset points", fontsize=6.3,
                arrowprops=dict(arrowstyle="-", lw=0.5), zorder=7)

c = ax[2]
xt, xl = [], []
for k, (gname, col) in enumerate((("KO", pc.RED), ("MO", pc.BLUE))):
    s = d[d.group == gname]
    for j, (cc, lab) in enumerate(LAY):
        x0 = 4 * k + j
        c.bar(x0 - 0.2, s[f"{cc}_K"].mean(), 0.4, color=col, alpha=0.35)
        c.bar(x0 + 0.2, s[f"{cc}_U"].mean(), 0.4, color=col)
        xt.append(x0); xl.append(lab)
    c.text(4 * k + 1, -0.19, gname, ha="center", va="top", fontsize=7.5, transform=c.get_xaxis_transform())
c.bar([np.nan], [np.nan], color=pc.GREY, alpha=0.35, label="per-spin")
c.bar([np.nan], [np.nan], color=pc.GREY, label="uniform-max")
c.set_xticks(xt); c.set_xticklabels(xl, fontsize=7.5)
c.set_ylim(0, 1.12)
c.set_ylabel(r"classical $P_{\rm succ}$")
c.legend(loc="upper center", ncol=2, frameon=False, fontsize=6.5, columnspacing=1.0, handlelength=1.4)
pc.panel_labels(ax)
fig.tight_layout(w_pad=0.9)
pc.save(fig, "fig6_intervention_v17")
say.write()
