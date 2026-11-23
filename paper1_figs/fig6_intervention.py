"""MSBQA paper 1, Fig. 6 (claim C5): per-spin (K) -> uniform max (U) detuning, quantum gap (weyl, h = 1) and
classical KPO success. Reads run_v4/phase1d/per_instance_1d.csv. Run from msbqa_phase1/:
    python paper1_figs/fig6_intervention.py
"""
import numpy as np
import pandas as pd

import p1common as pc

say = pc.Numbers("fig6_intervention")
d = pd.read_csv(pc.P("run_v4", "phase1d", "per_instance_1d.csv")).set_index("inst")
GROUPS = [("KO", pc.RED, "KO: crossing removed"), ("MO", pc.BLUE, "MO: crossing created"),
          ("BOTH", pc.PURPLE, "BOTH"), ("CTRL", pc.GREY, "CTRL: no crossing")]
gp = d[d["group"].notna() & d.Dmin_K.notna() & d.Dmin_U.notna()].copy()
gp["lr"] = np.log(gp.Dmin_U / gp.Dmin_K)
for gname, _, _ in GROUPS + [("NEG", None, None)]:
    s = gp[gp.group == gname]
    if len(s):
        say(f"{gname:5s} N {len(s):3d}  median Dmin_K {s.Dmin_K.median():.3e}  Dmin_U {s.Dmin_U.median():.3e}  "
            f"geo-median ratio {np.exp(s.lr.median()):.3f}  frac U>K {(s.lr > 0).mean():.2f}")
chg = gp[gp.group.isin(["KO", "MO"])]
okdir = ((chg.group == "KO") & (chg.lr > 0)) | ((chg.group == "MO") & (chg.lr < 0))
say(f"direction as predicted: {int(okdir.sum())} of {len(chg)}")
# paired regression ln(U/K) = a + b * (cross_max - cross_kpo) over the selected groups
dc = (gp.cross_max.clip(lower=0) - gp.cross_kpo.clip(lower=0)).astype(float)
A = np.c_[np.ones(len(gp)), dc]
coef = np.linalg.lstsq(A, gp.lr.values, rcond=None)[0]
say(f"paired regression over {len(gp)} selected instances: ln(U/K) = {coef[0]:+.2f} {coef[1]:+.2f} * delta_cross "
    f"-> factor {np.exp(-coef[1]):.2f} per crossing")
LAY = [("L2g", "L2"), ("L3", "L3"), ("L4", "L4")]
for gname in ("KO", "MO", "BOTH"):
    s = d[d.group == gname]
    say(f"classical {gname}: " + ", ".join(f"{lab} {s[f'{c}_K'].mean():.3f} -> {s[f'{c}_U'].mean():.3f}" for c, lab in LAY))

plt = pc.setup_mpl()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
for gname, col, lab in GROUPS:
    s = gp[gp.group == gname]
    ax[0].loglog(s.Dmin_K, s.Dmin_U, "o", ms=3, c=col, alpha=0.8, lw=0, label=lab)
lo = min(gp.Dmin_K.min(), gp.Dmin_U.min()) * 0.7
hi = max(gp.Dmin_K.max(), gp.Dmin_U.max()) * 1.4
ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.6)
ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
ax[0].set_xlabel(r"$\Delta_{\min}$, per-spin detuning"); ax[0].set_ylabel(r"$\Delta_{\min}$, uniform detuning")
ax[0].legend(loc="lower right", frameon=False, fontsize=5.8, handletextpad=0.2)

rng = np.random.default_rng(0)
for k, (gname, col, lab) in enumerate(GROUPS):
    s = gp[gp.group == gname]
    ax[1].scatter(k + rng.uniform(-0.18, 0.18, len(s)), s.lr, s=6, c=col, lw=0, alpha=0.8)
    ax[1].plot([k - 0.3, k + 0.3], [s.lr.median()] * 2, c="k", lw=1.2)
for t in (np.log(3), -np.log(3)):
    ax[1].axhline(t, c=pc.GREY, ls="--", lw=0.6)
ax[1].axhline(0, c="k", lw=0.4)
ax[1].set_xticks(range(len(GROUPS))); ax[1].set_xticklabels([gname for gname, _, _ in GROUPS], fontsize=7)
ax[1].set_ylabel(r"$\ln(\Delta_U/\Delta_K)$")

xt, xl = [], []
for k, (gname, col) in enumerate((("KO", pc.RED), ("MO", pc.BLUE))):
    s = d[d.group == gname]
    for j, (c, lab) in enumerate(LAY):
        x0 = 4 * k + j
        ax[2].bar(x0 - 0.2, s[f"{c}_K"].mean(), 0.4, color=col, alpha=0.35, label="per-spin" if (k, j) == (0, 0) else None)
        ax[2].bar(x0 + 0.2, s[f"{c}_U"].mean(), 0.4, color=col, label="uniform" if (k, j) == (0, 0) else None)
        xt.append(x0); xl.append(lab if j != 1 else f"{lab}\n{gname}")
ax[2].set_xticks(xt); ax[2].set_xticklabels(xl, fontsize=7)
ax[2].set_ylim(0, 1.12)
ax[2].set_ylabel(r"classical KPO $P_{\rm succ}$")
ax[2].legend(loc="upper center", ncol=2, frameon=False, fontsize=6)
pc.panel_labels(ax)
fig.tight_layout(w_pad=0.8)
pc.save(fig, "fig6_intervention")
say.write()
