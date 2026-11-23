"""MSBQA paper 1, Fig. 5 (claims C3, C4): quantum gap vs classical crossing, weyl family, n = 4.
 (a) p*(h) - p_c (crossing) and p*(h) - p_first (no crossing), with the quadratic extrapolation to h = 0 (p0_quad)
 (b) ln Delta_min vs 1/h; open symbols below the solver resolution FLOOR (not used in fits)
 (c) S from ln Delta = c - S/h (h <= 0.5, Delta > FLOOR, >= 3 points) vs ham_c * xsq_c, crossing instances
 (d) measured quantum failure 1 - P_Q vs LZ estimate exp(-x_LZ T), T = 700, both families (300 test instances)
Reads run_v4/phase1c/gap_vs_h.csv and run_v4/phase1b/per_instance_1b.csv. Run from msbqa_phase1/:
    python paper1_figs/fig5_gap_law.py
The numbers file reports the fit counts quoted in the paper (exponential better on X of 16 crossing / Y of 43 others,
Spearman(S, ham*xsq)); if they differ from the text, update the text.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import p1common as pc

say = pc.Numbers("fig5_gap_law")
FLOOR = 1e-6
HS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12]
HK = {1.0: "1", 0.7: "0.7", 0.5: "0.5", 0.35: "0.35", 0.25: "0.25", 0.18: "0.18", 0.12: "0.12"}
g = pd.read_csv(pc.P("run_v4", "phase1c", "gap_vs_h.csv")).set_index("inst")
g["cross"] = g["cross"].fillna(0).clip(lower=0).astype(int)
D = pd.DataFrame({h: g[f"Dmin_h{HK[h]}"] for h in HS})
PS = pd.DataFrame({h: g[f"pstar_h{HK[h]}"] for h in HS})
cr = g.cross == 1
say(f"instances {len(g)}: crossing {int(cr.sum())}, no crossing {int((~cr).sum())}")

# (a) gap position
ref = np.where(cr, g.p_c, g.p_first)
off0 = g.p0_quad - ref
for m, nm in ((cr, "crossing (vs p_c)"), (~cr, "no crossing (vs p_first)")):
    say(f"(a) {nm:26s}: p*(h->0) - ref median {off0[m].median():+.3f}, |.|<0.2 in {(off0[m].abs() < 0.2).mean():.0%}, "
        f"Spearman(p0, ref) {spearmanr(g.p0_quad[m], ref[m]).correlation:+.3f};  p*(h=1) - ref median "
        f"{(PS[1.0] - ref)[m].median():+.2f}")

# (b, c) tunnelling fits
rows = []
for i in g.index:
    hs = np.array([h for h in HS if h <= 0.5 and np.isfinite(D.loc[i, h]) and D.loc[i, h] > FLOOR])
    if len(hs) < 3:
        rows.append(dict(inst=i, npts=len(hs), S=np.nan, k=np.nan, exp_better=np.nan))
        continue
    y = np.log(D.loc[i, hs].values.astype(float))
    A1 = np.c_[np.ones_like(hs), -1.0 / hs]
    A2 = np.c_[np.ones_like(hs), np.log(hs)]
    c1, r1 = np.linalg.lstsq(A1, y, rcond=None)[:2]
    c2, r2 = np.linalg.lstsq(A2, y, rcond=None)[:2]
    rss1 = float(np.sum((A1 @ c1 - y) ** 2))
    rss2 = float(np.sum((A2 @ c2 - y) ** 2))
    rows.append(dict(inst=i, npts=len(hs), S=c1[1], k=c2[1], exp_better=float(rss1 < rss2)))
fit = pd.DataFrame(rows).set_index("inst")
g = g.drop(columns=["S", "k_pow", "rss_inv", "rss_pow"], errors="ignore").join(fit)   # report's own fits replaced
g["dx"] = g.ham_c * g.xsq_c
ok = g.npts >= 3
for m, nm in ((cr, "crossing"), (~cr, "no crossing")):
    s = g[m & ok]
    say(f"(b) {nm:12s}: fitted {len(s)} (excluded, < 3 points above floor: {int((m & ~ok).sum())}); "
        f"exp(-S/h) better on {int(s.exp_better.sum())} of {len(s)}; median S {s.S.median():.3f}")
s = g[cr & ok]
say(f"(c) crossing: Spearman(S, ham_c*xsq_c) {spearmanr(s.S, s.dx).correlation:+.3f} (N {len(s)}); "
    f"S/(ham_c*xsq_c): median {(s.S / s.dx).median():.2f}, range {(s.S / s.dx).min():.2f}-{(s.S / s.dx).max():.2f}")
for i in (6, 29, 94, 231):
    if i in s.index:
        say(f"    inst {i}: points {int(s.loc[i, 'npts'])}, S {s.loc[i, 'S']:.3f}, ham*xsq {s.loc[i, 'dx']:.3f}, "
            f"ratio {s.loc[i, 'S'] / s.loc[i, 'dx']:.2f}")
below = (D < FLOOR) & D.notna()
say(f"    points below {FLOOR:g}: {int(below.values.sum())} (instances {sorted(D.index[below.any(axis=1)].tolist())})")

# (d) LZ
b = pc.load_1b()
T = 700.0
lz = []
for fam, qc, xc in (("weyl", "Q_weyl", "xLZ_Wg"), ("goto", "Q_goto", "xLZ_Gg")):
    PLZ = np.exp(-b[xc] * T)
    fail = 1.0 - b[qc]
    lz.append((fam, PLZ, fail))
    big = PLZ > 0.01
    say(f"(d) {fam}: instances with P_LZ > 0.01: {int(big.sum())};  (P_LZ, 1-P_Q): " +
        ", ".join(f"#{i} {PLZ[i]:.3f}/{fail[i]:.3f}" for i in b.index[big]))
    small = PLZ < 1e-6
    say(f"    failure floor where P_LZ < 1e-6: median 1-P_Q {fail[small].median():.1e}, mean {fail[small].mean():.1e} "
        f"(N {int(small.sum())})")
    say(f"    crossing instances: mean LZ success {(1 - PLZ[b.cross == 1]).mean():.3f}, measured {b[qc][b.cross == 1].mean():.3f}")

# ------------------------------------------------------------- figure
plt = pc.setup_mpl()
from matplotlib import cm
from matplotlib.colors import Normalize
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
ax = ax.ravel()
norm = Normalize(0, np.nanmax(g.dx[cr]))
cmap = cm.plasma
for i in g.index:
    if cr[i]:
        col, lw, z = cmap(norm(g.dx[i]) * 0.85), 1.0, 2
        r = g.p_c[i]
    else:
        col, lw, z = pc.GREY, 0.5, 1
        r = g.p_first[i]
    hs = np.array(HS)
    ax[0].plot(hs, PS.loc[i].values - r, c=col, lw=lw, alpha=0.9 if cr[i] else 0.35, zorder=z)
    ax[0].plot([0], [g.p0_quad[i] - r], marker="o" if cr[i] else ".", ms=3, c=col, zorder=z + 1)
ax[0].axhline(0, c="k", lw=0.6)
ax[0].set_xlabel(r"$h$"); ax[0].set_ylabel(r"$p^*(h)-p_{\rm c}$  (grey: $p^*(h)-p_{\rm first}$)", fontsize=8)
ax[0].set_xlim(-0.03, 1.03)

for i in g.index:
    y = D.loc[i].values.astype(float)
    x = 1.0 / np.array(HS)
    good = y > FLOOR
    if cr[i]:
        col = cmap(norm(g.dx[i]) * 0.85)
        ax[1].plot(x[good], np.log(y[good]), "-o", ms=2.5, lw=0.8, c=col, zorder=2)
        bad = ~good & np.isfinite(y)
        ax[1].plot(x[bad], np.full(bad.sum(), np.log(FLOOR)), "v", ms=3, mfc="white", c=col, zorder=2)
    else:
        ax[1].plot(x[good], np.log(y[good]), "-", lw=0.5, c=pc.GREY, alpha=0.35, zorder=1)
ax[1].axhline(np.log(FLOOR), c=pc.GREY, ls=":", lw=0.6)
ax[1].set_xlabel(r"$1/h$"); ax[1].set_ylabel(r"$\ln\Delta_{\min}$")
sm = cm.ScalarMappable(norm=Normalize(0, np.nanmax(g.dx[cr]) / 0.85), cmap=cmap)
cb = fig.colorbar(sm, ax=ax[1], pad=0.02, fraction=0.05)
cb.set_label(r"$d_{\rm H}\,x_{\rm c}^2$", fontsize=8); cb.ax.tick_params(labelsize=7)

ax[2].scatter(s.dx, s.S, s=14, c=pc.RED, lw=0)
for i in s.index[s.dx > 1.0]:
    ax[2].annotate(str(i), (s.dx[i], s.S[i]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
xx = np.linspace(0, s.dx.max() * 1.05, 10)
ax[2].plot(xx, 2 * xx, c=pc.GREY, ls="--", lw=0.6, label=r"single cat: $S=2d_{\rm H}x_{\rm c}^2$")
ax[2].plot(xx, xx, c=pc.GREY, ls=":", lw=0.6, label=r"$S=d_{\rm H}x_{\rm c}^2$")
ax[2].set_xlabel(r"$d_{\rm H}\,x_{\rm c}^2$"); ax[2].set_ylabel(r"$S$ (fit $h\leq0.5$)")
ax[2].set_ylim(bottom=0)
ax[2].legend(loc="upper left", frameon=False)

for (fam, PLZ, fail), col, mk in zip(lz, (pc.RED, pc.BLUE), ("o", "s")):
    ax[3].loglog(np.clip(PLZ, 1e-8, 1), np.clip(fail, 1e-6, 1), mk, ms=3, c=col, alpha=0.6, lw=0, label=f"{fam} family")
ax[3].plot([1e-8, 1], [1e-8, 1], "k--", lw=0.6)
ax[3].set_xlim(1e-8, 1.5); ax[3].set_ylim(1e-6, 1.5)
ax[3].set_xlabel(r"$P_{\rm LZ}=\exp(-x_{\rm LZ}T)$ (clipped at $10^{-8}$)"); ax[3].set_ylabel(r"$1-P_{\rm Q}$ (measured, $T=700$)")
ax[3].legend(loc="upper left", frameon=False)
pc.panel_labels(ax)
fig.tight_layout(w_pad=0.8, h_pad=0.8)
pc.save(fig, "fig5_gap_law")
say.write()
