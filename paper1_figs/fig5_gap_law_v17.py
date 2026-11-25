"""MSBQA paper 1, Fig. 5 (v17): quantum gap vs classical crossing, weyl family, n = 4.
 (a) extrapolated gap position p*(h -> 0) (quadratic in h, all seven h) against the classical reference p_c
     (crossings) or p_first (no crossing); for crossings also p*(h = 1), joined to the extrapolation
 (b) ln Delta_min vs 1/h; crossings coloured by d_H x_c^2, no-crossing instances grey; triangles: below 1e-6
 (c) S from ln Delta = a - S/h (h <= 0.5, Delta > 1e-6, >= 3 points) vs d_H x_c^2 (16 crossings)
 (d) measured quantum failure 1 - P_Q vs P_LZ at T = 700, both families, only instances with P_LZ >= 1e-5.
     goto instances whose minimum gap is at p = 0 (the start of the ramp, where the Landau-Zener formula does not
     apply) are drawn as open squares; horizontal lines: mean failure of the instances with P_LZ < 1e-6
     (goto: those with an interior minimum)
Panels (a)-(c) are those of fig5_gap_law_v16.py. Run from msbqa_phase1/:
    python paper1_figs/fig5_gap_law_v17.py
Writes paper1_figs/fig5_gap_law_v17.pdf / .png and fig5_gap_law_v17_numbers.txt.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import p1common as pc

say = pc.Numbers("fig5_gap_law_v17")
FLOOR = 1e-6
HS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12]
HK = {1.0: "1", 0.7: "0.7", 0.5: "0.5", 0.35: "0.35", 0.25: "0.25", 0.18: "0.18", 0.12: "0.12"}
g = pd.read_csv(pc.P("run_v4", "phase1c", "gap_vs_h.csv")).set_index("inst")
g["cross"] = g["cross"].fillna(0).clip(lower=0).astype(int)
D = pd.DataFrame({h: g[f"Dmin_h{HK[h]}"] for h in HS})
PS = pd.DataFrame({h: g[f"pstar_h{HK[h]}"] for h in HS})
cr = g.cross == 1
ref = pd.Series(np.where(cr, g.p_c, g.p_first), index=g.index)
off0 = g.p0_quad - ref
for m, nm in ((cr, "crossing (vs p_c)"), (~cr, "no crossing (vs p_first)")):
    say(f"(a) {nm:26s}: p*(h->0) - ref median {off0[m].median():+.3f}, |.|<0.2 in {(off0[m].abs() < 0.2).mean():.0%}, "
        f"Spearman {spearmanr(g.p0_quad[m], ref[m]).correlation:+.3f};  p*(h=1) - ref median {(PS[1.0] - ref)[m].median():+.2f}")

rows = []
for i in g.index:
    hs = np.array([h for h in HS if h <= 0.5 and np.isfinite(D.loc[i, h]) and D.loc[i, h] > FLOOR])
    if len(hs) < 3:
        rows.append(dict(inst=i, npts=len(hs), S=np.nan, exp_better=np.nan))
        continue
    y = np.log(D.loc[i, hs].values.astype(float))
    A1 = np.c_[np.ones_like(hs), -1.0 / hs]
    A2 = np.c_[np.ones_like(hs), np.log(hs)]
    c1 = np.linalg.lstsq(A1, y, rcond=None)[0]
    c2 = np.linalg.lstsq(A2, y, rcond=None)[0]
    rows.append(dict(inst=i, npts=len(hs), S=c1[1],
                     exp_better=float(np.sum((A1 @ c1 - y) ** 2) < np.sum((A2 @ c2 - y) ** 2))))
fit = pd.DataFrame(rows).set_index("inst")
g = g.drop(columns=["S", "k_pow", "rss_inv", "rss_pow"], errors="ignore").join(fit)
g["dx"] = g.ham_c * g.xsq_c
ok = g.npts >= 3
for m, nm in ((cr, "crossing"), (~cr, "no crossing")):
    s = g[m & ok]
    say(f"(b) {nm:12s}: fitted {len(s)}; exp(-S/h) better on {int(s.exp_better.sum())} of {len(s)}")
s = g[cr & ok]
say(f"(c) Spearman(S, d_H x_c^2) {spearmanr(s.S, s.dx).correlation:+.3f} (N {len(s)}); S/(d_H x_c^2) median "
    f"{(s.S / s.dx).median():.2f}, range {(s.S / s.dx).min():.2f}-{(s.S / s.dx).max():.2f}")

b1 = pc.load_1b()
T = 700.0
LZ = []
for fam, qc, xc, sc in (("weyl", "Q_weyl", "xLZ_Wg", "pstar_Wg"), ("goto", "Q_goto", "xLZ_Gg", "pstar_Gg")):
    PLZ = np.exp(-b1[xc] * T)
    fail = 1.0 - b1[qc]
    start = b1[sc] == 0.0                      # minimum gap at p = 0: no avoided crossing, Eq. (LZ) does not apply
    floor = fail[(PLZ < 1e-6) & ~start].mean()
    shown = PLZ >= 1e-5
    LZ.append((fam, PLZ, fail, floor, shown, start))
    say(f"(d) {fam}: minimum gap at p = 0 on {int(start.sum())} of {len(b1)}; shown (P_LZ >= 1e-5) {int(shown.sum())}, of which "
        f"interior minimum {int((shown & ~start).sum())}, at p = 0 {int((shown & start).sum())}")
    say(f"    mean failure where P_LZ < 1e-6: all {fail[PLZ < 1e-6].mean():.1e} (N {int((PLZ < 1e-6).sum())}), interior minimum "
        f"{floor:.1e} (N {int(((PLZ < 1e-6) & ~start).sum())})")
    say("    interior minimum, P_LZ > 0.01 (P_LZ / measured): " +
        ", ".join(f"#{i} {PLZ[i]:.3f}/{fail[i]:.3f}" for i in b1.index[(PLZ > 0.01) & ~start]))
    if start.any():
        f0 = fail[start]
        say(f"    minimum at p = 0: failure > 1e-3 on {int((f0 > 1e-3).sum())} ({f0[f0 > 1e-3].min():.3f}-{f0.max():.3f}), "
            f"<= 1e-3 on {int((f0 <= 1e-3).sum())} (max {f0[f0 <= 1e-3].max():.1e})")

# ------------------------------------------------------------- figure
plt = pc.setup_mpl()
from matplotlib import cm
from matplotlib.colors import Normalize
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4))
ax = ax.ravel()
norm = Normalize(0, 3.5)
cmap = cm.plasma

a = ax[0]
for i in g.index[cr]:
    a.plot([ref[i], ref[i]], [PS.loc[i, 1.0], g.p0_quad[i]], c=pc.RED, lw=0.5, alpha=0.35, zorder=1)
a.scatter(ref[~cr], g.p0_quad[~cr], s=10, c=pc.GREY, alpha=0.6, lw=0, zorder=2, label=r"no crossing, $h\to0$ (vs $p_{\rm first}$)")
a.scatter(ref[cr], PS.loc[cr, 1.0], s=16, facecolors="white", edgecolors=pc.RED, lw=0.8, zorder=3,
          label=r"crossing, $h=1$")
a.scatter(ref[cr], g.p0_quad[cr], s=18, c=pc.RED, lw=0, zorder=4, label=r"crossing, $h\to0$")
lim = [0, 7]
a.plot(lim, lim, "k--", lw=0.6)
a.set_xlim(0, 4.6); a.set_ylim(0, 7)
a.set_xlabel(r"classical $p_{\rm c}$ (crossing) or $p_{\rm first}$ (no crossing)")
a.set_ylabel(r"position of minimum gap $p^*$")
a.legend(loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.3)
a.text(2.95, 2.65, "$p^*=p_{\\rm c}$", fontsize=7, ha="left", va="top")

b = ax[1]
x = 1.0 / np.array(HS)
for i in g.index[~cr]:
    y = D.loc[i].values.astype(float)
    good = y > FLOOR
    b.plot(x[good], np.log(y[good]), "-", lw=0.5, c=pc.GREY, alpha=0.4, zorder=1)
for i in g.index[cr]:
    y = D.loc[i].values.astype(float)
    good = y > FLOOR
    col = cmap(norm(g.dx[i]) * 0.9)
    b.plot(x[good], np.log(y[good]), "-o", ms=2.5, lw=1.0, c=col, zorder=2)
    bad = ~good & np.isfinite(y)
    b.plot(x[bad], np.full(bad.sum(), np.log(FLOOR)), "v", ms=3.5, mfc="white", c=col, ls="none", zorder=2)
b.axhline(np.log(FLOOR), c=pc.GREY, ls=":", lw=0.6)
b.text(8.4, np.log(FLOOR) + 0.4, r"resolution $10^{-6}K$", fontsize=6.5, ha="right", va="bottom", color=pc.GREY)
b.text(8.4, -0.6, "no crossing", fontsize=7, ha="right", va="bottom", color=pc.GREY)
b.set_xlabel(r"$1/h$"); b.set_ylabel(r"$\ln(\Delta_{\min}/K)$")
b.set_ylim(np.log(FLOOR) - 0.8, 1.0)
sm = cm.ScalarMappable(norm=Normalize(0, 3.5 / 0.9), cmap=cmap)
cb = fig.colorbar(sm, ax=b, pad=0.02, fraction=0.05)
cb.set_label(r"crossings: $d_{\rm H}x_{\rm c}^2$", fontsize=8); cb.ax.tick_params(labelsize=7)

c = ax[2]
c.scatter(s.dx, s.S, s=16, c=pc.RED, lw=0)
for i in s.index[s.dx > 1.0]:
    c.annotate(f"#{i}", (s.dx[i], s.S[i]), fontsize=6, xytext=(-16, -9) if i == 231 else (3, 2), textcoords="offset points")
xx = np.linspace(0, 3.5, 10)
c.plot(xx, 2 * xx, c=pc.GREY, ls="--", lw=0.7)
c.plot(xx, xx, c=pc.GREY, ls=":", lw=0.7)
c.text(1.75, 3.6, r"$S=2d_{\rm H}x_{\rm c}^2$", fontsize=7, color=pc.GREY, ha="right")
c.text(2.0, 1.75, r"$S=d_{\rm H}x_{\rm c}^2$", fontsize=7, color=pc.GREY, ha="left", va="top")
c.set_xlim(0, 3.5); c.set_ylim(0, 5)
c.set_xlabel(r"classical distance $d_{\rm H}x_{\rm c}^2$"); c.set_ylabel(r"fitted action $S$ ($h\leq0.5$)")

d = ax[3]
for (fam, PLZ, fail, floor, shown, start), col, mk in zip(LZ, (pc.RED, pc.BLUE), ("o", "s")):
    m = shown & ~start
    d.loglog(PLZ[m], np.clip(fail[m], 1e-6, 1), mk, ms=4, c=col, alpha=0.85, lw=0, zorder=3,
             label=f"{fam} ({int(m.sum())})")
    m = shown & start
    if m.any():
        d.loglog(PLZ[m], np.clip(fail[m], 1e-6, 1), mk, ms=3.6, mfc="none", mec=col, mew=0.6, alpha=0.55, lw=0, zorder=2,
                 label=f"{fam}, minimum gap at $p=0$ ({int(m.sum())})")
    d.plot([3e-2, 1.5], [floor, floor], c=col, ls=":", lw=0.9)
d.plot([1e-5, 1.5], [1e-5, 1.5], "k--", lw=0.6)
d.text(1.3, LZ[1][3] * 1.3, r"goto, mean for $P_{\rm LZ}<10^{-6}$", fontsize=6.3, color=pc.BLUE, va="bottom", ha="right")
d.text(1.3, LZ[0][3] * 1.3, r"weyl, mean for $P_{\rm LZ}<10^{-6}$", fontsize=6.3, color=pc.RED, va="bottom", ha="right")
d.set_xlim(1e-5, 1.5); d.set_ylim(1e-6, 1.5)
d.set_xlabel("Landau–Zener estimate " + r"$P_{\rm LZ}$")
d.set_ylabel(r"measured failure $1-P_{\rm Q}$ ($T=700/K$)")
d.legend(loc="upper left", frameon=False, fontsize=6.3, handletextpad=0.2, borderaxespad=0.2)
pc.panel_labels(ax)
fig.tight_layout(w_pad=1.0, h_pad=1.0)
pc.save(fig, "fig5_gap_law_v17")
say.write()
