"""MSBQA paper 1, Fig. 4 (v16 layout): anatomy of crossing instance #29 (weyl family), drawn for readability.
 (a) classical energy of the competing branch relative to the ground branch, E_A - E_B (V(x; p), paper Eq. 7),
     recomputed by damped Newton continuation as in code/branches.py; only the two branches that cross are drawn.
 (b) lowest even-parity quantum gap E1 - E0 for h = 1, 0.7, 0.5, 0.35, 0.25. Each curve joins the coarse p grid
     with the points evaluated by the Brent refinement and the Landau-Zener probe of gap_scan(_gpu).py, so it
     reaches the refined minimum. h = 0.18 and 0.12 are left out: their minima are below the 1e-6 resolution.
Inputs: instances/G4.npz, run_v4/branches/branches_weyl.csv, run_v4/parts/Wg*_29.json (from paper1_figdata.tgz).
Run from msbqa_phase1/:   python paper1_figs/fig4_crossing_anatomy_v16.py
Writes paper1_figs/fig4_crossing_anatomy_v16.pdf / .png and fig4_crossing_anatomy_v16_numbers.txt.
"""
import json
import os

import numpy as np

import p1common as pc
import kpo_common as kc

INST, PF, DP, PMAX = 29, 7.0, 0.005, 5.0
TAGS = [("Wg", 1.0), ("Wg07", 0.7), ("Wg05", 0.5), ("Wg035", 0.35), ("Wg025", 0.25)]
say = pc.Numbers("fig4_crossing_anatomy_v16")

J, Delta, meta = kc.load_instance(pc.P("instances"), "G4", INST)
xi0, K, n = float(meta["xi0"]), 1.0, len(Delta)
S_all = kc.all_spins(n).astype(float)
S = S_all[S_all[:, 0] > 0]
C = len(S)
g = int(np.argmin(kc.ising_energies(J)[S_all[:, 0] > 0]))

# ------------------------------------------------------------- classical branches (as fig4_crossing_anatomy.py)
pgrid = np.round(np.arange(PF, -0.5 * DP, -DP), 10)
pgrid[-1] = max(pgrid[-1], 0.0)
E = np.full((len(pgrid), C), np.nan)
X = S * np.sqrt(np.maximum(PF - Delta, 0.5) / K)
alive = np.ones(C, bool)


def grad_hess(X, p):
    gr = 2.0 * (Delta - p) * X + 2.0 * K * X ** 3 - 2.0 * xi0 * X @ J
    H = np.broadcast_to(-2.0 * xi0 * J, (C, n, n)).copy()
    H[:, np.arange(n), np.arange(n)] += 2.0 * (Delta - p) + 6.0 * K * X ** 2
    return gr, H


for k, p in enumerate(pgrid):
    for _ in range(60 if k == 0 else 8):
        gr, H = grad_hess(X, p)
        Hs = np.where(alive[:, None, None], H, np.eye(n))
        dx = np.linalg.solve(Hs, np.where(alive[:, None], gr, 0.0)[..., None])[..., 0]
        st = np.max(np.abs(dx), -1, keepdims=True)
        X = X - dx * np.minimum(1.0, 0.5 / np.maximum(st, 1e-300))
    gr, H = grad_hess(X, p)
    ok = np.max(np.abs(gr), -1) < 1e-9 * np.maximum(1.0, np.max(np.abs(X), -1) ** 3)
    ok &= np.all(S * X > 1e-7, -1)
    ok &= np.linalg.eigvalsh(np.where(alive[:, None, None], H, np.eye(n)))[:, 0] > 1e-10
    alive &= ok
    V = np.sum((Delta - p) * X ** 2 + 0.5 * K * X ** 4, -1) - xi0 * np.einsum("cj,jk,ck->c", X, J, X)
    E[k] = np.where(alive, V, np.nan)
    if not alive.any():
        break
pgrid, E = pgrid[::-1], E[::-1]
lowest = np.where(np.isfinite(E).any(1), np.nanargmin(np.where(np.isfinite(E), E, np.inf), 1), -1)
k_c = next(k for k in range(len(pgrid)) if np.all(lowest[k:] == g))
p_c = pgrid[k_c]
prev = lowest[(pgrid < p_c) & (lowest >= 0)]
A = int(prev[-1])
dE = E[:, A] - E[:, g]
say(f"#{INST}: p_c {p_c:.3f}; competing {S[A].astype(int)}, ground {S[g].astype(int)}; "
    f"min E_A - E_B {np.nanmin(dE):.4f} at p {pgrid[np.nanargmin(dE)]:.3f}")

# ------------------------------------------------------------- quantum gaps, grid + refinement points
curves = []
for tag, h in TAGS:
    r = json.load(open(pc.P("run_v4", "parts", f"{tag}_{INST}.json")))
    pts = list(zip(r["p_grid"], r["gap_grid"]))
    pts += [(p, w[1] - w[0]) for p, w in r.get("refine_evals", [])]
    pts += [tuple(q) for q in r.get("lz_points", [])]
    pts = np.array(sorted(set((float(p), float(v)) for p, v in pts)))
    curves.append((h, pts[:, 0], pts[:, 1], r["p_star"], r["Delta_min"]))
    say(f"h = {h:<4g}: p* {r['p_star']:.3f}, Delta_min {r['Delta_min']:.3e}, curve points {len(pts)}")

# ------------------------------------------------------------- figure
plt = pc.setup_mpl()
from matplotlib import cm
fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.7), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.35]))
a = ax[0]
m = np.isfinite(dE)
a.plot(pgrid[m], dE[m], c=pc.RED, lw=1.6)
a.axhline(0, c="k", lw=1.0)
a.fill_between(pgrid[m], dE[m], 0, where=dE[m] < 0, color=pc.RED, alpha=0.18, lw=0)
a.axvline(p_c, c="k", ls="--", lw=0.7)
a.set_ylim(-0.03, 0.06)
a.set_ylabel(r"$E_{\rm A}-E_{\rm B}$ (classical)")
a.text(0.12, -0.026, "A lower", color=pc.RED, fontsize=7, va="bottom")
a.text(p_c + 0.12, -0.026, "B lower", fontsize=7, va="bottom")
a.text(p_c, 0.061, r"$p_{\rm c}$", ha="center", va="bottom", fontsize=8)

b = ax[1]
cmap = cm.viridis
for i, (h, p, gap, ps, dm) in enumerate(curves):
    col = cmap(0.85 * i / (len(curves) - 1))
    keep = p <= PMAX + 1e-9
    b.semilogy(p[keep], gap[keep], c=col, lw=1.0)
    b.plot([ps], [dm], "o", ms=4, c=col, zorder=4)
    last = i == len(curves) - 1
    b.annotate(f"$h={h:g}$", (ps, dm), xytext=(5, 0) if last else (5, -2), textcoords="offset points",
               fontsize=7, color=col, va="center" if last else "top", ha="left")
b.axvline(p_c, c="k", ls="--", lw=0.7)
b.set_ylim(5e-7, 4)
b.set_xlim(0, PMAX)
b.set_xlabel(r"pump $p$")
b.set_ylabel(r"quantum gap $(E_1-E_0)/K$")
pc.panel_labels(ax)
fig.tight_layout(h_pad=0.5)
pc.save(fig, "fig4_crossing_anatomy_v16")
say.write()
