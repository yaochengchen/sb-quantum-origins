"""MSBQA paper 1, Fig. 4 (claims C2, C3): anatomy of one crossing instance (default #29, weyl family).
 (a) classical configuration branches E_s(p) - E_ground(p) of V(x; p) (paper Eq. 5), recomputed here for this one
     instance by damped Newton continuation from p = pf down to 0 (same method and step as code/branches.py);
 (b) quantum gap E1 - E0 (even parity) for h = 1 ... 0.12 from run_v4/parts/<tag>_<inst>.json.
Needs instances/G4.npz (as on pu1). Run from msbqa_phase1/:
    python paper1_figs/fig4_crossing_anatomy.py [--inst 29]
"""
import argparse
import json
import os

import numpy as np

import p1common as pc
import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst", type=int, default=29)
ap.add_argument("--inst-dir", default=pc.P("instances"))
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--dp", type=float, default=0.005)
a = ap.parse_args()
say = pc.Numbers("fig4_crossing_anatomy")
FLOOR = 1e-6
TAGS = [("Wg", 1.0), ("Wg07", 0.7), ("Wg05", 0.5), ("Wg035", 0.35), ("Wg025", 0.25), ("Wg018", 0.18), ("Wg012", 0.12)]

J, Delta, meta = kc.load_instance(a.inst_dir, "G4", a.inst)
xi0 = float(meta["xi0"])
K = 1.0
n = len(Delta)
S_all = kc.all_spins(n).astype(float)
reps = np.where(S_all[:, 0] > 0)[0]                  # Z2 representatives (s_0 = +1)
S = S_all[reps]
C = len(S)
E_is = kc.ising_energies(J)[reps]
g = int(np.argmin(E_is))
lam = np.linalg.eigvalsh(np.diag(Delta) - xi0 * J)
br = pc.load_branches().loc[a.inst]
say(f"instance {a.inst}: Delta {np.round(Delta, 4)}, lam1 {lam[0]:.4f}, lam2 {lam[1]:.4f}; "
    f"branches_weyl.csv: cross {int(br.cross)}, p_c {br.p_c}, ham_c {br.ham_c}, xsq_c {br.xsq_c:.3f}")

# ------------------------------------------------------------- classical branches (weyl: classical detuning = Delta)
pgrid = np.round(np.arange(a.pf, -0.5 * a.dp, -a.dp), 10)
pgrid[-1] = max(pgrid[-1], 0.0)
E = np.full((len(pgrid), C), np.nan)
X = S * np.sqrt(np.maximum(a.pf - Delta, 0.5) / K)
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
rel = E - E[:, [g]]
lowest = np.where(np.isfinite(E).any(1), np.nanargmin(np.where(np.isfinite(E), E, np.inf), 1), -1)
pc_here = np.nan
for k in range(len(pgrid)):
    if np.all(lowest[k:] == g):
        pc_here = pgrid[k]
        break
prev = lowest[(pgrid < pc_here) & (lowest >= 0)]
trap = int(prev[-1]) if len(prev) and prev[-1] != g else -1
born = {c: (pgrid[np.isfinite(E[:, c])][0] if np.isfinite(E[:, c]).any() else np.nan) for c in range(C)}
say(f"recomputed: p_c {pc_here:.3f} (csv {br.p_c}); ground cfg {S[g].astype(int)} born at {born[g]:.3f}; "
    f"replaced cfg {S[trap].astype(int) if trap >= 0 else None} born at {born.get(trap, np.nan):.3f}; "
    f"Hamming {int(min((S[g] != S[trap]).sum(), (S[g] == S[trap]).sum())) if trap >= 0 else 0}")

# ------------------------------------------------------------- quantum gaps
gaps = []
for tag, h in TAGS:
    f = pc.P("run_v4", "parts", f"{tag}_{a.inst}.json")
    if not os.path.exists(f):
        say(f"missing {f}")
        continue
    r = json.load(open(f))
    gaps.append((h, np.array(r["p_grid"]), np.array(r["gap_grid"]), r["p_star"], r["Delta_min"], r.get("edge", 0)))
    say(f"h = {h:<5g} cutoff {r['cutoff']}  p* {r['p_star']:.3f}  Delta_min {r['Delta_min']:.3e}  "
        f"{'(below solver resolution)' if r['Delta_min'] < FLOOR else ''}")

# ------------------------------------------------------------- figure
plt = pc.setup_mpl()
from matplotlib import cm
fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)
for c in range(C):
    if c == g:
        continue
    col = pc.RED if c == trap else pc.GREY
    ax[0].plot(pgrid, rel[:, c], c=col, lw=1.4 if c == trap else 0.7, zorder=3 if c == trap else 1,
               label="replaced (trap) configuration" if c == trap else None)
ax[0].axhline(0, c="k", lw=1.0, label="ground configuration")
ax[0].axvline(pc_here, c="k", ls="--", lw=0.7)
for l_ in lam[:2]:
    ax[0].axvline(l_, c=pc.GREY, ls=":", lw=0.6)
# zoom on the crossing: the energy difference of the two competing branches is small on the scale of the others
win = (pgrid <= pc_here + 1.5) & np.isfinite(rel[:, trap]) if trap >= 0 else np.zeros(len(pgrid), bool)
span = np.nanmax(np.abs(rel[win, trap])) if win.any() else 1.0
ax[0].set_ylim(-1.6 * span, 3.0 * span)
ax[0].set_ylabel(r"$E_{\mathbf{s}}-E_{\mathbf{s}_0}$ (classical)")
ax[0].legend(loc="upper right", frameon=False, fontsize=6.2)
ax[0].text(pc_here, -1.5 * span, r" $p_{\rm c}$", fontsize=8, va="bottom")
ax[0].text(lam[1], -1.5 * span, r" $\lambda_{1,2}$", fontsize=7, va="bottom", color=pc.GREY)
say(f"panel (a) y-range +-{span:.4f} (max |E_trap - E_ground| for p <= p_c + 1.5)")

cmap = cm.viridis
for i, (h, pg, gp, ps, dm, edge) in enumerate(gaps):
    col = cmap(i / max(len(gaps) - 1, 1))
    good = gp >= FLOOR
    ax[1].semilogy(pg[good], gp[good], c=col, lw=1.0, label=f"$h={h:g}$")
    ax[1].semilogy(pg[~good], np.full((~good).sum(), FLOOR), ls="none", marker=".", ms=2, c=col, alpha=0.4)
    ax[1].plot([ps], [max(dm, FLOOR)], marker="o" if dm >= FLOOR else "v", ms=4, c=col, mfc="white" if dm < FLOOR else col)
ax[1].axvline(pc_here, c="k", ls="--", lw=0.7)
ax[1].axhline(FLOOR, c=pc.GREY, lw=0.5, ls=":")
ax[1].set_xlabel(r"pump $p$")
ax[1].set_ylabel(r"$E_1-E_0$")
ax[1].set_xlim(0, a.pf)
ax[1].legend(loc="lower right", frameon=False, fontsize=6, ncol=2)
pc.panel_labels(ax)
fig.tight_layout(h_pad=0.4)
pc.save(fig, "fig4_crossing_anatomy")
say.write()
