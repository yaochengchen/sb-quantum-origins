"""MSBQA phase 1b, step 2: instantaneous spectrum of the quantum KPO network along the pump, even-parity sector
(the sector of the vacuum), minimum gap Delta_min, its position p*, and a Landau-Zener estimate. One instance per call.

usage (from msbqa_phase1/):
  python code/gap_scan.py --inst-dir instances --set G4 --inst 17 --family weyl --h 1 --cutoff 22 --pf 7 \
      --tag Wg --out run_v4/parts/Wg_17.json
  python code/gap_scan.py --inst-dir instances --set G4 --inst 0 --family weyl --h 1 --cutoff 22 --pf 7 --probe
      (a few eigensolves + an independent eigsh cross-check, an estimate of the time per task; writes nothing)

H(p) = diag + Hhop + p * Hpump: the KPOSector matrices of the time evolution (real symmetric; d = cutoff + 1,
same family / h conventions). g(p) = E_1(p) - E_0(p) inside the even sector.

Eigensolver (v3, 10-31). ARPACK (eigsh, v1/v2) is unreliable here: at p = 0 the total photon number is conserved,
H is block diagonal, and a single-vector Lanczos started from a random vector missed the vacuum (v1 probe:
E0 = +2.49 instead of 0) and failed even when started from vacuum + two-photon states (v2). v3:
  - p = 0 exactly: H(0) is diagonalised block by block (total photon number 0, 2, 4, ...; blocks of dimension
    <= ~1000); a block of N photons is skipped once its lower bound N w_min + (K h / 2)(N^2/n - N) lies above the
    m-th lowest level found (w_min = lowest single-photon energy; the Kerr bound is convexity of n(n-1)). This
    gives the m = k + guard lowest levels and eigenvectors of the even sector at p = 0 exactly.
  - p > 0: continuation with the block solver LOBPCG (scipy lobpcg, Jacobi-type preconditioner 1/(|diag - theta|+1)),
    started from the m eigenvectors of the previous p; the k lowest are reported, the 'guard' vectors catch levels
    that come down from above. Accepted only if (i) the residuals ||H v - E v|| of the k lowest are below
    10 * restol * max(1, |E|) and (ii) the Ritz values do not exceed the Rayleigh-Ritz values of the start block
    (upper bounds, Courant-Fischer). Otherwise LOBPCG is continued (up to 3 attempts); then the task stops with an
    error that prints the residuals, Ritz values and bounds.
  - refinement and Landau-Zener points start from the vectors stored at the grid minimum, never from random.

Steps:
  1. grid p = linspace(0, pf, --np): E_0 .. E_{k-1}
  2. the --nmin deepest local minima of g on the grid (end points included) are refined by bounded Brent
     (scipy minimize_scalar, --xatol) -> Delta_min = smallest refined g, p_star = its position
  3. Landau-Zener slope from g^2 ~ Delta^2 + v^2 (p - p*)^2 at p* +- (1e-3, 3e-3, 1e-2, 3e-2):
     v = sqrt(median of (g^2 - Delta^2) / dp^2). For a linear ramp p = pf t / T (hbar = 1, time unit 1/K as in
     the evolution) the two-level LZ estimate of the failure probability is P_LZ(T) = exp(-x_LZ T),
     x_LZ = pi Delta_min^2 / (2 v pf). For a broad (non-hyperbolic) minimum v^2 ~ Delta g'' and P_LZ is only a
     rough adiabaticity measure; the report compares it with the measured 1 - P_L0.
  edge = 1 if the deepest minimum lies in the first or last grid cell (no interior avoided crossing).
"""
import argparse
import os
import time
import warnings

import numpy as np
import scipy.sparse as sps
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--inst", type=int, required=True)
ap.add_argument("--family", required=True, choices=["goto", "weyl"])
ap.add_argument("--h", type=float, default=1.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--cutoff", type=int, required=True)
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--np", dest="npts", type=int, default=141)
ap.add_argument("--k", type=int, default=4, help="number of levels reported")
ap.add_argument("--guard", type=int, default=2, help="extra tracked levels (catch levels coming down)")
ap.add_argument("--restol", type=float, default=1e-8, help="relative residual tolerance of LOBPCG")
ap.add_argument("--iters", type=int, default=300, help="LOBPCG iterations per attempt (x attempt number)")
ap.add_argument("--nmin", type=int, default=3)
ap.add_argument("--xatol", type=float, default=1e-6)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--tag", default="gap")
ap.add_argument("--out", default="")
a = ap.parse_args()
if not a.probe:
    if not a.out:
        raise SystemExit("--out is required (or --probe)")
    if os.path.exists(a.out):
        raise SystemExit(0)

t0 = time.time()
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
xi0 = meta["xi0"]
n = len(Delta)
d = a.cutoff + 1
model = kc.KPOSector(J, Delta, a.K, xi0, a.h, d, a.family)
N = int(model.idx.size)
diag, Hh, Hp = model.diag, model.Hhop, model.Hp
m = a.k + a.guard
st = dict(calls=0, matvecs=0, retries=0, misconv=0, X=None)
rng = np.random.default_rng(20261031)


def Aop(p):
    def mm(Y):
        Y = np.asarray(Y)
        st["matvecs"] += 1 if Y.ndim == 1 else Y.shape[1]
        if Y.ndim == 1:
            return diag * Y + Hh @ Y + p * (Hp @ Y)
        return diag[:, None] * Y + Hh @ Y + p * (Hp @ Y)

    return LinearOperator((N, N), matvec=mm, matmat=mm, dtype=np.float64)


def exact_p0():
    """The m lowest levels and eigenvectors of H(0) in the even sector, block by block in the photon number."""
    occ = np.indices((d,) * n).reshape(n, -1)[:, model.idx]
    tot = occ.sum(0)
    del occ
    H0 = (sps.diags(diag) + Hh).tocsr()
    det = kc.quantum_detuning(Delta, a.K, a.h, a.family)
    wmin = float(np.linalg.eigvalsh(np.diag(det) - xi0 * np.asarray(J, dtype=np.float64))[0])
    blocks, levels, nb = [], [], 0                      # levels: (energy, block index, column)
    while True:
        sel = np.flatnonzero(tot == nb)
        if sel.size == 0:
            break
        lb = nb * wmin + 0.5 * a.K * a.h * (nb * nb / n - nb)
        # lb is a convex quadratic in nb, increasing for nb >= n/2, so once it is above the m-th level there,
        # every larger block is above it too
        if nb >= n / 2 and len(levels) >= m and lb > sorted(e for e, _, _ in levels)[m - 1]:
            break
        e, U = np.linalg.eigh(H0[sel][:, sel].toarray())
        blocks.append((nb, sel, U))
        levels += [(float(x), len(blocks) - 1, j) for j, x in enumerate(e)]
        nb += 2
    levels.sort()
    X = np.zeros((N, m))
    for c, (e, b, j) in enumerate(levels[:m]):
        X[blocks[b][1], c] = blocks[b][2][:, j]
    info = dict(blocks=[int(b[0]) for b in blocks], w_min=wmin,
                photon_number=[int(blocks[b][0]) for _, b, _ in levels[:m]])
    return np.array([e for e, _, _ in levels[:m]]), X, info


def solve(p):
    """Lowest m levels at p by LOBPCG continued from the tracked vectors st['X'] (updated on success)."""
    st["calls"] += 1
    X, _ = np.linalg.qr(st["X"])
    A = Aop(p)
    Hs = X.T @ A.matmat(X)
    mu = np.linalg.eigvalsh(0.5 * (Hs + Hs.T))              # Rayleigh-Ritz upper bounds of E_0 .. E_{m-1}
    scale = max(1.0, float(np.abs(mu).max()))
    dd = np.abs(diag - mu[0]) + 1.0
    M = LinearOperator((N, N), matvec=lambda y: np.ravel(y) / dd, matmat=lambda Y: np.asarray(Y) / dd[:, None],
                       dtype=np.float64)
    Y, why = X, ""
    for attempt in range(3):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w, V = lobpcg(A, Y, M=M, tol=a.restol * scale, maxiter=a.iters * (attempt + 1), largest=False)
        except Exception as exc:                            # e.g. an ill-conditioned Gram matrix: perturb, retry
            st["retries"] += 1
            why = f"lobpcg raised {type(exc).__name__}: {exc}"
            Y, _ = np.linalg.qr(X + 1e-4 * rng.standard_normal(X.shape))
            continue
        o = np.argsort(w)
        w, V = np.asarray(w)[o], V[:, o]
        res = np.linalg.norm(A.matmat(V) - V * w[None, :], axis=0)
        ok_res = bool(np.all(res[: a.k] <= 10 * a.restol * scale))
        ok_rr = bool(np.all(w <= mu + 1e-9 * scale))
        if ok_res and ok_rr:
            st["X"] = V
            return w
        if not ok_rr:
            st["misconv"] += 1
        st["retries"] += 1
        why = f"residuals {res[: a.k]}, Ritz {w}, bounds {mu}"
        Y = V
    raise RuntimeError(f"inst {a.inst}: eigensolve failed at p = {p}: {why}")


def spectrum(p):
    if p == 0.0:
        w, X, _ = exact_p0()
        st["X"] = X
        return w
    return solve(p)


def gap(p):
    w = spectrum(p)
    evals.append((float(p), w[: a.k].tolist()))
    return float(w[1] - w[0])


evals = []
t_build = time.time() - t0

if a.probe:
    print(f"inst {a.inst} family {a.family} h {a.h} cutoff {a.cutoff}: sector dim {N}, nnz hop {Hh.nnz} "
          f"pump {Hp.nnz}, build {t_build:.1f} s")
    tp = time.time()
    w0, X0, info0 = exact_p0()
    st["X"] = X0
    print(f"  p =  0.00 (exact, photon blocks {info0['blocks']}, w_min {info0['w_min']:.4f}): "
          f"E = {np.array2string(w0[: a.k], precision=8)}  photon numbers {info0['photon_number']}  "
          f"{time.time() - tp:.2f} s")
    times, cont = [], {}
    for p in (0.05, 0.5, 1.0, 1.05, 3.0, a.pf):
        tp, m0 = time.time(), st["matvecs"]
        w = solve(p)
        cont[p] = w
        times.append(time.time() - tp)
        print(f"  p = {p:5.2f}: E0 = {w[0]:+.10f}  gap = {w[1] - w[0]:.6e}  {times[-1]:.2f} s  "
              f"{st['matvecs'] - m0} matvecs")
    per = float(np.median(times))
    n_calls = a.npts + a.nmin * 30 + 16
    print(f"median {per:.2f} s per eigensolve, ~{n_calls} eigensolves per task -> ~{per * n_calls / 60:.0f} min "
          f"per task (single core); retries {st['retries']}, misconvergences {st['misconv']}")
    print("independent cross-check with eigsh (ARPACK, random start), k = %d:" % a.k)
    for p in (1.0, 3.0):
        try:
            we = np.sort(eigsh(Aop(p), k=a.k, which="SA", tol=1e-10, ncv=40, maxiter=20000,
                               return_eigenvectors=False))
            dE = np.abs(we[:2] - cont[p][:2]).max()
            print(f"  p = {p:4.2f}: eigsh E0, E1 = {we[0]:+.10f}, {we[1]:+.10f};  continuation "
                  f"{cont[p][0]:+.10f}, {cont[p][1]:+.10f};  max diff {dE:.1e}"
                  + ("  (eigsh higher: it missed a level, continuation is the lower = better one)"
                     if we[0] > cont[p][0] + 1e-8 or we[1] > cont[p][1] + 1e-8 else ""))
        except Exception as exc:
            print(f"  p = {p:4.2f}: eigsh failed ({type(exc).__name__})")
    ok = abs(w0[0]) < 1e-9 and info0["photon_number"][0] == 0
    print("check: p = 0 ground state = vacuum, E0 = 0" + ("  -> OK" if ok else "  -> WRONG, do not run the gap lines"))
    raise SystemExit(0)

pg = np.linspace(0.0, a.pf, a.npts)
Eg = np.zeros((a.npts, a.k))
g = np.zeros(a.npts)
cand = {}                                                     # grid index of a local minimum -> tracked vectors
prevX = None
for i, p in enumerate(pg):
    w = spectrum(float(p))
    Eg[i] = w[: a.k]
    g[i] = w[1] - w[0]
    Xi = st["X"].copy()
    if i >= 1:
        j = i - 1
        if (j == 0 or g[j] <= g[j - 1]) and g[j] <= g[i]:
            cand[j] = prevX
            if len(cand) > 2 * a.nmin:
                del cand[max(cand, key=lambda q: g[q])]
    prevX = Xi
if g[-1] <= g[-2]:
    cand[a.npts - 1] = prevX
if abs(Eg[0, 0]) > 1e-9:
    raise RuntimeError(f"inst {a.inst}: E0(p = 0) = {Eg[0, 0]} is not the vacuum energy 0")

loc = sorted(cand, key=lambda q: g[q])[: a.nmin]
minima, mvec = [], {}
for i in loc:
    lo, hi = pg[max(i - 1, 0)], pg[min(i + 1, a.npts - 1)]
    st["X"] = cand[i]
    r = minimize_scalar(gap, bounds=(lo, hi), method="bounded", options=dict(xatol=a.xatol))
    pm, gmin = float(r.x), float(r.fun)
    mvec[i] = st["X"].copy()
    if g[i] < gmin:                                          # Brent did not beat the grid point (e.g. an end point)
        pm, gmin = float(pg[i]), float(g[i])
        mvec[i] = cand[i]
    minima.append(dict(grid_index=int(i), p=pm, gap=gmin))
minima.sort(key=lambda q: q["gap"])
pstar, dmin = minima[0]["p"], minima[0]["gap"]
edge = int(minima[0]["grid_index"] in (0, a.npts - 1))   # deepest minimum in the first or last grid cell

st["X"] = mvec[minima[0]["grid_index"]]
lz_pts, v2s = [], []
for dp in (1e-3, 3e-3, 1e-2, 3e-2):
    for sgn in (-1.0, 1.0):
        pp = pstar + sgn * dp
        if 0.0 < pp <= a.pf:
            gg = gap(pp)
            lz_pts.append((float(pp), gg))
            if gg > dmin:
                v2s.append((gg ** 2 - dmin ** 2) / dp ** 2)
v = float(np.sqrt(np.median(v2s))) if v2s else float("nan")
x_lz = float(np.pi * dmin ** 2 / (2.0 * v * a.pf)) if v2s and v > 0 else float("nan")

out = dict(tag=a.tag, kind="gap", version=3, set=a.set, inst=a.inst, family=a.family, h=a.h, K=a.K,
           cutoff=a.cutoff, pf=a.pf, k=a.k, guard=a.guard, restol=a.restol, sector_dim=N, p_grid=pg.tolist(),
           E_grid=Eg.tolist(), gap_grid=g.tolist(), minima=minima, Delta_min=dmin, p_star=pstar, edge=edge,
           lz_points=lz_pts, v_LZ=v, x_LZ=x_lz, gap_at_pf=float(g[-1]), refine_evals=evals,
           eigsolves=st["calls"], matvecs=st["matvecs"], retries=st["retries"], misconv=st["misconv"],
           build_s=t_build, wall_s=time.time() - t0)
kc.write_json_atomic(a.out, out)
print(f"{a.set}#{a.inst} {a.family} h={a.h:g} cut={a.cutoff}: Delta_min={dmin:.3e} at p*={pstar:.4f} (edge {edge}) "
      f"v={v:.3e} x_LZ={x_lz:.3e}  solves={st['calls']} mv={st['matvecs']} retries={st['retries']} "
      f"{out['wall_s']:.0f}s")
