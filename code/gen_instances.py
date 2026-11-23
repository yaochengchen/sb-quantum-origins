"""Generate an instance set and its per-instance landscape / linear-stage diagnostics.

usage: python gen_instances.py --set G4 --n 4 --ninst 1000 --seed 2016 --xi0 0.5 --out ../instances
Writes <out>/<set>.npz (J[ninst,n,n], Delta[ninst,n], meta) and <out>/<set>_info.csv.

G4 follows Goto 2016 (Sci. Rep. 6, 21686): n = 4, J_ij (i<j) uniform in [-1, 1], J symmetric, J_ii = 0,
no local fields, Delta_i = xi0 * sum_j |J_ij| (his Eq. 7), xi0 = 0.5 K. The paper does not state the
distribution beyond "chosen randomly in the range of -1 to 1"; uniform is our assumption.

Diagnostics (info.csv):
  E0, E1, gap        ground energy, lowest energy above the ground manifold, E1 - E0
  n_gs               number of ground patterns (2 = one Z2 pair)
  gs_b               pattern index of the ground state with s_0 = +1
  n_locmin           single-flip local minima, counted per Z2 pair
  lam1, lam2         two smallest eigenvalues of M = diag(Delta) - xi0 J. The origin of the linearised KPO
                     network loses stability at p = lam1 (first bifurcation, Goto 2016 Eq. 6)
  trap               1 if sign(v1) (eigenvector of lam1) is NOT a ground state = Goto's "eigenvector trap"
  trap_excess        E(sign v1) - E0
  v1_gs_overlap      |v1 . s_gs| / (|v1| sqrt(n))
"""
import argparse
import json
import os

import numpy as np

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--ninst", type=int, default=1000)
ap.add_argument("--seed", type=int, default=2016)
ap.add_argument("--xi0", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rng = np.random.default_rng(a.seed)
n = a.n
Js = np.zeros((a.ninst, n, n))
Ds = np.zeros((a.ninst, n))
iu = np.triu_indices(n, 1)
S = kc.all_spins(n).astype(float)
rows = []
for k in range(a.ninst):
    J = np.zeros((n, n))
    J[iu] = rng.uniform(-1.0, 1.0, len(iu[0]))
    J = J + J.T
    Delta = kc.goto_detuning(J, a.xi0)
    Js[k], Ds[k] = J, Delta
    E = kc.ising_energies(J)
    gm = kc.ground_mask(E)
    E0 = E.min()
    above = E[~gm]
    E1 = above.min() if above.size else np.nan
    gs = np.flatnonzero(gm)
    gs_b = int([b for b in gs if S[b, 0] > 0][0])
    # single-flip local minima (strict: every flip raises E)
    nloc = 0
    for b in range(2 ** n):
        if all(E[b ^ (1 << i)] > E[b] for i in range(n)):
            nloc += 1
    M = np.diag(Delta) - a.xi0 * J
    w, U = np.linalg.eigh(M)
    v1 = U[:, 0]
    b1 = int(kc.pattern_index(np.where(v1 >= 0, 1, -1)))
    rows.append(dict(inst=k, E0=E0, E1=E1, gap=E1 - E0, n_gs=int(gm.sum()), gs_b=gs_b,
                     n_locmin=nloc // 2, lam1=w[0], lam2=w[1] if n > 1 else np.nan,
                     trap=int(not gm[b1]), trap_excess=E[b1] - E0,
                     v1_gs_overlap=abs(v1 @ S[gs_b]) / (np.linalg.norm(v1) * np.sqrt(n))))

meta = dict(set=a.set, n=n, ninst=a.ninst, seed=a.seed, xi0=a.xi0, K=1.0,
            J_dist="uniform(-1,1) upper triangle, symmetric, zero diagonal, no fields",
            Delta="xi0 * sum_j |J_ij| (Goto 2016 Eq. 7)")
np.savez(os.path.join(a.out, f"{a.set}.npz"), J=Js, Delta=Ds, meta=json.dumps(meta))
keys = list(rows[0].keys())
with open(os.path.join(a.out, f"{a.set}_info.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(f"{r[k]:.10g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
tr = np.mean([r["trap"] for r in rows])
print(f"{a.set}: {a.ninst} instances, n={n}, seed={a.seed}; eigenvector-trap fraction {tr:.3f}; "
      f"degenerate (n_gs>2): {sum(r['n_gs'] > 2 for r in rows)}")
