"""MSBQA 1d: instance set with a UNIFORM detuning, same couplings as the source set.

usage (from msbqa_phase1/):  python code/make_uniform_set.py --inst-dir instances --src G4 --dst G4U --mode max
Writes <inst-dir>/<dst>.npz (J identical to <src>, Delta_i = max_j Delta_j of the same instance for every i) and
<dst>_info.csv with the same columns as gen_instances.py (recomputed for the new detuning: lam1, lam2, trap, ...).
mode max: diag(Delta) - xi0 J is positive semidefinite on every instance (Gershgorin, since the source Delta_i =
xi0 sum_j |J_ij|), so the vacuum is the ground state at p = 0 and the quantum annealing is well defined.
mode mean: uniform mean_j Delta_j (not PSD on some instances; not used for the quantum runs).
Idempotent: if <dst> already exists it must be identical (J, Delta and the info file), otherwise the script stops.
Checks: the J-only columns of the info file (E0, E1, gap, n_gs, gs_b, n_locmin) equal those of the source set.
"""
import argparse
import json
import os

import numpy as np

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--src", default="G4")
ap.add_argument("--dst", default="G4U")
ap.add_argument("--mode", default="max", choices=["max", "mean"])
a = ap.parse_args()

src = np.load(os.path.join(a.inst_dir, f"{a.src}.npz"))
J = src["J"].astype(np.float64)
D = src["Delta"].astype(np.float64)
meta = json.loads(str(src["meta"]))
xi0 = float(meta["xi0"])
ninst, n = D.shape
DU = np.repeat(D.max(1, keepdims=True) if a.mode == "max" else D.mean(1, keepdims=True), n, 1)
S = kc.all_spins(n).astype(float)


def info_row(k, Jk, Delta):
    """Exactly the diagnostics of gen_instances.py."""
    E = kc.ising_energies(Jk)
    gm = kc.ground_mask(E)
    E0 = E.min()
    above = E[~gm]
    E1 = above.min() if above.size else np.nan
    gs = np.flatnonzero(gm)
    gs_b = int([b for b in gs if S[b, 0] > 0][0])
    nloc = 0
    for b in range(2 ** n):
        if all(E[b ^ (1 << i)] > E[b] for i in range(n)):
            nloc += 1
    M = np.diag(Delta) - xi0 * Jk
    w, U = np.linalg.eigh(M)
    v1 = U[:, 0]
    b1 = int(kc.pattern_index(np.where(v1 >= 0, 1, -1)))
    return dict(inst=k, E0=E0, E1=E1, gap=E1 - E0, n_gs=int(gm.sum()), gs_b=gs_b,
                n_locmin=nloc // 2, lam1=w[0], lam2=w[1] if n > 1 else np.nan,
                trap=int(not gm[b1]), trap_excess=E[b1] - E0,
                v1_gs_overlap=abs(v1 @ S[gs_b]) / (np.linalg.norm(v1) * np.sqrt(n)))


rows = [info_row(k, J[k], DU[k]) for k in range(ninst)]
keys = list(rows[0].keys())
text = ",".join(keys) + "\n" + "".join(
    ",".join(f"{r[k]:.10g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n" for r in rows)

# check: columns that depend only on J must equal the source info file
si = np.genfromtxt(os.path.join(a.inst_dir, f"{a.src}_info.csv"), delimiter=",", names=True)
for col in ("E0", "E1", "gap", "n_gs", "gs_b", "n_locmin"):
    new = np.array([float(r[col]) for r in rows])
    if not np.allclose(new, si[col].astype(float), rtol=1e-8, atol=1e-8, equal_nan=True):
        raise SystemExit(f"check failed: column {col} differs from {a.src}_info.csv")
lam1 = np.array([r["lam1"] for r in rows])
npsd = int(np.sum(lam1 < -1e-12))
if a.mode == "max" and npsd:
    raise SystemExit(f"check failed: {npsd} instances not PSD with the max detuning")

meta2 = dict(meta)
meta2.update(set=a.dst, src_set=a.src, detuning_mode=a.mode,
             Delta=f"uniform {a.mode}_j of the {a.src} detuning xi0 * sum_j |J_ij| (same value for every spin)")
pz = os.path.join(a.inst_dir, f"{a.dst}.npz")
pi = os.path.join(a.inst_dir, f"{a.dst}_info.csv")
if os.path.exists(pz) or os.path.exists(pi):
    if not (os.path.exists(pz) and os.path.exists(pi)):
        raise SystemExit(f"only one of {pz}, {pi} exists: remove it and run again")
    old = np.load(pz)
    same = (np.array_equal(old["J"], J) and np.array_equal(old["Delta"], DU) and open(pi).read() == text)
    if not same:
        raise SystemExit(f"{pz} exists with different content: not overwritten (remove it by hand if intended)")
    print(f"{a.dst}: exists and identical ({ninst} instances), nothing written")
else:
    tmp = os.path.join(a.inst_dir, f"{a.dst}.tmp.npz")
    np.savez(tmp, J=J, Delta=DU, meta=json.dumps(meta2))
    os.replace(tmp, pz)
    with open(pi + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(pi + ".tmp", pi)
    print(f"{a.dst}: written {ninst} instances (mode {a.mode}) -> {pz}, {pi}")
print(f"{a.dst}: trap fraction {np.mean([r['trap'] for r in rows]):.3f} ({sum(r['trap'] for r in rows)}), "
      f"not PSD {npsd}, lam1 min {lam1.min():.3e}; J-only columns equal to {a.src}: OK")
