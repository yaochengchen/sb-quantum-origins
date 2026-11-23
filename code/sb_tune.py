"""MSBQA phase 1b, step 1b: bSB and dSB as algorithms (Goto, Endo, Suzuki et al., Sci. Adv. 7, eabe7953 (2021)),
on a grid of hyperparameters, one instance per call. Unlike L4/L5 of the ablation chain these use the standard
SB normalisation (a0 = 1, walls at |x| = 1, c0 scaled with the coupling strength) and the algorithm's own time
step, so they are judged as algorithms, not as approximations of an ODE.

usage (from msbqa_phase1/):
  python code/sb_tune.py --inst-dir instances --set G4 --inst 17 --R 1000 --tag SBT --out run_v4/parts/SBT_17.json

Update, symplectic Euler (y first, then x with the new y), k = 0 .. Nsteps-1, a_k = a0 (k + 1) / Nsteps:
    y_i += dt * ( -(a0 - a_k) x_i + c0 * sum_j J_ij f(x_j) ),    f(x) = x (bSB) or sign(x) (dSB)
    x_i += dt * a0 * y_i
    |x_i| > 1  ->  x_i = sign(x_i), y_i = 0
c0 = c / (sigma sqrt(n)), sigma = sqrt(sum_{i != j} J_ij^2 / (n (n - 1))). Readout sign(x).
Grid (all combinations): algo {bSB, dSB} x c x dt x Nsteps x A (initial x, y uniform(-A, A)).
Common random numbers: one draw of uniform(-1, 1) for x and y per instance, scaled by A.

JSON: grid axes and P_succ, E_res as nested lists of shape (2, n_c, n_dt, n_steps, n_A).
"""
import argparse
import os
import time

import numpy as np

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--inst", type=int, required=True)
ap.add_argument("--R", type=int, default=1000)
ap.add_argument("--c", default="0.25,0.5,0.7,1.0,1.5")
ap.add_argument("--dts", default="0.1,0.25,0.5,1.0,1.25")
ap.add_argument("--steps", default="100,300,1000,3000,10000")
ap.add_argument("--A", default="1e-6,1e-3,0.1,1.0")
ap.add_argument("--seed-base", type=int, default=20261031)
ap.add_argument("--tag", default="SBT")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(0)

t0 = time.time()
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
n = J.shape[0]
Jm = np.asarray(J, dtype=np.float64)
E = kc.ising_energies(J)
gm = kc.ground_mask(E)
cs = [float(v) for v in a.c.split(",")]
dts = [float(v) for v in a.dts.split(",")]
steps = [int(v) for v in a.steps.split(",")]
As = [float(v) for v in a.A.split(",")]
algos = ["bSB", "dSB"]
sigma = float(np.sqrt((Jm ** 2).sum() / (n * (n - 1))))
a0 = 1.0
rng = np.random.default_rng([a.seed_base, a.inst])
U0 = rng.uniform(-1.0, 1.0, (a.R, n))
V0 = rng.uniform(-1.0, 1.0, (a.R, n))
# all (c, A) combinations in one batch of shape (G, R, n), G = n_c * n_A, index g = ic * n_A + iA
G = len(cs) * len(As)
c0 = np.array([c / (sigma * np.sqrt(n)) for c in cs for _ in As])[:, None, None]
amp = np.array([A for _ in cs for A in As])[:, None, None]

PS = np.zeros((len(algos), len(cs), len(dts), len(steps), len(As)))
ER = np.zeros_like(PS)
for ia, algo in enumerate(algos):
    for idt, dt in enumerate(dts):
        for ins, ns in enumerate(steps):
            x = amp * U0[None, :, :]
            y = amp * V0[None, :, :]
            for k in range(ns):
                ak = a0 * (k + 1) / ns
                drive = x if algo == "bSB" else np.sign(x)
                y += dt * (-(a0 - ak) * x + c0 * (drive @ Jm))
                x += dt * a0 * y
                hit = np.abs(x) > 1.0
                if hit.any():
                    x[hit] = np.sign(x[hit])
                    y[hit] = 0.0
            b = kc.pattern_index(np.where(x >= 0, 1, -1))          # (G, R)
            for g in range(G):
                P = np.bincount(b[g], minlength=2 ** n) / a.R
                ps, er = kc.summarize_patterns(P, E)
                ic, iA = divmod(g, len(As))
                PS[ia, ic, idt, ins, iA] = ps
                ER[ia, ic, idt, ins, iA] = er

out = dict(tag=a.tag, kind="sbtune", set=a.set, inst=a.inst, R=a.R, a0=a0, sigma=sigma, algos=algos, c=cs,
           dt=dts, steps=steps, A=As, axes=["algo", "c", "dt", "steps", "A"], P_succ=PS.tolist(), E_res=ER.tolist(),
           wall_s=time.time() - t0)
kc.write_json_atomic(a.out, out)
best = [float(PS[i].max()) for i in range(len(algos))]
print(f"{a.set}#{a.inst} sb_tune: best bSB {best[0]:.3f} dSB {best[1]:.3f}  {out['wall_s']:.0f}s")
