"""MSBQA 1d: 'Ising window' test of why bSB escapes the eigenvector trap (one instance per call).

usage (from msbqa_phase1/):
  python code/iwindow.py --inst-dir instances --set G4 --inst 1 --R 1000 --tag IW --out run_v4/parts/IW_1.json

Hypothesis (doc MSBQA_1c消融复核与下一步, sec. 3): with walls and a UNIFORM detuning D, at p = D the force inside the
walls is c J x only, so the box potential is the Ising energy and every configuration that is not a single-flip
local minimum is unstable there; a trajectory that spends enough time near p = D leaves the trap configuration.
With the per-spin detuning Delta_i there is no common p with Delta_i - p = 0 for all spins. Far above D (p = pf)
every corner is stable, so extra time there should not help.

Dynamics: exactly L4 / sb_ablate.py (symplectic Euler, y first, then x, inelastic walls at w = sqrt(pf/K)):
    y_i += dt * ( -(D_i - p) x_i + xi0 sum_j J_ij x_j ),   x_i += dt * D_i * y_i,   |x_i| > w -> x_i = w sign x_i, y_i = 0
    initial x, y uniform(-1e-6, 1e-6) with the L4 seed (seed_base, inst, 4): the same initial conditions in every
    configuration (common random numbers); readout sign(x).
Schedules (ramp p = pf t / T, T = 700, pf = 7, dt = 0.0025, i.e. the L4 rate 0.01):
    base           the ramp only (T = 700)
    hold@Dbar tau  the ramp, stopped for a time tau at p = Dbar = mean_i Delta_i, then continued to pf
    hold@pf tau    the ramp, then a time tau at p = pf (control: same total time, spent where every corner is stable)
Detunings: kpo (D_i = Delta_i, per spin) and mean (D_i = Dbar for every spin).
Checks (phase1d_report.py): 'kpo base' = sb_ablate 'L4' and 'mean base' = sb_ablate 'L4 +D' bit for bit.
JSON: per configuration P_succ, P (pattern probabilities), nsteps, dt.
"""
import argparse
import math
import os
import time

import numpy as np

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", default="G4")
ap.add_argument("--inst", type=int, required=True)
ap.add_argument("--R", type=int, default=1000)
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--T", type=float, default=700.0)
ap.add_argument("--dt", type=float, default=0.0025)
ap.add_argument("--taus", default="100,300,1000")
ap.add_argument("--seed-base", type=int, default=20261029)
ap.add_argument("--tag", default="IW")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(0)

t0 = time.time()
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
xi0 = float(meta["xi0"])
n = len(Delta)
Jm = np.asarray(J, dtype=np.float64)
E = kc.ising_energies(J)
Dbar = float(np.mean(Delta))
af, T, A, w = a.pf, a.T, 1e-6, math.sqrt(a.pf / a.K)
DET = {"kpo": np.asarray(Delta, dtype=np.float64), "mean": np.full(n, Dbar)}
TAUS = [float(x) for x in a.taus.split(",") if x.strip()]
CONFIGS = []
for det in ("kpo", "mean"):
    CONFIGS.append((f"{det} base", det, None, 0.0))
    CONFIGS += [(f"{det} hold@Dbar {tau:g}", det, Dbar, tau) for tau in TAUS]
    CONFIGS += [(f"{det} hold@pf {tau:g}", det, None, tau) for tau in TAUS]


def run(D, hold_p, tau):
    rng = np.random.default_rng([a.seed_base, a.inst, kc.LAYERS.index("L4")])
    x = rng.uniform(-A, A, (a.R, n))
    y = rng.uniform(-A, A, (a.R, n))
    Ttot = T + tau
    nsteps = int(math.ceil(Ttot / a.dt))                     # as sb_ablate.py / kpo_common.classical_run
    dt = Ttot / nsteps
    th = hold_p / af * T if hold_p is not None else None     # time at which the ramp reaches hold_p
    t = 0.0
    for _ in range(nsteps):
        t += dt
        if th is not None and t >= th:
            p = hold_p if t < th + tau else af * min(max((t - tau) / T, 0.0), 1.0)
        else:
            p = af * min(max(t / T, 0.0), 1.0)                # hold@pf: the ramp is capped at af after T
        y += dt * (-(D - p) * x + xi0 * (x @ Jm))
        x += dt * D * y
        hit = np.abs(x) > w
        if hit.any():
            x[hit] = w * np.sign(x[hit])
            y[hit] = 0.0
    b = kc.pattern_index(np.where(x >= 0, 1, -1))
    P = np.bincount(b, minlength=2 ** n) / a.R
    ps, er = kc.summarize_patterns(P, E)
    return dict(P=P.tolist(), P_succ=ps, E_res=er, nsteps=nsteps, dt=dt, Ttot=Ttot, hold_p=hold_p, tau=tau)


out = dict(tag=a.tag, kind="iwindow", set=a.set, inst=a.inst, R=a.R, Dbar=Dbar, xi0=xi0, pf=af, T=T, A=A, w=w,
           configs={})
for name, det, hold_p, tau in CONFIGS:
    tl = time.time()
    r = run(DET[det], hold_p, tau)
    r.update(det=det, wall_s=time.time() - tl)
    out["configs"][name] = r
out["wall_s"] = time.time() - t0
kc.write_json_atomic(a.out, out)
print(f"{a.set}#{a.inst} iwindow: " + "  ".join(f"{k}={v['P_succ']:.3f}" for k, v in out["configs"].items())
      + f"  {out['wall_s']:.0f}s")
