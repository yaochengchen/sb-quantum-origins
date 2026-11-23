"""MSBQA phase 1c, step 3: one-knob ablation between L4 (bSB-type dynamics with the KPO parameters, ~0.96) and the
standard bSB of sb_tune.py (Goto 2021 normalisation, bSB*, ~1.000). Which ingredient fixes the trap instances?
One instance per call.

usage (from msbqa_phase1/):
  python code/sb_ablate.py --inst-dir instances --set G4 --inst 17 --R 1000 --tag SBA --out run_v4/parts/SBA_17.json

L4 and bSB are the same integrator (symplectic Euler, y first, then x, inelastic walls), written in KPO units:
    t += dt;  a = a_f * min(t / T, 1)
    y_i += dt * ( -(D_i - a) x_i + c * sum_j J_ij x_j ),   x_i += dt * D_i * y_i,   |x_i| > w -> x_i = w sign x_i, y_i = 0
    initial x, y uniform(-A, A), readout sign(x).
L4 = kpo_common.classical_run 'L4' exactly (bitwise: same seed (seed_base, inst, 4), same formula):
    D = Delta_i (per spin), c = xi0, a_f = pf = 7, T = 700, dt = 0.0025, A = 1e-6, w = sqrt(pf/K).
bSB* (sb_tune best on the training set: c = 0.5, dt = 1, 10000 steps, A = 1e-3, walls at 1, a0 = 1) in the same
units with Dbar = mean_i Delta_i (rescaling time by Dbar maps it exactly onto sb_tune's a0 = 1 form):
    D = Dbar (uniform), c = 0.5 Dbar / (sigma sqrt n), a_f = Dbar, T = 10000 / Dbar, dt = 1 / Dbar, A = 1e-3, w = 1.
(The dynamics is linear apart from the walls, so only A / w matters, not A and w separately.)

Knobs (each switched alone, from L4 towards bSB* and from bSB* back towards L4):
  D     kpo: Delta_i per spin               bsb: uniform Dbar
  c     kpo: xi0                            bsb: 0.5 Dbar / (sigma sqrt n)
  af    kpo: pf = 7 (~ 9 Dbar)              bsb: Dbar          (also af = 2 Dbar, 4 Dbar from L4)
  T     kpo: 700                            bsb: 10000 / Dbar
  dt    kpo: 0.0025 (ODE-accurate)          bsb: 1 / Dbar      (algorithmic step)
  init  kpo: A = 1e-6, w = sqrt(pf/K)       bsb: A = 1e-3, w = 1
JSON: per configuration the knob settings, the numbers used, P (pattern probabilities) and P_succ.
"""
import argparse
import math
import os
import time

import numpy as np

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--inst", type=int, required=True)
ap.add_argument("--R", type=int, default=1000)
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--seed-base", type=int, default=20261029)
ap.add_argument("--tag", default="SBA")
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
gm = kc.ground_mask(E)
Dbar = float(np.mean(Delta))
sigma = float(np.sqrt((Jm ** 2).sum() / (n * (n - 1))))
VAL = {
    "D": {"kpo": np.asarray(Delta, dtype=np.float64), "bsb": np.full(n, Dbar)},
    "c": {"kpo": xi0, "bsb": 0.5 * Dbar / (sigma * math.sqrt(n))},
    "af": {"kpo": a.pf, "bsb": Dbar, "2D": 2.0 * Dbar, "4D": 4.0 * Dbar},
    "T": {"kpo": 700.0, "bsb": 10000.0 / Dbar},
    "dt": {"kpo": 0.0025, "bsb": 1.0 / Dbar},
    "init": {"kpo": (1e-6, math.sqrt(a.pf / a.K)), "bsb": (1e-3, 1.0)},
}
KNOBS = ["D", "c", "af", "T", "dt", "init"]
L4 = {k: "kpo" for k in KNOBS}
B = {k: "bsb" for k in KNOBS}
CONFIGS = [("L4", L4)]
CONFIGS += [(f"L4 +{k}", {**L4, k: "bsb"}) for k in KNOBS]
CONFIGS += [("L4 +af=2D", {**L4, "af": "2D"}), ("L4 +af=4D", {**L4, "af": "4D"})]
CONFIGS += [("bSB*", B)]
CONFIGS += [(f"bSB* -{k}", {**B, k: "kpo"}) for k in KNOBS]


def run(cfg):
    D = VAL["D"][cfg["D"]]
    c = VAL["c"][cfg["c"]]
    af = VAL["af"][cfg["af"]]
    T = VAL["T"][cfg["T"]]
    dt = VAL["dt"][cfg["dt"]]
    A, w = VAL["init"][cfg["init"]]
    rng = np.random.default_rng([a.seed_base, a.inst, kc.LAYERS.index("L4")])
    x = rng.uniform(-A, A, (a.R, n))
    y = rng.uniform(-A, A, (a.R, n))
    nsteps = int(math.ceil(T / dt))                          # as kpo_common.classical_run
    dt = T / nsteps
    t = 0.0
    for _ in range(nsteps):
        t += dt
        p = af * min(max(t / T, 0.0), 1.0)
        y += dt * (-(D - p) * x + c * (x @ Jm))
        x += dt * D * y
        hit = np.abs(x) > w
        if hit.any():
            x[hit] = w * np.sign(x[hit])
            y[hit] = 0.0
    b = kc.pattern_index(np.where(x >= 0, 1, -1))
    P = np.bincount(b, minlength=2 ** n) / a.R
    ps, er = kc.summarize_patterns(P, E)
    return dict(P=P.tolist(), P_succ=ps, E_res=er, nsteps=nsteps, dt=dt, T=T, af=af, c=c, A=A, w=w,
                D=np.asarray(D).tolist())


out = dict(tag=a.tag, kind="ablate", set=a.set, inst=a.inst, R=a.R, Dbar=Dbar, sigma=sigma, xi0=xi0, configs={})
for name, cfg in CONFIGS:
    tl = time.time()
    r = run(cfg)
    r.update(knobs=cfg, wall_s=time.time() - tl)
    out["configs"][name] = r
out["wall_s"] = time.time() - t0
kc.write_json_atomic(a.out, out)
print(f"{a.set}#{a.inst} ablate: " + "  ".join(f"{k}={v['P_succ']:.3f}" for k, v in out["configs"].items())
      + f"  {out['wall_s']:.0f}s")
