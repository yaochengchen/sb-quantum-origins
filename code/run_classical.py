"""L1-L5: classical layers of the ablation chain for one instance (R trajectories each).

usage: python run_classical.py --inst-dir ../instances --set G4 --inst 17 --family goto --T 700 --pf 7 --h 1 \
           --R 1000 --dt 0.01 --layers L1,L2w,L2g,L3,L4,L5 --out RUN/parts/G0c_17.json
Layers (see kpo_common.LAYER_DOC): L1 TWA | L2w same eqs, tiny noise | L2g Goto 2016 classical model |
L3 aSB | L4 bSB-type | L5 dSB-type. All share K, xi0, Delta_i, p(t) and the sign readout of x.
--family only changes L1 and L2w (the Weyl flow of the chosen quantum H); L2g..L5 do not depend on it.
Random streams: layer l of instance i uses seed (seed_base, i, l) -> identical initial conditions across
plan lines that differ only in dt (convergence check). Skips if --out exists; writes atomically.
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
ap.add_argument("--T", type=float, default=700.0)
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--h", type=float, default=1.0)
ap.add_argument("--family", default="goto", choices=["goto", "weyl"],
                help="quantum model whose Weyl flow L1/L2w integrate (see kpo_common)")
ap.add_argument("--R", type=int, default=1000)
ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--layers", default=",".join(kc.LAYERS))
ap.add_argument("--xwall", type=float, default=0.0, help="wall position for L4/L5; 0 -> sqrt(pf/K)")
ap.add_argument("--seed-base", type=int, default=20261029)
ap.add_argument("--tag", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(0)

t0 = time.time()
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
xi0 = meta["xi0"]
n = len(Delta)
E = kc.ising_energies(J)
pfun = kc.linear_schedule(a.pf, a.T)
xw = a.xwall if a.xwall > 0 else math.sqrt(a.pf / a.K)
out = dict(tag=a.tag, kind="classical", family=a.family, set=a.set, inst=a.inst, T=a.T, pf=a.pf, K=a.K, h=a.h, R=a.R,
           dt=a.dt, xwall=xw, layers={})
for li, layer in enumerate(a.layers.split(",")):
    tl = time.time()
    rng = np.random.default_rng([a.seed_base, a.inst, kc.LAYERS.index(layer)])
    b = kc.classical_run(layer, J, Delta, a.K, xi0, a.h, pfun, a.T, a.dt, a.R, rng, xw, a.family)
    P = np.bincount(b, minlength=2 ** n) / a.R
    psucc, eres = kc.summarize_patterns(P, E)
    out["layers"][layer] = dict(P=P.tolist(), P_succ=psucc, E_res=eres, wall_s=time.time() - tl)
out["wall_s"] = time.time() - t0
kc.write_json_atomic(a.out, out)
print(f"{a.set}#{a.inst} " + " ".join(f"{k}={v['P_succ']:.3f}" for k, v in out["layers"].items())
      + f"  {out['wall_s']:.0f}s")
