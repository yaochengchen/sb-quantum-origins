"""MSBQA phase 1b, step 1a: initial-noise scan of the classical layers (fair classical baseline) and a direct
test of the bifurcation-delay hypothesis. One instance per call.

usage (from msbqa_phase1/):
  python code/noise_scan.py --inst-dir instances --set G4 --inst 17 --T 700 --pf 7 --R 1000 --dt 0.0025 \
      --tag NS --out run_v4/parts/NS_17.json --raw run_v4/raw/NS_17.npz
Options: --layers L2g,L3,L4,L5   --eps 1e-8,1e-6,1e-4,1e-3,1e-2,3e-2,0.1,0.3,1   --thr 0.1   --every 10

For every layer and every eps: R trajectories with initial noise uniform(-eps, eps) on both quadratures
(Re z, Im z for L2g; x, y for L3-L5). Apart from eps the equations, integrators and random streams are those
of kpo_common.classical_run / run_classical.py:
  L2g  Goto 2016 App. E classical KPO model, RK4
  L3   aSB with KPO parameters, symplectic Euler        L4  bSB-type        L5  dSB-type (walls at sqrt(pf/K))
The random stream of layer l of instance i is seeded (seed_base, i, kc.LAYERS.index(l)) as in run_classical.py
and the uniform draws are scaled by eps, so eps = 1e-6 reproduces the G0c results of run_v4 exactly (checked in
phase1b_report.py), and all eps use common random numbers (the same underlying draws).

Departure from the origin: every --every steps, the first time max_i |z_i| (L2g) or max_i |x_i| (L3-L5)
exceeds --thr (default 0.1) is the departure; we store p_dep = p(t_dep) and the shares q1 = (v1.x)^2/|x|^2,
q2 = (v2.x)^2/|x|^2 of the state at departure (x = Re z for L2g), where v1, v2 are the eigenvectors of the two
smallest eigenvalues lam1, lam2 of M = diag(Delta) - xi0 J (for L3-L5 the growing linear mode is the one of
diag(Delta)(M - p), close to but not exactly v1). Trajectories that never depart get p_dep = nan. The departure
is only meaningful for eps well below thr (eps <= 1e-3 clearly, 1e-2 marginally); for larger eps it is at t ~ 0.

Outputs (written atomically; the JSON is written last and marks the task as done):
  --raw  NPZ: p_dep, q1, q2 (float32) and success (bool), shape (n_layers, n_eps, R); layers, eps, lam1, lam2
  --out  JSON: per layer a list over eps of {P (pattern probabilities), P_succ, E_res, departure summary}
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
ap.add_argument("--R", type=int, default=1000)
ap.add_argument("--dt", type=float, default=0.0025)
ap.add_argument("--layers", default="L2g,L3,L4,L5")
ap.add_argument("--eps", default="1e-8,1e-6,1e-4,1e-3,1e-2,3e-2,0.1,0.3,1")
ap.add_argument("--thr", type=float, default=0.1, help="departure threshold on max_i |z_i| or max_i |x_i|")
ap.add_argument("--every", type=int, default=10, help="check the departure every this many steps")
ap.add_argument("--xwall", type=float, default=0.0, help="wall position for L4/L5; 0 -> sqrt(pf/K)")
ap.add_argument("--seed-base", type=int, default=20261029)
ap.add_argument("--tag", default="NS")
ap.add_argument("--out", required=True)
ap.add_argument("--raw", required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(0)

t0 = time.time()
layers = a.layers.split(",")
for l in layers:
    if l not in ("L2g", "L3", "L4", "L5"):
        raise SystemExit(f"noise_scan.py: layer {l} not supported (L2g, L3, L4, L5)")
eps_list = [float(e) for e in a.eps.split(",")]
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
xi0 = meta["xi0"]
n = len(Delta)
K = a.K
Jm = np.asarray(J, dtype=np.float64)
E = kc.ising_energies(J)
gm = kc.ground_mask(E)
pfun = kc.linear_schedule(a.pf, a.T)
xw = a.xwall if a.xwall > 0 else math.sqrt(a.pf / K)
w, U = np.linalg.eigh(np.diag(Delta) - xi0 * Jm)
lam1, lam2 = float(w[0]), float(w[1])
v1, v2 = U[:, 0], U[:, 1]
nsteps = int(math.ceil(a.T / a.dt))
dt = a.T / nsteps
thr2 = a.thr ** 2


def run(layer, eps, rng):
    """Same integration as kpo_common.classical_run (noise amplitude eps), plus departure tracking.
    Returns final x (R, n), p_dep (R,), x at departure (R, n)."""
    R = a.R
    p_dep = np.full(R, np.nan)
    x_dep = np.zeros((R, n))
    gone = np.zeros(R, dtype=bool)

    def check(amp2, t, xnow):
        new = (~gone) & (amp2 > thr2)
        if new.any():
            p_dep[new] = pfun(t)
            x_dep[new] = xnow[new]
            gone[new] = True

    if layer == "L2g":
        z = rng.uniform(-eps, eps, (R, n)) + 1j * rng.uniform(-eps, eps, (R, n))
        Deff = Delta.copy()

        def f(z, t):
            return -1j * ((Deff + K * (z.real ** 2 + z.imag ** 2)) * z - pfun(t) * np.conj(z) - xi0 * (z @ Jm))

        t = 0.0
        for s in range(nsteps):                              # classical RK4, as in kpo_common
            k1 = f(z, t)
            k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(z + dt * k3, t + dt)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            if (s + 1) % a.every == 0:
                check((z.real ** 2 + z.imag ** 2).max(1), t, z.real)
        x = z.real
    else:
        x = rng.uniform(-eps, eps, (R, n))
        y = rng.uniform(-eps, eps, (R, n))
        t = 0.0
        for s in range(nsteps):                              # symplectic Euler, p at the new time, as in kpo_common
            t += dt
            p = pfun(t)
            if layer == "L3":
                y += dt * (-(K * x * x - p + Delta) * x + xi0 * (x @ Jm))
                x += dt * Delta * y
            else:
                drive = x if layer == "L4" else xw * np.sign(x)
                y += dt * (-(Delta - p) * x + xi0 * (drive @ Jm))
                x += dt * Delta * y
                hit = np.abs(x) > xw
                if hit.any():
                    x[hit] = xw * np.sign(x[hit])
                    y[hit] = 0.0
            if (s + 1) % a.every == 0:
                check((x * x).max(1), t, x)
    return x, p_dep, x_dep


nL, nE = len(layers), len(eps_list)
P_DEP = np.full((nL, nE, a.R), np.nan, dtype=np.float32)
Q1 = np.full((nL, nE, a.R), np.nan, dtype=np.float32)
Q2 = np.full((nL, nE, a.R), np.nan, dtype=np.float32)
SUCC = np.zeros((nL, nE, a.R), dtype=bool)
out = dict(tag=a.tag, kind="noise", set=a.set, inst=a.inst, T=a.T, pf=a.pf, K=K, R=a.R, dt=a.dt, xwall=xw,
           thr=a.thr, every=a.every, seed_base=a.seed_base, eps=eps_list, lam1=lam1, lam2=lam2, layers={})
for li, layer in enumerate(layers):
    out["layers"][layer] = []
    for ei, eps in enumerate(eps_list):
        tl = time.time()
        rng = np.random.default_rng([a.seed_base, a.inst, kc.LAYERS.index(layer)])
        x, p_dep, x_dep = run(layer, eps, rng)
        b = kc.pattern_index(np.where(x >= 0, 1, -1))
        succ = gm[b]
        P = np.bincount(b, minlength=2 ** n) / a.R
        psucc, eres = kc.summarize_patterns(P, E)
        gone = np.isfinite(p_dep)
        nx2 = (x_dep ** 2).sum(1)
        q1 = np.full(a.R, np.nan)
        q2 = np.full(a.R, np.nan)
        ok = gone & (nx2 > 0)
        q1[ok] = (x_dep[ok] @ v1) ** 2 / nx2[ok]
        q2[ok] = (x_dep[ok] @ v2) ** 2 / nx2[ok]
        P_DEP[li, ei], Q1[li, ei], Q2[li, ei], SUCC[li, ei] = p_dep, q1, q2, succ
        before = gone & (p_dep <= lam2)
        after = gone & (p_dep > lam2)
        dep = dict(frac_departed=float(gone.mean()),
                   p_dep_q10=float(np.quantile(p_dep[gone], 0.1)) if gone.any() else float("nan"),
                   p_dep_med=float(np.median(p_dep[gone])) if gone.any() else float("nan"),
                   p_dep_q90=float(np.quantile(p_dep[gone], 0.9)) if gone.any() else float("nan"),
                   frac_after_lam2=float(after.sum() / max(gone.sum(), 1)),
                   n_before=int(before.sum()), n_after=int(after.sum()),
                   succ_before=float(succ[before].mean()) if before.any() else float("nan"),
                   succ_after=float(succ[after].mean()) if after.any() else float("nan"),
                   q1_mean=float(np.nanmean(q1)) if ok.any() else float("nan"),
                   q2_mean=float(np.nanmean(q2)) if ok.any() else float("nan"))
        out["layers"][layer].append(dict(eps=eps, P=P.tolist(), P_succ=psucc, E_res=eres, dep=dep,
                                         wall_s=time.time() - tl))
out["wall_s"] = time.time() - t0

os.makedirs(os.path.dirname(os.path.abspath(a.raw)), exist_ok=True)
tmp = a.raw + ".tmp"
with open(tmp, "wb") as fh:                                  # file handle: np.savez must not append '.npz'
    np.savez_compressed(fh, p_dep=P_DEP, q1=Q1, q2=Q2, success=SUCC, layers=np.array(layers),
                        eps=np.array(eps_list), lam1=lam1, lam2=lam2)
os.replace(tmp, a.raw)
kc.write_json_atomic(a.out, out)
print(f"{a.set}#{a.inst} noise scan: " + "  ".join(
    f"{l}:" + ",".join(f"{r['P_succ']:.2f}" for r in out["layers"][l]) for l in layers) + f"  {out['wall_s']:.0f}s")
