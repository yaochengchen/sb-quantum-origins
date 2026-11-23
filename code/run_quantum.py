"""L0: exact closed-system quantum KPO network for one instance (truncated Fock space, Lanczos propagator).

usage: python run_quantum.py --inst-dir ../instances --set G4 --inst 17 --family goto --T 700 --pf 7 --h 1 \
           --cutoff 18 --tol 1e-8 --dtmax 0.1 --out RUN/parts/G0q_17.json [--engine csr|tensor]
cutoff = maximal photon number per mode (Goto 2016: 18 for the four-spin runs) -> local dimension d = cutoff+1.
family: goto (Goto 2016 H, calibration) or weyl (detuning + K h, ablation chain); see kpo_common.
engine: csr (default; even-parity sector, scipy sparse) or tensor (matrix-free, full space; the v1 engine,
        needed for n >= 6). Both give the same state to the Lanczos tolerance (selftest.py).
Skips immediately if --out exists (restart-safe); writes atomically.
Output: pattern probabilities P[b] (quadrature-sign readout, Goto 2016 App. B), success probability
(ground patterns, both Z2 partners), residual energy, truncation diagnostics, propagator statistics.
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
ap.add_argument("--family", default="goto", choices=["goto", "weyl"])
ap.add_argument("--engine", default="csr", choices=["csr", "tensor"])
ap.add_argument("--T", type=float, default=700.0)
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--h", type=float, default=1.0)
ap.add_argument("--cutoff", type=int, default=18)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--dtmax", type=float, default=0.1)            # selftest 4c: 0.1 -> 7e-6, 0.25 -> 7e-4
ap.add_argument("--krylov", type=int, default=24)
ap.add_argument("--tag", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(0)

t0 = time.time()
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
xi0 = meta["xi0"]
d = a.cutoff + 1
model = kc.KPOQuantum(J, Delta, a.K, xi0, a.h, d, a.family)          # readout + diagnostics (and tensor engine)
pfun = kc.linear_schedule(a.pf, a.T)
if a.engine == "csr":
    sec = kc.KPOSector(J, Delta, a.K, xi0, a.h, d, a.family)
    t_build = time.time() - t0
    psi_s, st = kc.lanczos_propagate(sec.matvec, sec.vacuum(), a.T, pfun, tol=a.tol, m=a.krylov,
                                     dt_max=a.dtmax)
    psi = sec.embed(psi_s)
else:
    t_build = time.time() - t0
    psi, st = kc.lanczos_propagate(model.matvec, model.vacuum(), a.T, pfun, tol=a.tol, m=a.krylov,
                                   dt_max=a.dtmax)
Theta = kc.theta_matrix(d)
P = kc.pattern_probabilities(model, psi, Theta)
E = kc.ising_energies(J)
psucc, eres = kc.summarize_patterns(P, E)
pops = model.mode_populations(psi)
nbar = (pops * np.arange(d)[None, :]).sum(1)
res = dict(tag=a.tag, kind="quantum", layer="L0", family=a.family, engine=a.engine, set=a.set, inst=a.inst,
           T=a.T, pf=a.pf, K=a.K, h=a.h, cutoff=a.cutoff, tol=a.tol, dtmax=a.dtmax, krylov=a.krylov,
           P=P.tolist(), P_succ=psucc, E_res=eres, P_sum=float(P.sum()),
           norm=float(np.linalg.norm(psi)), parity_leak=model.parity_leak(psi),
           top2_pop_max=float(pops[:, -2:].sum(1).max()), nbar=nbar.tolist(),
           steps=st["steps"], rejects=st["rejects"], matvecs=st["matvecs"], err_sum=st["err_sum"],
           build_s=t_build, wall_s=time.time() - t0)
kc.write_json_atomic(a.out, res)
print(f"{a.set}#{a.inst} {a.family}/{a.engine} T={a.T} h={a.h} cut={a.cutoff}: P_succ={psucc:.4f} "
      f"E_res={eres:.4f} top2={res['top2_pop_max']:.1e} steps={st['steps']} mv={st['matvecs']} "
      f"{res['wall_s']:.0f}s")
