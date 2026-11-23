"""Speed probe for one G0q quantum task (CSR engine) under the load of the node it runs on. Writes nothing.

usage (from code/):  python probe_speed.py [--inst 0] [--window 3]
Run it ON a node of the running array job, so that it sees the same memory-bandwidth contention as the tasks.
Prints:
  1 sector dimension, nonzeros, build time
  2 time of one matvec: current KPOSector.matvec, and a variant that applies one combined REAL matrix to the
    (N, 2) real view of the complex vector (no complex up-cast of the CSR data); their max difference must be ~1e-13
  3 time of one 24-vector Lanczos exponential (= half a time step) with both matvecs
  4 step sizes the adaptive propagator actually takes (same tol / dtmax as the plan) in two short windows:
      early: from the vacuum, t in [0, window]
      late : from a product of even cat states |a_i> + |-a_i>, a_i = sqrt(max(p - Delta_i, 0)), at t = T - window
             (a proxy for the end of the anneal, where the state has the most photons and dt is smallest)
  5 projected wall time of one full task at the current load (average of the early and late step rates)
"""
import argparse
import functools
import math
import os
import time

import numpy as np
import scipy.sparse as sps

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", default="../instances")
ap.add_argument("--set", default="G4")
ap.add_argument("--inst", type=int, default=0)
ap.add_argument("--cutoff", type=int, default=18)
ap.add_argument("--T", type=float, default=700.0)
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--dtmax", type=float, default=0.1)
ap.add_argument("--window", type=float, default=3.0)
a = ap.parse_args()

print(f"host {os.uname().nodename}  load average {os.getloadavg()}", flush=True)
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
xi0, d, n = meta["xi0"], a.cutoff + 1, len(Delta)
t0 = time.time()
sec = kc.KPOSector(J, Delta, 1.0, xi0, 1.0, d, "goto")
N = sec.idx.size
print(f"[1] n {n}  d {d}  sector dim {N}  nnz hop {sec.Hhop.nnz}  pump {sec.Hp.nnz}  "
      f"build {time.time() - t0:.1f} s", flush=True)

rng = np.random.default_rng(0)
v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
v /= np.linalg.norm(v)
p_test = 3.0
Hc = (sps.diags(sec.diag) + sec.Hhop + p_test * sec.Hp).tocsr()     # real, one matrix for this p
Hc.sort_indices()


def mv_real(x, p=None):
    """H(p_test) x via the (N, 2) real view; p is ignored (timing only)."""
    return np.ascontiguousarray(Hc @ x.view(np.float64).reshape(-1, 2)).reshape(-1).view(np.complex128)


def per_call(f, reps):
    f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


diff = np.abs(mv_real(v) - sec.matvec(v, p_test)).max()
t_cur = per_call(lambda: sec.matvec(v, p_test), 20)
t_real = per_call(lambda: mv_real(v), 20)
print(f"[2] matvec current {t_cur * 1e3:.1f} ms   real-view {t_real * 1e3:.1f} ms   "
      f"(x{t_cur / t_real:.1f})   max diff {diff:.1e}", flush=True)

V = np.empty((25, N), dtype=np.complex128)
st = dict(matvecs=0)
t_kc = per_call(lambda: kc._krylov_exp(sec.matvec, v, p_test, 0.05, 24, V, st), 3)
t_kr = per_call(lambda: kc._krylov_exp(mv_real, v, p_test, 0.05, 24, V, st), 3)
print(f"[3] one Lanczos exponential (m 24): current {t_kc:.2f} s   real-view {t_kr:.2f} s   "
      f"(matvecs are {24 * t_cur / t_kc * 100:.0f}% of it with the current matvec)", flush=True)

pfun = kc.linear_schedule(a.pf, a.T)
W = a.window
tol_w = a.tol * W / a.T                     # same per-step tolerance tol * dt / T as in a full run

t0 = time.time()
_, st_e = kc.lanczos_propagate(sec.matvec, sec.vacuum(), W, pfun, tol=tol_w, dt_max=a.dtmax)
w_e = time.time() - t0
print(f"[4] early window [0, {W:g}]: {st_e['steps']} steps (mean dt {W / st_e['steps']:.3f}), "
      f"rejects {st_e['rejects']}, {w_e:.0f} s", flush=True)

ts = a.T - W
p_s = pfun(ts)


def cat(alpha):
    k = np.arange(d)
    c = np.exp(-0.5 * alpha * alpha + k * math.log(max(alpha, 1e-300)) - np.array([math.lgamma(x + 1) for x in k]))
    c = c * (1 + (-1.0) ** k) if alpha > 0 else (k == 0).astype(float)
    return c / np.linalg.norm(c)


alphas = np.sqrt(np.maximum(p_s - Delta, 0.0))
full = functools.reduce(np.kron, [cat(x) for x in alphas])
psi_s = full[sec.idx].astype(np.complex128)
psi_s /= np.linalg.norm(psi_s)
nbar = [float((cat(x) ** 2 * np.arange(d)).sum()) for x in alphas]
t0 = time.time()
_, st_l = kc.lanczos_propagate(sec.matvec, psi_s, W, lambda t: pfun(ts + t), tol=tol_w, dt_max=a.dtmax)
w_l = time.time() - t0
print(f"[4] late window [{ts:g}, {a.T:g}] from cat states (nbar per mode {np.round(nbar, 1).tolist()}): "
      f"{st_l['steps']} steps (mean dt {W / st_l['steps']:.3f}), rejects {st_l['rejects']}, {w_l:.0f} s",
      flush=True)

steps_full = a.T / W * 0.5 * (st_e["steps"] + st_l["steps"])
sec_per_step = (w_e + w_l) / (st_e["steps"] + st_l["steps"])
print(f"[5] projected per task at this load: ~{steps_full:.0f} steps x {sec_per_step:.2f} s/step = "
      f"{steps_full * sec_per_step / 3600:.1f} h   (with the real-view matvec roughly "
      f"{steps_full * sec_per_step * t_kr / t_kc / 3600:.1f} h)", flush=True)
