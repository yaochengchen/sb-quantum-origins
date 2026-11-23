"""Self-tests of the GPU engine (kpo_gpu) against the CPU engine (kpo_common), about a minute on one GPU.

Prints PASS/FAIL per check, exits non-zero on any failure (gpu/run_gpu.sh then starts nothing).
  1 H psi: GPU kernel == KPOSector.matvec for a batch of different instances (n=3, d=6, both families, h=0.7;
    n=4, d=8, goto, h=1)
  2 one Lanczos exponential (m 24) of a batch == kpo_common._krylov_exp instance by instance
  3 propagation on a fixed step sequence (dt 0.05, T 60, Goto ramp rate), batch of 3 == CPU engine per instance
  4 adaptive propagation of a batch of 2 (linear p, dt_max 0.05) vs solve_ivp DOP853 on the dense H
  5 physics: two-spin ferromagnet of Goto 2016 App. D (cutoff 20, T 500) ends in (up,up)/(down,down), P > 0.95
  6 [INFO] time per step for a production-size batch (13 instances, n = 4, cutoff 18)
"""
import os
import sys
import time

import cupy as cp
import numpy as np
from scipy.integrate import solve_ivp

import kpo_common as kc
import kpo_gpu as kg

fails = []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


def dense_H(J, Delta, K, xi0, h, d, p):
    """full-space dense H (Goto family; pass Delta + K h for the weyl family), as in selftest.py"""
    n = len(Delta)
    a = np.diag(np.sqrt(np.arange(1, d)), 1)
    I = np.eye(d)

    def op(o, i):
        m = np.array([[1.0]])
        for k in range(n):
            m = np.kron(m, o if k == i else I)
        return m

    H = np.zeros((d ** n, d ** n))
    for i in range(n):
        H += Delta[i] * op(a.T @ a, i) + 0.5 * K * h * op(a.T @ a.T @ a @ a, i) - 0.5 * p * op(a @ a + a.T @ a.T, i)
        for j in range(n):
            if i != j:
                H -= xi0 * J[i, j] * op(a.T, i) @ op(a, j)
    return H


def rand_J(rng, B, n):
    J = rng.uniform(-1, 1, (B, n, n))
    J = np.triu(J, 1)
    return J + J.transpose(0, 2, 1)


rng = np.random.default_rng(7)
dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
print(f"device: {dev.decode() if isinstance(dev, bytes) else dev}   cupy {cp.__version__}", flush=True)

# 1 H psi
for (n, d, fam, h) in ((3, 6, "goto", 0.7), (3, 6, "weyl", 0.7), (4, 8, "goto", 1.0)):
    B = 3
    Js = rand_J(rng, B, n)
    Ds = np.stack([kc.goto_detuning(J, 0.5) for J in Js])
    g = kg.SectorBatchGPU(Js, Ds, 1.0, 0.5, h, d, fam)
    X = rng.standard_normal((B, g.N)) + 1j * rng.standard_normal((B, g.N))
    Yg = cp.asnumpy(g.matvec(cp.asarray(X), 0.9))
    e1 = max(np.abs(Yg[b] - kc.KPOSector(Js[b], Ds[b], 1.0, 0.5, h, d, fam).matvec(X[b], 0.9)).max()
             for b in range(B))
    check(f"GPU H psi == KPOSector.matvec (n={n}, d={d}, {fam}, h={h})", e1 < 1e-12, f"max {e1:.1e}")

# 2 one Lanczos exponential
n, d, B = 4, 8, 3
Js = rand_J(rng, B, n)
Ds = np.stack([kc.goto_detuning(J, 0.5) for J in Js])
g = kg.SectorBatchGPU(Js, Ds, 1.0, 0.5, 1.0, d)
secs = [kc.KPOSector(Js[b], Ds[b], 1.0, 0.5, 1.0, d) for b in range(B)]
X = rng.standard_normal((B, g.N)) + 1j * rng.standard_normal((B, g.N))
X /= np.linalg.norm(X, axis=1, keepdims=True)
Xg, Yg = cp.asarray(X), cp.empty((B, g.N), dtype=cp.complex128)
errg, kg_ = g.krylov_exp(Xg, 1.3, 0.05, Yg)
Yg = cp.asnumpy(Yg)
e2, kmatch = 0.0, True
for b in range(B):
    Vc = np.empty((25, g.N), dtype=np.complex128)
    yc, ec, kcpu = kc._krylov_exp(secs[b].matvec, X[b], 1.3, 0.05, 24, Vc, dict(matvecs=0))
    e2 = max(e2, np.abs(Yg[b] - yc).max())
    kmatch &= int(kg_[b]) == kcpu
check("GPU Lanczos exponential == kpo_common._krylov_exp (n=4, d=8, batch 3)", e2 < 1e-11 and kmatch,
      f"max {e2:.1e}  k {kg_.tolist()}")

# 3 fixed step sequence, batch of 3 vs CPU per instance
pf3 = kc.linear_schedule(7.0 * 60 / 700, 60.0)
kw = dict(tol=1.0, dt_init=0.05, dt_max=0.05)
psg, stg = g.propagate(g.vacuum(), 60.0, pf3, **kw)
psg = cp.asnumpy(psg)
e3, same = 0.0, True
for b in range(B):
    psc, stc = kc.lanczos_propagate(secs[b].matvec, secs[b].vacuum(), 60.0, pf3, **kw)
    e3 = max(e3, np.abs(psg[b] - psc).max())
    same &= stc["steps"] == stg["steps"] and stc["rejects"] == 0
check("GPU propagation == CPU engine, fixed dt 0.05 (n=4, d=8, batch 3)", same and stg["rejects"] == 0 and e3 < 1e-9,
      f"max {e3:.1e}  steps {stg['steps']}")

# 4 adaptive, batch of 2, vs DOP853 on the dense H
n, d, T4, pf4 = 2, 8, 20.0, 3.0
J4 = np.array([[[0.0, -0.6], [-0.6, 0.0]], [[0.0, 0.8], [0.8, 0.0]]])
D4 = np.stack([kc.goto_detuning(J, 0.5) for J in J4])
g4 = kg.SectorBatchGPU(J4, D4, 1.0, 0.5, 1.0, d)
ps4, st4 = g4.propagate(g4.vacuum(), T4, kc.linear_schedule(pf4, T4), tol=1e-10, dt_max=0.05)
ps4 = cp.asnumpy(ps4)
e4 = 0.0
for b in range(2):
    H0 = dense_H(J4[b], D4[b], 1.0, 0.5, 1.0, d, 0.0)
    Hp = dense_H(J4[b], D4[b], 1.0, 0.5, 1.0, d, 1.0) - H0
    psi0 = np.zeros(d ** n, dtype=np.complex128)
    psi0[0] = 1.0
    sol = solve_ivp(lambda t, y: -1j * ((H0 + pf4 * t / T4 * Hp) @ y), (0, T4), psi0, method="DOP853",
                    rtol=1e-11, atol=1e-12, t_eval=[T4])
    e4 = max(e4, np.abs(g4.embed(ps4[b]) - sol.y[:, -1]).max())
check("GPU adaptive propagation vs DOP853 (n=2, d=8, batch 2, dt_max 0.05)", e4 < 1e-6,
      f"max {e4:.1e}  steps {st4['steps']}")

# 5 two-spin ferromagnet, Goto 2016 App. D
J7 = np.array([[0.0, 1.0], [1.0, 0.0]])
D7 = kc.goto_detuning(J7, 0.5)
g7 = kg.SectorBatchGPU(J7[None], D7[None], 1.0, 0.5, 1.0, 21)
ps7, st7 = g7.propagate(g7.vacuum(), 500.0, kc.linear_schedule(5.0, 500.0), tol=1e-8)
psi7 = g7.embed(cp.asnumpy(ps7)[0])
P7 = kc.pattern_probabilities(kc.KPOQuantum(J7, D7, 1.0, 0.5, 1.0, 21), psi7, kc.theta_matrix(21))
p7, _ = kc.summarize_patterns(P7, kc.ising_energies(J7))
check("GPU two-spin ferromagnet success > 0.95", p7 > 0.95, f"P_succ={p7:.5f} steps={st7['steps']}")

# 6 timing, production size
inst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instances", "G4.npz")
if os.path.exists(inst):
    z = np.load(inst)
    J6, D6 = z["J"][:13].astype(np.float64), z["Delta"][:13].astype(np.float64)
else:
    J6 = rand_J(rng, 13, 4)
    D6 = np.stack([kc.goto_detuning(J, 0.5) for J in J6])
g6 = kg.SectorBatchGPU(J6, D6, 1.0, 0.5, 1.0, 19)
kw6 = dict(tol=1.0, dt_init=0.05, dt_max=0.05)
g6.propagate(g6.vacuum(), 0.05, lambda t: 3.5, **kw6)            # compile + warm-up
cp.cuda.Device().synchronize()
t0 = time.time()
_, st6 = g6.propagate(g6.vacuum(), 1.0, lambda t: 3.5, **kw6)
cp.cuda.Device().synchronize()
ms = (time.time() - t0) / st6["steps"] * 1e3
print(f"[INFO] 13 instances, n 4, cutoff 18 (sector dim {g6.N}): {ms:.1f} ms per step "
      f"-> {ms * 1e4 / 6e4:.1f} min per 10 000 steps", flush=True)

print("SELFTEST_GPU_OK" if not fails else f"SELFTEST_GPU_FAILED: {fails}", flush=True)
sys.exit(1 if fails else 0)
