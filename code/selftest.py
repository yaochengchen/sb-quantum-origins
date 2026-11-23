"""Self-tests run by the prep job before anything is submitted (one to a few minutes on one core).

Prints PASS/FAIL per check and exits non-zero on any failure, so the dependent Slurm jobs do not start.
  1 theta_matrix: diagonal 1/2, parity zeros, <1|Theta|0> = 1/sqrt(2 pi) (Goto 2016 App. B formula)
  2 matrix-free H equals the dense Kronecker-product H (n=2, d=5) and is Hermitian (n=3, d=6);
    2b the CSR even-sector engine equals the tensor engine (both families); weyl family = Delta + K h
  4b CSR-engine propagation equals tensor-engine propagation (n=4, d=8) on an identical fixed step sequence
  4c [INFO only] Lanczos vs DOP853 at Goto's ramp rate (T 700, pf 7) for dt_max 0.05 / 0.1 / 0.25 / 1.0
  6b weyl family: L2w and L2g integrate identical equations
  3 Lanczos propagator, constant p: matches scipy.linalg.expm
  4 Lanczos propagator, linear p(t): matches solve_ivp (DOP853, rtol 1e-11) on the dense H
  5 parity leak from the vacuum ~ 0; pattern probabilities sum to |psi|^2
  6 classical: Weyl equations at h -> 0 reproduce Goto's normal-ordered model trajectory by trajectory
  7 physics: two-spin ferromagnet of Goto 2016 App. D (Delta = xi0 = 0.5, p: 0 -> 5 over T = 500,
    cutoff 20) must end in (up,up)/(down,down) with probability > 0.95
"""
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import kpo_common as kc

fails = []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        fails.append(name)


def dense_H(J, Delta, K, xi0, h, d, p):
    n = len(Delta)
    a = np.diag(np.sqrt(np.arange(1, d)), 1)
    I = np.eye(d)

    def op(o, i):
        m = np.array([[1.0]])
        for k in range(n):
            m = np.kron(m, o if k == i else I)
        return m

    nn = a.T @ a
    H = np.zeros((d ** n, d ** n))
    for i in range(n):
        H += Delta[i] * op(nn, i) + 0.5 * K * h * op(a.T @ a.T @ a @ a, i) - 0.5 * p * op(a @ a + a.T @ a.T, i)
        for j in range(n):
            if i != j:
                H -= xi0 * J[i, j] * op(a.T, i) @ op(a, j)
    return H


rng = np.random.default_rng(1)

# 1 theta
d = 19
Th = kc.theta_matrix(d)
dg = np.abs(np.diag(Th) - 0.5).max()
mm, nn_ = np.indices(Th.shape)
par = np.abs(Th[((mm - nn_) % 2 == 0) & (mm != nn_)]).max()
check("theta diag = 1/2", dg < 1e-10, f"max dev {dg:.1e}")
check("theta parity zeros", par < 1e-10, f"max {par:.1e}")
check("theta <1|0> = 1/sqrt(2pi)", abs(Th[1, 0] - 1 / math.sqrt(2 * math.pi)) < 1e-10, f"{Th[1, 0]:.12f}")
ev = np.linalg.eigvalsh(Th)
check("theta eigenvalues in [0,1]", ev.min() > -1e-10 and ev.max() < 1 + 1e-10, f"[{ev.min():.2e},{ev.max():.6f}]")

# 2 matvec vs dense, hermiticity
J2 = np.array([[0.0, 0.7], [0.7, 0.0]])
D2 = kc.goto_detuning(J2, 0.5)
m2 = kc.KPOQuantum(J2, D2, 1.0, 0.5, 1.0, 5)
H2 = dense_H(J2, D2, 1.0, 0.5, 1.0, 5, 1.3)
v = rng.standard_normal(25) + 1j * rng.standard_normal(25)
err = np.abs(m2.matvec(v, 1.3) - H2 @ v).max()
check("matvec == dense H (n=2,d=5)", err < 1e-12, f"max {err:.1e}")
J3 = rng.uniform(-1, 1, (3, 3)); J3 = np.triu(J3, 1); J3 = J3 + J3.T
m3 = kc.KPOQuantum(J3, kc.goto_detuning(J3, 0.5), 1.0, 0.5, 0.7, 6)
u = rng.standard_normal(216) + 1j * rng.standard_normal(216)
w = rng.standard_normal(216) + 1j * rng.standard_normal(216)
herm = abs(np.vdot(u, m3.matvec(w, 0.9)) - np.conj(np.vdot(w, m3.matvec(u, 0.9))))
check("H Hermitian (n=3,d=6)", herm < 1e-10, f"{herm:.1e}")

# 2b CSR even-sector engine == tensor engine (both families), on the even sector
for fam in ("goto", "weyl"):
    mt = kc.KPOQuantum(J3, kc.goto_detuning(J3, 0.5), 1.0, 0.5, 0.7, 6, fam)
    ms = kc.KPOSector(J3, kc.goto_detuning(J3, 0.5), 1.0, 0.5, 0.7, 6, fam)
    vs = rng.standard_normal(ms.idx.size) + 1j * rng.standard_normal(ms.idx.size)
    e2b = np.abs(ms.embed(ms.matvec(vs, 0.9)) - mt.matvec(ms.embed(vs), 0.9)).max()
    check(f"CSR sector matvec == tensor matvec ({fam})", e2b < 1e-12, f"max {e2b:.1e}")
mw = kc.KPOQuantum(J2, D2, 1.0, 0.5, 0.7, 5, "weyl")
Hw = dense_H(J2, D2 + 0.7, 1.0, 0.5, 0.7, 5, 1.3)
ew = np.abs(mw.matvec(v, 1.3) - Hw @ v).max()
check("weyl family == dense H with Delta + K h", ew < 1e-12, f"max {ew:.1e}")

# 3 constant p vs expm
d = 8
J2b = np.array([[0.0, -0.6], [-0.6, 0.0]])
D2b = kc.goto_detuning(J2b, 0.5)
mb = kc.KPOQuantum(J2b, D2b, 1.0, 0.5, 1.0, d)
Hb = dense_H(J2b, D2b, 1.0, 0.5, 1.0, d, 1.1)
psi0 = mb.vacuum()
psiL, _ = kc.lanczos_propagate(mb.matvec, psi0, 5.0, lambda t: 1.1, tol=1e-10)
psiE = expm(-1j * 5.0 * Hb) @ psi0
e3 = np.abs(psiL - psiE).max()
check("Lanczos const-p vs expm", e3 < 1e-7, f"max {e3:.1e}")

# 4 linear p(t) vs DOP853
T4, pf4 = 20.0, 3.0
H0 = dense_H(J2b, D2b, 1.0, 0.5, 1.0, d, 0.0)
Hp = dense_H(J2b, D2b, 1.0, 0.5, 1.0, d, 1.0) - H0
sol = solve_ivp(lambda t, y: -1j * ((H0 + pf4 * t / T4 * Hp) @ y), (0, T4), psi0, method="DOP853",
                rtol=1e-11, atol=1e-12)
psiL4, st4 = kc.lanczos_propagate(mb.matvec, psi0, T4, kc.linear_schedule(pf4, T4), tol=1e-10, dt_max=0.05)
e4 = np.abs(psiL4 - sol.y[:, -1]).max()
check("Lanczos linear-p vs DOP853 (dt_max 0.05)", e4 < 1e-6, f"max {e4:.1e} steps {st4['steps']}")
for dtm in (0.25, 1.0):                                  # informational: error at production-like step caps
    psiLx, stx = kc.lanczos_propagate(mb.matvec, psi0, T4, kc.linear_schedule(pf4, T4), tol=1e-10, dt_max=dtm)
    print(f"[INFO] Lanczos linear-p vs DOP853 (dt_max {dtm}): max {np.abs(psiLx - sol.y[:, -1]).max():.1e} "
          f"steps {stx['steps']}")

# 4c informational: the same comparison at Goto's ramp rate (p: 0 -> 7 over T = 700, as in the plan), to choose
#    the plan column dtmax. Dense H (n = 2, d = 8): this checks only the time stepping, not the engines.
T4c, pf4c = 700.0, 7.0
sol4c = solve_ivp(lambda t, y: -1j * ((H0 + pf4c * t / T4c * Hp) @ y), (0, T4c), psi0, method="DOP853",
                  rtol=1e-11, atol=1e-12, t_eval=[T4c])


def dense_matvec(v, p):
    return H0 @ v + p * (Hp @ v)


for dtm in (0.05, 0.1, 0.25, 1.0):
    psiLx, stx = kc.lanczos_propagate(dense_matvec, psi0, T4c, kc.linear_schedule(pf4c, T4c), tol=1e-10,
                                      dt_max=dtm)
    print(f"[INFO] Goto ramp (T 700, pf 7) vs DOP853 (dt_max {dtm}): "
          f"max {np.abs(psiLx - sol4c.y[:, -1]).max():.1e} steps {stx['steps']}")

# 4b CSR engine propagation == tensor engine propagation (n=4, d=8, Goto ramp rate, T = 60).
#    Both engines are forced onto the same fixed step sequence: dt_init = dt_max = 0.05 and a tolerance so loose
#    that no step is rejected or shortened (the Krylov error at dt 0.05, m 24 is at round-off level), so they
#    differ only by matvec round-off. With adaptive steps (first version of this test) the step controller
#    reacts to round-off-level differences in the Krylov error estimate, the two engines take different step
#    sequences, and the difference is then the Magnus truncation error (5e-6 on n013), not an engine error.
J4b = rng.uniform(-1, 1, (4, 4)); J4b = np.triu(J4b, 1); J4b = J4b + J4b.T
D4b = kc.goto_detuning(J4b, 0.5)
pf4b = kc.linear_schedule(7.0 * 60 / 700, 60.0)          # first 60/700 of the Goto ramp
mt4 = kc.KPOQuantum(J4b, D4b, 1.0, 0.5, 1.0, 8)
ms4 = kc.KPOSector(J4b, D4b, 1.0, 0.5, 1.0, 8)
kw4b = dict(tol=1.0, dt_init=0.05, dt_max=0.05)
pt4, stt4 = kc.lanczos_propagate(mt4.matvec, mt4.vacuum(), 60.0, pf4b, **kw4b)
ps4, sts4 = kc.lanczos_propagate(ms4.matvec, ms4.vacuum(), 60.0, pf4b, **kw4b)
e4b = np.abs(ms4.embed(ps4) - pt4).max()
same4b = stt4["steps"] == sts4["steps"] and stt4["rejects"] == 0 and sts4["rejects"] == 0
check("CSR engine propagation == tensor engine (n=4, d=8, fixed dt 0.05)", same4b and e4b < 1e-9,
      f"max {e4b:.1e} steps {stt4['steps']}/{sts4['steps']} rejects {stt4['rejects']}/{sts4['rejects']}")

# 5 parity, probability sum
check("parity leak ~ 0", mb.parity_leak(psiL4) < 1e-20, f"{mb.parity_leak(psiL4):.1e}")
P = kc.pattern_probabilities(mb, psiL4, kc.theta_matrix(d))
check("sum P = |psi|^2", abs(P.sum() - np.vdot(psiL4, psiL4).real) < 1e-10, f"{P.sum():.12f}")

# 6 classical: Weyl at h -> 0 == normal ordered (same initial conditions)
J4 = rng.uniform(-1, 1, (4, 4)); J4 = np.triu(J4, 1); J4 = J4 + J4.T
D4 = kc.goto_detuning(J4, 0.5)
pf = kc.linear_schedule(7.0, 100.0)
bw = kc.classical_run("L2w", J4, D4, 1.0, 0.5, 1e-9, pf, 100.0, 0.01, 64, np.random.default_rng(5), 2.6)
bg = kc.classical_run("L2g", J4, D4, 1.0, 0.5, 1e-9, pf, 100.0, 0.01, 64, np.random.default_rng(5), 2.6)
check("Weyl(h->0) == Goto classical, per trajectory", (bw == bg).mean() >= 0.95, f"{(bw == bg).mean():.3f}")
bw1 = kc.classical_run("L2w", J4, D4, 1.0, 0.5, 1.0, pf, 100.0, 0.01, 64, np.random.default_rng(5), 2.6, "weyl")
bg1 = kc.classical_run("L2g", J4, D4, 1.0, 0.5, 1.0, pf, 100.0, 0.01, 64, np.random.default_rng(5), 2.6, "weyl")
check("weyl family: L2w == L2g at h = 1 (identical equations)", (bw1 == bg1).mean() >= 0.95,
      f"{(bw1 == bg1).mean():.3f}")

# 7 two-spin ferromagnet, Goto 2016 App. D
J7 = np.array([[0.0, 1.0], [1.0, 0.0]])
m7 = kc.KPOQuantum(J7, kc.goto_detuning(J7, 0.5), 1.0, 0.5, 1.0, 21)
s7 = kc.KPOSector(J7, kc.goto_detuning(J7, 0.5), 1.0, 0.5, 1.0, 21)
psi7s, st7 = kc.lanczos_propagate(s7.matvec, s7.vacuum(), 500.0, kc.linear_schedule(5.0, 500.0), tol=1e-8)
psi7 = s7.embed(psi7s)
P7 = kc.pattern_probabilities(m7, psi7, kc.theta_matrix(21))
ps7, _ = kc.summarize_patterns(P7, kc.ising_energies(J7))
check("two-spin ferromagnet success > 0.95", ps7 > 0.95, f"P_succ={ps7:.5f} steps={st7['steps']}")

print("SELFTEST_OK" if not fails else f"SELFTEST_FAILED: {fails}")
sys.exit(1 if fails else 0)
