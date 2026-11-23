"""Shared model code for MSBQA phase 1 (KPO network: quantum L0, classical L1-L5).

Units: K = 1 is NOT assumed; every rate is passed explicitly. Goto 2016 (Sci. Rep. 6, 21686) uses
K = 1, xi0 = 0.5, Delta_i = xi0 * sum_j |J_ij| (his Eq. 7), p(t) linear 0 -> 7 over T = 700, cutoff 18.

Quantum Hamiltonian (hbar = 1), Goto 2016 Eq. 5, with an effective-hbar family h (h = 1 is Goto's model):
    H(t) = sum_i [ Delta_i n_i + (K h / 2) a_i^+2 a_i^2 - (p(t)/2) (a_i^2 + a_i^+2) ]
           - xi0 * sum_{i != j} J_ij a_i^+ a_j
(Goto writes -(xi0/2) sum_{i,j} J_ij (a_i^+ a_j + a_i a_j^+); for i != j this equals the line above.)
With z = sqrt(h) * alpha the classical flow is independent of h, so h -> 0 is the classical limit at fixed
classical trajectories; only the initial vacuum noise (Var Re z = h/4) and the final photon number
(|alpha|^2 ~ (p_f - Delta)/(K h)) change with h.

Spin readout: s_i = sign of the x quadrature of mode i (Goto 2016 App. B). Bits: s_i = 1 - 2*bit_i,
pattern index b = sum_i bit_i 2^i. The same convention is used for quantum, classical and Ising energies.
"""
import json
import math
import os

import numpy as np

# ----------------------------------------------------------------------------- Ising helpers


def all_spins(n):
    """(2^n, n) array of spins, row b has s_i = 1 - 2*((b >> i) & 1)."""
    b = np.arange(2 ** n)[:, None]
    bits = (b >> np.arange(n)[None, :]) & 1
    return (1 - 2 * bits).astype(np.int8)


def ising_energies(J):
    """E(s) = -(1/2) sum_{i,j} J_ij s_i s_j for every pattern (Goto 2016 Eq. 4)."""
    S = all_spins(J.shape[0]).astype(np.float64)
    return -0.5 * np.einsum("bi,ij,bj->b", S, J, S)


def ground_mask(E, rtol=1e-12):
    e0 = E.min()
    return E <= e0 + rtol * max(1.0, abs(e0))


def pattern_index(spins):
    """spins: (..., n) array of +-1 -> pattern index."""
    bits = (np.asarray(spins) < 0).astype(np.int64)
    return (bits << np.arange(bits.shape[-1])).sum(-1)


def goto_detuning(J, xi0):
    """Goto 2016 Eq. 7: Delta_i = xi0 * sum_j |J_ij| (makes M positive semidefinite at p = 0)."""
    return xi0 * np.abs(J).sum(1)


# ----------------------------------------------------------------------------- instances


def load_instance(inst_dir, set_name, idx):
    d = np.load(os.path.join(inst_dir, f"{set_name}.npz"))
    J = d["J"][idx].astype(np.float64)
    Delta = d["Delta"][idx].astype(np.float64)
    meta = json.loads(str(d["meta"]))
    return J, Delta, meta


# ----------------------------------------------------------------------------- quantum model
#
# Two model families (plan column 'family'):
#   goto : H exactly as Goto 2016 (detuning Delta_i). Its Weyl-ordered classical flow has the detuning
#          Delta_i - K h, i.e. at h = 1 a shift of -K that is larger than Delta_i itself (smoke test 10-29:
#          L2w far below L2g). Used for the G0 calibration only.
#   weyl : detuning Delta_i + K h in the quantum H (as in Jiapei's P0-P1.5 model). Its Weyl flow is then
#          exactly Goto's classical equations for every h, so L1 (TWA) and L2 differ only in the initial noise.
#          Used for the ablation chain and the hbar scan. Both families coincide for h -> 0.


def quantum_detuning(Delta, K, h, family):
    if family == "goto":
        return np.asarray(Delta, dtype=np.float64)
    if family == "weyl":
        return np.asarray(Delta, dtype=np.float64) + K * h
    raise ValueError(f"unknown family {family}")


class KPOQuantum:
    """Matrix-free KPO network in the full truncated Fock space, psi stored as a tensor of shape (d,)*n.

    Every ladder operator acts on one axis by slicing, so a matvec costs O(n^2 d^n) with no stored matrix.
    The vacuum lies in the even total-photon-number sector, which H conserves; we do not restrict to it
    (a factor 2 in memory and time), which keeps the code simple and lets the self-test check parity.
    """

    def __init__(self, J, Delta, K, xi0, h, d, family="goto"):
        self.n = n = len(Delta)
        self.d = d
        self.J = np.asarray(J, dtype=np.float64)
        self.xi0 = float(xi0)
        self.shape = (d,) * n
        self.D = d ** n
        det = quantum_detuning(Delta, K, h, family)
        k = np.arange(d, dtype=np.float64)
        diag = np.zeros(self.shape)
        for i in range(n):
            s = [1] * n
            s[i] = d
            diag += (det[i] * k + 0.5 * K * h * k * (k - 1)).reshape(s)
        self.diag = diag.reshape(-1)
        self.sq1 = np.sqrt(k[1:])                 # a|k> = sqrt(k)|k-1>
        self.sq2 = np.sqrt(k[1:-1] * k[2:])       # a^2|k> = sqrt(k(k-1))|k-2>
        self._views = [(d ** i, d ** (n - i - 1)) for i in range(n)]

    # --- single-axis ladder operators (out is overwritten) ---------------------------------
    def _v(self, x, i):
        L, R = self._views[i]
        return x.reshape(L, self.d, R)

    def a(self, x, i, out):
        X, O = self._v(x, i), self._v(out, i)
        O[:, :-1, :] = self.sq1[None, :, None] * X[:, 1:, :]
        O[:, -1, :] = 0.0
        return out

    def adag(self, x, i, out):
        X, O = self._v(x, i), self._v(out, i)
        O[:, 1:, :] = self.sq1[None, :, None] * X[:, :-1, :]
        O[:, 0, :] = 0.0
        return out

    def a2_plus_adag2(self, x, i, out):
        X, O = self._v(x, i), self._v(out, i)
        O[...] = 0.0
        O[:, :-2, :] += self.sq2[None, :, None] * X[:, 2:, :]
        O[:, 2:, :] += self.sq2[None, :, None] * X[:, :-2, :]
        return out

    def apply_dense_axis(self, M, x, i):
        """(M acting on mode i) x for a dense d x d matrix M."""
        X = self._v(x, i)
        return np.einsum("kj,ljr->lkr", M, X, optimize=True).reshape(-1)

    # --- H(t) psi --------------------------------------------------------------------------
    def matvec(self, x, p):
        n = self.n
        out = self.diag * x
        tmp = np.empty_like(x)
        for i in range(n):                                   # pump: -(p/2) (a^2 + a^+2)
            out -= (0.5 * p) * self.a2_plus_adag2(x, i, tmp)
        phi = [self.a(x, j, np.empty_like(x)) for j in range(n)]
        chi = np.empty_like(x)
        for i in range(n):                                   # hopping: -xi0 sum_{j != i} J_ij a_i^+ a_j
            chi[...] = 0.0
            for j in range(n):
                if j != i and self.J[i, j] != 0.0:
                    chi += self.J[i, j] * phi[j]
            out -= self.xi0 * self.adag(chi, i, tmp)
        return out

    def vacuum(self):
        psi = np.zeros(self.D, dtype=np.complex128)
        psi[0] = 1.0
        return psi

    def mode_populations(self, psi):
        """(n, d) array: marginal photon-number distribution of every mode."""
        P = (np.abs(psi) ** 2).reshape(self.shape)
        out = np.empty((self.n, self.d))
        for i in range(self.n):
            axes = tuple(a for a in range(self.n) if a != i)
            out[i] = P.sum(axis=axes)
        return out

    def parity_leak(self, psi):
        """Total probability in the odd total-photon-number sector (must stay ~0 from the vacuum)."""
        k = np.indices(self.shape).sum(0).reshape(-1)
        return float((np.abs(psi[(k % 2) == 1]) ** 2).sum())


class KPOSector:
    """Same H as KPOQuantum, stored as scipy CSR matrices restricted to the even total-photon-number sector
    (the sector of the vacuum; H conserves the parity). Faster than the tensor engine for n <= 4-5.
    matvec(v, p) = diag*v + Hhop v + p Hpump v on sector vectors; embed() maps back to the full space.
    Memory grows like n^2 d^n nonzeros, so use the tensor engine for n >= 6.
    """

    def __init__(self, J, Delta, K, xi0, h, d, family="goto"):
        import scipy.sparse as sps
        self.n = n = len(Delta)
        self.d = d
        self.D = d ** n
        det = quantum_detuning(Delta, K, h, family)
        occ = np.indices((d,) * n).reshape(n, -1)
        self.idx = np.flatnonzero(occ.sum(0) % 2 == 0)
        diag = np.zeros(self.D)
        for i in range(n):
            k = occ[i].astype(np.float64)
            diag += det[i] * k + 0.5 * K * h * k * (k - 1)
        self.diag = diag[self.idx]
        a = sps.diags(np.sqrt(np.arange(1, d, dtype=np.float64)), 1, shape=(d, d), format="csr")
        ad = a.T.tocsr()
        I = sps.identity(d, format="csr")

        def op(o, i):
            m = sps.identity(1, format="csr")
            for k_ in range(n):
                m = sps.kron(m, o if k_ == i else I, format="csr")
            return m

        A = [op(a, i) for i in range(n)]
        Ad = [op(ad, i) for i in range(n)]
        hop = sps.csr_matrix((self.D, self.D))
        for i in range(n):
            for j in range(n):
                if i != j and J[i, j] != 0.0:
                    hop = hop - xi0 * J[i, j] * (Ad[i] @ A[j])
        pump = sps.csr_matrix((self.D, self.D))
        for i in range(n):
            pump = pump - 0.5 * (A[i] @ A[i] + Ad[i] @ Ad[i])
        ix = self.idx
        self.Hhop = hop.tocsr()[ix][:, ix].tocsr()
        self.Hp = pump.tocsr()[ix][:, ix].tocsr()
        self.Hhop.sort_indices()
        self.Hp.sort_indices()

    def matvec(self, v, p):
        # The CSR matrices are real. A real CSR matrix times a complex vector makes scipy convert the whole
        # matrix to complex on every call; acting on the (N, 2) real view (columns Re, Im) avoids that copy.
        v = np.ascontiguousarray(v, dtype=np.complex128)
        X = v.view(np.float64).reshape(-1, 2)
        Y = self.Hhop @ X
        Y += p * (self.Hp @ X)
        return self.diag * v + np.ascontiguousarray(Y).reshape(-1).view(np.complex128)

    def vacuum(self):
        v = np.zeros(self.idx.size, dtype=np.complex128)
        v[0] = 1.0                                  # idx[0] == 0 is the vacuum
        return v

    def embed(self, v):
        full = np.zeros(self.D, dtype=np.complex128)
        full[self.idx] = v
        return full


def theta_matrix(d, npts=40001):
    """<m| Theta(x_hat) |n>, x_hat = (a + a^+)/sqrt(2): probability operator for 'x quadrature > 0'.

    Hermite functions on a grid, Simpson rule on [0, xmax]. Diagonal = 1/2 exactly, entries with m - n
    even and m != n vanish by parity (checked in selftest.py). Same quantity as Goto 2016 App. B, I^(+).
    """
    xmax = math.sqrt(2 * d + 1) + 10.0
    x = np.linspace(0.0, xmax, npts)
    phi = np.empty((d, npts))
    phi[0] = math.pi ** -0.25 * np.exp(-0.5 * x * x)
    if d > 1:
        phi[1] = math.sqrt(2.0) * x * phi[0]
    for k in range(1, d - 1):
        phi[k + 1] = math.sqrt(2.0 / (k + 1)) * x * phi[k] - math.sqrt(k / (k + 1)) * phi[k - 1]
    w = np.ones(npts)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= (x[1] - x[0]) / 3.0
    T = (phi * w) @ phi.T
    return 0.5 * (T + T.T)


def pattern_probabilities(model, psi, Theta):
    """P(b) = <psi| prod_i Pi_{s_i}^{(i)} |psi>, Pi_+ = Theta, Pi_- = 1 - Theta. Depth-first, O(n D) memory."""
    n = model.n
    P = np.zeros(2 ** n)

    def rec(v, i, b):
        if i == n:
            P[b] = np.vdot(psi, v).real
            return
        vp = model.apply_dense_axis(Theta, v, i)
        rec(vp, i + 1, b)                      # s_i = +1 -> bit 0
        rec(v - vp, i + 1, b | (1 << i))       # s_i = -1 -> bit 1

    rec(psi, 0, 0)
    return P


# ----------------------------------------------------------------------------- time evolution


def _krylov_exp(matvec, psi, p, tau, m, V, st):
    """exp(-i tau H(p)) psi by Lanczos with full reorthogonalisation.
    Returns (new psi, error estimate beta_k |[exp(-i T_k tau) e_1]_k|, Krylov dimension k)."""
    nrm = np.linalg.norm(psi)
    V[0] = psi / nrm
    alpha = np.zeros(m)
    beta = np.zeros(m)
    k = m
    for j in range(m):
        w = matvec(V[j], p)
        st["matvecs"] += 1
        alpha[j] = np.vdot(V[j], w).real
        w -= alpha[j] * V[j]
        if j > 0:
            w -= beta[j - 1] * V[j - 1]
        c = np.conj(V[: j + 1] @ np.conj(w))            # full reorthogonalisation (c_l = <V_l|w>)
        w -= c @ V[: j + 1]
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-12 * max(1.0, abs(alpha[j])):
            k = j + 1
            beta[j] = 0.0
            break
        V[j + 1] = w / beta[j]
    Tm = np.diag(alpha[:k]) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
    ev, U = np.linalg.eigh(Tm)
    cvec = U @ (np.exp(-1j * ev * tau) * U[0])
    err = beta[k - 1] * abs(cvec[k - 1])
    return nrm * (cvec @ V[:k]), err, k


# fourth-order commutator-free Magnus (Alvermann & Fehske, J. Comput. Phys. 230, 5930 (2011), CFET4:2):
#   U(t+dt, t) ~= exp(-i dt (a1 H(t1) + a2 H(t2))) exp(-i dt (a2 H(t1) + a1 H(t2))),   right factor first,
#   t1,2 = t + (1/2 -+ sqrt(3)/6) dt,  a1 = (3 - 2 sqrt 3)/12,  a2 = (3 + 2 sqrt 3)/12,  a1 + a2 = 1/2.
# With H(t) = H0 + p(t) Hp every factor is exp(-i (dt/2) (H0 + p_eff Hp)).
_CF_C1, _CF_C2 = 0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6
_CF_A1, _CF_A2 = (3 - 2 * math.sqrt(3)) / 12, (3 + 2 * math.sqrt(3)) / 12


def lanczos_propagate(matvec, psi, T, pfun, tol=1e-8, m=24, dt_init=0.02, dt_max=1.0, dt_min=1e-7):
    """Adaptive short-iterative Lanczos for i d psi/dt = H(t) psi, H(t) = matvec(., p(t)), H linear in p.

    Time dependence: fourth-order commutator-free Magnus step (CFET4:2, see above), i.e. two Lanczos
    exponentials of length dt/2 with p_A = 2(a2 p1 + a1 p2) (applied first) and p_B = 2(a1 p1 + a2 p2).
    The (second-order) midpoint rule of the first version failed the DOP853 self-test (6.4e-5 at dt 0.05).
    Krylov error estimates of the two half steps are added and required to be <= tol * dt / T. The Magnus
    truncation error is not part of the estimate; it is checked by selftest.py and the dt_max plan line.
    Returns psi(T) and a stats dict.
    """
    N = psi.size
    V = np.empty((m + 1, N), dtype=np.complex128)
    t, dt = 0.0, dt_init
    st = dict(steps=0, rejects=0, matvecs=0, err_sum=0.0)
    while t < T * (1 - 1e-14):
        dt = min(dt, dt_max, T - t)
        while True:
            p1, p2 = pfun(t + _CF_C1 * dt), pfun(t + _CF_C2 * dt)
            pA = 2.0 * (_CF_A2 * p1 + _CF_A1 * p2)
            pB = 2.0 * (_CF_A1 * p1 + _CF_A2 * p2)
            psi1, e1, k1 = _krylov_exp(matvec, psi, pA, 0.5 * dt, m, V, st)
            psi2, e2, k2 = _krylov_exp(matvec, psi1, pB, 0.5 * dt, m, V, st)
            err, k = e1 + e2, min(k1, k2)
            tol_step = tol * dt / T
            if err <= tol_step or dt <= dt_min:
                break
            st["rejects"] += 1
            dt *= max(0.1, 0.9 * (tol_step / err) ** (1.0 / k))
        psi = psi2
        t += dt
        st["steps"] += 1
        st["err_sum"] += err
        grow = 2.0 if err == 0.0 else min(2.0, 0.9 * (tol_step / err) ** (1.0 / k))
        dt *= max(0.5, grow)
    return psi, st


def linear_schedule(p_f, T):
    return lambda t: p_f * min(max(t / T, 0.0), 1.0)


# ----------------------------------------------------------------------------- classical layers

LAYERS = ("L1", "L2w", "L2g", "L3", "L4", "L5")
LAYER_DOC = {
    "L1":  "TWA: Weyl flow of the quantum H of the family, vacuum Wigner initial state (Var Re z = h/4)",
    "L2w": "same Weyl flow as L1, initial noise uniform +-1e-6 (only the noise differs from L1); "
           "identical equations to L2g in the 'weyl' family",
    "L2g": "Goto 2016 App. E classical model (normal-ordered mean field), initial noise uniform +-1e-6",
    "L3":  "aSB (Goto 2019 separable H, per-spin Delta_i): xdot = Delta y, ydot = -(K x^2 - p + Delta) x + xi0 J x",
    "L4":  "bSB-type: ydot = -(Delta - p) x + xi0 J x, xdot = Delta y, inelastic walls at |x| = x_w",
    "L5":  "dSB-type: as L4 but coupling xi0 J (x_w sign x)",
}


def classical_run(layer, J, Delta, K, xi0, h, pfun, T, dt, R, rng, x_wall, family="goto"):
    """Integrate R trajectories of one classical layer; return final signs as pattern indices (R,).

    L1 / L2w integrate the Weyl flow of the quantum H of the given family: detuning
    quantum_detuning(...) - K h, i.e. Delta - K h for 'goto' and exactly Delta for 'weyl'.
    L2g is always Goto's classical model (detuning Delta), the common h -> 0 limit of both families.
    """
    n = len(Delta)
    nsteps = int(math.ceil(T / dt))
    dt = T / nsteps
    Jm = np.asarray(J, dtype=np.float64)
    if layer in ("L1", "L2w", "L2g"):
        if layer == "L1":
            z = math.sqrt(h) / 2.0 * (rng.standard_normal((R, n)) + 1j * rng.standard_normal((R, n)))
            Deff = quantum_detuning(Delta, K, h, family) - K * h
        elif layer == "L2w":
            z = rng.uniform(-1e-6, 1e-6, (R, n)) + 1j * rng.uniform(-1e-6, 1e-6, (R, n))
            Deff = quantum_detuning(Delta, K, h, family) - K * h
        else:
            z = rng.uniform(-1e-6, 1e-6, (R, n)) + 1j * rng.uniform(-1e-6, 1e-6, (R, n))
            Deff = Delta.copy()

        def f(z, t):
            return -1j * ((Deff + K * (z.real ** 2 + z.imag ** 2)) * z - pfun(t) * np.conj(z) - xi0 * (z @ Jm))

        t = 0.0
        for _ in range(nsteps):                              # classical RK4
            k1 = f(z, t)
            k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(z + dt * k3, t + dt)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        x = z.real
    else:
        x = rng.uniform(-1e-6, 1e-6, (R, n))
        y = rng.uniform(-1e-6, 1e-6, (R, n))
        t = 0.0
        for _ in range(nsteps):                              # symplectic Euler, p at the new time (Goto 2019)
            t += dt
            p = pfun(t)
            if layer == "L3":
                y += dt * (-(K * x * x - p + Delta) * x + xi0 * (x @ Jm))
                x += dt * Delta * y
            else:
                drive = x if layer == "L4" else x_wall * np.sign(x)
                y += dt * (-(Delta - p) * x + xi0 * (drive @ Jm))
                x += dt * Delta * y
                hit = np.abs(x) > x_wall
                if hit.any():
                    x[hit] = x_wall * np.sign(x[hit])
                    y[hit] = 0.0
    return pattern_index(np.where(x >= 0, 1, -1))


def summarize_patterns(P, E):
    """P: probabilities over patterns (sum 1). Returns success prob. and residual energy."""
    gm = ground_mask(E)
    return float(P[gm].sum()), float(P @ E - E.min())


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
