"""GPU engine (CuPy) for the L0 quantum KPO network: a batch of instances evolved together on one GPU.

Same model and same propagator as kpo_common (KPOSector + lanczos_propagate):
    H(t) = sum_i [ Delta_i n_i + (K h / 2) n_i (n_i - 1) - (p(t)/2) (a_i^2 + a_i^+2) ] - xi0 sum_{i != j} J_ij a_i^+ a_j
on the even total-photon-number sector, fourth-order commutator-free Magnus steps (CFET4:2), Lanczos with full
reorthogonalisation (same operation order as kpo_common._krylov_exp), same adaptive step control.

Differences from the CPU engine:
  * matrix-free: one CUDA kernel computes H psi for every instance of the batch from the occupation numbers
    (no stored matrix); the Lanczos vector operations are fused into three more kernels per Krylov vector;
  * the batch shares one time grid: a step is accepted when the LARGEST Krylov error estimate of the batch is
    within tolerance, and the step size follows that largest estimate. For one instance this is exactly the
    control of kpo_common.lanczos_propagate; for a batch the steps are at most as large, i.e. at least as accurate.
Precision: complex128 / float64 throughout (on RTX 6000-class cards FP64 is slow, but this computation is
limited by memory traffic, not arithmetic).
Basis order: the same as KPOSector (sector state s = s-th even entry of the C-ordered (d,)*n tensor, mode 0 the
slowest index), so embed() and the readout of kpo_common apply unchanged.
"""
import math
import time

import cupy as cp
import numpy as np

import kpo_common as kc

NMAX = 8
_SRC = r"""
#include <cupy/complex.cuh>
#define NMAX 8
typedef complex<double> cd;

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Sum of v over the block, valid in thread 0. Every thread of the block must call it; blockDim.x % 32 == 0.
__device__ double block_sum(double v, double* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();                      // sh may still be read by the previous call
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0;
    if (wid == 0) v = warp_sum(v);
    return v;
}

// Y_b = H_b(p) X_b on the even sector, and acc[b*astride + aoff] += Re <X_b|Y_b>.
// X_b = X + xoff + b*xs, Y_b = Y + yoff + b*ys (sector vectors of length N).
extern "C" __global__ void hmv(const cd* __restrict__ X, const long long xoff, const long long xs,
                               cd* __restrict__ Y, const long long yoff, const long long ys,
                               const int* __restrict__ idx, const int* __restrict__ pos,
                               const double* __restrict__ det, const double* __restrict__ jx,
                               const double kh, const double p, const int n, const int d, const int N,
                               double* __restrict__ acc, const int astride, const int aoff)
{
    __shared__ double sh[32];
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 0.0;
    if (s < N) {
        const cd* x = X + xoff + (long long)b * xs;
        const int f = idx[s];
        int k[NMAX], st[NMAX];
        int r = f, stride = 1;
        for (int i = n - 1; i >= 0; --i) { k[i] = r % d; r /= d; st[i] = stride; stride *= d; }
        const double* de = det + (long long)b * n;
        const double* J = jx + (long long)b * n * n;
        const cd x0 = x[s];
        double dg = 0.0;
        for (int i = 0; i < n; ++i) dg += de[i] * k[i] + 0.5 * kh * k[i] * (k[i] - 1);
        cd out = dg * x0;
        const double hp = 0.5 * p;
        for (int i = 0; i < n; ++i) {             // pump -(p/2)(a_i^2 + a_i^+2)
            if (k[i] + 2 <= d - 1) out -= (hp * sqrt((double)(k[i] + 1) * (k[i] + 2))) * x[pos[f + 2 * st[i]]];
            if (k[i] >= 2)         out -= (hp * sqrt((double)k[i] * (k[i] - 1))) * x[pos[f - 2 * st[i]]];
        }
        for (int i = 0; i < n; ++i) {             // hopping -xi0 J_ij a_i^+ a_j: <k| -> |k - e_i + e_j>
            if (k[i] < 1) continue;
            const double si = sqrt((double)k[i]);
            for (int j = 0; j < n; ++j) {
                if (j == i || k[j] + 1 > d - 1) continue;
                const double c = J[i * n + j];
                if (c == 0.0) continue;
                out -= (c * si * sqrt((double)(k[j] + 1))) * x[pos[f - st[i] + st[j]]];
            }
        }
        Y[yoff + (long long)b * ys + s] = out;
        a = (conj(x0) * out).real();
    }
    a = block_sum(a, sh);
    if (threadIdx.x == 0) atomicAdd(acc + (long long)b * astride + aoff, a);
}

// V_b holds m+1 Krylov vectors (batch stride vb). w = V_{j+1}:  w -= alpha_j V_j + beta_{j-1} V_{j-1};
// then c_l = <V_l|w>, l = 0..j, accumulated into cre/cim[(b*m + j)*m + l].
extern "C" __global__ void ortho1(cd* __restrict__ V, const long long vb, const int N, const int j, const int m,
                                  const double* __restrict__ alpha, const double* __restrict__ beta,
                                  double* __restrict__ cre, double* __restrict__ cim)
{
    __shared__ double sh[32];
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    cd* Vb = V + (long long)b * vb;
    cd w(0.0, 0.0);
    if (s < N) {
        w = Vb[(long long)(j + 1) * N + s] - alpha[b * m + j] * Vb[(long long)j * N + s];
        if (j > 0) w -= beta[b * m + j - 1] * Vb[(long long)(j - 1) * N + s];
        Vb[(long long)(j + 1) * N + s] = w;
    }
    double* cr = cre + ((long long)b * m + j) * m;
    double* ci = cim + ((long long)b * m + j) * m;
    for (int l = 0; l <= j; ++l) {
        cd t(0.0, 0.0);
        if (s < N) t = conj(Vb[(long long)l * N + s]) * w;
        const double re = block_sum(t.real(), sh);
        const double im = block_sum(t.imag(), sh);
        if (threadIdx.x == 0) { atomicAdd(cr + l, re); atomicAdd(ci + l, im); }
    }
}

// w = V_{j+1}:  w -= sum_{l<=j} c_l V_l;  nrm2[b*m + j] += |w|^2
extern "C" __global__ void ortho2(cd* __restrict__ V, const long long vb, const int N, const int j, const int m,
                                  const double* __restrict__ cre, const double* __restrict__ cim,
                                  double* __restrict__ nrm2)
{
    __shared__ double sh[32];
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 0.0;
    if (s < N) {
        cd* Vb = V + (long long)b * vb;
        const double* cr = cre + ((long long)b * m + j) * m;
        const double* ci = cim + ((long long)b * m + j) * m;
        cd w = Vb[(long long)(j + 1) * N + s];
        for (int l = 0; l <= j; ++l) w -= cd(cr[l], ci[l]) * Vb[(long long)l * N + s];
        Vb[(long long)(j + 1) * N + s] = w;
        a = w.real() * w.real() + w.imag() * w.imag();
    }
    a = block_sum(a, sh);
    if (threadIdx.x == 0) atomicAdd(nrm2 + b * m + j, a);
}

// beta_j = |w|; V_{j+1} = w / beta_j, or 0 with beta_j = 0 and kdim = min(kdim, j+1) on breakdown
// (beta_j < 1e-12 max(1, |alpha_j|), as in kpo_common._krylov_exp).
extern "C" __global__ void lzscale(cd* __restrict__ V, const long long vb, const int N, const int j, const int m,
                                   const double* __restrict__ alpha, const double* __restrict__ nrm2,
                                   double* __restrict__ beta, int* __restrict__ kdim)
{
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const double bt = sqrt(nrm2[b * m + j]);
    const bool brk = bt < 1e-12 * fmax(1.0, fabs(alpha[b * m + j]));
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        beta[b * m + j] = brk ? 0.0 : bt;
        if (brk) atomicMin(kdim + b, j + 1);
    }
    if (s < N) {
        cd* q = V + (long long)b * vb + (long long)(j + 1) * N + s;
        *q = brk ? cd(0.0, 0.0) : (*q) / bt;
    }
}

// out[b] += sum_s |X_b[s]|^2
extern "C" __global__ void sqnorm(const cd* __restrict__ X, const long long xs, const int N,
                                  double* __restrict__ out)
{
    __shared__ double sh[32];
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 0.0;
    if (s < N) {
        const cd x = X[(long long)b * xs + s];
        a = x.real() * x.real() + x.imag() * x.imag();
    }
    a = block_sum(a, sh);
    if (threadIdx.x == 0) atomicAdd(out + b, a);
}

// V_0 = X / |X|
extern "C" __global__ void lzinit(const cd* __restrict__ X, const long long xs, cd* __restrict__ V,
                                  const long long vb, const int N, const double* __restrict__ n0)
{
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < N) V[(long long)b * vb + s] = X[(long long)b * xs + s] / sqrt(n0[b]);
}

// Y_b = |X_b| sum_{l<m} cv[b, l] V_l   (cv is zero beyond the Krylov dimension of instance b)
extern "C" __global__ void combine(const cd* __restrict__ V, const long long vb, const int N, const int m,
                                   const cd* __restrict__ cv, const double* __restrict__ n0,
                                   cd* __restrict__ Y, const long long ys)
{
    const int b = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    const cd* Vb = V + (long long)b * vb;
    const cd* c = cv + (long long)b * m;
    cd acc(0.0, 0.0);
    for (int l = 0; l < m; ++l) acc += c[l] * Vb[(long long)l * N + s];
    Y[(long long)b * ys + s] = sqrt(n0[b]) * acc;
}
"""

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC)
    return _MOD


_i32, _i64, _f64 = np.int32, np.int64, np.float64


class SectorBatchGPU:
    """B instances with the same n, cutoff, K, h and family; J (B, n, n), Delta (B, n), xi0 scalar or (B,)."""

    def __init__(self, Js, Deltas, K, xi0s, h, d, family="goto", m=24, block=256):
        Js = np.asarray(Js, dtype=np.float64)
        Deltas = np.asarray(Deltas, dtype=np.float64)
        B, n = Deltas.shape
        self.B, self.n = B, n
        if n > NMAX:
            raise ValueError(f"n = {n} > {NMAX}")
        if Js.shape != (B, n, n):
            raise ValueError(f"J has shape {Js.shape}, expected {(B, n, n)}")
        self.d, self.m = d, m
        self.D = D = d ** n
        if D >= 2 ** 31:
            raise ValueError("d^n too large for int32 indices")
        occ = np.indices((d,) * n).reshape(n, -1)
        idx = np.flatnonzero(occ.sum(0) % 2 == 0)
        self.idx_host = idx
        self.N = N = idx.size
        pos = np.full(D, -1, dtype=np.int32)
        pos[idx] = np.arange(N, dtype=np.int32)
        det = np.stack([kc.quantum_detuning(Deltas[b], K, h, family) for b in range(B)])
        jx = np.asarray(xi0s, dtype=np.float64).reshape(-1, 1, 1) * Js
        self.idx = cp.asarray(idx.astype(np.int32))
        self.pos = cp.asarray(pos)
        self.det = cp.asarray(np.ascontiguousarray(det))
        self.jx = cp.asarray(np.ascontiguousarray(np.broadcast_to(jx, (B, n, n))))
        self.kh = float(K * h)
        self.V = cp.zeros((B, m + 1, N), dtype=cp.complex128)
        self.buf = cp.zeros((3, B, N), dtype=cp.complex128)
        nb = B * m
        self.acc = cp.zeros(3 * nb + B + 2 * nb * m, dtype=cp.float64)   # one memset per Krylov build
        self.alpha = self.acc[0:nb].reshape(B, m)
        self.beta = self.acc[nb:2 * nb].reshape(B, m)
        self.nrm2 = self.acc[2 * nb:3 * nb].reshape(B, m)
        self.n0 = self.acc[3 * nb:3 * nb + B]
        self.cre = self.acc[3 * nb + B:3 * nb + B + nb * m]
        self.cim = self.acc[3 * nb + B + nb * m:]
        self.kdim = cp.empty(B, dtype=cp.int32)
        self.cv = cp.zeros((B, m), dtype=cp.complex128)
        self.grid = ((N + block - 1) // block, B)
        self.block = (block,)
        mod = _module()
        self._hmv = mod.get_function("hmv")
        self._o1 = mod.get_function("ortho1")
        self._o2 = mod.get_function("ortho2")
        self._sc = mod.get_function("lzscale")
        self._sq = mod.get_function("sqnorm")
        self._init = mod.get_function("lzinit")
        self._comb = mod.get_function("combine")

    # --- states -------------------------------------------------------------------------------
    def vacuum(self):
        v = cp.zeros((self.B, self.N), dtype=cp.complex128)
        v[:, 0] = 1.0                                   # idx[0] == 0 is the vacuum
        return v

    def embed(self, v):
        """one sector vector (host) -> full (d,)*n space (host), as KPOSector.embed"""
        full = np.zeros(self.D, dtype=np.complex128)
        full[self.idx_host] = v
        return full

    # --- H psi (self-test) ------------------------------------------------------------------------
    def matvec(self, X, p):
        """H(p) X for X of shape (B, N) (C-contiguous complex128 on the GPU)."""
        X = cp.ascontiguousarray(X, dtype=cp.complex128)
        Y = cp.empty_like(X)
        dummy = cp.zeros(self.B, dtype=cp.float64)
        N = self.N
        self._hmv(self.grid, self.block, (X, _i64(0), _i64(N), Y, _i64(0), _i64(N), self.idx, self.pos, self.det,
                                          self.jx, _f64(self.kh), _f64(p), _i32(self.n), _i32(self.d), _i32(N),
                                          dummy, _i32(1), _i32(0)))
        return Y

    # --- one Lanczos exponential -----------------------------------------------------------------
    def krylov_exp(self, X, p, tau, Y):
        """Y_b = exp(-i tau H_b(p)) X_b for every instance (X, Y: distinct C-contiguous (B, N) arrays).
        Returns (error estimates (B,), Krylov dimensions (B,)) on the host, as kpo_common._krylov_exp."""
        B, N, m = self.B, self.N, self.m
        V, vb = self.V, _i64((m + 1) * N)
        g, bl = self.grid, self.block
        self.acc.fill(0.0)
        self.kdim.fill(m)
        self._sq(g, bl, (X, _i64(N), _i32(N), self.n0))
        self._init(g, bl, (X, _i64(N), V, vb, _i32(N), self.n0))
        for j in range(m):
            self._hmv(g, bl, (V, _i64(j * N), vb, V, _i64((j + 1) * N), vb, self.idx, self.pos, self.det, self.jx,
                              _f64(self.kh), _f64(p), _i32(self.n), _i32(self.d), _i32(N),
                              self.alpha, _i32(m), _i32(j)))
            self._o1(g, bl, (V, vb, _i32(N), _i32(j), _i32(m), self.alpha, self.beta, self.cre, self.cim))
            self._o2(g, bl, (V, vb, _i32(N), _i32(j), _i32(m), self.cre, self.cim, self.nrm2))
            self._sc(g, bl, (V, vb, _i32(N), _i32(j), _i32(m), self.alpha, self.nrm2, self.beta, self.kdim))
        ab = cp.asnumpy(self.acc[:2 * B * m])
        al, be = ab[:B * m].reshape(B, m), ab[B * m:].reshape(B, m)
        kd = cp.asnumpy(self.kdim)
        cv = np.zeros((B, m), dtype=np.complex128)
        err = np.zeros(B)
        for k in np.unique(kd):
            sel = np.flatnonzero(kd == k)
            ii = np.arange(k)
            T = np.zeros((sel.size, k, k))
            T[:, ii, ii] = al[sel, :k]
            if k > 1:
                T[:, ii[:-1], ii[1:]] = be[sel, :k - 1]
                T[:, ii[1:], ii[:-1]] = be[sel, :k - 1]
            ev, U = np.linalg.eigh(T)
            c = np.einsum("bij,bj->bi", U, np.exp(-1j * ev * tau) * U[:, 0, :])
            cv[sel, :k] = c
            err[sel] = be[sel, k - 1] * np.abs(c[:, k - 1])
        self.cv.set(cv)
        self._comb(g, bl, (V, vb, _i32(N), _i32(m), self.cv, self.n0, Y, _i64(N)))
        return err, kd

    # --- time evolution ---------------------------------------------------------------------------
    def propagate(self, psi, T, pfun, tol=1e-8, dt_init=0.02, dt_max=1.0, dt_min=1e-7, log=None, nlog=10):
        """Batched kpo_common.lanczos_propagate: psi (B, N) on the GPU -> (psi(T) (B, N) on the GPU, stats).
        stats: steps, rejects, matvecs (per instance), err_sum (per instance, host array)."""
        B, m = self.B, self.m
        P = self.buf
        cp.copyto(P[0], psi)
        cur = 0
        t, dt = 0.0, dt_init
        st = dict(steps=0, rejects=0, matvecs=0, err_sum=np.zeros(B))
        t0, next_log = time.time(), 1
        while t < T * (1 - 1e-14):
            dt = min(dt, dt_max, T - t)
            i1, i2 = [i for i in range(3) if i != cur]
            while True:
                p1, p2 = pfun(t + kc._CF_C1 * dt), pfun(t + kc._CF_C2 * dt)
                pA = 2.0 * (kc._CF_A2 * p1 + kc._CF_A1 * p2)
                pB = 2.0 * (kc._CF_A1 * p1 + kc._CF_A2 * p2)
                e1, k1 = self.krylov_exp(P[cur], pA, 0.5 * dt, P[i1])
                e2, k2 = self.krylov_exp(P[i1], pB, 0.5 * dt, P[i2])
                st["matvecs"] += 2 * m
                errb = e1 + e2
                err = float(errb.max())
                k = int(min(k1.min(), k2.min()))
                tol_step = tol * dt / T
                if err <= tol_step or dt <= dt_min:
                    break
                st["rejects"] += 1
                dt *= max(0.1, 0.9 * (tol_step / err) ** (1.0 / k))
            cur = i2
            t += dt
            st["steps"] += 1
            st["err_sum"] += errb
            grow = 2.0 if err == 0.0 else min(2.0, 0.9 * (tol_step / err) ** (1.0 / k))
            dt *= max(0.5, grow)
            if log is not None and t >= next_log * T / nlog * (1 - 1e-12):
                log(f"t {t:8.2f}/{T:g}  steps {st['steps']:6d}  rejects {st['rejects']:4d}  dt {dt:.3g}  "
                    f"{time.time() - t0:7.0f} s")
                next_log = int(math.floor(t / T * nlog + 1e-9)) + 1
        return P[cur].copy(), st
