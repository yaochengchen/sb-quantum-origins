"""MSBQA phase 1c (GPU, v6: fused CUDA kernel for H X, own LOBPCG, own kernels for the dense operations): instantaneous spectrum of the quantum KPO network along the pump,
even-parity sector, minimum gap Delta_min, its position p*, Landau-Zener slope -- the GPU version of gap_scan.py (v3)
for large sectors (small h, n = 6). One instance per call, any n <= 8.

usage (from msbqa_phase1/, one GPU):
  CUDA_VISIBLE_DEVICES=0 python code/gap_scan_gpu.py --inst-dir instances --set G4 --inst 6 --family weyl --h 0.18 \
      --cutoff 53 --pmax 5 --tag Wg018 --out run_v4/parts/Wg018_6.json
  ... --probe   (quick, < ~1 min: kernel checks against the GPU CSR matrices and the CPU engine, dense kernels vs
                 cuBLAS, timings, four continuation eigensolves; writes nothing)

Same physics, conventions and algorithm as gap_scan.py v3 (see there). GPU specifics:
  * H(p) X for a block X of m vectors (C order, shape (N, m)) is ONE fused CUDA kernel ('hblk', compiled once per
    block width so that the m accumulators stay in registers): each thread takes
    one sector state, decodes its occupations from the full index, and gathers the diagonal, pump and hopping terms
    from the neighbouring states through the full-index -> sector-position table. No matrix is stored. Same matrix
    elements and basis order as kpo_common.KPOSector and as the validated phase-1 time-evolution kernel 'hmv'
    (kpo_gpu.py), in real arithmetic:
        diag  sum_i [ det_i k_i + (K h / 2) k_i (k_i - 1) ]
        pump  -(p/2) sqrt((k_i+1)(k_i+2)) X[k + 2e_i]   and   -(p/2) sqrt(k_i (k_i-1)) X[k - 2e_i]
        hop   -xi0 J_ij sqrt(k_i (k_j+1)) X[k - e_i + e_j]   (i != j)
  * v1 used cupyx CSR matrices (cuSPARSE SpMM). The CSR path is kept as an independent check (--probe compares
    kernel and CSR, and kernel and KPOSector for small sectors) and as a fallback; built only when needed.
  * p = 0 is solved exactly block by block in the photon number (blocks built on the CPU from the occupations).
  * p > 0: LOBPCG continued from the previous p, accepted only if the residuals of the k lowest are small and the
    Ritz values do not exceed the Rayleigh-Ritz bounds of the start block. The 10-31 probes showed that
    cupyx.scipy.sparse.linalg.lobpcg spends ~98 % of its time in its own dense operations on the long vectors at
    1.4e7 states (kernel 7.7 ms per H X with 6 vectors, yet ~15 s per eigensolve of ~25 iterations), so the
    default is lean_lobpcg: the same LOBPCG iteration with few passes over the long vectors. v5: the v4 probe showed
    that the remaining time per iteration is in the tall-skinny Gram products (N x m)^T (N x m), which cuBLAS runs on
    very few thread blocks; v5 split the long dimension into batches, which did not help (21 ms either way). v6: own
    CUDA kernels 'gramk' (A^T B) and 'lincomb' (Y0 + A C) for all dense operations on the long vectors, checked against
    cuBLAS at the start of every run (check_dense; cuBLAS is used on a mismatch). Attempts of one
    eigensolve: lean, lean (perturbed start, more iterations), cupyx lobpcg with the kernel ('cupyx_fallbacks'),
    cupyx lobpcg with CSR ('csr_fallbacks'), scipy lobpcg on the CPU ('cpu_fallbacks'). --probe also runs cupyx
    lobpcg from the same start at p = 0.1, 1, 3 and compares E0, E1 (independent check of lean_lobpcg).
  * --pmax: the scan covers p in [0, pmax] with step --dpgrid; a minimum in the first/last grid cell is edge = 1.
  * top2: largest population (over the grid) of states with some k_i >= d - 2 in the tracked ground state.
"""
import argparse
import math
import os
import time
import warnings

import numpy as np

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--inst", type=int, required=True)
ap.add_argument("--family", required=True, choices=["goto", "weyl"])
ap.add_argument("--h", type=float, default=1.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--cutoff", type=int, required=True)
ap.add_argument("--pf", type=float, default=7.0, help="final pump of the time evolution (for x_LZ)")
ap.add_argument("--pmax", type=float, default=0.0, help="upper end of the scan (0 -> pf)")
ap.add_argument("--dpgrid", type=float, default=0.05)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--guard", type=int, default=2)
ap.add_argument("--restol", type=float, default=1e-8)
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--nmin", type=int, default=3)
ap.add_argument("--xatol", type=float, default=1e-6)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--tag", default="gapgpu")
ap.add_argument("--out", default="")
a = ap.parse_args()
if not a.probe:
    if not a.out:
        raise SystemExit("--out is required (or --probe)")
    if os.path.exists(a.out):
        raise SystemExit(0)

import cupy as cp                                                    # noqa: E402
import cupyx.scipy.sparse as csp                                      # noqa: E402
from cupyx.scipy.sparse.linalg import LinearOperator as GOp           # noqa: E402
from cupyx.scipy.sparse.linalg import lobpcg as globpcg               # noqa: E402
from scipy.optimize import minimize_scalar                            # noqa: E402

t0 = time.time()
J, Delta, meta = kc.load_instance(a.inst_dir, a.set, a.inst)
J = np.asarray(J, dtype=np.float64)
xi0 = float(meta["xi0"])
n = len(Delta)
d = a.cutoff + 1
K, h = a.K, a.h
pmax = a.pmax if a.pmax > 0 else a.pf
det = np.asarray(kc.quantum_detuning(Delta, K, h, a.family), dtype=np.float64)
m = a.k + a.guard
if n > 8:
    raise SystemExit("n <= 8 only")
STRIDE = [d ** (n - 1 - i) for i in range(n)]

_SRC = r"""
#define NMAX 8
#define M MVAL
extern "C" __global__ void hblk(const double* __restrict__ X, double* __restrict__ Y,
                                const long long* __restrict__ idx, const int* __restrict__ pos,
                                const double* __restrict__ det, const double* __restrict__ jx,
                                const double kh, const double p, const int n, const int d,
                                const long long N)
{
    const long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    const long long f = idx[s];
    int k[NMAX];
    long long st[NMAX];
    long long r = f, stride = 1;
    for (int i = n - 1; i >= 0; --i) { k[i] = (int)(r % d); r /= d; st[i] = stride; stride *= d; }
    double dg = 0.0;
    for (int i = 0; i < n; ++i) dg += det[i] * k[i] + 0.5 * kh * k[i] * (k[i] - 1);
    double acc[M];
    const double* xs = X + s * M;
#pragma unroll
    for (int c = 0; c < M; ++c) acc[c] = dg * xs[c];
    const double hp = 0.5 * p;
    if (hp != 0.0) {
        for (int i = 0; i < n; ++i) {                 // pump -(p/2)(a_i^2 + a_i^+2)
            if (k[i] + 2 <= d - 1) {
                const double cf = hp * sqrt((double)(k[i] + 1) * (k[i] + 2));
                const double* xt = X + (long long)pos[f + 2 * st[i]] * M;
#pragma unroll
                for (int c = 0; c < M; ++c) acc[c] -= cf * xt[c];
            }
            if (k[i] >= 2) {
                const double cf = hp * sqrt((double)k[i] * (k[i] - 1));
                const double* xt = X + (long long)pos[f - 2 * st[i]] * M;
#pragma unroll
                for (int c = 0; c < M; ++c) acc[c] -= cf * xt[c];
            }
        }
    }
    for (int i = 0; i < n; ++i) {                     // hopping -xi0 J_ij a_i^+ a_j: <k| ... |k - e_i + e_j>
        if (k[i] < 1) continue;
        const double si = sqrt((double)k[i]);
        for (int j = 0; j < n; ++j) {
            if (j == i || k[j] + 1 > d - 1) continue;
            const double cj = jx[i * n + j];
            if (cj == 0.0) continue;
            const double cf = cj * si * sqrt((double)(k[j] + 1));
            const double* xt = X + (long long)pos[f - st[i] + st[j]] * M;
#pragma unroll
            for (int c = 0; c < M; ++c) acc[c] -= cf * xt[c];
        }
    }
    double* ys = Y + s * M;
#pragma unroll
    for (int c = 0; c < M; ++c) ys[c] = acc[c];
}
"""
MMAX = 16
_KERNS = {}


def kern(mw):
    """The kernel compiled for block width mw (accumulators in registers); cached."""
    if mw not in _KERNS:
        _KERNS[mw] = cp.RawKernel(_SRC.replace("MVAL", str(mw)), "hblk")
    return _KERNS[mw]


# ------------------------------------------------------------------ sector index on the GPU
def build_index():
    D = d ** n
    if D >= 2 ** 62:
        raise SystemExit("sector too large")
    f = cp.arange(D, dtype=cp.int64)
    tot = cp.zeros(D, dtype=cp.int32)
    r = f.copy()
    for i in range(n - 1, -1, -1):
        tot += (r % d).astype(cp.int32)
        r //= d
    del r
    idx = cp.flatnonzero((tot % 2) == 0).astype(cp.int64)            # sector -> full index (ascending)
    del tot, f
    Ns = int(idx.size)
    if Ns >= 2 ** 31 - 1:
        raise SystemExit("sector dimension does not fit int32 positions")
    pos = cp.full(D, -1, dtype=cp.int32)
    pos[idx] = cp.arange(Ns, dtype=cp.int32)
    ks = cp.empty((n, Ns), dtype=cp.int16)
    r = idx.copy()
    for i in range(n - 1, -1, -1):
        ks[i] = (r % d).astype(cp.int16)
        r //= d
    del r
    diag = cp.zeros(Ns, dtype=cp.float64)
    for i in range(n):
        kf = ks[i].astype(cp.float64)
        diag += det[i] * kf + 0.5 * K * h * kf * (kf - 1.0)
    hi = cp.max(ks, axis=0) >= d - 2                                  # truncation check
    return Ns, idx, pos, ks, diag, hi


tb = time.time()
Ns, IDX, POS, KS, DIAG, HI = build_index()
DET_G = cp.asarray(det)
JX_G = cp.asarray((xi0 * J).ravel())
cp.cuda.Device().synchronize()
t_build = time.time() - tb
CSR = {}


def build_csr():
    """cupyx CSR matrices of the same H (independent code path; v1 of this script)."""
    if CSR:
        return CSR["Hh"], CSR["Hp"]
    src_all = cp.arange(Ns, dtype=cp.int32)

    def assemble(terms):
        counts = [int(cp.count_nonzero(mk)) for mk, _, _ in terms]
        nnz = sum(counts)
        rows = cp.empty(nnz, dtype=cp.int32)
        cols = cp.empty(nnz, dtype=cp.int32)
        vals = cp.empty(nnz, dtype=cp.float64)
        o = 0
        for (mk, shift, vf), c in zip(terms, counts):
            if c == 0:
                continue
            rows[o:o + c] = POS[IDX[mk] + shift]
            cols[o:o + c] = src_all[mk]
            vals[o:o + c] = vf(mk)
            o += c
        if int(cp.count_nonzero(rows < 0)):
            raise RuntimeError("target outside the even sector (bug)")
        M = csp.coo_matrix((vals, (rows, cols)), shape=(Ns, Ns)).tocsr()
        M.sum_duplicates()
        return M

    terms = []
    for i in range(n):
        def vpump(mk, i=i):
            kf = KS[i][mk].astype(cp.float64)
            return -0.5 * cp.sqrt(kf * (kf - 1.0))
        terms.append((KS[i] >= 2, -2 * STRIDE[i], vpump))
    lower = assemble(terms)
    Hp = (lower + lower.T.tocsr()).tocsr()
    terms = []
    for i in range(n):
        for j in range(n):
            if i == j or J[i, j] == 0.0:
                continue

            def vhop(mk, i=i, j=j):
                return (-xi0 * J[i, j]) * cp.sqrt(KS[j][mk].astype(cp.float64) * (KS[i][mk].astype(cp.float64) + 1.0))
            terms.append(((KS[j] >= 1) & (KS[i] <= d - 2), STRIDE[i] - STRIDE[j], vhop))
    CSR["Hh"], CSR["Hp"] = assemble(terms), Hp
    return CSR["Hh"], CSR["Hp"]


st = dict(calls=0, matvecs=0, retries=0, misconv=0, lean_iters=0, cupyx_fallbacks=0, csr_fallbacks=0,
          cpu_fallbacks=0, X=None, top2=0.0)


def hkern(Y, p):
    """H(p) Y with the fused kernel; Y of shape (N,) or (N, m)."""
    Y = cp.asarray(Y, dtype=cp.float64)
    one = Y.ndim == 1
    Y2 = Y.reshape(Ns, 1) if one else Y
    mm_ = Y2.shape[1]
    st["matvecs"] += mm_
    args = (IDX, POS, DET_G, JX_G, np.float64(K * h), np.float64(p), np.int32(n), np.int32(d), np.int64(Ns))
    if mm_ <= MMAX:
        Xc = cp.ascontiguousarray(Y2)
        out = cp.empty_like(Xc)
        kern(mm_)(((Ns + 255) // 256,), (256,), (Xc, out) + args)
    else:
        out = cp.empty((Ns, mm_), dtype=cp.float64)
        for c0 in range(0, mm_, MMAX):
            c1 = min(c0 + MMAX, mm_)
            Xc = cp.ascontiguousarray(Y2[:, c0:c1])
            Yc = cp.empty_like(Xc)
            kern(c1 - c0)(((Ns + 255) // 256,), (256,), (Xc, Yc) + args)
            out[:, c0:c1] = Yc
    return out.reshape(Ns) if one else out


def hcsr(Y, p):
    Hh, Hp = build_csr()
    Y = cp.asarray(Y, dtype=cp.float64)
    st["matvecs"] += 1 if Y.ndim == 1 else Y.shape[1]
    if Y.ndim == 1:
        return DIAG * Y + Hh @ Y + p * (Hp @ Y)
    return DIAG[:, None] * Y + Hh @ Y + p * (Hp @ Y)


def Aop(p, engine="kernel"):
    f = hkern if engine == "kernel" else hcsr

    def mm(Y):
        return f(Y, p)
    return GOp((Ns, Ns), matvec=mm, matmat=mm, dtype=cp.float64)


def exact_p0():
    """m lowest levels of H(0) in the even sector, block by block in the photon number (CPU blocks, GPU vectors)."""
    tot = KS.astype(cp.int32).sum(0)
    wmin = float(np.linalg.eigvalsh(np.diag(det) - xi0 * J)[0])
    blocks, levels, nb = [], [], 0
    while True:
        sel = cp.flatnonzero(tot == nb)
        if int(sel.size) == 0:
            break
        lb = nb * wmin + 0.5 * K * h * (nb * nb / n - nb)
        if nb >= n / 2 and len(levels) >= m and lb > sorted(e for e, _, _ in levels)[m - 1]:
            break
        occ = KS[:, sel].get().astype(np.int64).T                     # (B, n)
        full = occ @ np.array(STRIDE, dtype=np.int64)
        loc = {int(x): q for q, x in enumerate(full)}
        B = len(full)
        H0 = np.zeros((B, B))
        for q in range(B):
            kq = occ[q].astype(np.float64)
            H0[q, q] = float(np.sum(det * kq + 0.5 * K * h * kq * (kq - 1.0)))
            for i in range(n):
                for j in range(n):
                    if i == j or J[i, j] == 0.0 or occ[q, j] < 1 or occ[q, i] > d - 2:
                        continue
                    t = loc.get(int(full[q] + STRIDE[i] - STRIDE[j]))
                    if t is not None:
                        H0[t, q] += -xi0 * J[i, j] * math.sqrt(kq[j] * (kq[i] + 1.0))
        e, U = np.linalg.eigh(0.5 * (H0 + H0.T))
        blocks.append((nb, sel, U))
        levels += [(float(x), len(blocks) - 1, j) for j, x in enumerate(e)]
        nb += 2
    levels.sort()
    X = cp.zeros((Ns, m), dtype=cp.float64)
    for c, (e, b, j) in enumerate(levels[:m]):
        X[blocks[b][1], c] = cp.asarray(blocks[b][2][:, j])
    return np.array([e for e, _, _ in levels[:m]]), X, [int(blocks[b][0]) for _, b, _ in levels[:m]]


def cpu_lobpcg(p, X):
    from scipy.sparse.linalg import LinearOperator, lobpcg
    Hh, Hp = build_csr()
    dg, hh, hp = DIAG.get(), Hh.get(), Hp.get()
    A = LinearOperator((Ns, Ns), matvec=lambda y: dg * np.ravel(y) + hh @ np.ravel(y) + p * (hp @ np.ravel(y)),
                       matmat=lambda Y: dg[:, None] * Y + hh @ Y + p * (hp @ Y), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, V = lobpcg(A, X.get(), tol=a.restol * 10, maxiter=4 * a.iters, largest=False)
    return np.asarray(w), cp.asarray(V)


# v6: dense operations on the long vectors with own CUDA kernels instead of cuBLAS. The v5 probe (h = 0.12, 1.42e7
# states): X^T X with 6 columns took 21 ms with cuBLAS, batched or not, i.e. ~30x slower than reading the data;
# probable reason: DGEMM computes on fixed output tiles (6 x 6 -> padded) and this card's FP64 rate is 1/64 of FP32.
# Per LOBPCG iteration ~19 such Gram products and ~16 (N x a)(a x b) updates took ~0.6 s of the 0.62 s.
#   gramk:   A^T B, one thread per row (grid-stride), a*b accumulators in registers, warp + block reduction,
#            one partial per thread block (summed by CuPy; deterministic).
#   lincomb: Y = beta Y0 + A C, one thread per row, C (a x b) small.
# Both are compared with cuBLAS at the start of every run (check_dense); on a mismatch the cuBLAS path is used.
_DSRC = r"""
#define NA NAVAL
#define NB NBVAL
#define NWARP 8
extern "C" __global__ void gramk(const double* __restrict__ A, const double* __restrict__ B,
                                 double* __restrict__ part, const long long N)
{
    double acc[NA * NB];
#pragma unroll
    for (int c = 0; c < NA * NB; ++c) acc[c] = 0.0;
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x; s < N; s += step) {
        double av[NA], bv[NB];
#pragma unroll
        for (int i = 0; i < NA; ++i) av[i] = A[s * NA + i];
#pragma unroll
        for (int j = 0; j < NB; ++j) bv[j] = B[s * NB + j];
#pragma unroll
        for (int i = 0; i < NA; ++i) {
#pragma unroll
            for (int j = 0; j < NB; ++j) acc[i * NB + j] += av[i] * bv[j];
        }
    }
    __shared__ double sh[NWARP * NA * NB];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < NA * NB; ++c) {
        double v = acc[c];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[wid * NA * NB + c] = v;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < NA * NB; c += blockDim.x) {
        double t = 0.0;
        for (int w = 0; w < NWARP; ++w) t += sh[w * NA * NB + c];
        part[(long long)blockIdx.x * NA * NB + c] = t;
    }
}

extern "C" __global__ void lincomb(const double* __restrict__ A, const double* __restrict__ C,
                                   const double* Y0, double* Y, const double beta, const long long N)
{
    const long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    double acc[NB];
    if (beta != 0.0) {
#pragma unroll
        for (int j = 0; j < NB; ++j) acc[j] = beta * Y0[s * NB + j];
    } else {
#pragma unroll
        for (int j = 0; j < NB; ++j) acc[j] = 0.0;
    }
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const double x = A[s * NA + i];
#pragma unroll
        for (int j = 0; j < NB; ++j) acc[j] += x * C[i * NB + j];
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) Y[s * NB + j] = acc[j];
}
"""
_DMODS = {}
DENSE = {"kernel": True}
NSM = int(cp.cuda.Device().attributes["MultiProcessorCount"])


def dmod(na, nb):
    """Module with gramk / lincomb compiled for widths (na, nb); cached."""
    if (na, nb) not in _DMODS:
        _DMODS[(na, nb)] = cp.RawModule(code=_DSRC.replace("NAVAL", str(na)).replace("NBVAL", str(nb)))
    return _DMODS[(na, nb)]


def f64c(Z):
    return cp.ascontiguousarray(cp.asarray(Z, dtype=cp.float64))


def gram(A_, B_):
    """A_^T B_ for tall-skinny (N x a), (N x b) arrays (own kernel; cuBLAS if disabled or a*b > 64)."""
    A_, B_ = f64c(A_), f64c(B_)
    Nr, na = A_.shape
    nb_ = B_.shape[1]
    if not DENSE["kernel"] or na * nb_ > 64:
        return A_.T @ B_
    nblk = max(1, min((Nr + 255) // 256, 8 * NSM))
    part = cp.empty((nblk, na * nb_), dtype=cp.float64)
    dmod(na, nb_).get_function("gramk")((nblk,), (256,), (A_, B_, part, np.int64(Nr)))
    return part.sum(axis=0).reshape(na, nb_)


def lin(Y0, terms):
    """Y0 + sum_t A_t @ C_t (Y0 may be None) for tall-skinny A_t (N x a_t) and small C_t (a_t x b); new array."""
    out = None
    for A_, C in terms:
        A_, C = f64c(A_), f64c(C)
        Nr, na = A_.shape
        nb_ = C.shape[1]
        base = out if out is not None else (f64c(Y0) if Y0 is not None else None)
        if not DENSE["kernel"] or na > 32 or nb_ > 32:
            prod = A_ @ C
            out = prod if base is None else base + prod
            continue
        res = cp.empty((Nr, nb_), dtype=cp.float64)
        y0 = base if base is not None else res
        dmod(na, nb_).get_function("lincomb")(((Nr + 255) // 256,), (256,),
                                              (A_, C, y0, res, np.float64(1.0 if base is not None else 0.0),
                                               np.int64(Nr)))
        out = res
    return out


def mm(A_, C):
    """A_ @ C for tall-skinny A_ and small C."""
    return lin(None, [(A_, C)])


def colnorm(R):
    """Euclidean norms of the columns of a tall-skinny array, as a NumPy array."""
    return np.sqrt(np.maximum(np.diag(gram(R, R).get()), 0.0))


def check_dense(verbose=True):
    """Own dense kernels vs cuBLAS on random vectors of the full length (< 1 s); falls back to cuBLAS on a mismatch."""
    rng = np.random.default_rng(3)
    A_ = cp.asarray(rng.standard_normal((Ns, m)))
    B_ = cp.asarray(rng.standard_normal((Ns, m - 1)))
    C = cp.asarray(rng.standard_normal((m, m - 1)))
    g1, g2 = gram(A_, B_), A_.T @ B_
    rg = float(cp.abs(g1 - g2).max() / cp.abs(g2).max())
    l1, l2 = lin(B_, [(A_, C)]), B_ + A_ @ C
    rl = float(cp.abs(l1 - l2).max() / cp.abs(l2).max())
    ok = rg < 1e-10 and rl < 1e-12

    def tm(fn):
        fn()
        cp.cuda.Device().synchronize()
        tt = time.time()
        for _ in range(5):
            fn()
        cp.cuda.Device().synchronize()
        return (time.time() - tt) / 5 * 1e3
    t = (tm(lambda: gram(A_, A_)), tm(lambda: A_.T @ A_), tm(lambda: mm(A_, cp.eye(m))), tm(lambda: A_ @ cp.eye(m)))
    if not ok:
        DENSE["kernel"] = False
    if verbose:
        print(f"  dense kernels vs cuBLAS: gram rel diff {rg:.1e}, lincomb rel diff {rl:.1e}"
              + ("  -> OK" if ok else "  -> WRONG, using cuBLAS") + f";  X^T X {t[0]:.2f} ms (cuBLAS {t[1]:.2f}), "
              f"X C {t[2]:.2f} ms (cuBLAS {t[3]:.2f}), {m} columns", flush=True)
    return ok


def orthonormalize(X):
    """Orthonormal basis of the columns of X (SVQB twice; replaces cp.linalg.qr, slow on tall-skinny arrays)."""
    Q, _ = svqb(X)
    Q, _ = svqb(Q)
    return Q


def svqb(Y, AY=None, drop=1e-12):
    """Orthonormalise the columns of Y (and apply the same transform to AY) by the eigen-decomposition of the
    column-normalised Gram matrix (SVQB, Stathopoulos & Wu 2002); directions below drop x the largest are removed
    (returns None, None if nothing is left)."""
    G = gram(Y, Y).get()
    dsc = 1.0 / np.sqrt(np.maximum(np.diag(G), 1e-300))
    th, U = np.linalg.eigh(0.5 * (G + G.T) * dsc[:, None] * dsc[None, :])
    keep = th > drop * max(float(th.max()), 1e-300)
    if not keep.any():
        return None, None
    T = np.ascontiguousarray(dsc[:, None] * U[:, keep] / np.sqrt(th[keep])[None, :])
    return mm(Y, T), (mm(AY, T) if AY is not None else None)


def lean_lobpcg(p, X0, dd, tol_abs, maxit):
    """Block LOBPCG (Knyazev 2001) in the stabilised form (W and P orthonormalised against the current block X, as in
    BLOPEX), written for long vectors: per iteration one fused-kernel application A W; all other work is
    (N x m)^T (N x m) Gram products and (N x m)(m x m) updates (cuBLAS, nothing concatenated) plus small dense
    eigenproblems on the CPU; AX and AP are carried along, not recomputed. Rayleigh-Ritz on [X, W, P] with a
    Gram-matrix guard (column-normalised, directions below 1e-10 dropped). Returns (Ritz values (m,), X, iterations,
    converged, residual norms); convergence = residual norms of the k lowest <= tol_abs."""
    X = orthonormalize(X0)
    AX = hkern(X, p)
    Hs = gram(X, AX).get()
    lam, C = np.linalg.eigh(0.5 * (Hs + Hs.T))
    X, AX = mm(X, C), mm(AX, C)
    P = AP = None
    res = np.full(m, np.inf)
    for it in range(maxit):
        R = AX - X * cp.asarray(lam)[None, :]
        res = colnorm(R)
        if np.all(res[: a.k] <= tol_abs):
            return lam, X, it, True, res
        W = R / dd[:, None]
        W = lin(W, [(X, -gram(X, W))])
        W, _ = svqb(W)
        if W is None:
            return lam, X, it, bool(np.all(res[: a.k] <= tol_abs)), res
        W = lin(W, [(X, -gram(X, W))])                              # second pass (re-orthogonalisation)
        W, _ = svqb(W)
        if W is None:
            return lam, X, it, bool(np.all(res[: a.k] <= tol_abs)), res
        AW = hkern(W, p)
        blocks = [(X, AX), (W, AW)]
        if P is not None:
            c1 = -gram(X, P)
            c2 = -gram(W, P)
            P = lin(P, [(X, c1), (W, c2)])
            AP = lin(AP, [(AX, c1), (AW, c2)])
            P, AP = svqb(P, AP)
            if P is not None:
                blocks.append((P, AP))
        wid = [b[0].shape[1] for b in blocks]
        off = np.concatenate([[0], np.cumsum(wid)])
        tot = int(off[-1])
        G = np.zeros((tot, tot))
        H = np.zeros((tot, tot))
        for i in range(len(blocks)):
            for j in range(i, len(blocks)):
                gij = gram(blocks[i][0], blocks[j][0]).get()
                hij = gram(blocks[i][0], blocks[j][1]).get()
                G[off[i]:off[i + 1], off[j]:off[j + 1]] = gij
                H[off[i]:off[i + 1], off[j]:off[j + 1]] = hij
                if j > i:
                    G[off[j]:off[j + 1], off[i]:off[i + 1]] = gij.T
                    H[off[j]:off[j + 1], off[i]:off[i + 1]] = hij.T
        dsc = 1.0 / np.sqrt(np.maximum(np.diag(G), 1e-300))
        Gn = G * dsc[:, None] * dsc[None, :]
        Hn = 0.5 * (H + H.T) * dsc[:, None] * dsc[None, :]
        th, U = np.linalg.eigh(0.5 * (Gn + Gn.T))
        keep = th > 1e-10 * th.max()
        B = U[:, keep] / np.sqrt(th[keep])[None, :]
        Hr = B.T @ Hn @ B
        lam_all, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
        Cf = dsc[:, None] * (B @ Q[:, :m])                          # (tot, m) coefficients on the blocks
        lam = lam_all[:m]
        Cb = [np.ascontiguousarray(Cf[off[i]:off[i + 1]]) for i in range(len(blocks))]
        Pn = lin(None, [(blocks[q][0], Cb[q]) for q in range(1, len(blocks))])
        APn = lin(None, [(blocks[q][1], Cb[q]) for q in range(1, len(blocks))])
        X = lin(Pn, [(X, Cb[0])])
        AX = lin(APn, [(AX, Cb[0])])
        P, AP = Pn, APn
    return lam, X, maxit, False, res


ENGINES = ["lean", "lean", "cupyx", "csr", "cpu"]                    # attempts of one eigensolve, in this order


def solve(p):
    st["calls"] += 1
    X = orthonormalize(st["X"])
    A = Aop(p)
    Hs = gram(X, A.matmat(X)).get()
    mu = np.linalg.eigvalsh(0.5 * (Hs + Hs.T))
    scale = max(1.0, float(np.abs(mu).max()))
    dd = cp.abs(DIAG - mu[0]) + 1.0
    M = GOp((Ns, Ns), matvec=lambda y: cp.ravel(y) / dd, matmat=lambda Y: cp.asarray(Y) / dd[:, None],
            dtype=cp.float64)
    Y, why = X, ""
    for attempt, eng in enumerate(ENGINES):
        try:
            if eng == "lean":
                w, V, its, conv, _ = lean_lobpcg(p, Y, dd, a.restol * scale, a.iters * (attempt + 1))
                st["lean_iters"] += its
                if not conv:
                    raise RuntimeError(f"lean LOBPCG did not converge in {its} iterations")
            elif eng in ("cupyx", "csr"):
                st[f"{eng}_fallbacks"] += 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w, V = globpcg(Aop(p, "kernel" if eng == "cupyx" else "csr"), Y, M=M, tol=a.restol * scale,
                                   maxiter=3 * a.iters, largest=False)
                w = cp.asnumpy(w)
            else:
                st["cpu_fallbacks"] += 1
                w, V = cpu_lobpcg(p, Y)
        except Exception as exc:
            st["retries"] += 1
            why = f"{eng} raised {type(exc).__name__}: {exc}"
            Q = X + 1e-4 * cp.asarray(np.random.default_rng(st["calls"] * 10 + attempt).standard_normal(X.shape))
            Y = orthonormalize(Q)
            continue
        o = np.argsort(w)
        w = w[o]
        V = V[:, cp.asarray(o)]
        res = colnorm(A.matmat(V) - V * cp.asarray(w)[None, :])
        ok_res = bool(np.all(res[: a.k] <= 10 * a.restol * scale))
        ok_rr = bool(np.all(w <= mu + 1e-9 * scale))
        if ok_res and ok_rr:
            st["X"] = V
            st["top2"] = max(st["top2"], float(cp.sum(V[:, 0][HI] ** 2)))
            return w
        if not ok_rr:
            st["misconv"] += 1
        st["retries"] += 1
        why = f"{eng}: residuals {res[: a.k]}, Ritz {w}, bounds {mu}"
        Y = V
    raise RuntimeError(f"inst {a.inst}: eigensolve failed at p = {p}: {why}")


def spectrum(p):
    if p == 0.0:
        w, X, _ = exact_p0()
        st["X"] = X
        return w
    return solve(p)


def gap(p):
    w = spectrum(p)
    evals.append((float(p), w[: a.k].tolist()))
    return float(w[1] - w[0])


evals = []

if a.probe:
    mem = cp.get_default_memory_pool().used_bytes() / 2 ** 30
    print(f"inst {a.inst} set {a.set} n {n} family {a.family} h {h} cutoff {a.cutoff} pmax {pmax}: sector dim {Ns}, "
          f"index build {t_build:.1f} s, GPU memory in use {mem:.1f} GB (no matrix stored)")
    rv = cp.asarray(np.random.default_rng(1).standard_normal((Ns, m)))
    tc = time.time()
    build_csr()
    cp.cuda.Device().synchronize()
    print(f"  CSR matrices for the checks: {time.time() - tc:.1f} s, nnz hop {CSR['Hh'].nnz} pump {CSR['Hp'].nnz}")
    for p in (0.0, 1.3):
        yk, yc = hkern(rv, p), hcsr(rv, p)
        dif = float(cp.abs(yk - yc).max())
        ref = float(cp.abs(yc).max())
        print(f"  check kernel vs CSR, H(p={p}) on {m} random vectors: max |diff| {dif:.2e}"
              + ("  -> OK" if dif < 1e-11 * max(1.0, ref) else "  -> WRONG"))
    if Ns <= 400000:
        model = kc.KPOSector(J, Delta, K, xi0, h, d, a.family)
        v = np.random.default_rng(2).standard_normal(Ns)
        for p in (0.0, 1.3):
            ref = model.diag * v + model.Hhop @ v + p * (model.Hp @ v)
            got = hkern(cp.asarray(v), p).get()
            dif = np.abs(got - ref).max()
            print(f"  check kernel vs kpo_common.KPOSector (CPU), p={p}: max |diff| {dif:.2e}"
                  + ("  -> OK" if dif < 1e-10 * max(1.0, np.abs(ref).max()) else "  -> WRONG"))
    for name, fn in (("kernel", hkern), ("CSR", hcsr)):
        fn(rv, 1.0)
        cp.cuda.Device().synchronize()
        tt = time.time()
        for _ in range(10):
            fn(rv, 1.0)
        cp.cuda.Device().synchronize()
        print(f"  timing {name:6s}: {(time.time() - tt) / 10 * 1e3:8.2f} ms per H X with {m} vectors")
    CSR.clear()
    del rv
    cp.get_default_memory_pool().free_all_blocks()
    check_dense()
    cp.get_default_memory_pool().free_all_blocks()
    tp = time.time()
    w0, X0, phot = exact_p0()
    st["X"] = X0
    print(f"  p =  0.00 (exact): E = {np.array2string(w0[: a.k], precision=8)}  photon numbers {phot}  "
          f"{time.time() - tp:.2f} s")
    # v6: quick probe (the cupyx comparisons and the large jumps of v4/v5 took ~15 min at h = 0.12; dropped)
    times = []
    for p in (0.05, 0.1, 0.15, 0.2):
        tp, m0, i0 = time.time(), st["matvecs"], st["lean_iters"]
        w = solve(p)
        cp.cuda.Device().synchronize()
        times.append(time.time() - tp)
        nit = st["lean_iters"] - i0
        print(f"  p = {p:5.2f}: E0 = {w[0]:+.10f}  gap = {w[1] - w[0]:.6e}  {times[-1]:.2f} s  "
              f"{st['matvecs'] - m0} matvecs, {nit} lean-LOBPCG iterations ({times[-1] / max(nit, 1) * 1e3:.0f} ms each)",
              flush=True)
    small = float(np.median(times[1:]))
    npts = int(round(pmax / a.dpgrid)) + 1
    ncall = npts + a.nmin * 30 + 16
    print(f"per eigensolve: {small:.2f} s for continuation steps; ~{ncall} eigensolves per task -> "
          f"~{small * ncall / 60:.1f} min per task (dpgrid {a.dpgrid}; larger steps take longer); retries "
          f"{st['retries']}, misconvergences {st['misconv']}, fallbacks cupyx/CSR/CPU "
          f"{st['cupyx_fallbacks']}/{st['csr_fallbacks']}/{st['cpu_fallbacks']}")
    print("check: p = 0 ground state = vacuum, E0 = 0" + ("  -> OK" if abs(w0[0]) < 1e-9 and phot[0] == 0 else
                                                           "  -> WRONG"))
    raise SystemExit(0)

check_dense()
cp.get_default_memory_pool().free_all_blocks()
npts = int(round(pmax / a.dpgrid)) + 1
pg = np.linspace(0.0, pmax, npts)
Eg = np.zeros((npts, a.k))
g = np.zeros(npts)
cand, prevX = {}, None
for i, p in enumerate(pg):
    w = spectrum(float(p))
    Eg[i] = w[: a.k]
    g[i] = w[1] - w[0]
    Xi = st["X"].copy()
    if i >= 1:
        j = i - 1
        if (j == 0 or g[j] <= g[j - 1]) and g[j] <= g[i]:
            cand[j] = prevX
            if len(cand) > 2 * a.nmin:
                del cand[max(cand, key=lambda q: g[q])]
    prevX = Xi
if g[-1] <= g[-2]:
    cand[npts - 1] = prevX
if abs(Eg[0, 0]) > 1e-9:
    raise RuntimeError(f"inst {a.inst}: E0(p = 0) = {Eg[0, 0]} is not the vacuum energy 0")

loc = sorted(cand, key=lambda q: g[q])[: a.nmin]
minima, mvec = [], {}
for i in loc:
    lo, hi = pg[max(i - 1, 0)], pg[min(i + 1, npts - 1)]
    st["X"] = cand[i]
    r = minimize_scalar(gap, bounds=(lo, hi), method="bounded", options=dict(xatol=a.xatol))
    pm, gmin = float(r.x), float(r.fun)
    mvec[i] = st["X"].copy()
    if g[i] < gmin:
        pm, gmin = float(pg[i]), float(g[i])
        mvec[i] = cand[i]
    minima.append(dict(grid_index=int(i), p=pm, gap=gmin))
minima.sort(key=lambda q: q["gap"])
pstar, dmin = minima[0]["p"], minima[0]["gap"]
edge = int(minima[0]["grid_index"] in (0, npts - 1))

st["X"] = mvec[minima[0]["grid_index"]]
lz_pts, v2s = [], []
for dp in (1e-3, 3e-3, 1e-2, 3e-2):
    for sgn in (-1.0, 1.0):
        pp = pstar + sgn * dp
        if 0.0 < pp <= pmax:
            gg = gap(pp)
            lz_pts.append((float(pp), gg))
            if gg > dmin:
                v2s.append((gg ** 2 - dmin ** 2) / dp ** 2)
v = float(np.sqrt(np.median(v2s))) if v2s else float("nan")
x_lz = float(np.pi * dmin ** 2 / (2.0 * v * a.pf)) if v2s and v > 0 else float("nan")
mem = cp.get_default_memory_pool().total_bytes() / 2 ** 30

out = dict(tag=a.tag, kind="gap", version=3, engine="gpu-kernel", code="gap_scan_gpu v6", dense_kernel=DENSE["kernel"], set=a.set, inst=a.inst, n=n, family=a.family, h=h,
           K=K, cutoff=a.cutoff, pf=a.pf, pmax=pmax, dpgrid=a.dpgrid, k=a.k, guard=a.guard, restol=a.restol,
           sector_dim=Ns, p_grid=pg.tolist(), E_grid=Eg.tolist(), gap_grid=g.tolist(), minima=minima, Delta_min=dmin,
           p_star=pstar, edge=edge, lz_points=lz_pts, v_LZ=v, x_LZ=x_lz, gap_at_pmax=float(g[-1]), top2=st["top2"],
           refine_evals=evals, eigsolves=st["calls"], matvecs=st["matvecs"], retries=st["retries"],
           misconv=st["misconv"], lean_iters=st["lean_iters"], cupyx_fallbacks=st["cupyx_fallbacks"],
           csr_fallbacks=st["csr_fallbacks"], cpu_fallbacks=st["cpu_fallbacks"],
           build_s=t_build, gpu_mem_gb=mem, wall_s=time.time() - t0)
kc.write_json_atomic(a.out, out)
print(f"{a.set}#{a.inst} {a.family} h={h:g} cut={a.cutoff} pmax={pmax:g}: Delta_min={dmin:.3e} at p*={pstar:.4f} "
      f"(edge {edge}) x_LZ={x_lz:.3e} top2={st['top2']:.1e} solves={st['calls']} fb={st['cupyx_fallbacks']}/"
      f"{st['csr_fallbacks']}/{st['cpu_fallbacks']} {out['wall_s']:.0f}s")
