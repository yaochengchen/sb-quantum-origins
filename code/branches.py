"""MSBQA phase 1c, step 1: classical configuration branches of the KPO network and the configuration-crossing test.

usage (from msbqa_phase1/; one process, about a minute):
  python code/branches.py --inst-dir instances --run run_v4
Writes RUN/branches/branches_weyl.csv, branches_goto_h1.csv and RUN/branches/report_branches.txt.

Hypothesis (1b results, sec. 0.6): on eigenvector-trap instances the classical trajectory commits near lam1 to a
non-ground configuration; at a larger pump p_c the ground configuration becomes the lowest classical branch, the
classical trajectory cannot switch, while the quantum state follows the ground state through an avoided crossing
at p*. Test: p* (quantum, gap_scan.py) ~ p_c (classical, here), the minimum gap is small where such a crossing
exists, and ln Delta_min scales like a tunnelling exponent (~ 1/h).

Classical branches. With z = sqrt(h) alpha the quantum H (detuning Delta_q, Kerr K h) has the classical energy
H_cl / h; for real z = x (the stationary cat amplitudes; p > 0 selects the x quadrature)
    V(x; p) = sum_i [ (Dc_i - p) x_i^2 + (K/2) x_i^4 ] - xi0 sum_{i != j} J_ij x_i x_j,
    Dc = Delta_q - K h  (Weyl detuning, as kpo_common.classical_run):  weyl family -> Delta (h-independent),
                                                                       goto family at h = 1 -> Delta - K.
A configuration s (sign pattern, s_0 = +1 representative of the Z2 pair) has a branch at p if V has a local
minimum (Hessian positive definite) with sign(x) = s. The 2^(n-1) branches are followed by damped Newton from
p = pf (where every orthant has a minimum) down to p = 0 in steps --dp, all instances and configurations at once;
a branch dies at the first p where Newton does not converge to a sign-consistent minimum (saddle-node), and is
not followed further. E_s(p) = V(x_s(p); p) in classical units (quantum energy ~ E_s / h).

Per instance (CSV columns):
  p_first      smallest p at which any branch exists (first classical bifurcation into a configuration)
  first_cfg / first_is_ground / first_is_trapcfg   the configuration of that first branch; trapcfg = sign(v1)
  gs_lowest_at_pf   1 if the ground configuration is the lowest branch at p = pf
  cross        1 if some other configuration is the lowest branch at some p and the ground one takes over at p_c
               (0: the ground branch is the lowest wherever a branch exists; -1: not the lowest even at pf)
  cross_kind   1 = energy crossing (the ground branch already existed below p_c), 2 = the ground branch is born
               at p_c (saddle-node) while another configuration is the lowest
  p_c          smallest p from which the ground configuration is the lowest branch up to pf (= p_exist of the
               ground branch if cross = 0)
  ham_c        Hamming distance (mod Z2) between the ground configuration and the one it replaces at p_c
  xsq_c        mean x_i^2 of the ground branch at p_c (classical; photon number per mode ~ xsq_c / h)
  p_gmin_cl, gmin_cl   position and value of the minimum over p of E_(2nd lowest) - E_(lowest) among existing
               branches (classical analogue of the gap; divide by h for quantum units)
Report: joined with RUN/phase1b/per_instance_1b.csv (quantum p*, Delta_min at h = 1, 0.7, 0.5, both families;
classical success at eps*), see the printed sections.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

import kpo_common as kc

ap = argparse.ArgumentParser()
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", default="G4")
ap.add_argument("--run", default="run_v4")
ap.add_argument("--out", default="")
ap.add_argument("--pf", type=float, default=7.0)
ap.add_argument("--K", type=float, default=1.0)
ap.add_argument("--dp", type=float, default=0.005)
ap.add_argument("--ninst", type=int, default=0, help="first N instances only (0 = all)")
a = ap.parse_args()
OUT = a.out or os.path.join(a.run, "branches")
os.makedirs(OUT, exist_ok=True)
L = []


def say(s=""):
    print(s)
    L.append(s)


d = np.load(os.path.join(a.inst_dir, f"{a.set}.npz"))
meta = json.loads(str(d["meta"]))
xi0 = float(meta["xi0"])
Jall, Dall = d["J"].astype(np.float64), d["Delta"].astype(np.float64)
if a.ninst:
    Jall, Dall = Jall[: a.ninst], Dall[: a.ninst]
I, n = Dall.shape
K = a.K
info = np.genfromtxt(os.path.join(a.inst_dir, f"{a.set}_info.csv"), delimiter=",", names=True)[:I]
S = kc.all_spins(n).astype(np.float64)
S = S[S[:, 0] > 0]                                            # Z2 representatives, s_0 = +1
C = S.shape[0]
cfg_b = kc.pattern_index(S)                                   # pattern index of each configuration
gidx = np.array([int(np.flatnonzero(cfg_b == int(info["gs_b"][i]))[0]) for i in range(I)])
trapcfg = np.zeros(I, dtype=int)
for i in range(I):
    w, U = np.linalg.eigh(np.diag(Dall[i]) - xi0 * Jall[i])
    v1 = U[:, 0] * (1.0 if U[0, 0] >= 0 else -1.0)
    sv = np.where(v1 >= 0, 1.0, -1.0)
    trapcfg[i] = int(np.flatnonzero((S == sv).all(1))[0])


def follow(shift):
    """Branches for classical detuning Dc = Delta - shift. Returns pgrid (ascending), E (P, I, C), XSQ (P, I, C)."""
    Dc = (Dall - shift)[:, None, :] * np.ones((1, C, 1))     # (I, C, n)
    Jb = Jall[:, None, :, :]                                  # (I, 1, n, n)
    Sb = np.broadcast_to(S[None], (I, C, n))
    pgrid = np.round(np.arange(a.pf, -0.5 * a.dp, -a.dp), 10)
    pgrid[-1] = max(pgrid[-1], 0.0)
    P = len(pgrid)
    E = np.full((P, I, C), np.nan)
    XSQ = np.full((P, I, C), np.nan)
    alive = np.ones((I, C), dtype=bool)
    eye = np.broadcast_to(np.eye(n), (I, C, n, n))
    X = Sb * np.sqrt(np.maximum(a.pf - Dc, 0.5) / K)

    def grad_hess(X, p):
        JX = np.einsum("ijk,ick->icj", Jall, X)
        g = 2.0 * (Dc - p) * X + 2.0 * K * X ** 3 - 2.0 * xi0 * JX
        H = -2.0 * xi0 * np.broadcast_to(Jb, (I, C, n, n)).copy()
        H[..., np.arange(n), np.arange(n)] += 2.0 * (Dc - p) + 6.0 * K * X ** 2
        return g, H

    def newton(X, p, iters):
        for _ in range(iters):
            g, H = grad_hess(X, p)
            Hs = np.where(alive[..., None, None], H, eye)
            gs = np.where(alive[..., None], g, 0.0)
            try:
                dx = np.linalg.solve(Hs, gs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                dx = np.einsum("icjk,ick->icj", np.linalg.pinv(Hs), gs)
            step = np.max(np.abs(dx), -1, keepdims=True)
            dx = dx * np.minimum(1.0, 0.5 / np.maximum(step, 1e-300))
            X = X - dx
        return X

    for k, p in enumerate(pgrid):
        X = newton(X, p, 60 if k == 0 else 8)
        g, H = grad_hess(X, p)
        ok = np.max(np.abs(g), -1) < 1e-9 * np.maximum(1.0, np.max(np.abs(X), -1) ** 3)
        ok &= np.all(Sb * X > 1e-7, -1)
        ok &= np.linalg.eigvalsh(np.where(alive[..., None, None], H, eye))[..., 0] > 1e-10
        alive &= ok
        V = np.sum((Dc - p) * X ** 2 + 0.5 * K * X ** 4, -1) - xi0 * np.einsum("icj,ijk,ick->ic", X, Jall, X)
        E[k] = np.where(alive, V, np.nan)
        XSQ[k] = np.where(alive, np.mean(X ** 2, -1), np.nan)
        if not alive.any():
            break
    return pgrid[::-1].copy(), E[::-1].copy(), XSQ[::-1].copy()


def analyse(pg, E, XSQ):
    rows = []
    for i in range(I):
        Ei = E[:, i, :]                                       # (P, C), ascending p
        ex = np.isfinite(Ei)
        pex = np.array([pg[ex[:, c]].min() if ex[:, c].any() else np.nan for c in range(C)])
        g = gidx[i]
        anyb = ex.any(1)
        r = dict(inst=i, trap=int(info["trap"][i]), lam1=float(info["lam1"][i]), lam2=float(info["lam2"][i]))
        if not anyb.any():
            rows.append(r)
            continue
        first = int(np.nanargmin(pex))
        low = np.full(len(pg), -1)
        low[anyb] = np.nanargmin(np.where(ex[anyb], Ei[anyb], np.inf), 1)
        r.update(p_first=float(pex[first]), first_cfg=int(cfg_b[first]), first_is_ground=int(first == g),
                 first_is_trapcfg=int(first == trapcfg[i]), gs_lowest_at_pf=int(low[-1] == g))
        idx = np.flatnonzero(anyb)
        bad = idx[low[idx] != g]
        if low[-1] != g:
            r.update(cross=-1, p_c=np.nan, ham_c=np.nan, xsq_c=np.nan)
        elif bad.size == 0:
            r.update(cross=0, p_c=float(pex[g]), ham_c=0, xsq_c=float(XSQ[idx[0], i, g]))
        else:
            kc_ = bad.max() + 1                               # first grid point from which g is the lowest
            prev = low[bad.max()]
            h_ = int(np.sum(S[prev] != S[g]))
            r.update(cross=1, cross_kind=1 if ex[bad.max(), g] else 2, p_c=float(pg[kc_]), ham_c=min(h_, n - h_),
                     xsq_c=float(XSQ[kc_, i, g]))
        srt = np.sort(np.where(ex, Ei, np.inf), 1)
        gap = np.where(ex.sum(1) >= 2, srt[:, 1] - srt[:, 0], np.inf)
        if np.isfinite(gap).any():
            kk = int(np.argmin(gap))
            r.update(p_gmin_cl=float(pg[kk]), gmin_cl=float(gap[kk]))
        rows.append(r)
    return rows


COLS = ["inst", "trap", "lam1", "lam2", "p_first", "first_cfg", "first_is_ground", "first_is_trapcfg",
        "gs_lowest_at_pf", "cross", "cross_kind", "p_c", "ham_c", "xsq_c", "p_gmin_cl", "gmin_cl"]
FAMS = {"weyl": 0.0, "goto_h1": 1.0 * K}
res = {}
for fam, shift in FAMS.items():
    pg, E, XSQ = follow(shift)
    rows = analyse(pg, E, XSQ)
    res[fam] = rows
    with open(os.path.join(OUT, f"branches_{fam}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def col(rows, k):
    return np.array([r.get(k, np.nan) if r.get(k, "") != "" else np.nan for r in rows], dtype=float)


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def f3(v):
    return "  -  " if v is None or not np.isfinite(v) else f"{v:+.3f}"


say(f"MSBQA phase 1c: classical configuration branches ({I} instances, n = {n}, dp = {a.dp})")
for fam, rows in res.items():
    tr = col(rows, "trap") == 1
    cr = col(rows, "cross")
    say(f"\n== {fam} (classical detuning Delta - {FAMS[fam]:g})")
    say(f"   ground configuration lowest at p = pf: {np.mean(col(rows, 'gs_lowest_at_pf') == 1) * 100:.1f} %;  "
        f"first branch = ground: {np.mean(col(rows, 'first_is_ground') == 1) * 100:.1f} %;  "
        f"first branch = sign(v1): {np.mean(col(rows, 'first_is_trapcfg') == 1) * 100:.1f} %")
    say(f"   crossing (another configuration lowest first, ground takes over at p_c): {np.sum(cr == 1)} of {I} "
        f"({np.mean(cr == 1) * 100:.1f} %); among trap instances {np.mean(cr[tr] == 1) * 100:.1f} %, "
        f"non-trap {np.mean(cr[~tr] == 1) * 100:.1f} %;  ground never lowest at pf: {np.sum(cr == -1)};  "
        f"kind: energy crossing {np.sum(col(rows, 'cross_kind') == 1)}, ground branch born at p_c "
        f"{np.sum(col(rows, 'cross_kind') == 2)}")
    pc, l1, l2 = col(rows, "p_c"), col(rows, "lam1"), col(rows, "lam2")
    m = cr == 1
    if m.any():
        say(f"   p_c on crossing instances: median p_c - lam1 {np.nanmedian(pc[m] - l1[m]):.3f}, "
            f"p_c - lam2 {np.nanmedian(pc[m] - l2[m]):.3f};  Hamming(mod Z2) at p_c: "
            + ", ".join(f"{h}: {np.sum(col(rows, 'ham_c')[m] == h)}" for h in range(1, n // 2 + 1)))

# ------------------------------------------------------------------ join with the quantum gaps (phase 1b CSV)
pi = os.path.join(a.run, "phase1b", "per_instance_1b.csv")
if a.set != "G4" or not os.path.exists(pi):
    say(f"\n(quantum comparison skipped: it uses {pi}, which belongs to the set G4)")
else:
    with open(pi) as fh:
        Q = {int(r["inst"]): r for r in csv.DictReader(fh)}

    def qcol(name, insts):
        out = []
        for i in insts:
            v = Q.get(i, {}).get(name, "")
            out.append(float(v) if v not in ("", None) else np.nan)
        return np.array(out)

    qkeys = list(next(iter(Q.values())).keys())
    l2g = next((k for k in qkeys if k.startswith("L2g eps*")), None)
    l4 = next((k for k in qkeys if k.startswith("L4 eps*")), None)
    PAIRS = [("weyl", "Wg", 1.0, "Q_weyl"), ("weyl", "Wg07", 0.7, None), ("weyl", "Wg05", 0.5, None),
             ("goto_h1", "Gg", 1.0, "Q_goto")]
    for fam, gt, h, qn in PAIRS:
        rows = res[fam]
        insts = [r["inst"] for r in rows if r["inst"] in Q]
        rr = [rows[i] for i in insts]
        ps, dm = qcol(f"pstar_{gt}", insts), qcol(f"Dmin_{gt}", insts)
        ok = np.isfinite(ps)
        if not ok.any():
            continue
        cr, pc, pgm = col(rr, "cross"), col(rr, "p_c"), col(rr, "p_gmin_cl")
        gcl, ham, xsq = col(rr, "gmin_cl"), col(rr, "ham_c"), col(rr, "xsq_c")
        tr = col(rr, "trap") == 1
        say(f"\n== quantum {gt} ({fam}, h = {h:g}) vs classical branches, {int(ok.sum())} instances")
        m = ok & (cr == 1)
        if m.any():
            dd = ps[m] - pc[m]
            say(f"   crossing instances ({int(m.sum())}): p* - p_c median {np.median(dd):+.3f}, |p* - p_c| < 0.1: "
                f"{np.mean(np.abs(dd) < 0.1) * 100:.0f} %, < 0.3: {np.mean(np.abs(dd) < 0.3) * 100:.0f} %;  "
                f"Spearman(p*, p_c) {f3(sp(ps[m], pc[m]))}")
            say(f"   Delta_min: crossing median {np.median(dm[m]):.3e} vs no crossing "
                f"{np.median(dm[ok & (cr == 0)]) if (ok & (cr == 0)).any() else float('nan'):.3e};  "
                f"Spearman(ln Delta_min, ham_c * xsq_c / h) on crossing {f3(sp(np.log(dm[m]), ham[m] * xsq[m] / h))}")
        dd = ps[ok] - pgm[ok]
        say(f"   all: p* - p_gmin_cl (classical gap minimum) median {np.nanmedian(dd):+.3f}, |.| < 0.1: "
            f"{np.mean(np.abs(dd) < 0.1) * 100:.0f} %;  Spearman(p*, p_gmin_cl) {f3(sp(ps[ok], pgm[ok]))};  "
            f"Spearman(Delta_min, gmin_cl / h) {f3(sp(dm[ok], gcl[ok] / h))}")
        if qn:
            qq = qcol(qn, insts)
            for name, k in (("L2g eps*", l2g), ("L4 eps*", l4)):
                if k is None:
                    continue
                cl = qcol(k, insts)
                say(f"   {name:8s} mean success   crossing {np.nanmean(cl[cr == 1]):.3f}   no crossing "
                    f"{np.nanmean(cl[cr == 0]):.3f}   |   trap {np.nanmean(cl[tr]):.3f}   non-trap {np.nanmean(cl[~tr]):.3f};"
                    f"  Spearman(Q - {name}, cross) {f3(sp(qq - cl, cr == 1))}, (Q - {name}, trap) {f3(sp(qq - cl, tr))}")

    # tunnelling scaling across h (weyl family; instances with Delta_min at h = 1, 0.7, 0.5)
    rows = res["weyl"]
    insts = [r["inst"] for r in rows if r["inst"] in Q]
    D = np.stack([qcol(f"Dmin_{t}", insts) for t in ("Wg", "Wg07", "Wg05")], 1)
    hs = np.array([1.0, 0.7, 0.5])
    ok = np.all(np.isfinite(D) & (D > 0), 1)
    if ok.sum() >= 3:
        lnD = np.log(D[ok])
        sl_inv, r2_inv, r2_pow = [], [], []
        for y in lnD:
            for xv, store in ((1.0 / hs, "inv"), (np.log(hs), "pow")):
                A_ = np.vstack([xv, np.ones(3)]).T
                coef, res_, *_ = np.linalg.lstsq(A_, y, rcond=None)
                ss = np.sum((y - y.mean()) ** 2)
                r2 = 1.0 - (res_[0] / ss if res_.size and ss > 0 else 0.0)
                if store == "inv":
                    sl_inv.append(-coef[0])
                    r2_inv.append(r2)
                else:
                    r2_pow.append(r2)
        sl_inv, r2_inv, r2_pow = map(np.array, (sl_inv, r2_inv, r2_pow))
        rr = [rows[i] for i in np.array(insts)[ok]]
        cr, ham, xsq = col(rr, "cross"), col(rr, "ham_c"), col(rr, "xsq_c")
        say(f"\n== tunnelling scaling, weyl, Delta_min at h = 1, 0.7, 0.5 ({int(ok.sum())} instances)")
        say(f"   fit ln Delta = c - S/h: median S {np.median(sl_inv):+.3f}, median R^2 {np.median(r2_inv):.3f};  "
            f"fit ln Delta = c + k ln h: median R^2 {np.median(r2_pow):.3f};  1/h fits better in "
            f"{np.mean(r2_inv > r2_pow) * 100:.0f} % (3 points per instance: weak test)")
        m = cr == 1
        if m.sum() >= 3:
            say(f"   crossing instances ({int(m.sum())}): median S {np.median(sl_inv[m]):+.3f} (no crossing "
                f"{np.median(sl_inv[cr == 0]) if (cr == 0).any() else float('nan'):+.3f});  "
                f"Spearman(S, ham_c * xsq_c) {f3(sp(sl_inv[m], ham[m] * xsq[m]))}")

with open(os.path.join(OUT, "report_branches.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"written {OUT}/branches_*.csv and {OUT}/report_branches.txt")
