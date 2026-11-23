"""MSBQA 1c re-check, step 1: do the classical configuration crossings disappear when the detuning is uniform?

usage (from msbqa_phase1/; 3 processes, about 1-2 minutes):
  python code/branches_uniform.py --inst-dir instances --run run_v4
  quick test (first 50 instances, a few seconds):  ... --ninst 50 --out run_v4/branches_uniform_test
Writes RUN/branches_uniform/branches_weyl_{kpo,mean,max}.csv and report_branches_uniform.txt.

Hypothesis (doc MSBQA_1c消融复核与下一步, sec. 2): with Goto's per-spin detuning Delta_i = xi0 sum_j |J_ij| the
cat amplitudes x_i^2 ~ (p - Delta_i)/K differ between spins, so at intermediate p the configuration energies are a
weighted Ising energy and a wrong configuration can be the lowest branch (the 54 'energy crossings' of
branches.py, weyl family). With a uniform detuning the amplitudes are (nearly) equal and the crossings should
disappear. Decision: crossings 54 -> ~0 under 'mean' and 'max': hypothesis holds, go to the quantum run with a
uniform detuning; crossings remain: hypothesis is wrong, stop.

Same method as branches.py (weyl family, classical detuning Dc = detuning, h-independent): the 2^(n-1) branches
(local minima of V with sign(x) = s, s_0 = +1) are followed by damped Newton from p = pf down to 0, all instances
and configurations at once. V(x; p) = sum_i [(Dc_i - p) x_i^2 + (K/2) x_i^4] - xi0 sum_{i != j} J_ij x_i x_j.
Detuning modes:
  kpo    Delta_i per spin (Goto Eq. 7)             -> must reproduce RUN/branches/branches_weyl.csv (checked)
  mean   uniform Dbar = mean_i Delta_i             (diag - xi0 J not PSD on some instances: origin unstable at p = 0)
  max    uniform max_i Delta_i                     (PSD on every instance by Gershgorin; the choice for the quantum run)
For a uniform detuning the first bifurcating mode is the top eigenvector of J; trapU = 1 if its sign pattern is
not a ground state (trapK = the old 'trap' column, defined with the per-spin detuning).

Extra CSV columns (besides those of branches.py):
  trapK, trapU, psd       old trap flag; uniform-detuning trap flag; diag(detuning) - xi0 J positive semidefinite
  lam1, lam2              of diag(detuning) - xi0 J for this mode (first / second linear instability)
  cv_c                    amplitude spread std(x_i^2)/mean(x_i^2) of the ground branch at p_c
  cv_prev                 the same for the branch it replaces at p_c (crossing instances only)
  cv_mid                  the same for the lowest branch at p = lam1 + 1 (where the crossings sit, median p_c - lam1 ~ 1)
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np

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
ap.add_argument("--modes", default="kpo,mean,max")
a = ap.parse_args()
MODES = [m.strip() for m in a.modes.split(",") if m.strip()]
for m in MODES:
    if m not in ("kpo", "mean", "max"):
        raise SystemExit(f"unknown mode {m}")
OUT = a.out or os.path.join(a.run, "branches_uniform")

d = np.load(os.path.join(a.inst_dir, f"{a.set}.npz"))
Jall, Dall = d["J"].astype(np.float64), d["Delta"].astype(np.float64)
xi0 =float(json.loads(str(d["meta"]))["xi0"])
if a.ninst:
    Jall, Dall = Jall[: a.ninst], Dall[: a.ninst]
I, n = Dall.shape
K = a.K
info = np.genfromtxt(os.path.join(a.inst_dir, f"{a.set}_info.csv"), delimiter=",", names=True)[:I]
S = kc.all_spins(n).astype(np.float64)
S = S[S[:, 0] > 0]                                            # Z2 representatives, s_0 = +1
C = S.shape[0]
cfg_b = kc.pattern_index(S)
gidx = np.array([int(np.flatnonzero(cfg_b == int(info["gs_b"][i]))[0]) for i in range(I)])
GM = np.array([kc.ground_mask(kc.ising_energies(Jall[i])) for i in range(I)])   # (I, 2^n) ground patterns


def detuning(mode):
    if mode == "kpo":
        return Dall.copy()
    if mode == "mean":
        return np.repeat(Dall.mean(1, keepdims=True), n, 1)
    return np.repeat(Dall.max(1, keepdims=True), n, 1)


def linear(Dm):
    """lam1, lam2, sign(v1) configuration index, v1-is-wrong flag, PSD flag of diag(Dm) - xi0 J."""
    lam1, lam2 = np.zeros(I), np.zeros(I)
    cfg = np.zeros(I, dtype=int)
    wrong = np.zeros(I, dtype=bool)
    for i in range(I):
        w, U = np.linalg.eigh(np.diag(Dm[i]) - xi0 * Jall[i])
        v1 = U[:, 0] * (1.0 if U[0, 0] >= 0 else -1.0)
        sv = np.where(v1 >= 0, 1.0, -1.0)
        cfg[i] = int(np.flatnonzero((S == sv).all(1))[0])
        wrong[i] = not GM[i][int(kc.pattern_index(sv))]
        lam1[i], lam2[i] = w[0], w[1]
    return lam1, lam2, cfg, wrong, lam1 >= -1e-12


def follow(Dm):
    """Branches for classical detuning Dm (I, n); as branches.follow. Returns pgrid (ascending), E, XSQ, CV (P, I, C)."""
    Dc = Dm[:, None, :] * np.ones((1, C, 1))                  # (I, C, n)
    Jb = Jall[:, None, :, :]
    Sb = np.broadcast_to(S[None], (I, C, n))
    pgrid = np.round(np.arange(a.pf, -0.5 * a.dp, -a.dp), 10)
    pgrid[-1] = max(pgrid[-1], 0.0)
    P = len(pgrid)
    E = np.full((P, I, C), np.nan)
    XSQ = np.full((P, I, C), np.nan)
    CV = np.full((P, I, C), np.nan)
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
        X2 = X ** 2
        E[k] = np.where(alive, V, np.nan)
        XSQ[k] = np.where(alive, X2.mean(-1), np.nan)
        CV[k] = np.where(alive, X2.std(-1) / np.maximum(X2.mean(-1), 1e-300), np.nan)
        if not alive.any():
            break
    return pgrid[::-1].copy(), E[::-1].copy(), XSQ[::-1].copy(), CV[::-1].copy()


def analyse(mode):
    t0 = time.time()
    Dm = detuning(mode)
    lam1, lam2, v1cfg, _, psd = linear(Dm)
    pg, E, XSQ, CV = follow(Dm)
    rows = []
    for i in range(I):
        Ei = E[:, i, :]
        ex = np.isfinite(Ei)
        pex = np.array([pg[ex[:, c]].min() if ex[:, c].any() else np.nan for c in range(C)])
        g = gidx[i]
        anyb = ex.any(1)
        r = dict(inst=i, trapK=int(info["trap"][i]), trapU=int(TRAPU[i]), psd=int(psd[i]),
                 lam1=float(lam1[i]), lam2=float(lam2[i]))
        if not anyb.any():
            rows.append(r)
            continue
        first = int(np.nanargmin(pex))
        low = np.full(len(pg), -1)
        low[anyb] = np.nanargmin(np.where(ex[anyb], Ei[anyb], np.inf), 1)
        r.update(p_first=float(pex[first]), first_cfg=int(cfg_b[first]), first_is_ground=int(first == g),
                 first_is_v1cfg=int(first == v1cfg[i]), gs_lowest_at_pf=int(low[-1] == g))
        idx = np.flatnonzero(anyb)
        bad = idx[low[idx] != g]
        if low[-1] != g:
            r.update(cross=-1)
        elif bad.size == 0:
            r.update(cross=0, p_c=float(pex[g]), ham_c=0, xsq_c=float(XSQ[idx[0], i, g]),
                     cv_c=float(CV[idx[0], i, g]))
        else:
            kc_ = bad.max() + 1
            prev = low[bad.max()]
            h_ = int(np.sum(S[prev] != S[g]))
            r.update(cross=1, cross_kind=1 if ex[bad.max(), g] else 2, p_c=float(pg[kc_]), ham_c=min(h_, n - h_),
                     xsq_c=float(XSQ[kc_, i, g]), cv_c=float(CV[kc_, i, g]), cv_prev=float(CV[bad.max(), i, prev]))
        km = int(np.argmin(np.abs(pg - (lam1[i] + 1.0))))
        if anyb[km]:
            r.update(cv_mid=float(CV[km, i, low[km]]))
        srt = np.sort(np.where(ex, Ei, np.inf), 1)
        gap = np.where(ex.sum(1) >= 2, srt[:, 1] - srt[:, 0], np.inf)
        if np.isfinite(gap).any():
            kk = int(np.argmin(gap))
            r.update(p_gmin_cl=float(pg[kk]), gmin_cl=float(gap[kk]))
        rows.append(r)
    return mode, rows, time.time() - t0


# trapU: sign of the top eigenvector of J (= first mode for any uniform detuning) is not a ground state
_, _, _, TRAPU, _ = linear(np.repeat(Dall.mean(1, keepdims=True), n, 1))
COLS = ["inst", "trapK", "trapU", "psd", "lam1", "lam2", "p_first", "first_cfg", "first_is_ground", "first_is_v1cfg",
        "gs_lowest_at_pf", "cross", "cross_kind", "p_c", "ham_c", "xsq_c", "cv_c", "cv_prev", "cv_mid",
        "p_gmin_cl", "gmin_cl"]


def col(rows, k):
    return np.array([float(r[k]) if r.get(k, "") != "" else np.nan for r in rows], dtype=float)


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return {int(r["inst"]): r for r in csv.DictReader(fh)}


def fv(r, k):
    v = r.get(k, "") if r else ""
    return float(v) if v not in ("", None) else np.nan


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    t0 = time.time()
    with Pool(len(MODES)) as pool:
        res = {m: (rows, dt) for m, rows, dt in pool.map(analyse, MODES)}
    for m, (rows, _) in res.items():
        with open(os.path.join(OUT, f"branches_weyl_{m}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in COLS})

    trK = col(next(iter(res.values()))[0], "trapK") == 1
    trU = TRAPU.astype(bool)
    say(f"MSBQA 1c re-check: configuration branches with per-spin vs uniform detuning, weyl family "
        f"({I} instances, n = {n}, dp = {a.dp}, wall {time.time() - t0:.0f} s)")
    say(f"trapK (old trap, per-spin detuning) {trK.sum()}, trapU (top eigenvector of J wrong) {trU.sum()}, "
        f"both {(trK & trU).sum()}")

    # -------- check: mode kpo must reproduce branches.py (RUN/branches/branches_weyl.csv)
    if "kpo" in res:
        old = read_csv(os.path.join(a.run, "branches", "branches_weyl.csv"))
        if old is None:
            say("check: RUN/branches/branches_weyl.csv not found, kpo mode not compared")
        else:
            rows = res["kpo"][0]
            ii = [r["inst"] for r in rows if r["inst"] in old]
            dc = sum(int(float(old[i]["cross"])) != int(rows[i]["cross"]) for i in ii
                     if old[i]["cross"] != "" and "cross" in rows[i])
            dp_ = [abs(fv(old[i], "p_c") - rows[i]["p_c"]) for i in ii if "p_c" in rows[i]]
            dp_ = np.array([x for x in dp_ if np.isfinite(x)])
            say(f"check: kpo mode vs branches.py on {len(ii)} instances: cross differs on {dc} (should be 0), "
                f"max |p_c diff| {dp_.max() if dp_.size else float('nan'):.2e} (should be 0)")
            tk = col(rows, "trapK") == 1
            say(f"check: trapK equals sign(v1) wrong for the kpo detuning: "
                f"{int(np.sum(linear(Dall)[3] != tk))} mismatches (should be 0)")

    # -------- per mode summary
    say("\n   mode   PSD    gs lowest@pf  first=ground  first=v1   cross: all   trapK   trapU   other   "
        "kind1/kind2   median cv_mid (cross / no)")
    for m, (rows, dt) in res.items():
        cr = col(rows, "cross")
        cvm = col(rows, "cv_mid")
        oth = ~trK & ~trU
        say(f"   {m:5s} {int(col(rows, 'psd').sum()):5d}   {np.mean(col(rows, 'gs_lowest_at_pf') == 1) * 100:9.1f} %"
            f"   {np.mean(col(rows, 'first_is_ground') == 1) * 100:9.1f} %"
            f"   {np.mean(col(rows, 'first_is_v1cfg') == 1) * 100:6.1f} %"
            f"   {np.sum(cr == 1):6d}  {np.sum(cr[trK] == 1):6d}  {np.sum(cr[trU] == 1):6d}  {np.sum(cr[oth] == 1):6d}"
            f"   {np.sum(col(rows, 'cross_kind') == 1):5d}/{np.sum(col(rows, 'cross_kind') == 2):<5d}"
            f"   {np.nanmedian(cvm[cr == 1]) if (cr == 1).any() else float('nan'):.3f} / "
            f"{np.nanmedian(cvm[cr == 0]):.3f}   ({dt:.0f} s)")
        if (cr == -1).any():
            say(f"          ground configuration never the lowest at pf: {np.sum(cr == -1)} instances "
                f"{[r['inst'] for r in rows if r.get('cross') == -1][:10]}")

    # -------- what happens to the kpo crossings under a uniform detuning
    if "kpo" in res:
        crk = col(res["kpo"][0], "cross") == 1
        say(f"\n   the {crk.sum()} kpo-crossing instances under the other modes (cross = 1 / 0 / -1):")
        for m in [x for x in res if x != "kpo"]:
            cm = col(res[m][0], "cross")
            say(f"     {m:5s}: {np.sum(cm[crk] == 1)} / {np.sum(cm[crk] == 0)} / {np.sum(cm[crk] == -1)};   "
                f"new crossings on instances without a kpo crossing: {np.sum(cm[~crk] == 1)}")
        rk = res["kpo"][0]
        cvc, cvp = col(rk, "cv_c"), col(rk, "cv_prev")
        say(f"   kpo amplitude spread std(x^2)/mean(x^2) at p_c: ground branch median {np.nanmedian(cvc[crk]):.3f}, "
            f"replaced branch {np.nanmedian(cvp[crk]):.3f};  no-crossing instances (ground branch at birth) "
            f"{np.nanmedian(cvc[col(rk, 'cross') == 0]):.3f}")

    # -------- join: ablation (500 instances) and quantum (1b, 300 instances)
    AB = read_csv(os.path.join(a.run, "phase1c", "ablate_per_instance.csv"))
    QB = read_csv(os.path.join(a.run, "phase1b", "per_instance_1b.csv"))
    if a.set == "G4" and AB is not None:
        say("\n   ablation success (R = 1000), by crossing status of the matching detuning:")
        for m, cfgs in (("kpo", ("L4", "bSB* -D")), ("mean", ("L4 +D", "bSB*"))):
            if m not in res:
                continue
            rows = res[m][0]
            ii = [r["inst"] for r in rows if r["inst"] in AB and r.get("cross") in (0, 1)]
            cr = np.array([res[m][0][i]["cross"] for i in ii])
            for c in cfgs:
                v = np.array([fv(AB[i], c) for i in ii])
                say(f"     {m:5s} {c:8s}: crossing {np.nanmean(v[cr == 1]) if (cr == 1).any() else float('nan'):.3f} "
                    f"({np.sum(cr == 1)})   no crossing {np.nanmean(v[cr == 0]):.3f} ({np.sum(cr == 0)})")
    if a.set == "G4" and QB is not None and "kpo" in res:
        rows = res["kpo"][0]
        ii = [r["inst"] for r in rows if r["inst"] in QB and r.get("cross") == 1]
        if ii:
            say(f"\n   kpo-crossing test instances with quantum data ({len(ii)}): Q_weyl {np.nanmean([fv(QB[i], 'Q_weyl') for i in ii]):.3f}, "
                f"Dmin_Wg median {np.nanmedian([fv(QB[i], 'Dmin_Wg') for i in ii]):.3e}  "
                f"(prediction for the uniform-detuning quantum run: Dmin rises to ~1)")
            for m in [x for x in ("mean", "max") if x in res]:
                say(f"     {m:5s}: of these, still crossing {sum(res[m][0][i].get('cross') == 1 for i in ii)}")

    # -------- verdict
    if "kpo" in res:
        n0 = int(np.sum(col(res["kpo"][0], "cross") == 1))
        parts = [f"{m} {int(np.sum(col(res[m][0], 'cross') == 1))}" for m in res if m != "kpo"]
        say(f"\nVERDICT: crossings kpo {n0} -> " + ", ".join(parts)
            + ".  ~0: amplitude-heterogeneity hypothesis holds -> quantum run with uniform detuning (max).  "
              "Still many: hypothesis wrong, stop.")

    with open(os.path.join(OUT, "report_branches_uniform.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"written {OUT}/branches_weyl_*.csv and {OUT}/report_branches_uniform.txt")
