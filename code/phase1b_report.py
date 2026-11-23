"""MSBQA phase 1b report: fair classical baseline vs quantum, bifurcation-delay test, quantum gap.
Works on whatever is finished (sections without data are skipped); can be run at any time.

usage (from msbqa_phase1/):  python code/phase1b_report.py --run run_v4 --inst-dir instances --out run_v4/phase1b
Writes OUT/report_1b.txt and OUT/per_instance_1b.csv (test instances).

Inputs in RUN/parts: NS_* (noise_scan.py), SBT_* (sb_tune.py), Gg_/Wg_/Wg07_/Wg05_/Wg035_* (gap_scan.py) and the
phase-1 results G0q, Wq, WqT200, WqT2000, Wq07, Wq05, Wq035 (quantum), G0c (classical, check of NS at eps = 1e-6),
Wc, Wc07 .. Wc0001 (TWA = layer L1 at the different h).
Test set = instances <= --test-hi (0-299, the instances with quantum results); training set = the others
(300-499 noise scan, 300-999 sb_tune). The noise amplitude eps* of each layer and the SB hyperparameters are chosen
on the training set only. TWA (L1 at the best h, lines Wc .. Wc0001, mostly only instances 0-99) is a reference
line, not part of the fair baselines / portfolio: its h is chosen on the test set, and its noise (std sqrt(h)/2)
is covered by the noise scan of L2g, which integrates the same equations.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", default="G4")
ap.add_argument("--out", default="")
ap.add_argument("--ns", default="NS")
ap.add_argument("--sbt", default="SBT")
ap.add_argument("--test-hi", type=int, default=299)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
OUT = a.out or os.path.join(a.run, "phase1b")
os.makedirs(OUT, exist_ok=True)
L = []


def say(s=""):
    print(s)
    L.append(s)


def load(tag):
    d = {}
    for p in glob.glob(os.path.join(a.run, "parts", f"{tag}_*.json")):
        try:
            with open(p) as fh:
                r = json.load(fh)
        except Exception:
            continue
        if r.get("tag") == tag:
            d[int(r["inst"])] = r
    return d


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def f3(v):
    return "   -  " if v is None or not np.isfinite(v) else f"{v:6.3f}"


def fe(v):
    return "    -    " if v is None or not np.isfinite(v) else f"{v:9.2e}"


info = np.genfromtxt(os.path.join(a.inst_dir, f"{a.set}_info.csv"), delimiter=",", names=True)
lam1, lam2, trap, igap = info["lam1"], info["lam2"], info["trap"].astype(bool), info["gap"]

QTAGS = ["G0q", "GqT200", "GqT2000", "Wq", "WqT200", "WqT2000", "Wq07", "Wq05", "Wq035"]
qP = {t: {i: r["P_succ"] for i, r in load(t).items()} for t in QTAGS}
G0c = load("G0c")
NS = load(a.ns)
SBT = load(a.sbt)
TWTAGS = ["Wc", "Wc07", "Wc05", "Wc035", "Wc025", "Wc010", "Wc005", "Wc002", "Wc0005", "Wc0001"]
TW = {t: load(t) for t in TWTAGS}
GTAGS = ["Gg", "Wg", "Wg07", "Wg05", "Wg035"]
GAP = {t: load(t) for t in GTAGS}

say(f"MSBQA phase 1b report from {a.run}")
say("finished: " + "  ".join(f"{t} {len(v)}" for t, v in [(a.ns, NS), (a.sbt, SBT)] + list(GAP.items())
                             + [(t, qP[t]) for t in QTAGS]))
say()

# ------------------------------------------------------------------------------------------------ A. check
eps_star, P_ns, ns_layers, ns_eps = {}, {}, [], []
if NS:
    r0 = next(iter(NS.values()))
    ns_layers, ns_eps = list(r0["layers"].keys()), list(r0["eps"])
    say("== A. check: noise scan at eps = 1e-6 against the phase-1 G0c results (same seeds, should be 0)")
    for l in ns_layers:
        diffs = [abs(e["P_succ"] - G0c[i]["layers"][l]["P_succ"])
                 for i, r in NS.items() if i in G0c and l in G0c[i]["layers"]
                 for e in r["layers"][l] if e["eps"] == 1e-6]
        say(f"   {l:4s} n = {len(diffs):4d}  max |NS - G0c| = {max(diffs) if diffs else float('nan'):.2e}")
    say()

    # -------------------------------------------------------------------------------------------- B. noise scan
    train = sorted(i for i in NS if i > a.test_hi)
    test = sorted(i for i in NS if i <= a.test_hi)
    say(f"== B. noise scan (T = {r0['T']:g}, R = {r0['R']}, dt = {r0['dt']}): mean P_succ, training "
        f"{len(train)} / test {len(test)} instances")
    for l in ns_layers:
        P_ns[l] = {i: np.array([e["P_succ"] for e in NS[i]["layers"][l]]) for i in NS}
    say("      eps    " + "".join(f"  {l:>4s} train  test " for l in ns_layers))
    for k, e in enumerate(ns_eps):
        row = f"   {e:8.0e}  "
        for l in ns_layers:
            tr = np.mean([P_ns[l][i][k] for i in train]) if train else float("nan")
            te = np.mean([P_ns[l][i][k] for i in test]) if test else float("nan")
            row += f"    {f3(tr)} {f3(te)}"
        say(row)
    for l in ns_layers:
        base = train if train else test
        mtr = np.array([np.mean([P_ns[l][i][k] for i in base]) for k in range(len(ns_eps))])
        eps_star[l] = int(np.argmax(mtr))
        te = np.mean([P_ns[l][i][eps_star[l]] for i in test]) if test else float("nan")
        say(f"   {l}: eps* = {ns_eps[eps_star[l]]:g} (chosen on {'training' if train else 'TEST (no training data)'}"
            f" set, mean {mtr[eps_star[l]]:.3f}); test mean at eps* {te:.3f}")
    say()

    # -------------------------------------------------------------------------------------------- C. delay test
    say(f"== C. bifurcation delay (departure = first max|x| > {r0['thr']:g}; eps <= 1e-3 clean, 1e-2 marginal)")
    small = [k for k, e in enumerate(ns_eps) if e <= 1e-2]
    for l in ns_layers:
        say(f"   {l}:   eps     med(p_dep - lam1)  frac departed after lam2   mean P_succ   trap: P_succ   "
            f"non-trap: P_succ")
        for k in small:
            d = [NS[i]["layers"][l][k]["dep"] for i in NS]
            ii = list(NS)
            mp = np.nanmedian([dd["p_dep_med"] - lam1[i] for dd, i in zip(d, ii)])
            fa = np.nanmean([dd["frac_after_lam2"] for dd in d])
            ps = np.mean([P_ns[l][i][k] for i in ii])
            pt = np.mean([P_ns[l][i][k] for i in ii if trap[i]]) if any(trap[i] for i in ii) else float("nan")
            pn = np.mean([P_ns[l][i][k] for i in ii if not trap[i]])
            say(f"          {ns_eps[k]:8.0e}      {f3(mp)}             {f3(fa)}            {f3(ps)}        "
                f"{f3(pt)}          {f3(pn)}")
        # within (instance, eps): success of trajectories that depart before vs after lam2
        dif = []
        for i in NS:
            for k in small:
                dd = NS[i]["layers"][l][k]["dep"]
                if dd["n_before"] >= 20 and dd["n_after"] >= 20:
                    dif.append(dd["succ_before"] - dd["succ_after"])
        if dif:
            say(f"      same instance and eps, success(depart <= lam2) - success(depart > lam2): "
                f"{len(dif)} cases, mean {np.mean(dif):+.3f}, median {np.median(dif):+.3f}, "
                f"positive in {np.mean(np.array(dif) > 0) * 100:.0f} %")
        # within instance across eps: Spearman(frac after lam2, P_succ)
        rs, rt = [], []
        for i in NS:
            fa = [NS[i]["layers"][l][k]["dep"]["frac_after_lam2"] for k in small]
            ps = [P_ns[l][i][k] for k in small]
            r = sp(fa, ps)
            if np.isfinite(r):
                (rt if trap[i] else rs).append(r)
        if rs or rt:
            say(f"      within instance across eps <= 1e-2, Spearman(frac after lam2, P_succ): trap median "
                f"{f3(np.median(rt) if rt else float('nan'))} ({len(rt)} inst., negative in "
                f"{np.mean(np.array(rt) < 0) * 100 if rt else float('nan'):.0f} %), non-trap median "
                f"{f3(np.median(rs) if rs else float('nan'))} ({len(rs)} inst.)")
        k6 = ns_eps.index(1e-6) if 1e-6 in ns_eps else None
        tr_i = [i for i in NS if trap[i]]
        if k6 is not None and tr_i:
            say(f"      trap instances ({len(tr_i)}): P at eps 1e-6 {np.mean([P_ns[l][i][k6] for i in tr_i]):.3f}, "
                f"at eps* {np.mean([P_ns[l][i][eps_star[l]] for i in tr_i]):.3f}, per-instance best eps "
                f"{np.mean([P_ns[l][i].max() for i in tr_i]):.3f}")
    say()

# ------------------------------------------------------------------------------------------------ D. SB tuned
sbt_star = {}
if SBT:
    r0 = next(iter(SBT.values()))
    algos, cs, dts, steps, As = r0["algos"], r0["c"], r0["dt"], r0["steps"], r0["A"]
    PS = {i: np.array(r["P_succ"]) for i, r in SBT.items()}
    train = sorted(i for i in SBT if i > a.test_hi)
    test = sorted(i for i in SBT if i <= a.test_hi)
    base = train if train else test
    say(f"== D. bSB / dSB as algorithms (Goto 2021 normalisation), R = {r0['R']}; hyperparameters chosen on "
        f"{'training' if train else 'TEST'} set ({len(base)} inst.), evaluated on test ({len(test)} inst.)")
    mtr = np.mean([PS[i] for i in base], 0)
    mte = np.mean([PS[i] for i in test], 0) if test else np.full_like(mtr, np.nan)
    for ia, al in enumerate(algos):
        flat = np.argsort(mtr[ia].ravel())[::-1]
        for rank, fi in enumerate(flat[:5]):
            ic, idt, ins, iA = np.unravel_index(fi, mtr[ia].shape)
            say(f"   {al} #{rank + 1}: c {cs[ic]:<5g} dt {dts[idt]:<5g} steps {steps[ins]:<6d} A {As[iA]:<7g} "
                f"train {mtr[ia][ic, idt, ins, iA]:.3f}  test {mte[ia][ic, idt, ins, iA]:.3f}")
        ic, idt, ins, iA = np.unravel_index(flat[0], mtr[ia].shape)
        sbt_star[al] = ({i: PS[i][ia, ic, idt, ins, iA] for i in SBT},
                        f"c {cs[ic]:g}, dt {dts[idt]:g}, {steps[ins]} steps, A {As[iA]:g}")
        line = "      best per number of steps (train/test): "
        for ins2, nst in enumerate(steps):
            sub = mtr[ia][:, :, ins2, :]
            j = np.unravel_index(np.argmax(sub), sub.shape)
            line += f"{nst}: {sub[j]:.3f}/{mte[ia][j[0], j[1], ins2, j[2]]:.3f}  "
        say(line)
        for iA2, A in enumerate(As):
            sub = mtr[ia][:, :, :, iA2]
            j = np.unravel_index(np.argmax(sub), sub.shape)
            say(f"      initial amplitude A = {A:g}: best train {sub[j]:.3f}, test {mte[ia][j[0], j[1], j[2], iA2]:.3f}")
    say()

# ------------------------------------------------------------------------------------------------ E. fair comparison
fair = {}          # name -> {inst: P}
for l in ns_layers:
    fair[f"{l} eps*={ns_eps[eps_star[l]]:g}"] = {i: P_ns[l][i][eps_star[l]] for i in NS}
for al, (d, desc) in sbt_star.items():
    fair[f"{al}* ({desc})"] = d
twa_mean = {}
for t, d in TW.items():
    te = [i for i in d if i <= a.test_hi]
    if te:
        twa_mean[t] = (d[te[0]]["h"], np.mean([d[i]["layers"]["L1"]["P_succ"] for i in te]))
ref = {"L2g eps=1e-6 (Goto 2016 classical)": {i: r["layers"]["L2g"]["P_succ"] for i, r in G0c.items()}}
if twa_mean:
    tbest = max(twa_mean, key=lambda t: twa_mean[t][1])
    ref[f"TWA h*={twa_mean[tbest][0]:g} (reference, h chosen on test)"] = {i: r["layers"]["L1"]["P_succ"]
                                                                          for i, r in TW[tbest].items()}

csv_cols, csv_rows = {}, {}
rng = np.random.default_rng(0)
if fair:
    for qt, fam, gt in (("Wq", "weyl", "Wg"), ("G0q", "goto", "Gg")):
        Q = qP[qt]
        ii = sorted(i for i in Q if i <= a.test_hi and all(i in d for d in fair.values()))
        if not ii:
            continue
        q = np.array([Q[i] for i in ii])
        say(f"== E. quantum ({fam} family, {qt}, h = 1, T = 700) vs classical baselines, {len(ii)} test instances")
        say("   baseline                                               mean B  mean(Q-B)  #Q-B>0.1  #Q-B>0.2  #B-Q>0.02")
        B = {}
        for name, d in list(ref.items()) + list(fair.items()):
            kk = [k for k, i in enumerate(ii) if i in d]
            if not kk:
                continue
            b = np.array([d[ii[k]] for k in kk])
            qq = q[kk]
            if len(kk) == len(ii):
                B[name] = b
            say(f"   {name[:54]:54s} {b.mean():.3f}   {np.mean(qq - b):+.3f}    {np.sum(qq - b > 0.1):4d}      "
                f"{np.sum(qq - b > 0.2):4d}      {np.sum(b - qq > 0.02):4d}"
                + ("" if len(kk) == len(ii) else f"   (only {len(kk)} instances)"))
        names = [n for n in fair if n in B]
        best1 = max(names, key=lambda n: B[n].mean())
        port = np.max([B[n] for n in names], 0)
        which = np.array(names)[np.argmax([B[n] for n in names], 0)]
        say(f"   best single fair baseline: {best1} (mean {B[best1].mean():.3f})")
        say(f"   per-instance max over the {len(names)} fair baselines (portfolio, biased towards classical): "
            f"mean {port.mean():.3f}")
        for nm, b in (("best single", B[best1]), ("portfolio", port)):
            dq = q - b
            boot = [np.mean(dq[rng.integers(0, len(dq), len(dq))] > 0.1) for _ in range(a.boot)]
            say(f"   Q - {nm:11s} > 0.1: {np.sum(dq > 0.1)} of {len(dq)} ({np.mean(dq > 0.1) * 100:.0f} %, "
                f"bootstrap 95 % {np.quantile(boot, 0.025) * 100:.0f}-{np.quantile(boot, 0.975) * 100:.0f} %);  "
                f"> 0.2: {np.sum(dq > 0.2)};  B - Q > 0.02: {np.sum(-dq > 0.02)}")
        G = GAP.get(gt, {})
        say("   instances with Q - portfolio > 0.1:")
        say("     inst     Q    port  best baseline                          trap  Ising gap  lam1  lam2-lam1  "
            "Delta_min    p*")
        for k in np.argsort(-(q - port)):
            if q[k] - port[k] <= 0.1:
                break
            i = ii[k]
            dm = G[i]["Delta_min"] if i in G else float("nan")
            pst = G[i]["p_star"] if i in G else float("nan")
            say(f"     {i:4d}  {q[k]:.3f}  {port[k]:.3f}  {which[k][:38]:38s} {int(trap[i]):3d}  {igap[i]:8.3f}  "
                f"{lam1[i]:.3f}  {lam2[i] - lam1[i]:8.3f}  {fe(dm)}  {f3(pst)}")
        if G:
            gi = [k for k, i in enumerate(ii) if i in G]
            say(f"   Spearman(Delta_min, Q - portfolio) = {f3(sp([G[ii[k]]['Delta_min'] for k in gi], (q - port)[gi]))}"
                f" over {len(gi)} instances")
        say()
        for k, i in enumerate(ii):
            row = csv_rows.setdefault(i, {})
            row[f"Q_{fam}"] = q[k]
            row[f"portfolio_vs_{fam}"] = port[k]
            for n in B:
                row[n] = B[n][k]

# ------------------------------------------------------------------------------------------------ F. gaps
PAIRS = {"Gg": [("GqT200", 200.0), ("G0q", 700.0), ("GqT2000", 2000.0)], "Wg": [("WqT200", 200.0), ("Wq", 700.0), ("WqT2000", 2000.0)],
         "Wg07": [("Wq07", 700.0)], "Wg05": [("Wq05", 700.0)], "Wg035": [("Wq035", 700.0)]}
pooled_h = []
for gt, G in GAP.items():
    if not G:
        continue
    ii = sorted(G)
    r0 = G[ii[0]]
    D = np.array([G[i]["Delta_min"] for i in ii])
    ps = np.array([G[i]["p_star"] for i in ii])
    xl = np.array([G[i]["x_LZ"] for i in ii])
    ed = np.array([G[i]["edge"] for i in ii])
    l1, l2, tr = lam1[ii], lam2[ii], trap[ii]
    say(f"== F. quantum gap {gt} ({r0['family']}, h = {r0['h']:g}, cutoff {r0['cutoff']}), {len(ii)} instances")
    say(f"   Delta_min: median {np.median(D):.3e}, min {D.min():.3e} (inst {ii[int(np.argmin(D))]}); "
        f"minimum at an end point (no interior minimum): {int(ed.sum())}")
    inner = ed == 0
    if inner.any():
        say(f"   p* (interior minima, {int(inner.sum())}): below lam1 {np.mean(ps[inner] < l1[inner]) * 100:.0f} %, "
            f"between lam1 and lam2 {np.mean((ps[inner] >= l1[inner]) & (ps[inner] <= l2[inner])) * 100:.0f} %, "
            f"above lam2 {np.mean(ps[inner] > l2[inner]) * 100:.0f} %;  median p* - lam1 "
            f"{np.median(ps[inner] - l1[inner]):.3f}, median p* - lam2 {np.median(ps[inner] - l2[inner]):.3f}")
    say(f"   Delta_min: trap mean {np.mean(D[tr]) if tr.any() else float('nan'):.3e} / non-trap "
        f"{np.mean(D[~tr]):.3e};  Spearman with trap {f3(sp(D, tr))}, lam2 - lam1 {f3(sp(D, l2 - l1))}, "
        f"Ising gap {f3(sp(D, igap[ii]))}, lam1 {f3(sp(D, l1))}")
    pool = []
    for qt, T in PAIRS.get(gt, []):
        Q = qP.get(qt, {})
        jj = [i for i in ii if i in Q]
        if len(jj) < 3:
            continue
        pf_ = np.array([1.0 - Q[i] for i in jj])
        Dj = np.array([G[i]["Delta_min"] for i in jj])
        xj = np.array([G[i]["x_LZ"] for i in jj])
        plz = np.exp(-xj * T)
        say(f"   vs {qt} (T = {T:g}, {len(jj)} inst.): mean 1 - P_L0 {pf_.mean():.4f};  Spearman(1 - P_L0, "
            f"Delta_min) {f3(sp(pf_, Dj))}, (1 - P_L0, x_LZ T) {f3(sp(pf_, xj * T))}")
        say("      P_LZ bin          n   mean P_LZ   mean (1 - P_L0)")
        edges = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0001]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (plz >= lo) & (plz < hi)
            if m.any():
                say(f"      [{lo:.0e}, {min(hi, 1):.0e})  {int(m.sum()):4d}   {plz[m].mean():.2e}    {pf_[m].mean():.2e}")
        pool += [(pf_[k], xj[k] * T, Dj[k] ** 2 * T, T) for k in range(len(jj))]
        if T == 700.0 and r0["family"] == "weyl":
            pooled_h += [(pf_[k], xj[k] * T, r0["h"]) for k in range(len(jj))]
    if len({p[3] for p in pool}) > 1:
        pa = np.array(pool)
        say(f"   pooled over T ({len(pa)} points): Spearman(1 - P_L0, x_LZ T) {f3(sp(pa[:, 0], pa[:, 1]))}, "
            f"(1 - P_L0, Delta_min^2 T) {f3(sp(pa[:, 0], pa[:, 2]))}, (1 - P_L0, T) {f3(sp(pa[:, 0], pa[:, 3]))}")
    for k, i in enumerate(ii):
        if i <= a.test_hi:
            row = csv_rows.setdefault(i, {})
            row[f"Dmin_{gt}"], row[f"pstar_{gt}"], row[f"xLZ_{gt}"] = D[k], ps[k], xl[k]
    say()
if len({p[2] for p in pooled_h}) > 1:
    pa = np.array(pooled_h)
    say(f"== F'. weyl family, T = 700, pooled over h = {sorted(set(pa[:, 2].tolist()))}: Spearman(1 - P_L0, x_LZ T) "
        f"{f3(sp(pa[:, 0], pa[:, 1]))}, (1 - P_L0, h) {f3(sp(pa[:, 0], pa[:, 2]))}")
    say()

with open(os.path.join(OUT, "report_1b.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
if csv_rows:
    cols = sorted({c for r in csv_rows.values() for c in r})
    with open(os.path.join(OUT, "per_instance_1b.csv"), "w") as fh:
        fh.write("inst,trap,ising_gap,lam1,lam2," + ",".join(c.replace(",", ";") for c in cols) + "\n")
        for i in sorted(csv_rows):
            r = csv_rows[i]
            fh.write(f"{i},{int(trap[i])},{igap[i]:.6g},{lam1[i]:.6g},{lam2[i]:.6g},"
                     + ",".join(f"{r[c]:.6g}" if c in r else "" for c in cols) + "\n")
print(f"written {OUT}/report_1b.txt" + (f" and {OUT}/per_instance_1b.csv" if csv_rows else ""))
