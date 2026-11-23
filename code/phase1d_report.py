"""MSBQA 1d report: (A) checks, (B) does the quantum minimum gap follow the classical configuration crossing when
the detuning is changed (per-spin Delta_i -> uniform max_i Delta_i), (C) classical KPO layers under both
detunings, (D) the Ising-window test of bSB. Can be run at any time; it reports what has finished.

usage (from msbqa_phase1/):
  python code/phase1d_report.py --run run_v4 --inst-dir instances                 (full report)
  python code/phase1d_report.py --run run_v4 --inst-dir instances --checks-only   (set checks only; exit 1 on failure)
Inputs: RUN/branches_uniform/branches_weyl_{kpo,mean,max}.csv, RUN/tasks/sel_1d_groups.csv, instances G4 / G4U,
RUN/parts: Wg (per-spin gaps 0-299, phase 1b), Wg1d (per-spin gaps >= 300), WgU (uniform gaps), G0c (per-spin
classical, phase 1), G0cU (uniform classical), IW (Ising window), SBA (phase 1c ablation, for the check of IW).
Writes RUN/phase1d/report_1d.txt and per_instance_1d.csv.

Decision rules (written before the data):
  B  KO (crossing only per-spin): median Delta_U / Delta_K >= 3 and > 75 % of the instances with ratio > 1
     MO (crossing only uniform):  median Delta_U / Delta_K <= 1/3 and > 75 % with ratio < 1
     both met -> PASS (the quantum gap follows the classical crossing under an intervention);
     KO median ratio < 1.5 or MO median ratio > 1/1.5 -> FAIL; otherwise INCONCLUSIVE.  (Groups need >= 10.)
  D  'mean hold@Dbar 1000' on the mean-detuning crossing instances >= 0.90 and >= 'mean base' + 0.2, and
     'mean hold@pf 1000' - 'mean base' <= 0.05 there -> Ising window SUPPORTED; otherwise NOT SUPPORTED.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--checks-only", action="store_true")
a = ap.parse_args()
OUT = a.out or os.path.join(a.run, "phase1d")
L = []
FAILS = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def read_csv(path):
    with open(path) as fh:
        return {int(r["inst"]): r for r in csv.DictReader(fh)}


def num(v):
    return float(v) if v not in ("", None) else np.nan


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


def gmed(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.median(np.log(x)))) if x.size else float("nan")


def mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


# ------------------------------------------------------------------ inputs
BR = {m: read_csv(os.path.join(a.run, "branches_uniform", f"branches_weyl_{m}.csv")) for m in ("kpo", "mean", "max")}
CR = {m: {i: (int(num(r["cross"])) if r["cross"] not in ("", None) else None) for i, r in BR[m].items()} for m in BR}
dK = np.load(os.path.join(a.inst_dir, "G4.npz"))
JK, DK = dK["J"].astype(np.float64), dK["Delta"].astype(np.float64)
xi0 = float(json.loads(str(dK["meta"]))["xi0"])
NI, n = DK.shape
infoK = np.genfromtxt(os.path.join(a.inst_dir, "G4_info.csv"), delimiter=",", names=True)

# ------------------------------------------------------------------ A. checks of the uniform set
say("MSBQA 1d report")
say("\n== A. checks")
pU = os.path.join(a.inst_dir, "G4U.npz")
if not os.path.exists(pU):
    say("   G4U.npz missing (run code/make_uniform_set.py)")
    FAILS.append("G4U missing")
else:
    dU = np.load(pU)
    JU, DU = dU["J"].astype(np.float64), dU["Delta"].astype(np.float64)
    infoU = np.genfromtxt(os.path.join(a.inst_dir, "G4U_info.csv"), delimiter=",", names=True)
    okJ = np.array_equal(JU, JK)
    okD = np.array_equal(DU, np.repeat(DK.max(1, keepdims=True), n, 1))
    say(f"   G4U: J identical to G4: {okJ};  Delta = max_i Delta_i of G4 on every spin: {okD}")
    if not (okJ and okD):
        FAILS.append("G4U content")
    ii = [i for i in range(len(infoU)) if i in BR["max"]]
    dtrap = sum(int(infoU["trap"][i]) != int(num(BR["max"][i]["trapU"])) for i in ii)
    dl1 = max(abs(float(infoU["lam1"][i]) - num(BR["max"][i]["lam1"])) for i in ii)
    dl2 = max(abs(float(infoU["lam2"][i]) - num(BR["max"][i]["lam2"])) for i in ii)
    dgk = sum(int(infoK["trap"][i]) != int(num(BR["kpo"][i]["trapK"])) for i in ii)
    say(f"   G4U_info vs branches_weyl_max.csv on {len(ii)} instances: trap != trapU {dtrap} (should be 0), "
        f"max |lam1 diff| {dl1:.1e}, |lam2 diff| {dl2:.1e} (should be < 1e-8);  G4 trap != trapK {dgk} (0)")
    if dtrap or dgk or dl1 > 1e-8 or dl2 > 1e-8:
        FAILS.append("G4U info vs branches")
if a.checks_only:
    say("checks: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "OK"))
    sys.exit(1 if FAILS else 0)

GRP = {}
pg = os.path.join(a.run, "tasks", "sel_1d_groups.csv")
if os.path.exists(pg):
    GRP = {i: r["group"] for i, r in read_csv(pg).items()}

# per-spin gaps: Wg (phase 1b, 0-299) + Wg1d (>= 300); uniform gaps: WgU


def gaps(tags, sset):
    d = {}
    for tag in tags:
        for i, r in load(tag).items():
            if r.get("set") == sset and r.get("family") == "weyl" and abs(float(r.get("h", 0)) - 1.0) < 1e-12:
                d[i] = r
    return d


GK, GU = gaps(("Wg", "Wg1d"), "G4"), gaps(("WgU",), "G4U")
CK, CU = load("G0c"), load("G0cU")
IW, SBA = load("IW"), load("SBA")
say(f"   finished: per-spin gaps {len(GK)} (Wg + Wg1d), uniform gaps {len(GU)} (WgU), G0c {len(CK)}, G0cU {len(CU)}, "
    f"IW {len(IW)};  selected instances {len(GRP)}")
if IW:
    common = [i for i in IW if i in SBA]
    for cfg, ref in (("kpo base", "L4"), ("mean base", "L4 +D")):
        d = [abs(IW[i]["configs"][cfg]["P_succ"] - SBA[i]["configs"][ref]["P_succ"]) for i in common]
        say(f"   IW '{cfg}' vs sb_ablate '{ref}' on {len(d)} instances: max |diff| = "
            f"{max(d) if d else float('nan'):.2e} (should be 0)")
        if d and max(d) > 0:
            FAILS.append(f"IW {cfg}")
say("   checks: " + ("FAILED: " + ", ".join(FAILS) if FAILS else "OK"))

# local-minimum flags of the two trap configurations (per-spin: sign v1 of diag(Delta) - xi0 J; uniform: sign of the
# top eigenvector of J)
locK, locU = np.zeros(NI, bool), np.zeros(NI, bool)
for i in range(NI):
    for M, store in ((np.diag(DK[i]) - xi0 * JK[i], locK), (-JK[i], locU)):
        w, U = np.linalg.eigh(M)
        s = np.where(U[:, 0] >= 0, 1.0, -1.0)
        store[i] = bool(np.all(s * (JK[i] @ s) > 0))
trapK = np.array([int(num(BR["kpo"][i]["trapK"])) if i in BR["kpo"] else 0 for i in range(NI)], bool)
trapU = np.array([int(num(BR["kpo"][i]["trapU"])) if i in BR["kpo"] else 0 for i in range(NI)], bool)

# ------------------------------------------------------------------ B. gap intervention test
say("\n== B. quantum minimum gap (weyl, h = 1, cutoff 22) with per-spin (K) vs uniform max (U) detuning")
say("   group  pred      N   median D_K   median D_U   ratio U/K (geo. median)   frac U>K   median p*_K  p*_U   edge K/U")
VER = {}
for g, pred in (("KO", "opens"), ("MO", "closes"), ("BOTH", "-"), ("NEG", "-"), ("CTRL", "~1")):
    ids = [i for i, gg in GRP.items() if gg == g and i in GK and i in GU]
    if not ids:
        say(f"   {g:5s}  {pred:7s}   0")
        continue
    dk = np.array([GK[i]["Delta_min"] for i in ids])
    du = np.array([GU[i]["Delta_min"] for i in ids])
    rr = du / dk
    VER[g] = (len(ids), gmed(rr), float(np.mean(rr > 1)))
    say(f"   {g:5s}  {pred:7s} {len(ids):4d}   {np.median(dk):10.3e}   {np.median(du):10.3e}   {gmed(rr):10.3g}"
        f"                {np.mean(rr > 1):6.2f}     {np.median([GK[i]['p_star'] for i in ids]):6.2f}  "
        f"{np.median([GU[i]['p_star'] for i in ids]):6.2f}   {sum(GK[i]['edge'] for i in ids)}/{sum(GU[i]['edge'] for i in ids)}")
for g in ("BOTH", "NEG"):
    ids = sorted(i for i, gg in GRP.items() if gg == g and i in GK and i in GU)
    if ids:
        say(f"   {g}: " + "; ".join(f"#{i} {GK[i]['Delta_min']:.2e} -> {GU[i]['Delta_min']:.2e}" for i in ids))
for m, G in (("kpo", GK), ("max", GU)):
    ids = [i for i in G if CR[m].get(i) in (0, 1)]
    c = np.array([CR[m][i] == 1 for i in ids])
    d = np.array([G[i]["Delta_min"] for i in ids])
    if c.any() and (~c).any():
        auc = np.mean(d[c][:, None] < d[~c][None, :])
        pc = np.array([num(BR[m][i]["p_c"]) for i in np.array(ids)[c]])
        ps = np.array([G[i]["p_star"] for i in np.array(ids)[c]])
        say(f"   {m:4s} detuning: crossing {c.sum()} instances median Delta_min {np.median(d[c]):.3e}, no crossing "
            f"{(~c).sum()} median {np.median(d[~c]):.3e};  AUC(crossing -> smaller gap) {auc:.3f};  "
            f"median p* - p_c on crossing {np.nanmedian(ps - pc):+.2f}")
vb = "INCONCLUSIVE"
if "KO" in VER and "MO" in VER and VER["KO"][0] >= 10 and VER["MO"][0] >= 10:
    ko, mo = VER["KO"], VER["MO"]
    if ko[1] >= 3 and ko[2] > 0.75 and mo[1] <= 1 / 3 and (1 - mo[2]) > 0.75:
        vb = "PASS"
    elif ko[1] < 1.5 or mo[1] > 1 / 1.5:
        vb = "FAIL"
else:
    vb = "INCOMPLETE (KO and MO need >= 10 finished instances each)"
say(f"   VERDICT B: {vb}")

# ------------------------------------------------------------------ C. classical layers, both detunings
say("\n== C. classical KPO layers (T = 700, eps = 1e-6, R = 1000): per-spin G0c vs uniform G0cU")
for m, C in (("kpo", CK), ("max", CU)):
    ids = [i for i in C if CR[m].get(i) in (0, 1)]
    if not ids:
        continue
    c = np.array([CR[m][i] == 1 for i in ids])
    parts = []
    for lay in ("L2g", "L3", "L4"):
        v = np.array([C[i]["layers"][lay]["P_succ"] if lay in C[i]["layers"] else np.nan for i in ids])
        parts.append(f"{lay} {mean(v[c]):.3f} / {mean(v[~c]):.3f}")
    say(f"   {m:4s} detuning, {len(ids)} instances, crossing {c.sum()}: success crossing / no crossing:  " + ",  ".join(parts))
for g in ("KO", "MO", "BOTH"):
    ids = [i for i in range(NI) if i in CK and i in CU and CR["kpo"].get(i) in (0, 1) and CR["max"].get(i) in (0, 1)
           and {"KO": (1, 0), "MO": (0, 1), "BOTH": (1, 1)}[g] == (CR["kpo"][i], CR["max"][i])]
    if ids:
        say(f"   {g:4s} ({len(ids)} of all 1000): per-spin -> uniform  " + ",  ".join(
            f"{lay} {mean([CK[i]['layers'][lay]['P_succ'] for i in ids]):.3f} -> "
            f"{mean([CU[i]['layers'][lay]['P_succ'] for i in ids]):.3f}" for lay in ("L2g", "L3", "L4")))

# ------------------------------------------------------------------ D. Ising window
say("\n== D. Ising window (L4 dynamics, ramp rate 0.01; hold for tau at p = Dbar or at p = pf), mean success")
VD = "INCOMPLETE"
if IW:
    ids = sorted(IW)
    grp = {
        "all": ids,
        "kpo cross": [i for i in ids if CR["kpo"].get(i) == 1],
        "trapK": [i for i in ids if trapK[i]],
        "trapK locmin": [i for i in ids if trapK[i] and locK[i]],
        "mean cross": [i for i in ids if CR["mean"].get(i) == 1],
        "trapU": [i for i in ids if trapU[i]],
        "none": [i for i in ids if not trapK[i] and not trapU[i] and CR["kpo"].get(i) == 0 and CR["mean"].get(i) == 0],
    }
    say("   configuration        " + "".join(f"{g:>14s}" for g in grp))
    say("   (instances)          " + "".join(f"{len(v):14d}" for v in grp.values()))
    names = list(IW[ids[0]]["configs"].keys())
    tab = {}
    for nm in names:
        tab[nm] = {g: mean([IW[i]["configs"][nm]["P_succ"] for i in v]) for g, v in grp.items()}
        say(f"   {nm:20s} " + "".join(f"{tab[nm][g]:14.3f}" for g in grp))
    mc = grp["mean cross"]
    if len(mc) >= 5 and "mean hold@Dbar 1000" in tab and "mean hold@pf 1000" in tab:
        b0, hd, hp = tab["mean base"]["mean cross"], tab["mean hold@Dbar 1000"]["mean cross"], \
            tab["mean hold@pf 1000"]["mean cross"]
        VD = "SUPPORTED" if (hd >= 0.90 and hd >= b0 + 0.2 and hp - b0 <= 0.05) else "NOT SUPPORTED"
        say(f"   mean-detuning crossing instances ({len(mc)}): base {b0:.3f}, hold@Dbar 1000 {hd:.3f}, hold@pf 1000 {hp:.3f}")
    say(f"   VERDICT D: {VD}")

# ------------------------------------------------------------------ per-instance table
os.makedirs(OUT, exist_ok=True)
iw_names = list(IW[next(iter(IW))]["configs"].keys()) if IW else []
cols = ["inst", "group", "cross_kpo", "cross_max", "cross_mean", "trapK", "trapU", "locK", "locU", "Dmin_K", "Dmin_U",
        "pstar_K", "pstar_U", "pc_kpo", "pc_max"] + [f"{lay}_{s}" for s in ("K", "U") for lay in ("L2g", "L3", "L4")] \
    + [f"IW {nm}" for nm in iw_names]
with open(os.path.join(OUT, "per_instance_1d.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    def cv(m, i):
        v = CR[m].get(i)
        return "" if v is None else v

    for i in range(NI):
        row = [i, GRP.get(i, ""), cv("kpo", i), cv("max", i), cv("mean", i), int(trapK[i]), int(trapU[i]),
               int(locK[i]), int(locU[i]),
               GK[i]["Delta_min"] if i in GK else "", GU[i]["Delta_min"] if i in GU else "",
               GK[i]["p_star"] if i in GK else "", GU[i]["p_star"] if i in GU else "",
               BR["kpo"][i]["p_c"] if i in BR["kpo"] else "", BR["max"][i]["p_c"] if i in BR["max"] else ""]
        for C in (CK, CU):
            for lay in ("L2g", "L3", "L4"):
                row.append(C[i]["layers"][lay]["P_succ"] if i in C and lay in C[i]["layers"] else "")
        row += [IW[i]["configs"][nm]["P_succ"] if i in IW else "" for nm in iw_names]
        w.writerow(row)
with open(os.path.join(OUT, "report_1d.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"written {OUT}/report_1d.txt and per_instance_1d.csv")
