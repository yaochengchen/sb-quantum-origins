"""MSBQA 1e report: does the n = 4 picture hold at n = 6? Classical baselines (NS6 noise scan, SBT6 bSB / dSB) on
G6 0-299, quantum minimum gaps at h = 1 (W6g, edge cases redone as W6g7 over p in [0, 7]), Landau-Zener estimate
of the quantum success, side by side with n = 4 (G4 0-299, phase 1b). Can be run at any time.

usage (from msbqa_phase1/):  python code/phase1e_report.py --run run_v4 --inst-dir instances
Inputs: RUN/branches_G6/branches_weyl.csv, RUN/parts/{NS6,SBT6,W6g,W6g7}_*.json, RUN/branches/branches_weyl.csv and
RUN/phase1b/per_instance_1b.csv (n = 4). Writes RUN/phase1e/report_1e.txt and per_instance_1e.csv.
Hyperparameters are the ones chosen on the n = 4 training set (phase 1b), not re-tuned for n = 6:
  L2g eps* = 0.03, L3 eps* = 1e-3, L4 eps* = 1e-6 (and Goto's L2g eps = 1e-6);
  bSB* c 0.5, dt 1, 10000 steps, A 1e-3;  dSB* c 0.5, dt 0.5, 10000 steps, A 1e-3.
Quantum success at n = 6 is NOT simulated: P_Q(LZ) = 1 - exp(-x_LZ T), T = 700 (two-level Landau-Zener at the
minimum gap). Its calibration on n = 4 (same estimate vs the measured Q_weyl) is printed next to it.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--T", type=float, default=700.0)
a = ap.parse_args()
OUT = a.out or os.path.join(a.run, "phase1e")
L = []
EPS_STAR = {"L2g": 0.03, "L3": 1e-3, "L4": 1e-6}
SB_STAR = {"bSB": (0.5, 1.0, 10000, 1e-3), "dSB": (0.5, 0.5, 10000, 1e-3)}


def say(s=""):
    print(s, flush=True)
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


def read_csv(path):
    with open(path) as fh:
        return {int(r["inst"]): r for r in csv.DictReader(fh)}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def auc(small, large):
    s, g = np.asarray(small, float), np.asarray(large, float)
    s, g = s[np.isfinite(s)], g[np.isfinite(g)]
    return float(np.mean(s[:, None] < g[None, :])) if s.size and g.size else float("nan")


say("MSBQA 1e report: n = 6 (G6 0-299) next to n = 4 (G4 0-299)")
B6 = read_csv(os.path.join(a.run, "branches_G6", "branches_weyl.csv"))
NS6, SBT6, W6, W67 = load("NS6"), load("SBT6"), load("W6g"), load("W6g7")
say(f"finished: NS6 {len(NS6)}  SBT6 {len(SBT6)}  W6g {len(W6)}  W6g7 {len(W67)};  G6 branches {len(B6)}")

# ------------------------------------------------------------------ A. checks
say("\n== A. checks")
edge5 = sorted(i for i, r in W6.items() if int(r["edge"]) == 1)
say(f"   W6g (pmax 5, cutoff 18): edge = 1 on {len(edge5)}: {edge5}")
bad = []
for i in edge5:
    if i in W67:
        r7, r5 = W67[i], W6[i]
        ok = r7["Delta_min"] <= r5["Delta_min"] * (1 + 1e-6) + 1e-9
        say(f"   #{i}: pmax 5 Delta_min {r5['Delta_min']:.3e} at {r5['p_star']:.2f}  ->  pmax 7 {r7['Delta_min']:.3e} at "
            f"{r7['p_star']:.2f} (edge {r7['edge']}, top2 {r7.get('top2', float('nan')):.1e})"
            + ("" if ok else "   <- larger than over [0, 5]: CHECK"))
        if not ok:
            bad.append(i)
if W67:
    say(f"   W6g7: top2 max {max(r.get('top2', 0) for r in W67.values()):.1e} (should be < ~1e-4), "
        f"Delta_min(pmax 7) > Delta_min(pmax 5) on {len(bad)} (should be 0)")
if NS6:
    r0 = next(iter(NS6.values()))
    for l, e in EPS_STAR.items():
        if l not in r0["layers"] or not any(abs(x["eps"] - e) < 1e-12 * max(1, e) for x in r0["layers"][l]):
            say(f"   NS6: layer {l} has no eps = {e:g}: CHECK")
if SBT6:
    r0 = next(iter(SBT6.values()))
    for al, (c, dt, ns, A) in SB_STAR.items():
        if not (c in r0["c"] and dt in r0["dt"] and ns in r0["steps"] and A in r0["A"]):
            say(f"   SBT6: {al}* setting not in the grid: CHECK")


def gap6(i):
    """n = 6 gap result: W6g7 for the edge cases, W6g otherwise."""
    if i in W67:
        return W67[i]
    if i in W6 and int(W6[i]["edge"]) == 0:
        return W6[i]
    return None


def ns_at(r, layer, eps):
    for x in r["layers"][layer]:
        if abs(x["eps"] - eps) < 1e-12 * max(1, eps):
            return x["P_succ"]
    return np.nan


def sb_at(r, al):
    c, dt, ns, A = SB_STAR[al]
    ia = r["algos"].index(al)
    return np.array(r["P_succ"])[ia, r["c"].index(c), r["dt"].index(dt), r["steps"].index(ns), r["A"].index(A)]


# ------------------------------------------------------------------ per-instance n = 6
rows6 = []
for i in sorted(B6):
    b = B6[i]
    cr6 = num(b["cross"])
    r = dict(inst=i, trap=int(num(b["trap"])), cross=int(cr6) if np.isfinite(cr6) else None,
             p_c=num(b["p_c"]), ham_c=num(b["ham_c"]), xsq_c=num(b["xsq_c"]))
    if i in NS6:
        r["L2g_1e-6"] = ns_at(NS6[i], "L2g", 1e-6)
        for l, e in EPS_STAR.items():
            r[f"{l}_eps*"] = ns_at(NS6[i], l, e)
    if i in SBT6:
        for al in SB_STAR:
            r[f"{al}*"] = float(sb_at(SBT6[i], al))
    g = gap6(i)
    if g is not None:
        x = num(g.get("x_LZ"))
        r.update(Dmin=g["Delta_min"], pstar=g["p_star"], x_LZ=x, gap_src=g["tag"])
        r["PQ_LZ"] = 1.0 - float(np.exp(-x * a.T)) if np.isfinite(x) else np.nan
    rows6.append(r)

# ------------------------------------------------------------------ per-instance n = 4 (phase 1b)
P4 = read_csv(os.path.join(a.run, "phase1b", "per_instance_1b.csv"))
B4 = read_csv(os.path.join(a.run, "branches", "branches_weyl.csv"))
k4 = list(next(iter(P4.values())).keys())


def key4(prefix):
    return next((k for k in k4 if k.startswith(prefix)), None)


K4 = {"L2g_1e-6": key4("L2g eps=1e-6"), "L2g_eps*": key4("L2g eps*"), "L3_eps*": key4("L3 eps*"),
      "L4_eps*": key4("L4 eps*"), "bSB*": key4("bSB*"), "dSB*": key4("dSB*")}
rows4 = []
for i in sorted(P4):
    p = P4[i]
    cr4 = num(B4[i]["cross"]) if i in B4 else np.nan
    r = dict(inst=i, trap=int(num(p["trap"])), cross=int(cr4) if np.isfinite(cr4) else None,
             Dmin=num(p["Dmin_Wg"]), Q=num(p["Q_weyl"]))
    for k, col in K4.items():
        r[k] = num(p[col]) if col else np.nan
    x = num(p.get("xLZ_Wg"))
    r["PQ_LZ"] = 1.0 - float(np.exp(-x * a.T)) if np.isfinite(x) else np.nan
    rows4.append(r)


def sel(rows, f):
    return [r for r in rows if f(r)]


def col(rows, k):
    return [r.get(k, np.nan) for r in rows]


# ------------------------------------------------------------------ B. classical
say("\n== B. classical (T = 700, R = 1000; hyperparameters from the n = 4 training set), mean success")
groups = [("all", lambda r: True), ("crossing", lambda r: r["cross"] == 1), ("no crossing", lambda r: r["cross"] == 0),
          ("trap", lambda r: r["trap"] == 1)]
say("   " + f"{'':14s}" + "".join(f"{'n=4 ' + g:>18s}{'n=6 ' + g:>18s}" for g, _ in groups))
for k in ("L2g_1e-6", "L2g_eps*", "L3_eps*", "L4_eps*", "bSB*", "dSB*"):
    line = f"   {k:14s}"
    for g, f in groups:
        line += f"{mean(col(sel(rows4, f), k)):18.3f}{mean(col(sel(rows6, lambda r: f(r) and k in r), k)):18.3f}"
    say(line)
say("   " + f"{'(instances)':14s}" + "".join(
    f"{len(sel(rows4, f)):18d}{len(sel(rows6, lambda r: f(r) and 'L2g_eps*' in r)):18d}" for g, f in groups))
for nn, rows in (("n=4", rows4), ("n=6", rows6)):
    rr = [r for r in rows if "L2g_eps*" in r]
    if rr:
        say(f"   {nn}: trap {sum(r['trap'] for r in rr) / len(rr) * 100:.1f} %, crossing "
            f"{sum(r['cross'] == 1 for r in rr) / len(rr) * 100:.1f} %;  instances with bSB* < 0.9: "
            f"{sum(r.get('bSB*', 1) < 0.9 for r in rr)}, with max(L2g, L3, L4 at eps*) < 0.9: "
            f"{sum(max(r['L2g_eps*'], r['L3_eps*'], r['L4_eps*']) < 0.9 for r in rr)}")

# ------------------------------------------------------------------ C. quantum gaps and LZ estimate
say(f"\n== C. quantum minimum gap (weyl, h = 1) and Landau-Zener estimate P_Q(LZ) = 1 - exp(-x_LZ T), T = {a.T:g}")
for nn, rows in (("n=4", rows4), ("n=6", rows6)):
    rr = [r for r in rows if np.isfinite(r.get("Dmin", np.nan))]
    c = [r for r in rr if r["cross"] == 1]
    o = [r for r in rr if r["cross"] == 0]
    line = (f"   {nn}: {len(rr)} instances with gaps; Delta_min median crossing {med(col(c, 'Dmin')):.3e} ({len(c)}), "
            f"no crossing {med(col(o, 'Dmin')):.3e} ({len(o)}), AUC(crossing -> smaller) {auc(col(c, 'Dmin'), col(o, 'Dmin')):.3f};  "
            f"P_Q(LZ) mean crossing {mean(col(c, 'PQ_LZ')):.3f}, no crossing {mean(col(o, 'PQ_LZ')):.3f}")
    if nn == "n=4":
        line += f";  measured Q crossing {mean(col(c, 'Q')):.3f}, no crossing {mean(col(o, 'Q')):.3f}"
    else:
        pc = [r["pstar"] - r["p_c"] for r in c if np.isfinite(r.get("p_c", np.nan))]
        line += f";  median p* - p_c on crossing {med(pc):+.2f}"
    say(line)

# ------------------------------------------------------------------ D. where would the quantum machine win (n = 6)
say("\n== D. crossing instances with a gap: P_Q(LZ) vs classical (n = 4 uses the measured Q)")
for nn, rows, qk in (("n=4", rows4, "Q"), ("n=6", rows6, "PQ_LZ")):
    c = [r for r in rows if r["cross"] == 1 and np.isfinite(r.get(qk, np.nan)) and "L2g_eps*" in r]
    if not c:
        continue
    best_kpo = np.array([max(r["L2g_eps*"], r["L3_eps*"], r["L4_eps*"]) for r in c])
    q = np.array([r[qk] for r in c])
    bsb = np.array([r.get("bSB*", np.nan) for r in c])
    say(f"   {nn} ({len(c)}): mean {qk} {q.mean():.3f}, best KPO-type classical (max of L2g/L3/L4 at eps*) "
        f"{best_kpo.mean():.3f}, bSB* {np.nanmean(bsb):.3f};  {qk} - best KPO > 0.1 on {int(np.sum(q - best_kpo > 0.1))}, "
        f"bSB* - {qk} > 0.02 on {int(np.sum(bsb - q > 0.02))}, {qk} - bSB* > 0.1 on {int(np.sum(q - bsb > 0.1))}")

# ------------------------------------------------------------------ files
os.makedirs(OUT, exist_ok=True)
cols = ["inst", "trap", "cross", "p_c", "ham_c", "xsq_c", "L2g_1e-6", "L2g_eps*", "L3_eps*", "L4_eps*", "bSB*", "dSB*",
        "Dmin", "pstar", "x_LZ", "PQ_LZ", "gap_src"]
with open(os.path.join(OUT, "per_instance_1e.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for r in rows6:
        w.writerow(["" if r.get(k) is None else r.get(k, "") for k in cols])
with open(os.path.join(OUT, "report_1e.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"written {OUT}/report_1e.txt and per_instance_1e.csv")
