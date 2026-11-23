"""MSBQA phase 1c report for the gap runs: does the quantum avoided crossing p* go to the classical crossing p_c as
h -> 0, does ln Delta_min scale like a tunnelling exponent -S/h, and what changes at n = 6?

usage (from msbqa_phase1/):  python code/phase1c_gap_report.py --run run_v4 --inst-dir instances
Inputs: RUN/parts/{Wg,Wg07,Wg05,Wg035,Wg025,Wg018,Wg012,W6g}_*.json (gap_scan.py / gap_scan_gpu.py),
        RUN/branches/branches_weyl.csv (n = 4) and RUN/branches_G6/branches_weyl.csv (n = 6) from branches.py.
Writes RUN/phase1c/report_1c_gap.txt and RUN/phase1c/gap_vs_h.csv (one row per n = 4 instance).
Can be run at any time; uses what is finished.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or os.path.join(a.run, "phase1c")
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


def branches(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["inst"])] = {k: (float(v) if v not in ("", None) else np.nan) for k, v in r.items()}
    return out


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def f3(v):
    return "   -  " if v is None or not np.isfinite(v) else f"{v:+.3f}"


TAGS4 = ["Wg", "Wg07", "Wg05", "Wg035", "Wg025", "Wg018", "Wg012"]
G = {t: load(t) for t in TAGS4}
B4 = branches(os.path.join(a.run, "branches", "branches_weyl.csv"))
hval = {}
for t, d in G.items():
    if d:
        hval[t] = float(next(iter(d.values()))["h"])
tags = sorted(hval, key=lambda t: -hval[t])
say("MSBQA phase 1c: quantum minimum gap vs h (weyl family) and the classical configuration crossing")
say("finished: " + "  ".join(f"{t} (h={hval[t]:g}) {len(G[t])}" for t in tags) + f"  W6g {len(load('W6g'))}")

# ---------------------------------------------------------------- checks
say("\n== checks (truncation 'top2' should be < ~1e-4; 'edge' = minimum in the last grid cell, pmax too small)")
for t in tags:
    rs = list(G[t].values())
    top2 = [r.get("top2", np.nan) for r in rs]
    say(f"   {t:7s} h={hval[t]:<5g} n={len(rs):3d}  edge {sum(r['edge'] for r in rs):3d}  top2 max "
        f"{np.nanmax(top2) if np.isfinite(top2).any() else float('nan'):.1e}  misconv {sum(r.get('misconv', 0) for r in rs)}"
        f"  cpu fallbacks {sum(r.get('cpu_fallbacks', 0) for r in rs)}  pmax "
        f"{sorted({r.get('pmax', r['pf']) for r in rs})}")

# ---------------------------------------------------------------- per-instance table and fits
small = [t for t in tags if hval[t] < 0.45]                     # the small-h set defines the instances
base = small or tags
insts = sorted(set().union(*[set(G[t]) for t in base])) if base else []
rows = []
for i in insts:
    hs, ps, ds, ed = [], [], [], []
    for t in tags:
        if i in G[t]:
            r = G[t][i]
            hs.append(hval[t])
            ps.append(r["p_star"])
            ds.append(r["Delta_min"])
            ed.append(r["edge"])
    hs, ps, ds, ed = map(np.array, (hs, ps, ds, ed))
    b = B4.get(i, {})
    row = dict(inst=i, trap=b.get("trap", np.nan), cross=b.get("cross", np.nan), p_c=b.get("p_c", np.nan),
               p_first=b.get("p_first", np.nan), p_gmin_cl=b.get("p_gmin_cl", np.nan), ham_c=b.get("ham_c", np.nan),
               xsq_c=b.get("xsq_c", np.nan), nh=len(hs))
    for hh_, pp, dd_, ee in zip(hs, ps, ds, ed):
        row[f"pstar_h{hh_:g}"], row[f"Dmin_h{hh_:g}"], row[f"edge_h{hh_:g}"] = pp, dd_, ee
    ok = ed == 0
    # p*(h) -> h = 0: quadratic fit in h over all non-edge points (needs >= 4), linear over the 3 smallest h
    if ok.sum() >= 4:
        c2 = np.polyfit(hs[ok], ps[ok], 2)
        row["p0_quad"] = float(c2[-1])
        o = np.argsort(hs[ok])[:3]
        c1 = np.polyfit(hs[ok][o], ps[ok][o], 1)
        row["p0_lin3"] = float(c1[-1])
    # ln Delta_min: c - S/h versus c + k ln h, on non-edge points with h <= 0.5
    sm = ok & (hs <= 0.5 + 1e-9)
    if sm.sum() >= 3:
        y = np.log(ds[sm])
        res = {}
        for name, x in (("inv", 1.0 / hs[sm]), ("pow", np.log(hs[sm]))):
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            res[name] = (coef, float(np.sum((y - A @ coef) ** 2)))
        row["S"] = float(-res["inv"][0][0])
        row["k_pow"] = float(res["pow"][0][0])
        row["rss_inv"], row["rss_pow"] = res["inv"][1], res["pow"][1]
    rows.append(row)


def col(rs, k):
    return np.array([r.get(k, np.nan) for r in rs], dtype=float)


cr = col(rows, "cross")
say(f"\n== n = 4: {len(rows)} instances with gap results; classical crossing (branches.py) in {int(np.sum(cr == 1))}")
hs_all = [hval[t] for t in tags]
say("   crossing instances: p_c, then p* at h = " + ", ".join(f"{x:g}" for x in hs_all) + "  (* = edge)")
for r in rows:
    if r["cross"] != 1:
        continue
    ps = "  ".join((f"{r[f'pstar_h{x:g}']:6.3f}" + ("*" if r.get(f"edge_h{x:g}", 0) else " "))
                   if f"pstar_h{x:g}" in r else "    -  " for x in hs_all)
    say(f"     {r['inst']:4d}  p_c {r['p_c']:6.3f}  |  {ps}  |  p0 quad {f3(r.get('p0_quad', np.nan))} "
        f"lin3 {f3(r.get('p0_lin3', np.nan))}")
say("   crossing instances: Delta_min at the same h")
for r in rows:
    if r["cross"] != 1:
        continue
    ds = "  ".join(f"{r[f'Dmin_h{x:g}']:.2e}" if f"Dmin_h{x:g}" in r else "    -    " for x in hs_all)
    say(f"     {r['inst']:4d}  ham {r['ham_c']:.0f}  xsq_c {r['xsq_c']:.3f}  |  {ds}  |  S {f3(r.get('S', np.nan))}")

for lab, m in (("crossing", cr == 1), ("no crossing", cr == 0)):
    if not m.any():
        continue
    rs = [r for r, mm in zip(rows, m) if mm]
    p0q, p0l = col(rs, "p0_quad"), col(rs, "p0_lin3")
    pc, pfirst, pgm = col(rs, "p_c"), col(rs, "p_first"), col(rs, "p_gmin_cl")
    ref = pc if lab == "crossing" else pfirst
    refn = "p_c" if lab == "crossing" else "p_first"
    say(f"\n   {lab} ({len(rs)}): extrapolated p*(h -> 0) - {refn}: quadratic median {np.nanmedian(p0q - ref):+.3f} "
        f"(|.| < 0.2 in {np.mean(np.abs(p0q - ref)[np.isfinite(p0q)] < 0.2) * 100 if np.isfinite(p0q).any() else float('nan'):.0f} %), "
        f"linear-3 median {np.nanmedian(p0l - ref):+.3f};  Spearman(p0 quad, {refn}) {f3(sp(p0q, ref))};  "
        f"vs p_gmin_cl: median {np.nanmedian(p0q - pgm):+.3f}")
    S, rin, rpo = col(rs, "S"), col(rs, "rss_inv"), col(rs, "rss_pow")
    okk = np.isfinite(S)
    if okk.any():
        say(f"      ln Delta_min (h <= 0.5): -S/h fits better than k ln h in {np.mean(rin[okk] < rpo[okk]) * 100:.0f} % "
            f"of {int(okk.sum())};  median S {np.nanmedian(S):+.3f}, median k {np.nanmedian(col(rs, 'k_pow')):+.3f}")
        if lab == "crossing":
            w = col(rs, "ham_c") * col(rs, "xsq_c")
            say(f"      Spearman(S, ham_c * xsq_c) {f3(sp(S, w))};  median S / (ham_c * xsq_c) {np.nanmedian(S / w):.3f} "
                f"(WKB: S proportional to the distance between the two cat configurations squared)")

# ---------------------------------------------------------------- n = 6
G6 = load("W6g")
B6 = branches(os.path.join(a.run, "branches_G6", "branches_weyl.csv"))
if B6:
    b = list(B6.values())
    tr6, cr6 = col(b, "trap") == 1, col(b, "cross")
    say(f"\n== n = 6 (G6, first {len(b)} instances): trap {tr6.mean() * 100:.1f} %, classical crossing "
        f"{np.mean(cr6 == 1) * 100:.1f} % (n = 4: trap 9.1 %, crossing 5.4 %); crossing among trap "
        f"{np.mean(cr6[tr6] == 1) * 100 if tr6.any() else float('nan'):.0f} %")
if G6:
    ii = sorted(G6)
    D = np.array([G6[i]["Delta_min"] for i in ii])
    ps = np.array([G6[i]["p_star"] for i in ii])
    ed = np.array([G6[i]["edge"] for i in ii])
    c6 = np.array([B6.get(i, {}).get("cross", np.nan) for i in ii])
    pc6 = np.array([B6.get(i, {}).get("p_c", np.nan) for i in ii])
    top2 = np.array([G6[i].get("top2", np.nan) for i in ii])
    say(f"   W6g (h = 1, cutoff {G6[ii[0]]['cutoff']}, pmax {G6[ii[0]].get('pmax')}): {len(ii)} instances, edge "
        f"{int(ed.sum())}, top2 max {np.nanmax(top2):.1e}")
    for lab, m in (("crossing", c6 == 1), ("no crossing", c6 == 0)):
        if m.any():
            mm = m & (ed == 0)
            say(f"   {lab:11s} {int(m.sum()):3d}: Delta_min median {np.median(D[m]):.3e} (n = 4, h = 1: crossing 7.9e-02, "
                f"no crossing 1.0e+00);  p* - p_c median {np.nanmedian(ps[mm] - pc6[mm]) if mm.any() else float('nan'):+.3f}")

with open(os.path.join(OUT, "report_1c_gap.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
if rows:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "gap_vs_h.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
print(f"written {OUT}/report_1c_gap.txt")
