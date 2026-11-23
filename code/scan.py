"""hbar scan and annealing-time scan in the weyl family: does the quantum rescue survive towards the classical
limit, and does TWA catch up with it?

usage (from msbqa_phase1/):  python code/scan.py --run run_v4 --plan plan.txt --inst-dir instances [--out run_v4/scan]
Uses every weyl plan line whose tag does not start with 'cv' and groups them by (T, h). Reference point: T = 700,
h = 1 (Wq + Wc). Works on whatever is finished. Writes OUT/scan.txt and OUT/fig_scan.png.

Per (T, h) with a quantum line (instances that have both L0 and the classical layers):
  r_Q = P_L0 - P_L2w (quantum rescue), f = (P_L1 - P_L2w) / r_Q on r_Q > 0.2 (share reproduced by TWA),
  gap_QT = P_L0 - P_L1 (what TWA misses). Plan 3.4, B: >= 25 % of the r_Q > 0.2 instances have f < 0.5,
  and this does not vanish as h decreases.
The set R1 = instances with r_Q > 0.2 at the reference point is followed across h, so that a change with h is
not hidden by a change of which instances count as rescued.
L2w does not depend on h (its noise is +-1e-6 and its equations are h independent in the weyl family; same seeds),
so max |P_L2w(h) - P_L2w(1)| must be 0: printed as a consistency check.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--plan", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", default="G4")
ap.add_argument("--out", default="")
a = ap.parse_args()
out = a.out or os.path.join(a.run, "scan")
os.makedirs(out, exist_ok=True)
L = []


def say(s=""):
    print(s)
    L.append(s)


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


parts = defaultdict(dict)
for p in glob.glob(os.path.join(a.run, "parts", "*.json")):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    parts[r["tag"]][r["inst"]] = r

qtag, ctag = {}, {}                                    # (T, h) -> tag
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if len(f) != 14:
        continue
    tag, kind, family, sset = f[0], f[1], f[2], f[3]
    if family != "weyl" or sset != a.set or tag.startswith("cv"):
        continue
    key = (float(f[6]), float(f[8]))
    (qtag if kind == "quantum" else ctag)[key] = tag
info = np.genfromtxt(os.path.join(a.inst_dir, f"{a.set}_info.csv"), delimiter=",", names=True)
REF = (700.0, 1.0)


def pair(key):
    """instances, P_L0, top2, {layer: P} for one (T, h) with a quantum line; None if nothing finished."""
    if key not in qtag or key not in ctag:
        return None
    Q, C = parts[qtag[key]], parts[ctag[key]]
    insts = sorted(set(Q) & set(C))
    if not insts:
        return None
    PQ = np.array([Q[i]["P_succ"] for i in insts])
    top2 = np.array([Q[i]["top2_pop_max"] for i in insts])
    PC = {l: np.array([C[i]["layers"][l]["P_succ"] for i in insts]) for l in ("L1", "L2w", "L2g", "L3", "L4")}
    return np.array(insts), PQ, top2, PC


def row(key, d):
    insts, PQ, top2, PC = d
    rQ = PQ - PC["L2w"]
    sel = rQ > 0.2
    fr = (PC["L1"][sel] - PC["L2w"][sel]) / rQ[sel] if sel.any() else np.array([])
    medf = f"{np.median(fr):+.2f}" if fr.size else "   -"
    f05 = f"{100 * (fr < 0.5).mean():5.0f}" if fr.size else "    -"
    return (f"  {key[1]:5.3g} {key[0]:6.0f} {len(insts):4d}  {PQ.mean():.4f} {PC['L1'].mean():.4f} "
            f"{PC['L2w'].mean():.4f} {PC['L4'].mean():.4f}  {100 * sel.mean():5.1f}  {medf:>6s}  {f05}  "
            f"{np.mean(PQ - PC['L1']):+.4f}  {sp(PC['L1'], PQ):+.3f}  {top2.max():.1e}")


HDR = ("      h      T    n    P_L0   P_L1   P_L2w  P_L4   rQ>0.2%  med f  f<0.5%  mean(L0-L1)  "
       "Sp(L1,L0)  top2max")

# ------------------------------------------------------------------ 1 hbar scan
say("== 1. hbar scan, T = 700 (weyl family; each row: instances with both L0 and the classical layers)")
say(HDR)
hkeys = sorted([k for k in qtag if k[0] == 700.0], key=lambda k: -k[1])
done = {}
for k in hkeys:
    d = pair(k)
    if d is None:
        say(f"  {k[1]:5.3g} {k[0]:6.0f}    0  (not finished)")
        continue
    done[k] = d
    say(row(k, d))

# ------------------------------------------------------------------ 2 same instances at every h
common = None
for k, d in done.items():
    common = set(d[0]) if common is None else common & set(d[0])
if common and len(done) > 1:
    common = np.array(sorted(common))
    say(f"\n== 2. the same {len(common)} instances at every finished h (T = 700)")
    say(HDR)
    for k in hkeys:
        if k in done:
            insts, PQ, top2, PC = done[k]
            m = np.isin(insts, common)
            say(row(k, (insts[m], PQ[m], top2[m], {l: v[m] for l, v in PC.items()})))

# ------------------------------------------------------------------ 3 rescued set R1 followed across h
if REF in done:
    insts0, PQ0, _, PC0 = done[REF]
    R1 = insts0[(PQ0 - PC0["L2w"]) > 0.2]
    say(f"\n== 3. R1 = the {len(R1)} instances with r_Q > 0.2 at h = 1, T = 700, followed across h")
    say("      h    n(R1)  P_L0   P_L1   P_L2w  mean(L0-L1)  mean(L1-L2w)  share of R1 with f<0.5 (where r_Q>0.2)")
    for k in hkeys:
        if k not in done:
            continue
        insts, PQ, _, PC = done[k]
        m = np.isin(insts, R1)
        if not m.any():
            continue
        q, t1, c2 = PQ[m], PC["L1"][m], PC["L2w"][m]
        r = q - c2
        s = r > 0.2
        fr = (t1[s] - c2[s]) / r[s] if s.any() else np.array([])
        fs = f"{100 * (fr < 0.5).mean():.0f} % of {s.sum()}" if fr.size else "- (no r_Q > 0.2 left)"
        say(f"  {k[1]:5.3g}  {m.sum():5d}  {q.mean():.4f} {t1.mean():.4f} {c2.mean():.4f}  {np.mean(q - t1):+.4f}"
            f"      {np.mean(t1 - c2):+.4f}       {fs}")
    say("  reading: if mean(L0-L1) on R1 shrinks towards 0 as h decreases while mean(L0-L2w) stays, TWA catches up")
    say("  (rescue = vacuum fluctuations, outcome A at small h); if L0-L1 stays large, the rescue is beyond TWA (B).")

# ------------------------------------------------------------------ 4 TWA alone towards h -> 0
ref_q = done.get(REF)
say("\n== 4. TWA (L1) towards the classical limit, T = 700 (all classical weyl lines; L2w is h independent)")
say("      h      n   P_L1   P_L2w  mean|L1-L2w|  Sp(L1,L2w)  Sp(L1, L0 at h=1)  max|L2w(h)-L2w(1)|")
ck = sorted([k for k in ctag if k[0] == 700.0], key=lambda k: -k[1])
C_ref = parts[ctag[REF]] if REF in ctag else {}
for k in ck:
    C = parts[ctag[k]]
    insts = sorted(C)
    if not insts:
        say(f"  {k[1]:6.3g}    0  (not finished)")
        continue
    t1 = np.array([C[i]["layers"]["L1"]["P_succ"] for i in insts])
    c2 = np.array([C[i]["layers"]["L2w"]["P_succ"] for i in insts])
    s0 = "      -"
    if ref_q is not None:
        m = np.isin(insts, ref_q[0])
        mq = np.isin(ref_q[0], insts)
        if m.sum() >= 3:
            s0 = f"{sp(t1[m], ref_q[1][mq]):+.3f}"
    both = [i for i in insts if i in C_ref]
    dl = max((abs(C[i]["layers"]["L2w"]["P_succ"] - C_ref[i]["layers"]["L2w"]["P_succ"]) for i in both), default=float("nan"))
    say(f"  {k[1]:6.3g} {len(insts):4d}  {t1.mean():.4f} {c2.mean():.4f}   {np.mean(np.abs(t1 - c2)):.4f}"
        f"       {sp(t1, c2):+.3f}       {s0:>7s}            {dl:.1e}")

# ------------------------------------------------------------------ 5 annealing time
tkeys = sorted([k for k in qtag if k[1] == 1.0], key=lambda k: k[0])
if len(tkeys) > 1:
    say("\n== 5. annealing-time scan, h = 1")
    say(HDR)
    for k in tkeys:
        d = pair(k)
        say(row(k, d) if d is not None else f"  {k[1]:5.3g} {k[0]:6.0f}    0  (not finished)")
    if REF in done:
        say("  R1 (rescued at T = 700) at other T: mean P_L0 / P_L2w")
        for k in tkeys:
            d = pair(k)
            if d is None:
                continue
            m = np.isin(d[0], R1)
            if m.any():
                say(f"    T {k[0]:6.0f}: {d[1][m].mean():.4f} / {d[3]['L2w'][m].mean():.4f}  ({m.sum()} instances)")

# ------------------------------------------------------------------ 6 verdict helper
if done:
    say("\n== 6. plan 3.4, B criterion (>= 25 % of r_Q > 0.2 instances with f < 0.5) at each finished h, T = 700")
    for k in hkeys:
        if k not in done:
            continue
        insts, PQ, _, PC = done[k]
        rQ = PQ - PC["L2w"]
        s = rQ > 0.2
        if not s.any():
            say(f"  h {k[1]:5.3g}: no instance with r_Q > 0.2 ({len(insts)} instances)")
            continue
        fr = (PC["L1"][s] - PC["L2w"][s]) / rQ[s]
        say(f"  h {k[1]:5.3g}: {(fr < 0.5).mean() * 100:5.1f} % of {s.sum()} -> {'met' if (fr < 0.5).mean() >= 0.25 else 'not met'}")
    say("  (top2max column: population of the two highest Fock levels, should stay below ~1e-4)")

with open(os.path.join(out, "scan.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    hs = [k[1] for k in hkeys if k in done]
    if hs:
        ax[0].plot(hs, [done[(700.0, h)][1].mean() for h in hs], "o-", label="L0 quantum")
        ax[0].plot(hs, [done[(700.0, h)][3]["L1"].mean() for h in hs], "s-", label="L1 TWA (quantum h)")
    hc, t1m, c2m = [], [], []
    for k in ck:
        C = parts[ctag[k]]
        if C:
            hc.append(k[1])
            t1m.append(np.mean([r["layers"]["L1"]["P_succ"] for r in C.values()]))
            c2m.append(np.mean([r["layers"]["L2w"]["P_succ"] for r in C.values()]))
    if hc:
        ax[0].plot(hc, t1m, ":", color="gray", label="L1 TWA (all classical lines)")
        ax[0].plot(hc, c2m, "--", color="k", label="L2w classical")
    ax[0].set_xscale("log"); ax[0].set_xlabel("h (effective hbar)"); ax[0].set_ylabel("mean P_succ")
    ax[0].legend(fontsize=8); ax[0].set_title("weyl family, T = 700")
    if REF in done and hs:
        for i in R1:
            xs, ys = [], []
            for h in hs:
                insts, PQ, _, PC = done[(700.0, h)]
                j = np.flatnonzero(insts == i)
                if j.size:
                    xs.append(h); ys.append(PQ[j[0]] - PC["L1"][j[0]])
            ax[1].plot(xs, ys, "-", lw=0.8, alpha=0.7)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xscale("log"); ax[1].set_xlabel("h"); ax[1].set_ylabel("P_L0 - P_L1")
        ax[1].set_title(f"R1: instances rescued at h = 1 ({len(R1)})")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_scan.png"), dpi=150); plt.close(fig)
except Exception as e:                                  # figures are optional
    print(f"figure skipped: {e}")
print(f"written to {out}")
