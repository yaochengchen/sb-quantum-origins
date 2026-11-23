"""G0 calibration on the classical side: distribution of the classical success probability over all finished
instances (up to 1000, as in Goto 2016), to compare with his Fig. 2 (classical model, noise +-1e-6).

usage (from msbqa_phase1/):  python code/calib.py --run run_v4 --inst-dir instances [--q G0q --c G0c] [--out run_v4/calib]
Writes OUT/calib.txt and OUT/fig_calib_hist.png.

Printed: histograms (10 bins) of P_L2g over all classical instances and of P_L0 over the quantum ones; the
"almost always fails" tail (P < 0.1); success of the eigenvector-trap instances (Goto's failure mechanism);
how deterministic the classical outcome is (largest probability of one Z2 pattern pair per instance: ~1 means
every trajectory ends in the same pair, i.e. the +-1e-6 noise does not matter); correlations with the instance
features; the 15 instances with the lowest P_L2g.
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
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", default="G4")
ap.add_argument("--q", default="G0q")
ap.add_argument("--c", default="G0c")
ap.add_argument("--out", default="")
a = ap.parse_args()
out = a.out or os.path.join(a.run, "calib")
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
for tag in (a.q, a.c):
    for p in glob.glob(os.path.join(a.run, "parts", f"{tag}_*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r["tag"] == tag:
            parts[tag][r["inst"]] = r
Q, C = parts[a.q], parts[a.c]
if not C:
    raise SystemExit(f"no finished {a.c} instance in {a.run}/parts")
info = np.genfromtxt(os.path.join(a.inst_dir, f"{a.set}_info.csv"), delimiter=",", names=True)
ci = np.array(sorted(C))
inf = info[ci]
LAY = ("L1", "L2w", "L2g", "L3", "L4", "L5")
P = {l: np.array([C[i]["layers"][l]["P_succ"] for i in ci]) for l in LAY}
pat = np.array([C[i]["layers"]["L2g"]["P"] for i in ci])          # (ninst, 2^n) pattern distribution
nbits = int(round(np.log2(pat.shape[1])))
full = (1 << nbits) - 1
pairp = np.array([[row[b] + row[b ^ full] for b in range(1 << (nbits - 1))] for row in pat])
pmax = pairp.max(1)
trap = inf["trap"].astype(bool)
edges = np.linspace(0, 1, 11)


def hist(x):
    c, _ = np.histogram(x, edges)
    return " ".join(f"{v:4d}" for v in c)


say(f"== classical calibration: {a.c}, {len(ci)} instances (family {C[ci[0]]['family']}, T {C[ci[0]]['T']:g}, "
    f"dt {C[ci[0]]['dt']:g}, R {C[ci[0]]['R']})")
say("   bins of P_succ:            0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9-1.0")
say(f"   L2g (Goto App. E) counts: {hist(P['L2g'])}")
for l in ("L3", "L4"):
    say(f"   {l:3s}               counts: {hist(P[l])}")
if Q:
    qi = np.array(sorted(Q))
    PQ = np.array([Q[i]["P_succ"] for i in qi])
    say(f"   L0 quantum ({len(qi):4d} inst) : {hist(PQ)}")
say(f"\n   L2g: mean {P['L2g'].mean():.4f}, median {np.median(P['L2g']):.4f}, min {P['L2g'].min():.4f}; "
    f"P < 0.1: {(P['L2g'] < 0.1).sum()}, P < 0.5: {(P['L2g'] < 0.5).sum()}, P > 0.99: {(P['L2g'] > 0.99).sum()}")
say(f"   eigenvector-trap instances: {trap.sum()} ({100 * trap.mean():.1f} %); their P_L2g mean "
    f"{P['L2g'][trap].mean() if trap.any() else float('nan'):.3f}, median "
    f"{np.median(P['L2g'][trap]) if trap.any() else float('nan'):.3f}, < 0.5 in {(P['L2g'][trap] < 0.5).sum()}; "
    f"non-trap mean {P['L2g'][~trap].mean():.3f}")
say(f"   determinism of L2g (largest Z2-pair probability per instance): median {np.median(pmax):.3f}, "
    f"> 0.99 in {100 * (pmax > 0.99).mean():.1f} %, < 0.6 in {100 * (pmax < 0.6).mean():.1f} %")
say("   (if the classical outcome were fixed by the linear stage, pmax would be ~1 everywhere and P_L2g ~ 0 or 1)")
say("\n   Spearman of P_L2g with the instance features:")
for k in ("trap", "trap_excess", "gap", "lam1", "lam2", "n_locmin", "v1_gs_overlap"):
    say(f"     {k:14s} {sp(P['L2g'], inf[k]):+.3f}")
lg = inf["lam2"] - inf["lam1"]
say(f"     {'lam2-lam1':14s} {sp(P['L2g'], lg):+.3f}")
o = np.argsort(P["L2g"])[:15]
say("\n   15 lowest P_L2g:   inst  P_L2g  pmax   P_L1   P_L3   P_L4   P_L0   trap  gap    lam1   lam2-lam1  v1_gs_ovl")
for j in o:
    i = ci[j]
    q = f"{Q[i]['P_succ']:.3f}" if i in Q else "  -  "
    say(f"                     {i:4d}  {P['L2g'][j]:.3f}  {pmax[j]:.3f}  {P['L1'][j]:.3f}  {P['L3'][j]:.3f}  "
        f"{P['L4'][j]:.3f}  {q}  {int(trap[j])}   {inf['gap'][j]:.3f}  {inf['lam1'][j]:.3f}  {lg[j]:.3f}      "
        f"{inf['v1_gs_overlap'][j]:.3f}")
say("\n   Compare the L2g row with Goto 2016 Fig. 2 (classical). If his histogram has a clear group at P ~ 0 and ours")
say("   does not, the difference is in the setup (J distribution, noise, schedule), not in the statistics.")

with open(os.path.join(out, "calib.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(P["L2g"], edges); ax[0].set_title(f"L2g classical, {len(ci)} instances")
    if Q:
        ax[1].hist(PQ, edges); ax[1].set_title(f"L0 quantum, {len(qi)} instances")
    ax[2].scatter(pmax, P["L2g"], s=5, c=trap, cmap="coolwarm", vmin=0, vmax=1)
    ax[2].set_xlabel("largest Z2-pair probability (L2g)"); ax[2].set_ylabel("P_succ L2g")
    ax[2].set_title("red = eigenvector trap")
    for x in ax[:2]:
        x.set_xlabel("success probability"); x.set_ylabel("instances")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_calib_hist.png"), dpi=150); plt.close(fig)
except Exception as e:
    print(f"figure skipped: {e}")
print(f"written to {out}")
