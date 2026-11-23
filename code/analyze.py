"""Collect the parts and write the phase-1 report (calibration against Goto 2016, ablation chain, convergence).

usage: python analyze.py --run RUN --plan ../slurm/plan.txt --inst-dir INST [--q G0q --c G0c] [--out DIR]
Writes DIR/summary.txt, DIR/per_instance.csv, DIR/fig_goto2016_hist.png, DIR/fig_chain_scatter.png.

Definitions (MSBQA plan v1, sec. 3.3-3.4); the classical reference layer Lref is L2w for the 'weyl' family
(same equations as TWA, only the noise differs) and L2g for the 'goto' family (in the goto family L2w carries
the -K h Weyl shift, which at h = 1 destabilises the vacuum, so it is not a meaningful reference):
  r_Q = P_L0 - P_Lref                         quantum rescue relative to the classical equations
  f   = (P_L1 - P_Lref) / (P_L0 - P_Lref)     share of the rescue reproduced by TWA (only where r_Q > 0.2)
                                              (in the goto family L1 and L2g also differ by the shift)
  A: median f >= 0.8 and Spearman(P_L1, P_L0) >= 0.8
  B: >= 25 % of the r_Q > 0.2 instances have f < 0.5 (needs the hbar scan to be final)
  C: fewer than 5 % of the instances have r_Q > 0.2
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
ap.add_argument("--q", default="G0q")
ap.add_argument("--c", default="G0c")
ap.add_argument("--out", default="")
a = ap.parse_args()
out = a.out or os.path.join(a.run, "analysis")
os.makedirs(out, exist_ok=True)
L = []


def say(s=""):
    print(s)
    L.append(s)


# ------------------------------------------------------------------ load
parts = defaultdict(dict)                      # tag -> inst -> record
for p in glob.glob(os.path.join(a.run, "parts", "*.json")):
    try:
        r = json.load(open(p))
    except Exception:
        say(f"unreadable: {p}")
        continue
    parts[r["tag"]][r["inst"]] = r
plan = []
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if f:
        plan.append(f)

say("== completeness")
for f in plan:
    tag, lo, hi = f[0], int(f[4]), int(f[5])          # columns: tag kind family set lo hi ...
    have = sum(1 for i in range(lo, hi + 1) if i in parts[tag])
    say(f"  {tag:10s} {f[1]:9s} {f[2]:5s} {have}/{hi - lo + 1}")

qline = next((f for f in plan if f[0] == a.q), None)
if qline is None or not any(f[0] == a.c for f in plan):
    raise SystemExit(f"{a.q} / {a.c} not (both) in {a.plan} -> nothing to analyse")
fam, sset = qline[2], qline[3]
REF = "L2w" if fam == "weyl" else "L2g"
say(f"\n== analysing {a.q} + {a.c}: family {fam}, classical reference layer {REF}")
info = np.genfromtxt(os.path.join(a.inst_dir, f"{sset}_info.csv"), delimiter=",", names=True)
Q, C = parts[a.q], parts[a.c]
insts = sorted(set(Q) & set(C))
if not insts:
    raise SystemExit("no instance has both the quantum and the classical result yet")
idx = np.array(insts)
inf = info[idx]
PQ = np.array([Q[i]["P_succ"] for i in insts])
EQ = np.array([Q[i]["E_res"] for i in insts])
layers = list(C[insts[0]]["layers"].keys())
PC = {l: np.array([C[i]["layers"][l]["P_succ"] for i in insts]) for l in layers}
EC = {l: np.array([C[i]["layers"][l]["E_res"] for i in insts]) for l in layers}


def sp(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


# ------------------------------------------------------------------ numerical health of L0
say("\n== L0 numerical health")
top2 = np.array([Q[i]["top2_pop_max"] for i in insts])
leak = np.array([Q[i]["parity_leak"] for i in insts])
psum = np.array([abs(Q[i]["P_sum"] - 1) for i in insts])
say(f"  max population in the top two Fock levels: {top2.max():.2e} (median {np.median(top2):.1e})")
say(f"  max parity leak {leak.max():.1e};  max |sum P - 1| {psum.max():.1e}")
say(f"  steps median {np.median([Q[i]['steps'] for i in insts]):.0f}, matvecs median "
    f"{np.median([Q[i]['matvecs'] for i in insts]):.0f}, wall median {np.median([Q[i]['wall_s'] for i in insts]):.0f} s")

# ------------------------------------------------------------------ G0: calibration vs Goto 2016
say(f"\n== G0 calibration against Goto 2016 Fig. 2 ({len(insts)} instances, {a.q} vs {a.c}:L2g)")
say("   Goto 2016 gives Fig. 2 only as histograms: compare fig_goto2016_hist.png with his Fig. 2(a)-(d).")
for name, P, E in (("L0 quantum", PQ, EQ), ("L2g classical", PC["L2g"], EC["L2g"])):
    say(f"  {name:14s} P_succ mean {P.mean():.4f} median {np.median(P):.4f} | E_res mean {E.mean():.4f}")
fail = PC["L2g"] < 0.1
say(f"  instances where L2g 'almost always fails' (P < 0.1): {fail.sum()} "
    f"({100 * fail.mean():.1f} %); their L0 P_succ median {np.median(PQ[fail]) if fail.any() else float('nan'):.3f}")
trap = inf["trap"].astype(bool)
say(f"  eigenvector-trap fraction: all {trap.mean():.3f}; among L2g failures "
    f"{trap[fail].mean() if fail.any() else float('nan'):.3f}")
say(f"  Spearman(P_L0, P_L2g) = {sp(PQ, PC['L2g']):.3f}; mean(P_L0 - P_L2g) = {np.mean(PQ - PC['L2g']):+.4f}")

# ------------------------------------------------------------------ chain
say("\n== ablation chain (mean P_succ, mean E_res)")
say(f"  L0   {PQ.mean():.4f}  {EQ.mean():.4f}")
for l in layers:
    say(f"  {l:4s} {PC[l].mean():.4f}  {EC[l].mean():.4f}   Spearman with L0 {sp(PC[l], PQ):+.3f}")
rQ = PQ - PC[REF]
sel = rQ > 0.2
fr = (PC["L1"][sel] - PC[REF][sel]) / rQ[sel] if sel.any() else np.array([])
say(f"\n  r_Q = P_L0 - P_{REF} > 0.2 in {sel.sum()} instances ({100 * sel.mean():.1f} %)")
if sel.any():
    say(f"  f = TWA share of the rescue: median {np.median(fr):.3f}; f < 0.5 in {(fr < 0.5).mean() * 100:.1f} %")
s_all = sp(PC["L1"], PQ)
verdict = []
if sel.mean() < 0.05:
    verdict.append("C (quantum rescue rare)")
if sel.any() and np.median(fr) >= 0.8 and s_all >= 0.8:
    verdict.append("A (TWA reproduces the rescue)")
if sel.any() and (fr < 0.5).mean() >= 0.25:
    verdict.append("B candidate (needs the hbar scan)")
say(f"  provisional verdict (h = 1 only, one instance cell): {', '.join(verdict) or 'none of A/B/C'}")
say("  rescue vs instance features (Spearman with r_Q):")
for k in ("trap", "trap_excess", "gap", "lam1", "n_locmin", "v1_gs_overlap"):
    say(f"    {k:14s} {sp(rQ, inf[k]):+.3f}")

# ------------------------------------------------------------------ convergence
say("\n== convergence (G1): max |dP_succ| and max total-variation distance of the pattern distribution")


def tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()


for f in plan:
    tag, kind = f[0], f[1]
    # only lines that differ from the main pair in numerical settings (same family, set, T, pf, h)
    if tag in (a.q, a.c) or not parts[tag] or f[2:4] != qline[2:4] or f[6:9] != qline[6:9]:
        continue
    ref = Q if kind == "quantum" else C
    common = sorted(set(parts[tag]) & set(ref))
    if not common:
        continue
    if kind == "quantum":
        dP = max(abs(parts[tag][i]["P_succ"] - ref[i]["P_succ"]) for i in common)
        dT = max(tv(parts[tag][i]["P"], ref[i]["P"]) for i in common)
        say(f"  {tag:10s} vs {a.q}: {len(common)} inst  max|dP| {dP:.2e}  max TV {dT:.2e}")
    else:
        for l in layers:
            dP = max(abs(parts[tag][i]["layers"][l]["P_succ"] - ref[i]["layers"][l]["P_succ"]) for i in common)
            say(f"  {tag:10s} vs {a.c} {l:4s}: {len(common)} inst  max|dP| {dP:.3f}")

# ------------------------------------------------------------------ tables and figures
with open(os.path.join(out, "per_instance.csv"), "w") as fh:
    cols = ["inst"] + list(info.dtype.names[1:]) + ["P_L0", "E_L0"] + [f"P_{l}" for l in layers] + [f"E_{l}" for l in layers]
    fh.write(",".join(cols) + "\n")
    for k, i in enumerate(insts):
        row = [str(i)] + [f"{inf[c][k]:.8g}" for c in info.dtype.names[1:]] + [f"{PQ[k]:.6f}", f"{EQ[k]:.6f}"]
        row += [f"{PC[l][k]:.6f}" for l in layers] + [f"{EC[l][k]:.6f}" for l in layers]
        fh.write(",".join(row) + "\n")
with open(os.path.join(out, "summary.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    bins = np.linspace(0, 1, 21)
    ax[0, 0].hist(PQ, bins); ax[0, 0].set_title("L0 quantum: success probability")
    ax[0, 1].hist(PC["L2g"], bins); ax[0, 1].set_title("L2g classical (Goto App. E): success probability")
    eb = np.linspace(0, max(EQ.max(), EC["L2g"].max()) + 1e-9, 21)
    ax[1, 0].hist(EQ, eb); ax[1, 0].set_title("L0 quantum: residual energy")
    ax[1, 1].hist(EC["L2g"], eb); ax[1, 1].set_title("L2g classical: residual energy")
    for x in ax.flat:
        x.set_ylabel("instances")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_goto2016_hist.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
    for x, l in zip(ax.flat, layers):
        x.scatter(PC[l], PQ, s=6, c=trap, cmap="coolwarm", vmin=0, vmax=1)
        x.plot([0, 1], [0, 1], "k:", lw=0.8)
        x.set_xlabel(f"P_succ {l}"); x.set_ylabel("P_succ L0"); x.set_title(f"{l} (red = eigenvector trap)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_chain_scatter.png"), dpi=150); plt.close(fig)
except Exception as e:  # figures are optional
    say(f"figures skipped: {e}")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
print(f"written to {out}")
