"""Report for MSBQA phase 1c step 3 (sb_ablate.py): mean success of every ablation configuration, overall, on
trap / non-trap instances, and on the instances where L4 loses most; checks against phase 1 / 1b.

usage (from msbqa_phase1/):  python code/ablate_report.py --run run_v4 --inst-dir instances [--tag SBA]
Writes RUN/phase1c/report_ablate.txt and RUN/phase1c/ablate_per_instance.csv.
Checks: config 'L4' must equal the phase-1 G0c layer L4 bit for bit (same seed, same formula); config 'bSB*' should
agree statistically with sb_tune's chosen bSB* (different random numbers).
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--set", default="G4")
ap.add_argument("--tag", default="SBA")
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


A = load(a.tag)
if not A:
    raise SystemExit(f"no {a.tag}_*.json in {a.run}/parts")
info = np.genfromtxt(os.path.join(a.inst_dir, f"{a.set}_info.csv"), delimiter=",", names=True)
ii = sorted(A)
trap = info["trap"].astype(bool)[ii]
names = list(A[ii[0]]["configs"].keys())
P = {nm: np.array([A[i]["configs"][nm]["P_succ"] for i in ii]) for nm in names}
say(f"MSBQA phase 1c, step 3: L4 <-> bSB* one-knob ablation, {len(ii)} instances ({int(trap.sum())} trap), "
    f"R = {A[ii[0]]['R']}")

G0c = load("G0c")
common = [k for k, i in enumerate(ii) if i in G0c]
if common:
    d = [abs(P["L4"][k] - G0c[ii[k]]["layers"]["L4"]["P_succ"]) for k in common]
    say(f"check: config L4 vs phase-1 G0c L4 on {len(common)} instances: max |diff| = {max(d):.2e} (should be 0)")
SBT = load("SBT")
cs = [k for k, i in enumerate(ii) if i in SBT]
if cs:
    r0 = SBT[ii[cs[0]]]
    j = (r0["algos"].index("bSB"), r0["c"].index(0.5), r0["dt"].index(1.0), r0["steps"].index(10000),
         r0["A"].index(0.001))
    sb = np.array([np.array(SBT[ii[k]]["P_succ"])[j] for k in cs])
    say(f"check: config bSB* vs sb_tune bSB* (c 0.5, dt 1, 10000 steps, A 1e-3) on {len(cs)} instances: means "
        f"{P['bSB*'][cs].mean():.4f} / {sb.mean():.4f}, max |diff| {np.max(np.abs(P['bSB*'][cs] - sb)):.3f} "
        f"(statistical, R = 1000)")
say()
hard = P["L4"] < 0.9
say(f"   configuration        all     trap   non-trap  L4<0.9 ({int(hard.sum())})  #inst<0.9   knobs (bsb = bSB* value)")
for nm in names:
    p = P[nm]
    kn = A[ii[0]]["configs"][nm]["knobs"]
    ks = " ".join(f"{k}={v}" for k, v in kn.items() if v != "kpo") or "(all KPO)"
    say(f"   {nm:18s}  {p.mean():.4f}  {p[trap].mean() if trap.any() else float('nan'):.4f}   {p[~trap].mean():.4f}"
        f"    {p[hard].mean() if hard.any() else float('nan'):.4f}        {np.sum(p < 0.9):4d}      {ks}")
say()
say("reading: a knob that alone lifts L4 to ~1 on the trap / L4<0.9 instances (row 'L4 +knob'), and whose removal")
say("alone drops bSB* back (row 'bSB* -knob'), is the ingredient that makes SB better than its KPO parent model.")

with open(os.path.join(OUT, "report_ablate.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(OUT, "ablate_per_instance.csv"), "w") as fh:
    fh.write("inst,trap," + ",".join(nm.replace(",", ";") for nm in names) + "\n")
    for k, i in enumerate(ii):
        fh.write(f"{i},{int(trap[k])}," + ",".join(f"{P[nm][k]:.4f}" for nm in names) + "\n")
print(f"written {OUT}/report_ablate.txt and {OUT}/ablate_per_instance.csv")
