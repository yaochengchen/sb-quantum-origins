"""MSBQA 1d: instances for the quantum-gap intervention test, from the branch CSVs of branches_uniform.py.

usage (from msbqa_phase1/):
  python code/select_1d.py --branches-dir run_v4/branches_uniform --out-dir run_v4/tasks
Groups (weyl family, classical crossing with the per-spin detuning 'kpo' and the uniform detuning 'max'):
  KO    crossing only with kpo            (prediction: the quantum gap OPENS with the uniform detuning)
  MO    crossing only with max            (prediction: the quantum gap CLOSES with the uniform detuning)
  BOTH  crossing with both
  NEG   ground configuration not the lowest branch at pf with kpo or max (cross = -1)
  CTRL  the first --ncontrol instances below --ctrl-hi without a crossing in either (quantum data with kpo exist)
Writes OUT/sel_1d_all.txt (all selected, comma separated), OUT/sel_1d_ge300.txt (selected instances >= 300,
whose per-spin gap is not computed yet), OUT/sel_1d_groups.csv (inst, cross_kpo, cross_max, group).
"""
import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("--branches-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--ncontrol", type=int, default=40)
ap.add_argument("--ctrl-hi", type=int, default=299)
ap.add_argument("--have-hi", type=int, default=299, help="per-spin gaps (tag Wg) exist for instances 0..have-hi")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)


def read(m):
    with open(os.path.join(a.branches_dir, f"branches_weyl_{m}.csv")) as fh:
        return {int(r["inst"]): (int(float(r["cross"])) if r["cross"] not in ("", None) else None)
                for r in csv.DictReader(fh)}


K, M = read("kpo"), read("max")
if sorted(K) != sorted(M):
    raise SystemExit("kpo and max branch files cover different instances")
grp = {}
for i in sorted(K):
    ck, cm = K[i], M[i]
    if ck == -1 or cm == -1:
        grp[i] = "NEG"
    elif ck == 1 and cm == 1:
        grp[i] = "BOTH"
    elif ck == 1:
        grp[i] = "KO"
    elif cm == 1:
        grp[i] = "MO"
nctl = 0
for i in sorted(K):
    if nctl >= a.ncontrol:
        break
    if i <= a.ctrl_hi and i not in grp and K[i] == 0 and M[i] == 0:
        grp[i] = "CTRL"
        nctl += 1
sel = sorted(grp)
ge = [i for i in sel if i > a.have_hi]
with open(os.path.join(a.out_dir, "sel_1d_all.txt"), "w") as fh:
    fh.write(",".join(map(str, sel)) + "\n")
with open(os.path.join(a.out_dir, "sel_1d_ge300.txt"), "w") as fh:
    fh.write(",".join(map(str, ge)) + "\n")
with open(os.path.join(a.out_dir, "sel_1d_groups.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["inst", "cross_kpo", "cross_max", "group"])
    for i in sel:
        w.writerow([i, K[i], M[i], grp[i]])
cnt = {g: sum(1 for i in sel if grp[i] == g) for g in ("KO", "MO", "BOTH", "NEG", "CTRL")}
print(f"select_1d: {len(sel)} instances " + " ".join(f"{g} {c}" for g, c in cnt.items())
      + f";  >= {a.have_hi + 1} (per-spin gap still needed): {len(ge)}")
