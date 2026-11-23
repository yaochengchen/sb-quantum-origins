"""Pick the n = 6 instances for the GPU gap runs of phase 1c: all classical-crossing instances among the first --first
instances (at most --ncross) plus the first --ncontrol instances without a crossing, from branches.py's CSV.

usage: python code/select_1c.py --branches run_v4/branches_G6/branches_weyl.csv --out run_v4/tasks/g6_select.txt
Writes one comma-separated line of instance numbers (read by make_tasks_gpu_1c.py through '@file').
"""
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--branches", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--first", type=int, default=300)
ap.add_argument("--ncross", type=int, default=40)
ap.add_argument("--ncontrol", type=int, default=30)
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.branches)) if int(r["inst"]) < a.first]
cross = [int(r["inst"]) for r in rows if r["cross"] == "1"][: a.ncross]
ctrl = [int(r["inst"]) for r in rows if r["cross"] == "0"][: a.ncontrol]
sel = sorted(cross + ctrl)
with open(a.out, "w") as fh:
    fh.write(",".join(str(i) for i in sel) + "\n")
ntrap = sum(1 for r in rows if r["trap"] == "1")
print(f"{a.branches}: {len(rows)} instances, {ntrap} trap, {sum(1 for r in rows if r['cross'] == '1')} crossing; "
      f"selected {len(cross)} crossing + {len(ctrl)} controls -> {a.out}")
