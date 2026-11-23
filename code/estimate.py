"""Project the wall time of the run array from the smoke-test tasks (informational only).

usage: python estimate.py --run RUN --plan ../slurm/plan.txt --nodes 4 --workers 500
Median wall time per finished task, per plan tag, times the number of queued tasks of that tag, divided by
nodes * workers. The smoke test runs only a few dozen tasks at once, so under the full load of 500 processes
per node the per-task time can be larger (memory bandwidth); treat the result as a lower bound.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--plan", required=True)
ap.add_argument("--nodes", type=int, default=4)
ap.add_argument("--workers", type=int, default=500)
a = ap.parse_args()

walls = defaultdict(list)
for p in glob.glob(os.path.join(a.run, "parts", "*.json")):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    walls[r["tag"]].append(r["wall_s"])

plan = {}
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if f:
        plan[f[0]] = (f[1], int(f[5]) - int(f[4]) + 1)       # columns: tag kind family set lo hi ...

total = 0.0
for tag, (kind, ntask) in plan.items():
    w = walls.get(tag) or walls.get("G0q" if kind == "quantum" else "G0c")
    if not w:
        print(f"{tag:10s} {kind:9s} {ntask:5d} tasks   no timing sample")
        continue
    med = float(np.median(w))
    total += med * ntask
    print(f"{tag:10s} {kind:9s} {ntask:5d} tasks   median {med:8.1f} s/task (n={len(w)})"
          f"   {med * ntask / 3600:8.1f} core-h")
print(f"total {total / 3600:.1f} core-h -> ideal wall {total / (a.nodes * a.workers) / 60:.1f} min on "
      f"{a.nodes} nodes x {a.workers} workers (lower bound; the longest single task is a floor too: "
      f"{max((max(v) for v in walls.values()), default=0) / 60:.1f} min)")
