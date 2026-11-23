"""Per-GPU task lists for the phase-1c GPU gap runs (code/gap_scan_gpu.py), for run_gpu_1c.sh.

usage: python code/make_tasks_gpu_1c.py --plan plan_gpu_1c.txt --run run_v4 --code code --inst-dir instances --ngpu 8
Plan columns:  tag  set  family  h  cutoff  pmax  insts  [dpgrid]
  insts: comma-separated numbers and ranges (0-49,94,99) or @file (a file with such a list).
  dpgrid: grid step in p (optional, default 0.05; the minima are refined to 1e-6 in any case).
Output of a task: RUN/parts/<tag>_<inst>.json; tasks whose output exists are skipped (restart-safe).
Load balance: cost ~ sector dimension x grid points, longest-processing-time first over the GPUs.
Writes RUN/tasks/gpu1c_<g>.txt (one shell command per line).
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--ngpu", type=int, default=8)
ap.add_argument("--py", default="python")
a = ap.parse_args()
run, code, inst = map(os.path.abspath, (a.run, a.code, a.inst_dir))
os.makedirs(os.path.join(run, "parts"), exist_ok=True)
os.makedirs(os.path.join(run, "tasks"), exist_ok=True)


def parse_insts(s):
    if s.startswith("@"):
        with open(s[1:]) as fh:
            s = fh.read().strip()
    out = []
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


tasks, n_total, n_done = [], 0, 0
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if not f:
        continue
    if len(f) not in (7, 8):
        raise SystemExit(f"plan_gpu_1c line needs 7 or 8 columns: {raw.strip()}")
    tag, sset, family, h, cutoff, pmax, insts = f[:7]
    dpg = f[7] if len(f) == 8 else "0.05"
    n = int(np.load(os.path.join(inst, f"{sset}.npz"))["J"].shape[1])
    npts = int(round(float(pmax) / float(dpg))) + 1
    cost = (int(cutoff) + 1) ** n / 2 * npts
    nq = 0
    for i in parse_insts(insts):
        n_total += 1
        out = os.path.join(run, "parts", f"{tag}_{i}.json")
        if os.path.exists(out):
            n_done += 1
            continue
        cmd = (f"{a.py} {code}/gap_scan_gpu.py --inst-dir {inst} --set {sset} --inst {i} --family {family} --h {h}"
               f" --cutoff {cutoff} --pmax {pmax} --dpgrid {dpg} --tag {tag} --out {out} >> {run}/parts/{tag}.log 2>&1")
        tasks.append((cost, cmd))
        nq += 1
    print(f"  {tag:8s} {sset} n={n} {family} h={h} cutoff={cutoff} pmax={pmax} dp={dpg}: queued {nq}")

tasks.sort(key=lambda t: -t[0])
load = [0.0] * a.ngpu
lists = [[] for _ in range(a.ngpu)]
for c, cmd in tasks:
    g = min(range(a.ngpu), key=lambda q: load[q])
    load[g] += c
    lists[g].append(cmd)
for g in range(a.ngpu):
    with open(os.path.join(run, "tasks", f"gpu1c_{g}.txt"), "w") as fh:
        fh.write("".join(c + "\n" for c in lists[g]))
print(f"{n_total} tasks in plan, {n_done} already complete, {len(tasks)} queued over {a.ngpu} GPUs "
      f"(relative loads {', '.join(f'{x / max(load):.2f}' for x in load) if tasks else '-'})")
