"""Build the per-node task lists for the Slurm array (one line = one shell command = one task).

usage: python make_tasks.py --plan ../slurm/plan.txt --run RUN --code CODE --inst-dir INST --shards 4 [--tag X]
Plan columns (use '-' for not applicable):
  tag  kind  family  set  lo  hi  T  pf  h  cutoff  R  tol  dtmax  dtcl      (family: goto | weyl)
  kind = quantum   -> one task per instance: run_quantum.py  (cutoff, tol, dtmax used)
  kind = classical -> one task per instance: run_classical.py (R, dtcl used; all layers L1..L5)
Output file of a task: RUN/parts/<tag>_<inst>.json. Tasks whose output exists are skipped, so re-running this
script after an interrupted job only re-queues what is missing (safe restart).
Load balancing: rough cost weights, longest-processing-time first over the shards (one shard = one node).
Writes RUN/tasks/shard_<k>.txt (or shard_<tag>_<k>.txt) and RUN/tasks/manifest.csv.
--kind classical (or quantum): only the lines of that kind (run_all.sh uses --kind classical --shards 1 for the
CPU part on the GPU server; the quantum lines go to the GPUs through run_quantum_gpu.py).
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--py", default="python")
ap.add_argument("--tag", default="")
ap.add_argument("--kind", default="", choices=["", "quantum", "classical"], help="only plan lines of this kind")
a = ap.parse_args()
run, code, inst = map(os.path.abspath, (a.run, a.code, a.inst_dir))
os.makedirs(os.path.join(run, "parts"), exist_ok=True)
os.makedirs(os.path.join(run, "tasks"), exist_ok=True)

tasks, n_total, n_done = [], 0, 0
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if not f:
        continue
    if len(f) != 14:
        raise SystemExit(f"plan line needs 14 columns: {raw.strip()}")
    tag, kind, family, sset, lo, hi, T, pf, h, cutoff, R, tol, dtmax, dtcl = f
    if family not in ("goto", "weyl"):
        raise SystemExit(f"unknown family {family} in: {raw.strip()}")
    if a.kind and kind != a.kind:
        continue
    lo, hi = int(lo), int(hi)
    nq = 0
    n = int(np.load(os.path.join(inst, f"{sset}.npz"))["J"].shape[1])
    for i in range(lo, hi + 1):
        n_total += 1
        out = os.path.join(run, "parts", f"{tag}_{i}.json")
        if os.path.exists(out):
            n_done += 1
            continue
        common = (f"--inst-dir {inst} --set {sset} --inst {i} --family {family} --T {T} --pf {pf} --h {h}"
                  f" --tag {tag} --out {out}")
        if kind == "quantum":
            cmd = (f"{a.py} {code}/run_quantum.py {common} --cutoff {cutoff} --tol {tol} --dtmax {dtmax}"
                   f" >> {run}/parts/{tag}.log 2>&1")
            cost = (int(cutoff) + 1) ** n * n * n * float(T) * 1e-6 / float(h)
        elif kind == "classical":
            cmd = (f"{a.py} {code}/run_classical.py {common} --R {R} --dt {dtcl}"
                   f" >> {run}/parts/{tag}.log 2>&1")
            cost = int(R) * float(T) / float(dtcl) * n * 6 * 1e-8
        else:
            raise SystemExit(f"unknown kind {kind}")
        tasks.append((cost, tag, i, cmd))
        nq += 1
    print(f"  {tag:10s} {kind:9s} {family:5s} {sset} {lo}-{hi}: queued {nq}")

tasks.sort(key=lambda t: -t[0])
load = [0.0] * a.shards
shard = [[] for _ in range(a.shards)]
for t in tasks:
    k = min(range(a.shards), key=lambda s: load[s])
    load[k] += t[0]
    shard[k].append(t)
pre = f"shard_{a.tag}_" if a.tag else "shard_"
for k in range(a.shards):
    with open(os.path.join(run, "tasks", f"{pre}{k}.txt"), "w") as fh:
        fh.write("".join(t[3] + "\n" for t in shard[k]))
with open(os.path.join(run, "tasks", f"manifest_{a.tag}.csv" if a.tag else "manifest.csv"), "w") as fh:
    fh.write("shard,tag,inst,cost\n")
    for k in range(a.shards):
        for t in shard[k]:
            fh.write(f"{k},{t[1]},{t[2]},{t[0]:.4g}\n")
print(f"{n_total} tasks in plan, {n_done} already complete, {len(tasks)} queued over {a.shards} shards "
      f"(relative loads: {', '.join(f'{x / max(load):.3f}' for x in load) if tasks else '-'})")
