"""MSBQA 1e, GPU part: n = 6 gap scans (W6g, pmax 5, cutoff 18) whose minimum lay in the first or last grid cell
(edge = 1) are redone over the whole pump range p in [0, 7] with the cutoff of the n = 4 runs (22), tag W6g7.

usage (from msbqa_phase1/):
  python code/select_1e.py --run run_v4 --code code --inst-dir instances --ngpu 8 [--py python]
Writes RUN/tasks/g6_edge.txt (the instances) and RUN/tasks/gpu1e_<g>.txt (one gap_scan_gpu.py command per line,
round robin over the GPUs; tasks whose output exists are left out, so a second run only re-queues what is missing).
Does not touch the task lists of phase 1c (gpu1c_<g>.txt).
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--ngpu", type=int, default=8)
ap.add_argument("--py", default="python")
ap.add_argument("--src-tag", default="W6g")
ap.add_argument("--tag", default="W6g7")
ap.add_argument("--cutoff", type=int, default=22)
ap.add_argument("--pmax", type=float, default=7.0)
ap.add_argument("--dpgrid", type=float, default=0.1)
a = ap.parse_args()
run, code, inst = map(os.path.abspath, (a.run, a.code, a.inst_dir))
os.makedirs(os.path.join(run, "tasks"), exist_ok=True)

edge = []
nsrc = 0
for p in glob.glob(os.path.join(run, "parts", f"{a.src_tag}_*.json")):
    with open(p) as fh:
        r = json.load(fh)
    if r.get("tag") != a.src_tag:
        continue
    nsrc += 1
    if int(r.get("edge", 0)) == 1:
        if r.get("set") != "G6" or r.get("family") != "weyl" or abs(float(r.get("h", 0)) - 1.0) > 1e-12:
            raise SystemExit(f"unexpected settings in {p}")
        edge.append(int(r["inst"]))
edge.sort()
if nsrc == 0:
    raise SystemExit(f"no {a.src_tag}_*.json in {run}/parts")
with open(os.path.join(run, "tasks", "g6_edge.txt"), "w") as fh:
    fh.write(",".join(map(str, edge)) + "\n")
lists = [[] for _ in range(a.ngpu)]
nq = 0
for k, i in enumerate(edge):
    out = os.path.join(run, "parts", f"{a.tag}_{i}.json")
    if os.path.exists(out):
        continue
    cmd = (f"{a.py} {code}/gap_scan_gpu.py --inst-dir {inst} --set G6 --inst {i} --family weyl --h 1.0"
           f" --cutoff {a.cutoff} --pmax {a.pmax:g} --dpgrid {a.dpgrid:g} --tag {a.tag} --out {out}"
           f" >> {run}/parts/{a.tag}.log 2>&1")
    lists[nq % a.ngpu].append(cmd)
    nq += 1
for g in range(a.ngpu):
    with open(os.path.join(run, "tasks", f"gpu1e_{g}.txt"), "w") as fh:
        fh.write("".join(c + "\n" for c in lists[g]))
print(f"select_1e: {nsrc} {a.src_tag} results, {len(edge)} with edge = 1: {edge}; queued {nq} {a.tag} tasks "
      f"(cutoff {a.cutoff}, pmax {a.pmax:g}) over {a.ngpu} GPUs")
