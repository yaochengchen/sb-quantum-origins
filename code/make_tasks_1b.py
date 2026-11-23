"""Task list for MSBQA phase 1b (one line = one shell command = one instance), for run_1b.sh.

usage: python code/make_tasks_1b.py --plan plan_1b.txt --run run_v4 --code code --inst-dir instances [--py python]
Plan columns ('-' = not used):
  tag  kind  family  set  lo  hi  T  pf  h  cutoff  R  dt
  kind = noise  -> code/noise_scan.py  (T, pf, R, dt)
  kind = sbtune -> code/sb_tune.py     (R)
  kind = gap    -> code/gap_scan.py    (family, pf, h, cutoff)
  kind = classical -> code/run_classical.py, all layers L1..L5 (family, T, pf, h, R, dt), exactly as the classical
                      lines of plan.txt (used to extend Wc, Wc07, WcT200, WcT2000 to the new quantum instances)
  kind = ablate -> code/sb_ablate.py   (R)   (phase 1c: L4 <-> bSB* one-knob ablation)
Output of a task: RUN/parts/<tag>_<inst>.json (noise also RUN/raw/<tag>_<inst>.npz). Tasks whose JSON exists are
skipped, so running run_1b.sh again only re-queues what is missing. Tasks are written longest first (rough cost
weights), which is the right order for xargs -P.
Writes RUN/tasks/tasks_<plan file name without extension>.txt (e.g. tasks_plan_1b.txt), so that two plans
(e.g. plan_1b.txt and plan_gap_1b.txt) can run at the same time without overwriting each other's task list.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--inst-dir", required=True)
ap.add_argument("--py", default="python")
a = ap.parse_args()
run, code, inst = map(os.path.abspath, (a.run, a.code, a.inst_dir))
for sub in ("parts", "raw", "tasks"):
    os.makedirs(os.path.join(run, sub), exist_ok=True)

tasks, n_total, n_done = [], 0, 0
for raw in open(a.plan):
    f = raw.split("#")[0].split()
    if not f:
        continue
    if len(f) != 12:
        raise SystemExit(f"plan_1b line needs 12 columns: {raw.strip()}")
    tag, kind, family, sset, lo, hi, T, pf, h, cutoff, R, dt = f
    nq = 0
    for i in range(int(lo), int(hi) + 1):
        n_total += 1
        out = os.path.join(run, "parts", f"{tag}_{i}.json")
        if os.path.exists(out):
            n_done += 1
            continue
        common = f"--inst-dir {inst} --set {sset} --inst {i} --tag {tag} --out {out}"
        if kind == "noise":
            cmd = (f"{a.py} {code}/noise_scan.py {common} --raw {run}/raw/{tag}_{i}.npz --T {T} --pf {pf} --R {R}"
                   f" --dt {dt}")
            cost = 1500.0 * int(R) / 1000 * 0.0025 / float(dt) * float(T) / 700
        elif kind == "sbtune":
            cmd = f"{a.py} {code}/sb_tune.py {common} --R {R}"
            cost = 60.0 * int(R) / 1000
        elif kind == "gap":
            if family not in ("goto", "weyl"):
                raise SystemExit(f"gap line needs family goto or weyl: {raw.strip()}")
            cmd = f"{a.py} {code}/gap_scan.py {common} --family {family} --h {h} --cutoff {cutoff} --pf {pf}"
            cost = 1200.0 * ((int(cutoff) + 1) / 23.0) ** 5
        elif kind == "ablate":
            cmd = f"{a.py} {code}/sb_ablate.py {common} --R {R}"
            cost = 800.0 * int(R) / 1000
        elif kind == "classical":
            if family not in ("goto", "weyl"):
                raise SystemExit(f"classical line needs family goto or weyl: {raw.strip()}")
            cmd = (f"{a.py} {code}/run_classical.py --inst-dir {inst} --set {sset} --inst {i} --family {family}"
                   f" --T {T} --pf {pf} --h {h} --tag {tag} --out {out} --R {R} --dt {dt}")
            cost = 330.0 * int(R) / 1000 * 0.0025 / float(dt) * float(T) / 700
        else:
            raise SystemExit(f"unknown kind {kind} in: {raw.strip()}")
        tasks.append((cost, f"{cmd} >> {run}/parts/{tag}.log 2>&1"))
        nq += 1
    print(f"  {tag:8s} {kind:7s} {sset} {lo}-{hi}: queued {nq}")

tasks.sort(key=lambda t: -t[0])
path = os.path.join(run, "tasks", f"tasks_{os.path.splitext(os.path.basename(a.plan))[0]}.txt")
with open(path, "w") as fh:
    fh.write("".join(t[1] + "\n" for t in tasks))
print(f"{n_total} tasks in plan, {n_done} already complete, {len(tasks)} queued -> {path} "
      f"(rough total {sum(t[0] for t in tasks) / 3600:.0f} core-hours)")
