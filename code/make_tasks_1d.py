"""Task list for MSBQA 1d / 1e (one line = one shell command = one instance), for run_1d.sh / run_1e.sh.

usage: python code/make_tasks_1d.py --plan plan_1d.txt --run run_v4 --code code --inst-dir instances [--py python]
Plan columns ('-' = not used):
  tag  kind  family  set  insts  T  pf  h  cutoff  R  dt
  insts: numbers and ranges (0-499,602) or @file (a file with such a list, path relative to msbqa_phase1/)
  kind = gap       -> code/gap_scan.py   (family, pf, h, cutoff)            same command as make_tasks_1b.py
  kind = classical -> code/run_classical.py, layers L2g,L3,L4 only (family, T, pf, h, R, dt)
  kind = iwindow   -> code/iwindow.py    (R)
  kind = noise     -> code/noise_scan.py (T, pf, R, dt)                     same command as make_tasks_1b.py (1e)
  kind = sbtune    -> code/sb_tune.py    (R)                                same command as make_tasks_1b.py (1e)
Output of a task: RUN/parts/<tag>_<inst>.json; tasks whose JSON exists are skipped (restart-safe).
Tasks are written longest first. Writes RUN/tasks/tasks_<plan file name without extension>.txt.
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
for sub in ("parts", "tasks", "raw"):
    os.makedirs(os.path.join(run, sub), exist_ok=True)


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
    if len(f) != 11:
        raise SystemExit(f"plan_1d line needs 11 columns: {raw.strip()}")
    tag, kind, family, sset, insts, T, pf, h, cutoff, R, dt = f
    if not os.path.exists(os.path.join(inst, f"{sset}.npz")):
        raise SystemExit(f"instance set {sset} not found in {inst} (line: {raw.strip()})")
    ids = parse_insts(insts)
    nq = 0
    for i in ids:
        n_total += 1
        out = os.path.join(run, "parts", f"{tag}_{i}.json")
        if os.path.exists(out):
            n_done += 1
            continue
        common = f"--inst-dir {inst} --set {sset} --inst {i} --tag {tag} --out {out}"
        if kind == "gap":
            if family not in ("goto", "weyl"):
                raise SystemExit(f"gap line needs family goto or weyl: {raw.strip()}")
            cmd = f"{a.py} {code}/gap_scan.py {common} --family {family} --h {h} --cutoff {cutoff} --pf {pf}"
            cost = 2500.0 * ((int(cutoff) + 1) / 23.0) ** 5
        elif kind == "classical":
            if family not in ("goto", "weyl"):
                raise SystemExit(f"classical line needs family goto or weyl: {raw.strip()}")
            cmd = (f"{a.py} {code}/run_classical.py --inst-dir {inst} --set {sset} --inst {i} --family {family}"
                   f" --T {T} --pf {pf} --h {h} --tag {tag} --out {out} --R {R} --dt {dt} --layers L2g,L3,L4")
            cost = 170.0 * int(R) / 1000 * 0.0025 / float(dt) * float(T) / 700
        elif kind == "iwindow":
            cmd = f"{a.py} {code}/iwindow.py {common} --R {R}"
            cost = 400.0 * int(R) / 1000
        elif kind == "noise":
            cmd = (f"{a.py} {code}/noise_scan.py {common} --raw {run}/raw/{tag}_{i}.npz --T {T} --pf {pf} --R {R}"
                   f" --dt {dt}")
            cost = 1500.0 * int(R) / 1000 * 0.0025 / float(dt) * float(T) / 700
        elif kind == "sbtune":
            cmd = f"{a.py} {code}/sb_tune.py {common} --R {R}"
            cost = 60.0 * int(R) / 1000
        else:
            raise SystemExit(f"unknown kind {kind} in: {raw.strip()}")
        tasks.append((cost, f"{cmd} >> {run}/parts/{tag}.log 2>&1"))
        nq += 1
    print(f"  {tag:6s} {kind:9s} {sset:4s} {len(ids):4d} instances: queued {nq}")

tasks.sort(key=lambda t: -t[0])
path = os.path.join(run, "tasks", f"tasks_{os.path.splitext(os.path.basename(a.plan))[0]}.txt")
with open(path, "w") as fh:
    fh.write("".join(t[1] + "\n" for t in tasks))
print(f"{n_total} tasks in plan, {n_done} already complete, {len(tasks)} queued -> {path} "
      f"(rough total {sum(t[0] for t in tasks) / 3600:.0f} core-hours)")
