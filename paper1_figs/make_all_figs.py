"""MSBQA paper 1: make every figure and every quoted number on THIS computer, from msbqa_phase1 only.

    cd D:\\software\\Jiapei\\Make_SB_Quantum_Again\\msbqa_phase1
    python paper1_figs\\make_all_figs.py

Needs Python with numpy, scipy, pandas, matplotlib. About one minute. What it does:
 1. If the per-instance JSON files from pu1 are not in run_v4/parts/, unpacks them from paper1_figdata.tgz (in
    msbqa_phase1/; it holds run_v4/parts/{NS,Wc*,Wq,Wg*_29}_*.json and instances/G4.*).
 2. If instances/G4.npz or G6.npz is missing, regenerates it with code/gen_instances.py (seed 2016, deterministic;
    identical to the pu1 files).
 3. Checks every input file; anything missing is listed and the affected figure is skipped.
 4. Runs the figure scripts; each writes paper1_figs/<name>.pdf/.png/_numbers.txt; all numbers are collected in
    paper1_figs/ALL_NUMBERS.txt.
Nothing is read from msbqa_phase2.
"""
import glob
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def P(*a):
    return os.path.join(ROOT, *a)


TGZ = P("paper1_figdata.tgz")
CSV = {
    "1b": P("run_v4", "phase1b", "per_instance_1b.csv"),
    "gap": P("run_v4", "phase1c", "gap_vs_h.csv"),
    "abl": P("run_v4", "phase1c", "ablate_per_instance.csv"),
    "1d": P("run_v4", "phase1d", "per_instance_1d.csv"),
    "1e": P("run_v4", "phase1e", "per_instance_1e.csv"),
    "br4": P("run_v4", "branches", "branches_weyl.csv"),
    "br6": P("run_v4", "branches_G6", "branches_weyl.csv"),
}
GAP29 = [P("run_v4", "parts", f"{t}_29.json") for t in ("Wg", "Wg07", "Wg05", "Wg035", "Wg025", "Wg018", "Wg012")]
INST = {s: [P("instances", f"{s}.npz"), P("instances", f"{s}_info.csv")] for s in ("G4", "G6")}


def globs(pattern, nmin):
    return len(glob.glob(P("run_v4", "parts", pattern))) >= nmin


def parts_ok():
    return (all(os.path.exists(f) for f in GAP29) and globs("NS_*.json", 500) and globs("Wq_*.json", 300)
            and all(globs(f"{t}_*.json", 100) for t in ("Wc", "Wc07", "Wc05", "Wc035", "Wc025", "Wc010", "Wc005",
                                                         "Wc002", "Wc0005", "Wc0001")))


# 1. unpack the pu1 JSON files if needed
if not parts_ok():
    if os.path.exists(TGZ):
        print(f"unpacking {TGZ} ...")
        with tarfile.open(TGZ) as tf:
            ok = [m for m in tf.getmembers() if m.isfile() and not m.name.startswith(("/", ".."))
                  and m.name.split("/")[0] in ("run_v4", "instances") and ".." not in m.name.split("/")]
            tf.extractall(ROOT, members=ok)
        print(f"  {len(ok)} files")
    else:
        print(f"MISSING: {TGZ} (run_v4/parts JSON files from pu1); Fig. 3 and Fig. 4 need it")

# 2. instances (deterministic generator)
for s, n in (("G4", 4), ("G6", 6)):
    if not all(os.path.exists(f) for f in INST[s]):
        print(f"generating instances/{s} with code/gen_instances.py (seed 2016) ...")
        subprocess.run([sys.executable, "gen_instances.py", "--set", s, "--n", str(n), "--ninst", "1000", "--seed", "2016",
                        "--xi0", "0.5", "--out", os.path.join("..", "instances")], cwd=P("code"), check=True)

# 3. inputs per figure
NEEDS = {
    "fig1_schematic": [],
    "fig2_who_wins": [CSV["1b"], CSV["br4"]],
    "fig_classical_failure": [CSV["br4"], "parts:NS,Wc,Wq"],
    "fig4_crossing_anatomy": [CSV["br4"]] + INST["G4"] + GAP29,
    "fig5_gap_law": [CSV["gap"], CSV["1b"], CSV["br4"]],
    "fig6_intervention": [CSV["1d"]],
    "fig7_sb_n6": [CSV["abl"], CSV["1d"], CSV["1b"], CSV["br4"], CSV["1e"]],
    "check_classes_n6": [CSV["1b"], CSV["br4"], CSV["br6"]] + INST["G4"] + INST["G6"],
}


def missing(req):
    out = []
    for f in req:
        if f == "parts:NS,Wc,Wq":
            if not parts_ok():
                out.append("run_v4/parts/{NS,Wc*,Wq}_*.json (from paper1_figdata.tgz)")
        elif not os.path.exists(f):
            out.append(os.path.relpath(f, ROOT))
    return out


# 4. run
log = []
status = []
for name, req in NEEDS.items():
    miss = missing(req)
    if miss:
        status.append((name, "SKIPPED, missing: " + "; ".join(miss)))
        continue
    r = subprocess.run([sys.executable, os.path.join(HERE, name + ".py")], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    log.append(f"== {name}\n{r.stdout}{r.stderr}")
    status.append((name, "ok" if r.returncode == 0 else "FAILED (see ALL_NUMBERS.txt)"))
with open(os.path.join(HERE, "ALL_NUMBERS.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(log))
print()
for name, st in status:
    print(f"{name:24s} {st}")
print(f"\nnumbers: {os.path.join(HERE, 'ALL_NUMBERS.txt')}")
