"""MSBQA paper 1, manuscript v16 (pra_A_draft_v16.tex): make all seven figures on THIS computer from msbqa_phase1 only.

    cd D:\\software\\Jiapei\\Make_SB_Quantum_Again\\msbqa_phase1
    python paper1_figs\\make_paper1_v16_figs.py [--copy-to <manuscript folder>]

Needs Python with numpy, scipy, pandas, matplotlib (about one minute). Reads existing results only; no simulation.
 1. If the per-instance JSON files are not in run_v4/parts/, unpacks them from paper1_figdata.tgz (msbqa_phase1/).
 2. If instances/G4.npz is missing, regenerates it with code/gen_instances.py (seed 2016, identical to pu1).
 3. Runs the figure scripts below; each writes paper1_figs/<file>.pdf (and .png); printed numbers are collected in
    paper1_figs/ALL_NUMBERS_v16.txt.
 4. With --copy-to, copies the seven PDFs (the file names used in the manuscript) into that folder.

 Fig. 1  fig1_B1.py                       schematic (no data)
 Fig. 2  fig2_who_wins_v16.py             per-instance advantage over six baselines
 Fig. 3  fig_classical_failure_v16.py     early amplification, noise scan, TWA vs classical noise
 Fig. 4  fig4_crossing_anatomy_v16.py     instance #29: classical branches and quantum gaps
 Fig. 5  fig5_gap_law_v16.py              gap position, exponential law, action, Landau-Zener
 Fig. 6  fig6_intervention_v14.py         paired detuning intervention
 Fig. 7  fig7_sb_n6_v15.py                ablations, Ising-window holds, n = 4 vs 6
The analysis and simulation code that produced the result files is in msbqa_phase1/code/ (see msbqa_phase1/README*.md).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = lambda *a: os.path.join(ROOT, *a)
FIGS = [("fig1_B1", "fig1_B1.pdf"),
        ("fig2_who_wins_v16", "fig2_who_wins_v16.pdf"),
        ("fig_classical_failure_v16", "fig_classical_failure_v16.pdf"),
        ("fig4_crossing_anatomy_v16", "fig4_crossing_anatomy_v16.pdf"),
        ("fig5_gap_law_v16", "fig5_gap_law_v16.pdf"),
        ("fig6_intervention_v14", "fig6_intervention_v14.pdf"),
        ("fig7_sb_n6_v15", "fig7_sb_n6_v15.pdf")]

ap = argparse.ArgumentParser()
ap.add_argument("--copy-to", default="", help="folder of pra_A_draft_v16.tex; the PDFs are copied there")
a = ap.parse_args()


def parts_ok():
    need = [P("run_v4", "parts", f"{t}_29.json") for t in ("Wg", "Wg07", "Wg05", "Wg035", "Wg025")]
    return (all(os.path.exists(f) for f in need) and len(glob.glob(P("run_v4", "parts", "NS_*.json"))) >= 500
            and len(glob.glob(P("run_v4", "parts", "Wq_*.json"))) >= 100
            and all(len(glob.glob(P("run_v4", "parts", f"{t}_*.json"))) >= 100
                    for t in ("Wc", "Wc07", "Wc05", "Wc035", "Wc025", "Wc010", "Wc005", "Wc002", "Wc0005", "Wc0001")))


if not parts_ok():
    tgz = P("paper1_figdata.tgz")
    if not os.path.exists(tgz):
        raise SystemExit(f"missing {tgz} (run_v4/parts JSON files); Figs. 3 and 4 need it")
    print(f"unpacking {tgz} ...")
    with tarfile.open(tgz) as tf:
        ok = [m for m in tf.getmembers() if m.isfile() and not m.name.startswith(("/", ".."))
              and m.name.split("/")[0] in ("run_v4", "instances") and ".." not in m.name.split("/")]
        tf.extractall(ROOT, members=ok)
if not os.path.exists(P("instances", "G4.npz")):
    subprocess.run([sys.executable, "gen_instances.py", "--set", "G4", "--n", "4", "--ninst", "1000", "--seed", "2016",
                    "--xi0", "0.5", "--out", os.path.join("..", "instances")], cwd=P("code"), check=True)

log, status = [], []
for script, pdf in FIGS:
    r = subprocess.run([sys.executable, os.path.join(HERE, script + ".py")], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    log.append(f"== {script}\n{r.stdout}{r.stderr}")
    good = r.returncode == 0 and os.path.exists(os.path.join(HERE, pdf))
    status.append((script, "ok" if good else "FAILED (see ALL_NUMBERS_v16.txt)"))
    if good and a.copy_to:
        shutil.copy2(os.path.join(HERE, pdf), os.path.join(a.copy_to, pdf))
with open(os.path.join(HERE, "ALL_NUMBERS_v16.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(log))
for script, st in status:
    print(f"{script:28s} {st}")
print(f"numbers: {os.path.join(HERE, 'ALL_NUMBERS_v16.txt')}" + (f";  PDFs copied to {a.copy_to}" if a.copy_to else ""))
