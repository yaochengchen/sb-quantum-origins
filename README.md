# sb-quantum-origins

Code, instances, results, and figure scripts for

> Y. Chen and J. Zhuang, *Revisiting the quantum origins of simulated bifurcation: Tunneling bottlenecks and classical alternatives* (2026), arXiv:XXXX.XXXXX.

The paper compares closed Kerr-parametric-oscillator (KPO) networks with their classical counterparts and with tuned simulated-bifurcation (SB) algorithms on the same Ising instances: classical configuration branches, quantum spectra in the even-parity sector, direct Schrödinger propagation, truncated Wigner dynamics, and SB ablations.

## Reproduce the figures (about one minute, no simulation)

```bash
pip install -r requirements.txt
python paper1_figs/make_paper1_v16_figs.py
```

Run from this folder. The script unpacks `paper1_figdata.tgz` into `run_v4/parts/` if needed, writes the seven figures to `paper1_figs/` (`.pdf`, `.png`), and collects every printed number in `paper1_figs/ALL_NUMBERS_v16.txt`.

| Figure | Script | Inputs |
|---|---|---|
| 1 | `paper1_figs/fig1_B1.py` | schematic, no data |
| 2 | `paper1_figs/fig2_who_wins_v16.py` | `run_v4/phase1b/per_instance_1b.csv`, `run_v4/branches/branches_weyl.csv` |
| 3 | `paper1_figs/fig_classical_failure_v16.py` | `run_v4/parts/{NS,Wc*,Wq}_*.json`, `run_v4/branches/branches_weyl.csv` |
| 4 | `paper1_figs/fig4_crossing_anatomy_v16.py` | `instances/G4.npz`, `run_v4/parts/Wg*_29.json` |
| 5 | `paper1_figs/fig5_gap_law_v16.py` | `run_v4/phase1c/gap_vs_h.csv`, `run_v4/phase1b/per_instance_1b.csv` |
| 6 | `paper1_figs/fig6_intervention_v14.py` | `run_v4/phase1d/per_instance_1d.csv` |
| 7 | `paper1_figs/fig7_sb_n6_v15.py` | `run_v4/phase1c/ablate_per_instance.csv`, `run_v4/phase1d/`, `run_v4/phase1e/` |

## Layout

- `code/` — models and runs. `kpo_common.py` (KPO Hamiltonian, Magnus–Lanczos propagation, classical layers L1–L4), `kpo_gpu.py` (batched GPU propagation), `gap_scan.py` / `gap_scan_gpu.py` (LOBPCG gap solver), `branches.py` / `branches_uniform.py` (classical configuration branches), `noise_scan.py`, `sb_tune.py`, `sb_ablate.py`, `iwindow.py`, task builders, self-tests, and the `phase1*_report.py` analysis scripts.
- `instances/` — Ising instances G4 (n = 4, 1000), G4U (uniform-max detuning), G6 (n = 6, 300); `code/gen_instances.py --seed 2016` regenerates them.
- `run_v4/` — results: per-task JSON files in `parts/`, raw noise-scan trajectories in `raw/`, and per-phase summaries and reports (`phase1b`–`phase1e`, `branches*`, `analysis_*`).
- `paper1_figdata.tgz` — the per-task JSON files used by Figs. 3 and 4.
- `plan*.txt`, `run_*.sh`, `analyze_all.sh`, `gpu/` — run plans and launch scripts used for the production runs (one GPU server with eight NVIDIA RTX PRO 6000 GPUs).

## Re-running the simulations

Quantum runs need CUDA and CuPy; classical runs need only NumPy. The launch scripts read the plan files and skip finished tasks:

```bash
python code/selftest.py && python code/selftest_gpu.py
bash run_all.sh                                   # quantum and classical KPO lines of plan.txt; reports via analyze_all.sh
python code/branches.py --inst-dir instances --run run_v4            # classical configuration branches of G4
python code/branches_uniform.py --inst-dir instances --run run_v4    # branches with uniform detunings
bash run_1b.sh                                    # noise scan and SB grid (plan_1b.txt)
PLAN=plan_gap_1b.txt bash run_1b.sh               # gaps at h = 1 (both families) and h = 0.7, 0.5 (CPU)
PLAN=plan_1c.txt bash run_1b.sh                   # L4 <-> bSB* ablation; then python code/ablate_report.py --run run_v4 --inst-dir instances
bash run_gpu_1b.sh                                # quantum propagation on GPUs (plan_gpu_1b.txt)
bash run_gpu_1c.sh                                # small-h gaps, G6 branches and n = 6 gaps on GPUs (plan_gpu_1c.txt)
bash run_1d.sh                                    # detuning intervention and Ising-window holds (plan_1d.txt)
bash run_1e.sh                                    # n = 6 classical baselines and extended n = 6 gap scans
```

Each launcher writes its report to `run_v4/` (`report.txt`, `phase1b/` to `phase1e/`); the per-phase command details are in the headers of the scripts.

## Requirements

Python 3.10 or later with NumPy, SciPy, pandas, and Matplotlib; CuPy for the GPU runs.

## License

Code: MIT License (`LICENSE`). Instances, results, and figures: CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

If you use this code or data, please cite the paper above.
