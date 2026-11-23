#!/bin/bash
# MSBQA phase 1c, GPU part: minimum gaps on the 8 GPUs (code/gap_scan_gpu.py), plan_gpu_1c.txt. From msbqa_phase1/,
# in the conda env dsbtn; can run at the same time as the CPU ablation (plan_1c.txt):
#     nohup bash run_gpu_1c.sh > run_gpu_1c.log 2>&1 &
# Steps: 0 instances/G6 (n = 6, generated once, same generator and seed as G4)
#        1 classical branches of G6 (code/branches.py, first 300 instances) -> RUN/branches_G6/
#        2 instance selections for n = 6 (code/select_1c.py) -> RUN/tasks/g6_select_8_8.txt, g6_select_40_30.txt
#        3 per-GPU task lists (code/make_tasks_gpu_1c.py); one sequential worker per GPU
#        4 report (code/phase1c_gap_report.py) -> RUN/phase1c/report_1c_gap.txt
# Options (environment): RUN=run_v4  PLAN=plan_gpu_1c.txt (trimmed) or plan_gpu_1c_full.txt  NGPU=8  PY=python
# Restart-safe: finished parts are skipped; run the same command again after an interruption.
# Progress:  tail logs/gpu1c_0.log | wc -l run_v4/parts/Wg012.log run_v4/parts/W6g.log
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan_gpu_1c.txt}
NGPU=${NGPU:-8}
PY=${PY:-python}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p logs "$RUN/parts" "$RUN/tasks"
ts() { date '+%F %T'; }

$PY -c "import numpy, scipy, cupy; import cupyx.scipy.sparse.linalg as l; l.lobpcg" \
  || { echo "this Python needs numpy, scipy, cupy with cupyx.scipy.sparse.linalg.lobpcg"; exit 1; }
if [ ! -f instances/G6.npz ]; then
  (cd code && $PY gen_instances.py --set G6 --n 6 --ninst 1000 --seed 2016 --xi0 0.5 --out ../instances)
fi
if [ ! -f "$RUN/branches_G6/branches_weyl.csv" ]; then
  echo "$(ts) classical branches of G6 (first 300 instances)"
  $PY code/branches.py --inst-dir instances --set G6 --run "$RUN" --out "$RUN/branches_G6" --ninst 300
fi
for sel in "8 8" "40 30"; do                                    # n = 6 selections of the trimmed / full plan
  set -- $sel
  fsel="$RUN/tasks/g6_select_$1_$2.txt"
  [ -f "$fsel" ] || $PY code/select_1c.py --branches "$RUN/branches_G6/branches_weyl.csv" --out "$fsel" \
      --ncross "$1" --ncontrol "$2"
done
{
  echo "date: $(date -Is)  host: $(hostname)  gpus used: $NGPU"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
  $PY -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)"
  echo "sha256:"; sha256sum code/*.py "$PLAN" instances/G4.npz instances/G6.npz
} > "$RUN/provenance_gpu_1c.txt"

$PY code/make_tasks_gpu_1c.py --plan "$PLAN" --run "$RUN" --code code --inst-dir instances --ngpu "$NGPU" --py "$PY"
echo "$(ts) starting $NGPU GPU workers (logs/gpu1c_<g>.log; per-tag results: $RUN/parts/<tag>.log)"
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    while IFS= read -r cmd; do
      [ -z "$cmd" ] && continue
      bash -c "$cmd" || echo "$(ts) FAILED: $cmd"
    done < "$RUN/tasks/gpu1c_$g.txt"
  ) > "logs/gpu1c_$g.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
nfail=$(cat logs/gpu1c_*.log | grep -c "FAILED:" || true)
echo "$(ts) GPU part finished, $nfail failed tasks (see logs/gpu1c_*.log and $RUN/parts/<tag>.log)"

$PY code/phase1c_gap_report.py --run "$RUN" --inst-dir instances > logs/phase1c_gap_report.log 2>&1 \
  && echo "$(ts) report: $RUN/phase1c/report_1c_gap.txt" \
  || echo "$(ts) phase1c_gap_report.py failed: logs/phase1c_gap_report.log"
if [ "$nfail" -eq 0 ]; then echo "$(ts) ALL_DONE"; else echo "$(ts) FINISHED_WITH_FAILURES (run again)"; exit 1; fi
