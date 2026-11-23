#!/bin/bash
# MSBQA phase 1b, GPU part: the quantum lines of plan_gpu_1b.txt on all GPUs (one worker per GPU, the unchanged
# code/run_quantum_gpu.py of phase 1), then all reports. From the msbqa_phase1 directory, in the conda env dsbtn,
# at the same time as the CPU part:
#     nohup bash run_1b.sh > run_1b.log 2>&1 &
#     nohup bash run_gpu_1b.sh > run_gpu_1b.log 2>&1 &
# Options (environment): RUN=run_v4  PLAN=plan_gpu_1b.txt  NGPU=8  BATCH=64  PY=python
# Restart-safe: finished parts (RUN/parts/<tag>_<inst>.json) are skipped; run the same command again after an
# interruption. Progress: tail logs/gpu1b_0.log | wc -l run_v4/parts/G0q.log run_v4/parts/Wq.log
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan_gpu_1b.txt}
NGPU=${NGPU:-8}
BATCH=${BATCH:-64}
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p logs "$RUN/parts"
ts() { date '+%F %T'; }

[ -f instances/G4.npz ] || { echo "instances/G4.npz missing (run from msbqa_phase1/)"; exit 1; }
$PY -c "import numpy, scipy, cupy" || { echo "this Python needs numpy, scipy and cupy"; exit 1; }
{
  echo "date: $(date -Is)  host: $(hostname)  gpus used: $NGPU  batch: $BATCH"
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
  $PY -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)"
  echo "sha256:"; sha256sum code/*.py "$PLAN" instances/G4.npz
} > "$RUN/provenance_gpu_1b.txt"

echo "$(ts) starting $NGPU GPU workers on $PLAN (logs/gpu1b_<g>.log)"
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$g $PY code/run_quantum_gpu.py --plan "$PLAN" --run "$RUN" --inst-dir instances \
      --gpu "$g" --ngpu "$NGPU" --batch "$BATCH" > "logs/gpu1b_$g.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -eq 0 ]; then
  echo "$(ts) quantum part done"
else
  echo "$(ts) a GPU worker failed: see logs/gpu1b_*.log"
fi

# reports from everything finished so far (phase 1 reports now include the new instances; then phase 1b)
RUN="$RUN" PY="$PY" bash analyze_all.sh || echo "$(ts) analyze_all.sh failed"
$PY code/phase1b_report.py --run "$RUN" --inst-dir instances --out "$RUN/phase1b" > logs/phase1b_report.log 2>&1 \
  && echo "$(ts) report: $RUN/phase1b/report_1b.txt" \
  || echo "$(ts) phase1b_report.py failed: logs/phase1b_report.log"
if [ "$fail" -eq 0 ]; then
  echo "$(ts) ALL_DONE"
else
  echo "$(ts) FINISHED_WITH_FAILURES (run the same command again to redo only the missing instances)"
  exit 1
fi
