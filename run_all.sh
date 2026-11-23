#!/bin/bash
# MSBQA phase 1, everything on the GPU server (8 GPUs + its CPUs), one command. From the msbqa_phase1 directory,
# in the conda env that has numpy, scipy and cupy (dsbtn):
#     nohup bash run_all.sh > run_all.log 2>&1 &
# Steps:
#   1 instances/G4.npz: generated if missing (same command and seed as before -> same instances)
#   2 self-tests, CPU (code/selftest.py) and GPU (code/selftest_gpu.py) side by side; any failure stops here
#   3 at the same time: the quantum lines of the plan on the GPUs (one worker per GPU, instances i % NGPU == g)
#                       the classical lines on the CPUs (CPU_WORKERS single-threaded processes, niced so that
#                       the GPU drivers keep their cores)
#   4 analysis: bash analyze_all.sh -> RUN/report.txt (scan + calibration) and RUN/analysis_*/summary.txt
#     (analyze_all.sh can also be run by hand at any time for the results finished so far)
# Options (environment): RUN=run_v4  PLAN=plan.txt  NGPU=8  CPU_WORKERS=(nproc - NGPU - 4)  BATCH=64  PY=python
#                        SKIP_SELFTEST=1 (e.g. when re-running after adding plan lines)
# Restart-safe: finished parts (RUN/parts/<tag>_<inst>.json) are skipped. After an interruption, or after
# switching on more plan lines, run the same command again.
# Progress:  tail run_all.log | tail logs/gpu_0.log | cat RUN/parts/G0q.log | ls RUN/parts/*.json | wc -l
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan.txt}
NGPU=${NGPU:-8}
BATCH=${BATCH:-64}
PY=${PY:-python}
NCPU=$(nproc)
CPU_WORKERS=${CPU_WORKERS:-$((NCPU - NGPU - 4))}
[ "$CPU_WORKERS" -lt 1 ] && CPU_WORKERS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p logs "$RUN/parts"
ts() { date '+%F %T'; }

# 1 environment + instances
$PY -c "import numpy, scipy, cupy" || { echo "this Python needs numpy, scipy and cupy"; exit 1; }
if [ ! -f instances/G4.npz ]; then
  (cd code && $PY gen_instances.py --set G4 --n 4 --ninst 1000 --seed 2016 --xi0 0.5 --out ../instances)
fi
{
  echo "date: $(date -Is)  host: $(hostname)  cpus: $NCPU  gpus used: $NGPU  cpu workers: $CPU_WORKERS"
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
  $PY -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)"
  echo "sha256:"; sha256sum code/*.py "$PLAN" instances/G4.npz
} > "$RUN/provenance.txt"

# 2 self-tests (CPU and GPU at the same time)
if [ -z "${SKIP_SELFTEST:-}" ]; then
  echo "$(ts) self-tests: $RUN/selftest.txt, $RUN/selftest_gpu.txt"
  (cd code && $PY selftest.py) > "$RUN/selftest.txt" 2>&1 & p1=$!
  (cd code && CUDA_VISIBLE_DEVICES=0 $PY selftest_gpu.py) > "$RUN/selftest_gpu.txt" 2>&1 & p2=$!
  ok=1
  wait "$p1" || ok=0
  wait "$p2" || ok=0
  tail -n 1 "$RUN/selftest.txt" "$RUN/selftest_gpu.txt"
  if [ "$ok" -eq 0 ]; then
    echo "$(ts) SELFTEST_FAILED: nothing else started"
    exit 1
  fi
fi

# 3 quantum lines on the GPUs, classical lines on the CPUs, side by side
$PY code/make_tasks.py --plan "$PLAN" --run "$RUN" --code code --inst-dir instances --shards 1 \
    --kind classical --tag cpu --py "$PY"
CPU_TASKS=$RUN/tasks/shard_cpu_0.txt
echo "$(ts) starting $NGPU GPU workers (logs/gpu_<g>.log) and $(wc -l < "$CPU_TASKS") classical tasks on" \
     "$CPU_WORKERS CPU processes (results: $RUN/parts/<tag>.log)"
gpids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$g $PY code/run_quantum_gpu.py --plan "$PLAN" --run "$RUN" --inst-dir instances \
      --gpu "$g" --ngpu "$NGPU" --batch "$BATCH" > "logs/gpu_$g.log" 2>&1 &
  gpids+=($!)
done
nice -n 10 xargs -d '\n' -P "$CPU_WORKERS" -I{} bash -c '{}' < "$CPU_TASKS" > logs/cpu.log 2>&1 &
cpid=$!
fail=0
if wait "$cpid"; then
  echo "$(ts) classical part done"
else
  echo "$(ts) some classical tasks failed: see logs/cpu.log and $RUN/parts/<tag>.log"
  fail=1
fi
for pid in "${gpids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -eq 0 ]; then
  echo "$(ts) quantum part done"
else
  echo "$(ts) a GPU worker failed: see logs/gpu_*.log"
fi

# 4 analysis (also after a partial failure: the reports cover what is complete)
RUN="$RUN" PLAN="$PLAN" PY="$PY" bash analyze_all.sh || echo "$(ts) analyze_all.sh failed"
if [ "$fail" -eq 0 ]; then
  echo "$(ts) ALL_DONE"
else
  echo "$(ts) FINISHED_WITH_FAILURES (run the same command again to redo only the missing tasks)"
  exit 1
fi
