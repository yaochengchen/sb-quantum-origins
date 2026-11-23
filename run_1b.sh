#!/bin/bash
# MSBQA phase 1b, CPU part (plan_1b.txt), on the GPU server. From the msbqa_phase1 directory, in the conda env dsbtn,
# at the same time as the GPU part (run_gpu_1b.sh, plan_gpu_1b.txt):
#     nohup bash run_1b.sh > run_1b.log 2>&1 &
#     nohup bash run_gpu_1b.sh > run_gpu_1b.log 2>&1 &
# Steps: task list from plan_1b.txt (finished tasks skipped) -> all tasks on WORKERS single-threaded processes
#        -> report RUN/phase1b/report_1b.txt (code/phase1b_report.py; can be run by hand at any time).
# Options (environment): RUN=run_v4  PLAN=plan_1b.txt  WORKERS=(nproc - 12: leaves cores for the 8 GPU workers)  PY=python
# Restart-safe: after an interruption run the same command again.
# Gap lines (plan_gap_1b.txt) separately, e.g. at the same time with fewer workers:
#     PLAN=plan_gap_1b.txt WORKERS=40 nohup bash run_1b.sh > run_gap_1b.log 2>&1 &
# Task list, error log and provenance are named after the plan: RUN/tasks/tasks_<plan>.txt, logs/cpu_<plan>.log,
# RUN/provenance_<plan>.txt.
# Progress:  tail run_1b.log | wc -l run_v4/parts/NS.log run_v4/parts/SBT.log run_v4/parts/Wg.log | cat logs/cpu_plan_1b.log
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan_1b.txt}
PY=${PY:-python}
NCPU=$(nproc)
WORKERS=${WORKERS:-$((NCPU - 12))}
[ "$WORKERS" -lt 1 ] && WORKERS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p logs "$RUN/parts" "$RUN/raw"
ts() { date '+%F %T'; }
PNAME=$(basename "$PLAN" .txt)

[ -f instances/G4.npz ] || { echo "instances/G4.npz missing (run from msbqa_phase1/)"; exit 1; }
$PY -c "import numpy, scipy" || { echo "this Python needs numpy and scipy"; exit 1; }
{
  echo "date: $(date -Is)  host: $(hostname)  cpus: $NCPU  workers: $WORKERS"
  $PY -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
  echo "sha256:"; sha256sum code/*.py "$PLAN" instances/G4.npz
} > "$RUN/provenance_$PNAME.txt"

$PY code/make_tasks_1b.py --plan "$PLAN" --run "$RUN" --code code --inst-dir instances --py "$PY"
TASKS=$RUN/tasks/tasks_$PNAME.txt
echo "$(ts) starting $(wc -l < "$TASKS") tasks on $WORKERS CPU processes (per-tag logs: $RUN/parts/<tag>.log)"
fail=0
nice -n 10 xargs -d '\n' -P "$WORKERS" -I{} bash -c '{}' < "$TASKS" > "logs/cpu_$PNAME.log" 2>&1 || fail=1
if [ "$fail" -eq 0 ]; then
  echo "$(ts) all tasks done"
else
  echo "$(ts) some tasks failed: see logs/cpu_$PNAME.log and $RUN/parts/<tag>.log"
fi

$PY code/phase1b_report.py --run "$RUN" --inst-dir instances --out "$RUN/phase1b" > "logs/phase1b_report_$PNAME.log" 2>&1 \
  && echo "$(ts) report: $RUN/phase1b/report_1b.txt" \
  || echo "$(ts) phase1b_report.py failed: logs/phase1b_report_$PNAME.log"
if [ "$fail" -eq 0 ]; then
  echo "$(ts) ALL_DONE"
else
  echo "$(ts) FINISHED_WITH_FAILURES (run the same command again to redo only the missing tasks)"
  exit 1
fi
