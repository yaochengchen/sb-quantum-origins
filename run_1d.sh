#!/bin/bash
# MSBQA 1d, CPU only (plan_1d.txt). From the msbqa_phase1 directory, in the conda env dsbtn:
#     nohup bash run_1d.sh > run_1d.log 2>&1 &
# Does not touch the GPU job. Steps:
#   0. uniform set instances/G4U (code/make_uniform_set.py), instance selection (code/select_1d.py),
#      set checks (code/phase1d_report.py --checks-only): stops here if a check fails, before any long task
#   1. tasks of plan_1d.txt on WORKERS single-threaded processes (finished tasks skipped)
#   2. report RUN/phase1d/report_1d.txt (code/phase1d_report.py; can be run by hand at any time)
# Options (environment): RUN=run_v4  PLAN=plan_1d.txt  WORKERS=180 (capped at nproc - 8)  PY=python
# Restart-safe: after an interruption run the same command again.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan_1d.txt}
PY=${PY:-python}
NCPU=$(nproc)
WORKERS=${WORKERS:-180}
if [ "$WORKERS" -gt $((NCPU - 8)) ]; then WORKERS=$((NCPU - 8)); fi
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p logs "$RUN/parts" "$RUN/tasks" "$RUN/phase1d"
ts() { date '+%F %T'; }
PNAME=$(basename "$PLAN" .txt)

for f in instances/G4.npz instances/G4_info.csv "$RUN/branches_uniform/branches_weyl_kpo.csv" \
         "$RUN/branches_uniform/branches_weyl_mean.csv" "$RUN/branches_uniform/branches_weyl_max.csv"; do
  [ -f "$f" ] || { echo "$f missing (run from msbqa_phase1/; branches_uniform.py must have been run)"; exit 1; }
done

echo "$(ts) step 0: uniform set, selection, checks"
$PY code/make_uniform_set.py --inst-dir instances --src G4 --dst G4U --mode max
$PY code/select_1d.py --branches-dir "$RUN/branches_uniform" --out-dir "$RUN/tasks"
$PY code/phase1d_report.py --run "$RUN" --inst-dir instances --checks-only || { echo "$(ts) set checks FAILED: stopped"; exit 1; }

{
  echo "date: $(date -Is)  host: $(hostname)  cpus: $NCPU  workers: $WORKERS"
  $PY -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
  echo "sha256:"; sha256sum code/*.py "$PLAN" instances/G4.npz instances/G4U.npz "$RUN"/tasks/sel_1d_*.txt
} > "$RUN/provenance_$PNAME.txt"

$PY code/make_tasks_1d.py --plan "$PLAN" --run "$RUN" --code code --inst-dir instances --py "$PY"
TASKS=$RUN/tasks/tasks_$PNAME.txt
echo "$(ts) step 1: $(wc -l < "$TASKS") tasks on $WORKERS CPU processes (per-tag logs: $RUN/parts/<tag>.log)"
fail=0
nice -n 10 xargs -d '\n' -P "$WORKERS" -I{} bash -c '{}' < "$TASKS" > "logs/cpu_$PNAME.log" 2>&1 || fail=1
if [ "$fail" -eq 0 ]; then echo "$(ts) all tasks done"; else echo "$(ts) some tasks failed: see logs/cpu_$PNAME.log"; fi

$PY code/phase1d_report.py --run "$RUN" --inst-dir instances > "logs/phase1d_report.log" 2>&1 \
  && echo "$(ts) report: $RUN/phase1d/report_1d.txt" \
  || echo "$(ts) phase1d_report.py failed: logs/phase1d_report.log"
if [ "$fail" -eq 0 ]; then
  echo "$(ts) ALL_DONE"
else
  echo "$(ts) FINISHED_WITH_FAILURES (run the same command again to redo only the missing tasks)"
  exit 1
fi
