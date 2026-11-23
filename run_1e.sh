#!/bin/bash
# MSBQA 1e: the n = 6 check for paper 1 (the last computation planned for it). From msbqa_phase1/, in the conda env dsbtn:
#     nohup bash run_1e.sh > run_1e.log 2>&1 &
# CPU and GPU at the same time:
#   GPU  n = 6 gap scans with edge = 1 in W6g redone over p in [0, 7], cutoff 22 (code/select_1e.py -> tag W6g7)
#   CPU  plan_1e.txt: NS6 (noise scan) and SBT6 (bSB / dSB grid) on G6 0-299 (code/make_tasks_1d.py)
#   then the report RUN/phase1e/report_1e.txt (code/phase1e_report.py; can be run by hand at any time)
# Options (environment): RUN=run_v4  PLAN=plan_1e.txt  WORKERS=180 (capped at nproc - 12)  NGPU=8  PY=python
# Restart-safe: finished parts are skipped; run the same command again after an interruption.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan_1e.txt}
PY=${PY:-python}
NGPU=${NGPU:-8}
NCPU=$(nproc)
WORKERS=${WORKERS:-180}
if [ "$WORKERS" -gt $((NCPU - 12)) ]; then WORKERS=$((NCPU - 12)); fi
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
export PYTHONUNBUFFERED=1
mkdir -p logs "$RUN/parts" "$RUN/tasks" "$RUN/raw" "$RUN/phase1e"
ts() { date '+%F %T'; }
PNAME=$(basename "$PLAN" .txt)

for f in instances/G6.npz instances/G6_info.csv "$RUN/branches_G6/branches_weyl.csv" \
         "$RUN/branches/branches_weyl.csv" "$RUN/phase1b/per_instance_1b.csv"; do
  [ -f "$f" ] || { echo "$f missing (run from msbqa_phase1/)"; exit 1; }
done
{
  echo "date: $(date -Is)  host: $(hostname)  cpus: $NCPU  workers: $WORKERS  gpus: $NGPU"
  $PY -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
  echo "sha256:"; sha256sum code/*.py "$PLAN" instances/G6.npz
} > "$RUN/provenance_$PNAME.txt"

# ---- GPU part (background): one sequential worker per GPU
$PY code/select_1e.py --run "$RUN" --code code --inst-dir instances --ngpu "$NGPU" --py "$PY"
echo "$(ts) GPU: starting $NGPU workers (logs/gpu1e_<g>.log; results $RUN/parts/W6g7_*.json)"
gpids=()
for g in $(seq 0 $((NGPU - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    while IFS= read -r cmd; do
      [ -z "$cmd" ] && continue
      bash -c "$cmd" || echo "$(ts) FAILED: $cmd"
    done < "$RUN/tasks/gpu1e_$g.txt"
  ) > "logs/gpu1e_$g.log" 2>&1 &
  gpids+=($!)
done

# ---- CPU part (foreground)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
$PY code/make_tasks_1d.py --plan "$PLAN" --run "$RUN" --code code --inst-dir instances --py "$PY"
TASKS=$RUN/tasks/tasks_$PNAME.txt
echo "$(ts) CPU: $(wc -l < "$TASKS") tasks on $WORKERS processes (per-tag logs: $RUN/parts/<tag>.log)"
fail=0
nice -n 10 xargs -d '\n' -P "$WORKERS" -I{} bash -c '{}' < "$TASKS" > "logs/cpu_$PNAME.log" 2>&1 || fail=1
echo "$(ts) CPU part finished ($([ "$fail" -eq 0 ] && echo ok || echo "some tasks failed: logs/cpu_$PNAME.log"))"

for pid in "${gpids[@]}"; do wait "$pid" || true; done
nfail=$(cat logs/gpu1e_*.log | grep -c "FAILED:" || true)
echo "$(ts) GPU part finished, $nfail failed tasks"

$PY code/phase1e_report.py --run "$RUN" --inst-dir instances > logs/phase1e_report.log 2>&1 \
  && echo "$(ts) report: $RUN/phase1e/report_1e.txt" \
  || echo "$(ts) phase1e_report.py failed: logs/phase1e_report.log"
if [ "$fail" -eq 0 ] && [ "$nfail" -eq 0 ]; then
  echo "$(ts) ALL_DONE"
else
  echo "$(ts) FINISHED_WITH_FAILURES (run the same command again to redo only the missing tasks)"
  exit 1
fi
