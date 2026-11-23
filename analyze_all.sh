#!/bin/bash
# All reports from whatever is finished; can be run at any time, also while run_all.sh is still running:
#     bash analyze_all.sh                 (options: RUN=run_v4 PLAN=plan.txt PY=python)
# Writes
#   RUN/report.txt                     scan + calibration (read this first)
#   RUN/scan/scan.txt, fig_scan.png    hbar and annealing-time scans, weyl family (code/scan.py)
#   RUN/calib/calib.txt, fig_calib_hist.png   classical calibration on all G0c instances (code/calib.py)
#   RUN/analysis_goto, RUN/analysis_weyl, RUN/analysis_<tag>   analyze.py for every main quantum/classical pair
#     (pair = quantum tag and the same tag with its first 'q' replaced by 'c': G0q/G0c, Wq05/Wc05, WqT200/WcT200)
set -uo pipefail
cd "$(dirname "$0")"
RUN=${RUN:-run_v4}
PLAN=${PLAN:-plan.txt}
PY=${PY:-python}
mkdir -p logs
ts() { date '+%F %T'; }

for q in $(awk '{sub(/#.*/, "")} NF == 14 && $2 == "quantum" && $1 !~ /^cv/ {print $1}' "$PLAN"); do
  c=${q/q/c}
  case "$q" in
    G0q) o=analysis_goto ;;
    Wq)  o=analysis_weyl ;;
    *)   o=analysis_$q ;;
  esac
  if $PY code/analyze.py --run "$RUN" --plan "$PLAN" --inst-dir instances --q "$q" --c "$c" --out "$RUN/$o" \
       > "logs/analyze_$q.log" 2>&1; then
    echo "$(ts) $RUN/$o/summary.txt"
  else
    echo "$(ts) $q + $c: not analysed yet ($(tail -n 1 "logs/analyze_$q.log"))"
  fi
done
$PY code/scan.py --run "$RUN" --plan "$PLAN" --inst-dir instances --out "$RUN/scan" > logs/scan.log 2>&1 \
  || echo "$(ts) scan.py failed: logs/scan.log"
$PY code/calib.py --run "$RUN" --inst-dir instances --out "$RUN/calib" > logs/calib.log 2>&1 \
  || echo "$(ts) calib.py failed: logs/calib.log"
{
  echo "MSBQA phase 1 report, $(date -Is), from $RUN ($(ls "$RUN"/parts/*.json 2>/dev/null | wc -l) finished tasks)"
  echo
  cat "$RUN/scan/scan.txt" 2>/dev/null || echo "(no scan report)"
  echo
  cat "$RUN/calib/calib.txt" 2>/dev/null || echo "(no calibration report)"
} > "$RUN/report.txt"
echo "$(ts) $RUN/report.txt written"
