#!/bin/bash
# All quantum (L0) lines of slurm/plan.txt on one machine with several GPUs (no Slurm needed).
# From the msbqa_phase1 directory, in the Python env that has cupy:
#     nohup bash gpu/run_gpu.sh > gpu_run.log 2>&1 &
# Options (environment): NGPU=8  RUN=run_main  PLAN=slurm/plan.txt  TAGS=G0q,Wq (default: every quantum line)
#                        BATCH=64 (max instances evolved together on one GPU)  PY=python
# Steps:
#   1 instances/G4.npz: generated if missing (same command and seed as slurm/00_prep.sbatch -> same instances)
#   2 GPU self-test (code/selftest_gpu.py, on GPU 0); on failure nothing else starts
#   3 one worker per GPU (CUDA_VISIBLE_DEVICES=g) takes the instances i with i % NGPU == g.
#     Progress: logs/gpu_<g>.log. Finished instances: RUN/parts/<tag>_<i>.json and one line in RUN/parts/<tag>.log.
# Restart-safe: running it again only does the instances that are not finished.
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=${NGPU:-8}
RUN=${RUN:-run_main}
PLAN=${PLAN:-slurm/plan.txt}
BATCH=${BATCH:-64}
PY=${PY:-python}
mkdir -p logs "$RUN/parts"
$PY -c "import numpy, scipy, cupy" || { echo "this Python needs numpy, scipy and cupy (e.g. pip install scipy)"; exit 1; }
if [ ! -f instances/G4.npz ]; then
  (cd code && $PY gen_instances.py --set G4 --n 4 --ninst 1000 --seed 2016 --xi0 0.5 --out ../instances)
fi
echo "$(date -Is) GPU self-test"
(cd code && CUDA_VISIBLE_DEVICES=0 $PY selftest_gpu.py) | tee "$RUN/selftest_gpu.txt"
echo "$(date -Is) starting $NGPU GPU workers (progress: logs/gpu_<g>.log)"
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$g $PY code/run_quantum_gpu.py --plan "$PLAN" --run "$RUN" --inst-dir instances \
      --gpu "$g" --ngpu "$NGPU" --batch "$BATCH" ${TAGS:+--tags "$TAGS"} > "logs/gpu_$g.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -eq 0 ]; then
  echo "$(date -Is) ALL_GPU_DONE"
else
  echo "$(date -Is) SOME_GPU_WORKER_FAILED: see logs/gpu_*.log"
  exit 1
fi
