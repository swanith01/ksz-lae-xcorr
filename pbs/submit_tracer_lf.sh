#!/bin/bash
# Submit scripts/39 for several seeds (one job each).   SEEDS="1 2 3" bash pbs/submit_tracer_lf.sh
#   DRY=1 ... prints the qsub commands.   EXTRA_ARGS="--every 2" passes through (use ";" not "," between specs: qsub -v splits on commas).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p logs
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
for SEED in ${SEEDS}; do
  V="SEED=${SEED}"
  [ -n "${EXTRA_ARGS}" ] && V="${V},EXTRA_ARGS=\"${EXTRA_ARGS}\""
  if [ "${DRY:-0}" = "1" ]; then echo "DRY qsub -v ${V} pbs/pbs_tracer_lf.sh"
  else echo -n "seed ${SEED}: "; qsub -v "${V}" pbs/pbs_tracer_lf.sh; fi
done
