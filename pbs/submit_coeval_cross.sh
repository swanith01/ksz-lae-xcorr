#!/bin/bash
# Submit scripts/44 for several seeds (one job each).   SEEDS="1 2 3" bash pbs/submit_coeval_cross.sh     DRY=1 prints the qsub commands.
# Optional env: SPECS (";" between specs), WRAP_OFFSET, MAPS_ROOT, OUT_DIR.   Aggregate afterwards with scripts/45 (light).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p logs
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
for SEED in ${SEEDS}; do
  V="SEED=${SEED},WRAP_OFFSET=${WRAP_OFFSET:-0}"
  [ -n "${SPECS}" ] && V="${V},SPECS=\"${SPECS}\""
  [ -n "${MAPS_ROOT}" ] && V="${V},MAPS_ROOT=${MAPS_ROOT}"
  [ -n "${OUT_DIR}" ] && V="${V},OUT_DIR=${OUT_DIR}"
  if [ "${DRY:-0}" = "1" ]; then echo "DRY qsub -v ${V} pbs/pbs_coeval_cross.sh"
  else echo -n "seed ${SEED}: "; qsub -v "${V}" pbs/pbs_coeval_cross.sh; fi
done
