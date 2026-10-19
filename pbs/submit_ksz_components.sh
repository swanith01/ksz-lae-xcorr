#!/bin/bash
# =============================================================================
# submit_ksz_components.sh -- component decomposition for several seeds:
#   per seed:  scripts/41 (component maps)
#   then ONE:  scripts/42 (cross with every tracer already in MAPS_ROOT) after all seeds finished OK.
#   SEEDS="1 2 3 4 5 6 7 8 9 10" bash pbs/submit_ksz_components.sh        DRY=1 prints the qsub commands.
# Optional env: WRAP_OFFSET, MAPS_ROOT, COMP_DIR, OUT_DIR (see pbs_ksz_components_cross.sh).
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p logs
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
COMP_DIR="${COMP_DIR:-data/products_tracers/components}"
Q() { if [ "${DRY:-0}" = "1" ]; then echo "DRY qsub $*" >&2; echo "DRY.$RANDOM"; else qsub "$@"; fi; }
IDS=""
for SEED in ${SEEDS}; do
  J=$(Q -v "SEED=${SEED},WRAP_OFFSET=${WRAP_OFFSET:-0},EXTRA_ARGS=\"--out-dir ${COMP_DIR}\"" pbs/pbs_ksz_components.sh)
  echo "  seed ${SEED}: components ${J}"
  IDS="${IDS}:${J}"
done
X=$(Q -W depend=afterok${IDS} -v "WRAP_OFFSET=${WRAP_OFFSET:-0},COMP_DIR=${COMP_DIR},MAPS_ROOT=${MAPS_ROOT:-data/products_tracers/maps},OUT_DIR=${OUT_DIR:-paper/figure_scripts/output_components}" pbs/pbs_ksz_components_cross.sh)
echo "  component x tracer cross-power -> ${X} (starts after all seeds finish OK)"
echo "Monitor: qstat -u ${USER}"
