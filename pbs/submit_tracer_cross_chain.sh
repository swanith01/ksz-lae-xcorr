#!/bin/bash
# =============================================================================
# submit_tracer_cross_chain.sh -- the whole tracer cross-correlation suite for several seeds:
#   per seed:  scripts/37 (tracer window maps)       -- needs that seed's kSZ lpmaps pickle already in place
#   then ONE:  scripts/28 for every tracer (pbs_tracer_cross.sh) after all seeds finished OK.
#   SEEDS="1 2 3" bash pbs/submit_tracer_cross_chain.sh        DRY=1 prints the qsub commands.
# Optional env: EXTRA_ARGS (passed to scripts/37, use ";" between specs), WRAP_OFFSET, KSZ_CONFIG (default fiducial = 21cmFAST x_HII;
#   configs/variants/zreion_xh.yaml = zreion x_HII -- also set MAPS_ROOT/OUT_ROOT so the two do not overwrite each other).
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p logs
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
KSZ_CONFIG="${KSZ_CONFIG:-configs/fiducial.yaml}"
MAPS_ROOT="${MAPS_ROOT:-data/products_tracers/maps}"
OUT_ROOT="${OUT_ROOT:-paper/figure_scripts/output_tracers}"
Q() { if [ "${DRY:-0}" = "1" ]; then echo "DRY qsub $*" >&2; echo "DRY.$RANDOM"; else qsub "$@"; fi; }
IDS=""
for SEED in ${SEEDS}; do
  V="SEED=${SEED},WRAP_OFFSET=${WRAP_OFFSET:-0},EXTRA_ARGS=\"--config ${KSZ_CONFIG} --out-root ${MAPS_ROOT} ${EXTRA_ARGS}\""
  J=$(Q -v "${V}" pbs/pbs_tracer_maps.sh)
  echo "  seed ${SEED}: maps ${J}"
  IDS="${IDS}:${J}"
done
X=$(Q -W depend=afterok${IDS} -v "WRAP_OFFSET=${WRAP_OFFSET:-0},MAPS_ROOT=${MAPS_ROOT},OUT_ROOT=${OUT_ROOT},EXTRA_ARGS=\"--config ${KSZ_CONFIG}\"" pbs/pbs_tracer_cross.sh)
echo "  cross-power for every tracer -> ${X} (starts after all seeds finish OK)"
echo "Monitor: qstat -u ${USER}"
