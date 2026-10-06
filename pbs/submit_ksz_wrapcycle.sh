#!/bin/bash
# =============================================================================
# submit_ksz_wrapcycle.sh -- one scripts/26 job per seed in configs/fiducial.yaml
# (box.seeds), then ONE aggregation job (scripts/27) that starts only after
# ALL seed jobs finish OK (-W depend=afterok).
#
#   cd ~/ksz-lae-xcorr
#   bash pbs/submit_ksz_wrapcycle.sh                 # all 10 seeds
#   SEEDS="1 2" bash pbs/submit_ksz_wrapcycle.sh     # test on 2 seeds first
#   WRAP_OFFSET=100 bash pbs/submit_ksz_wrapcycle.sh # repeat with new rotation angles
#   QSUB_RES="select=1:ncpus=4:mem=48gb" bash pbs/submit_ksz_wrapcycle.sh   # smaller ask
#   SAVE_MAPS=1 bash pbs/submit_ksz_wrapcycle.sh     # ALSO save La Plante map inputs and
#                                                    # chain scripts/28 after all seeds
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

if [ -z "${SEEDS}" ]; then
  SEEDS=$(python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}/src')
from ksz_lae_xcorr.utils.config import load_config
print(' '.join(str(s) for s in load_config('${REPO_ROOT}/configs/fiducial.yaml').box.seeds))
")
fi
echo "Seeds: ${SEEDS}   wrap offset: ${WRAP_OFFSET:-0}"

RES_ARG=()
[ -n "${QSUB_RES}" ] && RES_ARG=(-l "${QSUB_RES}")

JOB_IDS=""
for SEED in ${SEEDS}; do
  JID=$(qsub "${RES_ARG[@]}" -v SEED=${SEED},WRAP_OFFSET=${WRAP_OFFSET:-0},SAVE_MAPS=${SAVE_MAPS:-0} pbs/pbs_ksz_wrapcycle.sh)
  echo "  seed ${SEED} -> ${JID}"
  JOB_IDS="${JOB_IDS}:${JID}"
done

AGG=$(qsub -W depend=afterok${JOB_IDS} -v WRAP_OFFSET=${WRAP_OFFSET:-0} pbs/pbs_ksz_wrapcycle_aggregate.sh)
echo "  aggregate -> ${AGG} (runs after all seeds finish OK)"
if [ "${SAVE_MAPS:-0}" = "1" ]; then
  LP=$(qsub -W depend=afterok${JOB_IDS} -v WRAP_OFFSET=${WRAP_OFFSET:-0} pbs/pbs_lp_cross.sh)
  echo "  lp cross (scripts/28) -> ${LP} (runs after all seeds finish OK)"
fi
echo "Monitor: qstat -u ${USER}"
