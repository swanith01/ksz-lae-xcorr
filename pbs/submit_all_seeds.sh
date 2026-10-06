#!/bin/bash
# =============================================================================
# submit_all_seeds.sh
# Submits one PBS job per seed listed in configs/fiducial.yaml (box.seeds).
#
# Usage:
#   chmod +x submit_all_seeds.sh
#   ./submit_all_seeds.sh
#
# To submit a single seed manually:
#   SEED=3 qsub -v SEED=3 pbs_coeval.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/pbs_coeval.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pull the seed list straight out of the fiducial config so there's a
# single source of truth -- no separate hardcoded seed list to drift out
# of sync with configs/fiducial.yaml.
SEEDS=$(python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}/src')
from ksz_lae_xcorr.utils.config import load_config
cfg = load_config('${REPO_ROOT}/configs/fiducial.yaml')
print(' '.join(str(s) for s in cfg.box.seeds))
")

echo "Seeds from configs/fiducial.yaml: ${SEEDS}"

for SEED in ${SEEDS}; do
    JOB_ID=$(qsub -v SEED=${SEED} "${PBS_SCRIPT}")
    echo "  Submitted seed ${SEED} -> job ${JOB_ID}"
done

echo ""
echo "All seeds submitted. Monitor with: qstat -u ${USER}"
