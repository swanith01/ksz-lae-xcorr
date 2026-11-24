#!/bin/bash
# =============================================================================
# submit_all_seeds.sh
# kSZ2 x LAE Project  —  22 Jun 2026
# =============================================================================
# Submits one PBS job per seed (seeds 1-5).
# Each job gets one full node: 64 cores, 450 GB RAM, 72 hr walltime.
#
# USAGE:
#   chmod +x submit_all_seeds.sh
#   ./submit_all_seeds.sh
#
# To submit a single seed manually:
#   SEED=3 qsub -v SEED=3 pbs_coeval.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/pbs_coeval.sh"

if [[ ! -f "${PBS_SCRIPT}" ]]; then
    echo "ERROR: pbs_coeval.sh not found in ${SCRIPT_DIR}"
    exit 1
fi

for SEED in 1 2 3 4 5 6 7 8 9 10; do
    JOB_ID=$(qsub -v SEED=${SEED} "${PBS_SCRIPT}")
    echo "  Submitted seed ${SEED} -> job ${JOB_ID}"
done

echo ""
echo "All 5 seeds submitted. Monitor with: qstat -u ${USER}"
