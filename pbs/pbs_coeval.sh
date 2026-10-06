#!/bin/bash
# =============================================================================
# pbs_coeval.sh
# PBS job script for one seed's coeval box run. Do not submit directly --
# use submit_all_seeds.sh, which passes SEED as an environment variable.
# =============================================================================

#PBS -N ksz_lae_coeval
#PBS -l select=1:ncpus=64:mem=450gb
#PBS -l walltime=72:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ksz-lae-xcorr

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
SEED="${SEED:?SEED must be set, e.g. qsub -v SEED=1 pbs_coeval.sh}"
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"

mkdir -p "${REPO_ROOT}/logs"

echo "======================================================================"
echo "  ksz-lae-xcorr coeval run"
echo "  Seed        : ${SEED}"
echo "  Node        : $(hostname)"
echo "  Started     : $(date)"
echo "  PBS job ID  : ${PBS_JOBID:-none}"
echo "======================================================================"

cd "${REPO_ROOT}"
python -u scripts/01_run_coeval_seed.py --seed "${SEED}" 2>&1

EXIT_CODE=$?
echo ""
echo "  Finished    : $(date)   Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
