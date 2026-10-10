#!/bin/bash
# =============================================================================
# pbs_zreion_on_21cmfast.sh -- scripts/34: apply zreion to a 21cmFAST seed's density, compare with the 21cmFAST
# ionisation field (history + morphology at equal x_HII), optionally write a zreion coeval root.
# Light: ~2-3 GB, a few minutes.   Env: SEED (default 1), EXTRA_ARGS (e.g. "--coarsen 3",
#   "--write-root /user1/swanith/zreion_coeval --match z").
#   qsub -v SEED=1 pbs/pbs_zreion_on_21cmfast.sh
# =============================================================================

#PBS -N zre_vs_21cm
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-p21c_v41}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

echo "== zreion vs 21cmFAST | seed ${SEED:-1} | $(hostname) | $(date) =="
python -u scripts/34_zreion_on_21cmfast.py --seed "${SEED:-1}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
