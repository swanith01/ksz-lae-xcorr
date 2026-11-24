#!/bin/bash
# =============================================================================
# pbs_coeval.sh
# kSZ2 x LAE Project  —  22 Jun 2026  (updated 27 Jun 2026)
# =============================================================================
# PBS job script for one seed.  Do not submit directly — use submit_all_seeds.sh
# which passes SEED as an environment variable.
#
# Resources: 1 node, 16 cores, 60 GB RAM, 72 hr walltime
# Updated: BOX_LEN=200, DIM=HII_DIM=512 (avoids PerturbHaloField crash).
# Peak RAM at 512^3: ~4 GB fields + ~2 GB halo catalog = well within 60 GB.
# =============================================================================

#PBS -N kSZ2_LAE_coeval
#PBS -l select=1:ncpus=32:mem=40gb
#PBS -l walltime=72:00:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41          # v4dev312 crashes; v4.1.0 passes the test

# Use all 64 cores for OpenMP threading in py21cmfast's C backend
export OMP_NUM_THREADS=32

# Seed passed in by submit_all_seeds.sh via -v SEED=N
SEED="${SEED:-1}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

PROJECT_ROOT="${HOME}/kSZ2_LAE_project_22Jun2026"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "======================================================================"
echo "  kSZ2 x LAE  coeval simulation"
echo "  Seed        : ${SEED}"
echo "  Node        : $(hostname)"
echo "  Started     : $(date)"
echo "  PBS job ID  : ${PBS_JOBID}"
echo "  OMP threads : ${OMP_NUM_THREADS}"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cd "${PBS_O_WORKDIR}"

python -u run_coeval_seed.py --seed "${SEED}" 2>&1

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "  Finished    : $(date)"
echo "  Exit code   : ${EXIT_CODE}"
echo "======================================================================"

exit ${EXIT_CODE}
