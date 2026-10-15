#!/bin/bash
# =============================================================================
# pbs_tracer_lf.sh -- scripts/39: halo mass function, UV LF and Lya LF of ONE seed (reads the halo MASS file + the LAE/LBG
# value files of every snapshot).   qsub -v SEED=1 pbs/pbs_tracer_lf.sh
# Optional env: SEED (required), EXTRA_ARGS (e.g. "--every 4").
# =============================================================================

#PBS -N trc_lf
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=04:00:00
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
export OMP_NUM_THREADS=2

: "${SEED:?set SEED, e.g. qsub -v SEED=1 pbs/pbs_tracer_lf.sh}"
echo "== tracer LF | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/39_tracer_luminosity_functions.py --seed "${SEED}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
