#!/bin/bash
# =============================================================================
# pbs_lp_cross.sh -- scripts/28: map-based La Plante D_l^cross vs z0 from the
# per-seed *_lpmaps.pkl files (written by scripts/26 --save-maps / SAVE_MAPS=1).
# Light (maps are 300x300): a few minutes.  Needs CAMB in the env for C_TT, or
# pass a CSV:  qsub -v EXTRA_ARGS="--cltt-file data/reference/cltt_raw.csv" pbs/pbs_lp_cross.sh
# Optional env: WRAP_OFFSET (default 0), EXTRA_ARGS (e.g. "--experiment CMB-S4").
# =============================================================================

#PBS -N lp_cross
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=00:30:00
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

echo "== lp cross vs z0 | $(hostname) | $(date) =="
python -u scripts/28_lp_cross_vs_z0.py --wrap-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
