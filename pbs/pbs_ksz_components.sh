#!/bin/bash
# =============================================================================
# pbs_ksz_components.sh -- scripts/41, ONE seed: filtered-and-squared kSZ maps from x_e only, v only, density only and the real
# integrand, each project-then-square (A) and square-then-project (B).  Rebuilds the seed's wrap-cycle lightcone.
#   qsub -v SEED=1 pbs/pbs_ksz_components.sh          (normally via pbs/submit_ksz_components.sh)
# Needs the kSZ products of >= 2 seeds (scripts/26 with SAVE_MAPS=1) for the filter.
# Memory: three float64 lightcones of 300x300x~3000 (~2 GB each) + chunked FFTs -> ~16 GB peak; 48 GB requested.
# Optional env: WRAP_OFFSET (default 0), EXTRA_ARGS (e.g. "--out-dir data/products_tracers/components_zreion --config configs/variants/zreion_xh.yaml").
# =============================================================================

#PBS -N ksz_comp
#PBS -l select=1:ncpus=8:mem=48gb
#PBS -l walltime=03:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e
: "${SEED:?SEED must be set: qsub -v SEED=1 pbs/pbs_ksz_components.sh}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-p21c_v41}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs
echo "== ksz components | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/41_ksz_component_maps.py --seed "${SEED}" --wrap-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
