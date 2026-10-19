#!/bin/bash
# =============================================================================
# pbs_ksz_components_cross.sh -- scripts/42: the component maps (scripts/41, all seeds) x the tracer window maps (scripts/37).
# Light (2D FFTs).   qsub pbs/pbs_ksz_components_cross.sh
# Optional env: MAPS_ROOT (data/products_tracers/maps), COMP_DIR (data/products_tracers/components),
#   OUT_DIR (paper/figure_scripts/output_components), WRAP_OFFSET (0), EXTRA_ARGS.
# =============================================================================

#PBS -N ksz_comp_x
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=02:00:00
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
echo "== component cross | $(hostname) | $(date) =="
python -u scripts/42_ksz_component_cross.py --maps-root "${MAPS_ROOT:-data/products_tracers/maps}" \
    --comp-dir "${COMP_DIR:-data/products_tracers/components}" --out-dir "${OUT_DIR:-paper/figure_scripts/output_components}" \
    --wrap-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
