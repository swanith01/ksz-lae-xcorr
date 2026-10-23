#!/bin/bash
# =============================================================================
# pbs_chunk_maps.sh -- scripts/48, ONE seed: per-window kSZ^2 chunk maps + the tracer maps of the same windows, to look at.
#   qsub -v SEED=1 pbs/pbs_chunk_maps.sh
# Needs the kSZ products of >= 2 seeds (scripts/26 --save-maps) and the tracer maps of scripts/37 for the seed.
# Optional env: WRAP_OFFSET (default 0), EXTRA_ARGS (e.g. "--z0 13;12;11;10;9;8;7 --tracers halo_M10.0,lae_L42.5").
# =============================================================================

#PBS -N chunk_maps
#PBS -l select=1:ncpus=8:mem=48gb
#PBS -l walltime=03:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e
: "${SEED:?SEED must be set: qsub -v SEED=1 pbs/pbs_chunk_maps.sh}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-p21c_v41}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs
echo "== chunk maps | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/48_chunk_maps.py --seed "${SEED}" --wrap-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
