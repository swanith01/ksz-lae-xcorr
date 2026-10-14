#!/bin/bash
# =============================================================================
# pbs_tracer_maps.sh -- scripts/37: tracer window maps (halo / LAE / LBG cuts) on the wrap-cycle lightcone, ONE seed.
# Needs the seed's kSZ products first (scripts/26 with SAVE_MAPS=1 -> seed<S>_wo<K>_lpmaps.pkl).
#   qsub -v SEED=1 pbs/pbs_tracer_maps.sh
# Optional env: SEED (required), WRAP_OFFSET (default 0), EXTRA_ARGS (e.g. "--specs halo:10;lae:42.5 --config configs/fiducial.yaml";
#   use ";" between specs, qsub -v splits on commas).
# Memory: ~0.5-1 GB of lightcone per tracer (15 default tracers ~ 12 GB) + one halo catalogue (~1 GB) -> 96 GB is generous.
# =============================================================================

#PBS -N trc_maps
#PBS -l select=1:ncpus=4:mem=96gb
#PBS -l walltime=08:00:00
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
export OMP_NUM_THREADS=4

: "${SEED:?set SEED, e.g. qsub -v SEED=1 pbs/pbs_tracer_maps.sh}"
echo "== tracer maps | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/37_tracer_lightcone_maps.py --seed "${SEED}" --wrap-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
