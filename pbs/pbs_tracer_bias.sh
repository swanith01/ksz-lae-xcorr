#!/bin/bash
# =============================================================================
# pbs_tracer_bias.sh -- scripts/35: bias b(z), n(z), N_obj of the halo / LAE / LBG tracer suite, ONE seed.
# Reads the full halo catalogue of every snapshot (tens of millions of haloes at low z) -> IO heavy, not CPU heavy.
#   qsub -v SEED=1 pbs/pbs_tracer_bias.sh
# Optional env: SEED (required), EXTRA_ARGS (e.g. "--specs halo:1e10;lae:42.5 --every 2"; use ";" between specs, qsub -v splits on commas).
# =============================================================================

#PBS -N trc_bias
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -l walltime=06:00:00
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

: "${SEED:?set SEED, e.g. qsub -v SEED=1 pbs/pbs_tracer_bias.sh}"
echo "== tracer bias | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/35_tracer_bias_suite.py --seed "${SEED}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
