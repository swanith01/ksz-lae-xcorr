#!/bin/bash
# =============================================================================
# pbs_ksz_wrapcycle.sh -- ONE seed of scripts/26 (wrap-cycle kSZ auto-power).
# Normally submitted via pbs/submit_ksz_wrapcycle.sh; manual single seed:
#   cd ~/ksz-lae-xcorr && qsub -v SEED=3 pbs/pbs_ksz_wrapcycle.sh
#   (resources are overridable: qsub -l select=1:ncpus=4:mem=48gb -v SEED=3 ...)
# Memory: ~3 float64 lightcones of 300x300x~3000 (~2 GB each) + FFT of the
# per-pixel kSZ slices (~2 x 4.4 GB complex) -> ~40 GB peak; 64 GB requested.
# Optional env: WRAP_OFFSET (default 0), EXTRA_ARGS (e.g. "--mode wrap").
# =============================================================================

#PBS -N ksz_wc
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e
: "${SEED:?SEED must be set: qsub -v SEED=3 pbs/pbs_ksz_wrapcycle.sh}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-p21c_v41}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

echo "== ksz wrap-cycle seed ${SEED} | node $(hostname) | $(date) | job ${PBS_JOBID:-none} =="
python -u scripts/26_ksz_auto_power_wrapcycle.py --seed "${SEED}" \
    --wrap-cycle-seed-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
