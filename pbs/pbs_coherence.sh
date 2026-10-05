#!/bin/bash
# =============================================================================
# pbs_coherence.sh
# Batch job: scripts/09 (kSZ auto-power P_diag/P_off + patchy-window D_diag,
# all seeds) then scripts/22 (the overview plot). Run from the repo root:
#
#   cd ~/ksz-lae-xcorr && qsub pbs/pbs_coherence.sh
#
# Writes data/products/coherence_decomposition.pkl, then
# paper/figure_scripts/output/ksz_auto_power_overview.{pdf,png}.
# Log: logs/ksz_coherence.<jobid>.OU (monitor with: tail -f).
# =============================================================================

#PBS -N ksz_coherence
#PBS -l select=1:ncpus=16:mem=200gb
#PBS -l walltime=24:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-p21c_v41}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

echo "== ksz coherence batch | node $(hostname) | $(date) | job ${PBS_JOBID:-none} =="
python -u scripts/09_coherence_decomposition.py
echo "== scripts/09 done: $(date) =="
python -u scripts/22_ksz_auto_power_plot.py
echo "== scripts/22 done: $(date) =="
