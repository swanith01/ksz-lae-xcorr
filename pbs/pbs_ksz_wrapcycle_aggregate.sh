#!/bin/bash
# =============================================================================
# pbs_ksz_wrapcycle_aggregate.sh -- scripts/27: combine the per-seed pickles
# into median + seed spread, CSVs and the overview figure. Cheap (seconds).
# Submitted with a dependency by pbs/submit_ksz_wrapcycle.sh, or by hand:
#   qsub pbs/pbs_ksz_wrapcycle_aggregate.sh
# Optional env: WRAP_OFFSET (default 0), EXTRA_ARGS (e.g. "--legacy-pkl ...").
# =============================================================================

#PBS -N ksz_wc_agg
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:20:00
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

echo "== ksz wrap-cycle aggregate | $(hostname) | $(date) =="
python -u scripts/27_ksz_auto_power_aggregate.py --wrap-offset "${WRAP_OFFSET:-0}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
