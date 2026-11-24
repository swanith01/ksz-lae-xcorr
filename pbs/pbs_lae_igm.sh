#!/bin/bash
# =============================================================================
# pbs_lae_igm.sh -- scripts/61: IGM attenuation of Jahaan's intrinsic LAEs on the repaired 300 Mpc boxes, one seed.
#   qsub -v SEED=1 pbs/pbs_lae_igm.sh            (from the repo root)
# After the box repair (sim_generation/2026-06_300Mpc_boxes/repair) for that seed has finished.
# Optional env: EXTRA_ARGS (e.g. "--spec fine", "--zmax 9").
# =============================================================================
#PBS -N lae_igm
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=12:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-p21c_v41}"
cd "${PBS_O_WORKDIR:-$(pwd)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== lae igm | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/61_lae_igm.py --seed "${SEED}" --workers 8 ${EXTRA_ARGS:-}
echo "== done $(date) =="
qstat -f "$PBS_JOBID" | grep resources_used || true
