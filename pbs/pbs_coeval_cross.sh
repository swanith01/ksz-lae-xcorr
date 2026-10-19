#!/bin/bash
# =============================================================================
# pbs_coeval_cross.sh -- scripts/44, ONE seed: D_ell^cross from coeval snapshot boxes next to the lightcone estimate (matter x Waters+16
# and the chosen tracer cuts).  Needs the kSZ products of >= 2 seeds (scripts/26 SAVE_MAPS=1) and, for the lightcone side of the
# tracers, the scripts/37 maps (MAPS_ROOT).
#   qsub -v SEED=1 pbs/pbs_coeval_cross.sh          (normally via pbs/submit_coeval_cross.sh)
# Optional env: WRAP_OFFSET (0), SPECS (default "halo:10;lae:42.5;lbg:-19"; ";" between specs -- qsub -v splits on commas),
#   MAPS_ROOT, OUT_DIR (data/products_tracers/coeval), EXTRA_ARGS.
# Cost: one pass over the ~40 snapshots between z=6 and 14 (3 boxes + one halo catalogue each) -> an hour or two.
# =============================================================================

#PBS -N coeval_x
#PBS -l select=1:ncpus=4:mem=48gb
#PBS -l walltime=06:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/

set -e
: "${SEED:?SEED must be set: qsub -v SEED=1 pbs/pbs_coeval_cross.sh}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-p21c_v41}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs
echo "== coeval cross | seed ${SEED} | $(hostname) | $(date) =="
python -u scripts/44_coeval_cross.py --seed "${SEED}" --wrap-offset "${WRAP_OFFSET:-0}" \
    --specs "${SPECS:-halo:10;lae:42.5;lbg:-19}" --maps-root "${MAPS_ROOT:-data/products_tracers/maps}" \
    --out-dir "${OUT_DIR:-data/products_tracers/coeval}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
