#!/bin/bash
# =============================================================================
# pbs_tracer_cross.sh -- scripts/28 once per tracer directory written by scripts/37 (needs >= 2 seeds there).
# Light: a few minutes per tracer (CAMB C_TT is recomputed each time; pass --cltt-file in EXTRA_ARGS to skip it).
#   qsub pbs/pbs_tracer_cross.sh
# Optional env: MAPS_ROOT (default data/products_tracers/maps), OUT_ROOT (default paper/figure_scripts/output_tracers),
#   WRAP_OFFSET (default 0), EXTRA_ARGS (passed to scripts/28, e.g. "--experiment CMB-S4").
# Results: <OUT_ROOT>/<tracer>/lp_cross_vs_z0_SO_wo0.csv (+ figure) for each tracer.
# =============================================================================

#PBS -N trc_cross
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=04:00:00
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

MAPS_ROOT="${MAPS_ROOT:-data/products_tracers/maps}"
OUT_ROOT="${OUT_ROOT:-paper/figure_scripts/output_tracers}"
echo "== tracer cross | $(hostname) | $(date) | maps ${MAPS_ROOT} -> ${OUT_ROOT} =="
for d in "${MAPS_ROOT}"/*/; do
  name="$(basename "${d}")"
  echo "---- ${name}"
  python -u scripts/28_lp_cross_vs_z0.py --wrap-offset "${WRAP_OFFSET:-0}" --in-dir "${d}" --out-dir "${OUT_ROOT}/${name}" ${EXTRA_ARGS:-} \
    || echo "  !! ${name} failed (continuing)"
done
echo "== done: $(date) =="
