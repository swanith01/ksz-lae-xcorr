#!/bin/bash
# =============================================================================
# pbs_sim_spectra.sh -- scripts/53: binned kSZ_f^2 x tracer and tracer auto spectra for every CMB experiment,
# from the EXISTING lpmaps pickles (matter x Waters bias + every tracer folder of scripts/37).  Light: ~1 min per folder.
#   qsub pbs/pbs_sim_spectra.sh
# Optional env: MATTER_DIR (default data/products/ksz_auto_wrapcycle), MAPS_ROOT (default data/products_tracers/maps),
#   OUT_DIR (default data/products_tracers/sim_spectra), EXTRA_ARGS (passed to scripts/53, e.g. "--ksz-reion lp").
# Then copy OUT_DIR to the desktop and run scripts/54 there.
# =============================================================================

#PBS -N sim_spec
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
echo "== sim spectra | $(hostname) | $(date) =="
python -u scripts/53_sim_spectra.py \
  --in-dir "${MATTER_DIR:-data/products/ksz_auto_wrapcycle}" --name matter_waters \
  --maps-root "${MAPS_ROOT:-data/products_tracers/maps}" \
  --out-dir "${OUT_DIR:-data/products_tracers/sim_spectra}" ${EXTRA_ARGS:-}
echo "== done: $(date) =="
