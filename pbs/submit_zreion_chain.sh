#!/bin/bash
# =============================================================================
# submit_zreion_chain.sh -- rerun the wrap-cycle kSZ auto-power + La Plante cross-power with the ionisation field
# REPLACED by zreion (density and velocity untouched), for several seeds, as one dependency chain per seed:
#     scripts/34 (write zreion coeval root, ~3 min)  ->  scripts/26 (kSZ maps, ~20 min)
# then ONE aggregation (scripts/27) and ONE La Plante cross (scripts/28) after all seeds finish OK.
#
#   cd ~/ksz-lae-xcorr
#   SEEDS="1 2" bash pbs/submit_zreion_chain.sh                     # MATCH=z (default): pure zreion x_HII(z)
#   SEEDS="1 2 3" MATCH=quantile bash pbs/submit_zreion_chain.sh    # 21cmFAST history, zreion morphology
#   DRY=1 SEEDS="1 2" bash pbs/submit_zreion_chain.sh               # print the qsub commands, submit nothing
#   QSUB_RES="select=1:ncpus=2:mem=64gb" ...                        # resources for the scripts/26 jobs (as before)
# Outputs: products under data/products_zreion[_q]/, figures under paper/figure_scripts/output_zreion[_q]/
# (separate from the 21cmFAST products, nothing is overwritten).  Heavy fields are NOT copied: the new coeval
# root holds symlinks to the original density/velocity files plus one new 108 MB neutral_fraction per snapshot.
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p logs

MATCH="${MATCH:-z}"
case "${MATCH}" in
  z)        CFG=configs/variants/zreion_xh.yaml;   ROOT=/user1/swanith/zreion_coeval;   OUT=paper/figure_scripts/output_zreion ;;
  quantile) CFG=configs/variants/zreion_xh_q.yaml; ROOT=/user1/swanith/zreion_coeval_q; OUT=paper/figure_scripts/output_zreion_q ;;
  *) echo "MATCH must be z or quantile"; exit 1 ;;
esac
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
echo "MATCH=${MATCH} config=${CFG} root=${ROOT} seeds: ${SEEDS}"
Q() { if [ "${DRY:-0}" = "1" ]; then echo "DRY qsub $*" >&2; echo "DRY.$RANDOM"; else qsub "$@"; fi; }

RES_ARG=()
[ -n "${QSUB_RES}" ] && RES_ARG=(-l "${QSUB_RES}")

JOB_IDS=""
for SEED in ${SEEDS}; do
  W=$(Q -v SEED=${SEED},EXTRA_ARGS="--config ${CFG} --write-root ${ROOT} --match ${MATCH} --out-dir ${OUT}" pbs/pbs_zreion_on_21cmfast.sh)
  J=$(Q "${RES_ARG[@]}" -W depend=afterok:${W} -v SEED=${SEED},WRAP_OFFSET=0,SAVE_MAPS=1,EXTRA_ARGS="--config ${CFG}" pbs/pbs_ksz_wrapcycle.sh)
  echo "  seed ${SEED}: write ${W} -> wrap-cycle ${J}"
  JOB_IDS="${JOB_IDS}:${J}"
done
AGG=$(Q -W depend=afterok${JOB_IDS} -v WRAP_OFFSET=0,EXTRA_ARGS="--config ${CFG} --out-dir ${OUT}" pbs/pbs_ksz_wrapcycle_aggregate.sh)
LP=$(Q -W depend=afterok${JOB_IDS} -v WRAP_OFFSET=0,EXTRA_ARGS="--config ${CFG} --out-dir ${OUT}" pbs/pbs_lp_cross.sh)
echo "  aggregate -> ${AGG};  lp cross -> ${LP}  (both start after all seeds finish OK)"
echo "Monitor: qstat -u ${USER}"
