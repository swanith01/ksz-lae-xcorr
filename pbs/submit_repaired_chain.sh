#!/bin/bash
# =============================================================================
# submit_repaired_chain.sh  (2026-11-25) -- redo the simulated-tracer products on the REPAIRED boxes
# (configs/repaired.yaml: evolved density, displaced halo positions, LAE catalogue from scripts/61).
# Per seed:  scripts/26 (kSZ wrap-cycle + La Plante map inputs)  ->  scripts/37 (tracer window maps)
#            scripts/35 (tracer bias + counts; independent)
# then, after all of those:  scripts/53 (binned spectra for every CMB experiment).
# Outputs:  data/products_repaired/ksz_auto_wrapcycle/   data/products_tracers_repaired/{bias,maps,sim_spectra}/
#   cd ~/ksz-lae-xcorr && bash pbs/submit_repaired_chain.sh            # all 10 seeds
#   SEEDS="1" bash pbs/submit_repaired_chain.sh                        # one seed
#   DRY=1 bash pbs/submit_repaired_chain.sh                            # print the qsub commands only
# Afterwards on the desktop: scripts/36 (bias table) on .../bias, then scripts/54-59 with the new spectra and bias table.
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p logs
CFG=configs/repaired.yaml
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
OUT_T=data/products_tracers_repaired
q() { if [ "${DRY:-0}" = "1" ]; then echo "DRY qsub $*" >&2; echo "dry.$RANDOM"; else qsub "$@"; fi; }

ALL=""
for S in ${SEEDS}; do
  J26=$(q -v "SEED=${S},WRAP_OFFSET=0,SAVE_MAPS=1,EXTRA_ARGS=--config ${CFG}" pbs/pbs_ksz_wrapcycle.sh)
  J35=$(q -v "SEED=${S},EXTRA_ARGS=--config ${CFG} --out-dir ${OUT_T}/bias" pbs/pbs_tracer_bias.sh)
  J37=$(q -W depend=afterok:${J26} -v "SEED=${S},WRAP_OFFSET=0,EXTRA_ARGS=--config ${CFG} --out-root ${OUT_T}/maps" pbs/pbs_tracer_maps.sh)
  echo "seed ${S}: kSZ ${J26}   bias ${J35}   tracer maps ${J37} (after kSZ)"
  ALL="${ALL}:${J26}:${J37}"
done
J53=$(q -W depend=afterok${ALL} -v "MATTER_DIR=data/products_repaired/ksz_auto_wrapcycle,MAPS_ROOT=${OUT_T}/maps,OUT_DIR=${OUT_T}/sim_spectra" pbs/pbs_sim_spectra.sh)
echo "spectra (scripts/53): ${J53}  (after all kSZ + tracer-map jobs)"
echo "Monitor: qstat -u ${USER}"
