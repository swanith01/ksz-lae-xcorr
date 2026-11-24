#!/bin/bash
#PBS -N repair300
#PBS -l select=1:ncpus=32:mem=120gb
#PBS -l walltime=08:00:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/ksz-lae-xcorr/logs/
# Submit with:  qsub -v SEED=1 pbs_repair.sh      (optional: -v SEED=1,ZMAX=6.0 for a quick trial)

source /user1/swanith/miniconda3/etc/profile.d/conda.sh
conda activate p21c_v41
export OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE
cd /user1/swanith/ksz-lae-xcorr/sim_generation/2026-06_300Mpc_boxes/repair
EXTRA=""
[ -n "${ZMAX:-}" ] && EXTRA="--zmax ${ZMAX}"
echo "seed ${SEED}  node $(hostname)  job ${PBS_JOBID}  $(date)"
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v python -u repair_seed.py --seed "${SEED}" ${EXTRA}
else
    python -u repair_seed.py --seed "${SEED}" ${EXTRA}
fi
echo "exit $?  $(date)"
qstat -f "$PBS_JOBID" | grep resources_used
