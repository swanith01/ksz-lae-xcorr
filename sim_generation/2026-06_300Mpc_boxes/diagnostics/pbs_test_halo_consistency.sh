#!/bin/bash
#PBS -N halo_consistency
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/ksz-lae-xcorr/logs/

source /user1/swanith/miniconda3/etc/profile.d/conda.sh
conda activate p21c_v41
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
cd /user1/swanith/ksz-lae-xcorr/sim_generation/2026-06_300Mpc_boxes/diagnostics
if [ -x /usr/bin/time ]; then /usr/bin/time -v python -u test_halo_consistency.py; else python -u test_halo_consistency.py; fi
qstat -f "$PBS_JOBID" | grep resources_used
