#!/bin/bash
#PBS -N downsample_test
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:30:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/downsample_test_${PBS_JOBID}.log

source /user1/swanith/miniconda3/etc/profile.d/conda.sh
conda activate p21c_v41
cd /user1/swanith/kSZ2_LAE_project_22Jun2026/
python downsample_density_300.py --seed 1
