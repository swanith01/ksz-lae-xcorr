#!/bin/bash
#PBS -N downsample_rest
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -j oe

source /user1/swanith/miniconda3/etc/profile.d/conda.sh
conda activate p21c_v41
cd /user1/swanith/kSZ2_LAE_project_22Jun2026/
python downsample_density_300.py --seed 2 3 4 5 6 7 8 9 10
