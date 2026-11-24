#!/bin/bash
#PBS -N quick_lc
#PBS -l select=1:ncpus=4:mem=30gb
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41
export OMP_NUM_THREADS=4

cd /user1/swanith/kSZ2_LAE_project_22Jun2026/codes/
python3 -u quick_lightcone_check.py
