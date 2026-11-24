#!/bin/bash
#PBS -N test_one_z
#PBS -l select=1:ncpus=8:mem=20gb
#PBS -l walltime=00:30:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41
export OMP_NUM_THREADS=8

cd /user1/swanith/kSZ2_LAE_project_22Jun2026/codes/
python3 -u -c "
import sys
sys.argv = ['run_coeval_seed.py', '--seed', '99']
exec(open('run_coeval_seed.py').read().replace(
    'del coeval', 'del coeval\n    break  # TEST: stop after first node'
))
"
