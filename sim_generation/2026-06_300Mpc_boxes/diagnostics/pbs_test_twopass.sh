#!/bin/bash
#PBS -N test_twopass
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -l walltime=00:30:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41
export OMP_NUM_THREADS=4

cd /user1/swanith/kSZ2_LAE_project_22Jun2026/codes/
python3 -u -c "
import sys
sys.argv = ['run_coeval_seed.py', '--seed', '96']
code = open('run_coeval_seed.py').read()
code = code.replace('BOX_LEN               = 300.0,', 'BOX_LEN               = 50.0,')
code = code.replace('HII_DIM               = 300,', 'HII_DIM               = 25,')
code = code.replace('DIM                   = 600,', 'DIM                   = 50,')
exec(code)
"
