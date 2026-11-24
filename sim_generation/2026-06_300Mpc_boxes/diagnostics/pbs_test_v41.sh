#!/bin/bash
#PBS -N p21c_v41_test
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41

export OMP_NUM_THREADS=4

echo "======================================================================"
echo "  py21cmfast v4.1.0 bounds-check test"
echo "  Node     : $(hostname)"
echo "  Started  : $(date)"
echo "======================================================================"

python -u - << EOF
import py21cmfast as p21c
import os, sys

TEST_DIR = "/user1/swanith/kSZ2_LAE_project_22Jun2026/test_cache"
os.makedirs(TEST_DIR, exist_ok=True)

p21c.config['direc'] = TEST_DIR
p21c.config['HALO_CATALOG_MEM_FACTOR'] = 3.0

print(f"py21cmfast version: {p21c.__version__}")
print(f"Cache dir: {TEST_DIR}")
print(f"Testing BOX_LEN=100, HII_DIM=100, DIM=200 ...")
sys.stdout.flush()

inputs = p21c.InputParameters(
    random_seed=1,
    simulation_options=p21c.SimulationOptions(
        BOX_LEN               = 100.0,
        HII_DIM               = 100,
        DIM                   = 200,
        N_THREADS             = 4,
        SAMPLER_MIN_MASS      = 1e8,
        SAMPLER_BUFFER_FACTOR = 2.0,
        Z_HEAT_MAX            = 20.0,
    ),
    matter_options=p21c.MatterOptions(
        KEEP_3D_VELOCITIES       = True,
        USE_INTERPOLATION_TABLES = 'hmf-interpolation',
    ),
    astro_options=p21c.AstroOptions(
        INHOMO_RECO  = True,
        USE_TS_FLUCT = True,
    ),
)

cache = p21c.OutputCache(TEST_DIR)

print("\n  Step 1: Computing initial conditions ...")
sys.stdout.flush()
try:
    ics = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=True)
    print("  ✓ Initial conditions done")
except Exception as e:
    print(f"  ✗ Initial conditions FAILED: {e}")
    sys.exit(1)

print("\n  Step 2: Running all nodes z=20 down to z=5 ...")
sys.stdout.flush()
last_z = None
try:
    for cv, _ in p21c.generate_coeval(inputs=inputs, cache=cache):
        print(f"    ✓ z={cv.redshift:.4f} OK")
        sys.stdout.flush()
        last_z = cv.redshift
        del cv
    print(f"\n  ✓ All nodes passed — reached z={last_z:.4f}")
except Exception as e:
    print(f"  ✗ FAILED at z={last_z}: {e}")
    sys.exit(1)

print("\n======================================================================")
print("  RESULT: PASSED — safe for production")
print("======================================================================")
EOF

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  VERDICT  : ✓ PASSED"
else
    echo "  VERDICT  : ✗ FAILED"
fi
echo "======================================================================"
exit ${EXIT_CODE}
