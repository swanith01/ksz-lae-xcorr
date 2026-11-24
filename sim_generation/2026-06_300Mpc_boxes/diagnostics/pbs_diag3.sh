#!/bin/bash
#PBS -N diag_initbox
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:15:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41

python3 << 'EOF'
import py21cmfast as p21c
import tempfile

with tempfile.TemporaryDirectory(dir="/user1/swanith/kSZ2_LAE_project_22Jun2026") as t:
    inputs = p21c.InputParameters(
        random_seed=1,
        simulation_options=p21c.SimulationOptions(
            BOX_LEN=50.0, HII_DIM=25, DIM=50, N_THREADS=2,
            SAMPLER_MIN_MASS=1e8, SAMPLER_BUFFER_FACTOR=2.0, Z_HEAT_MAX=20.0,
        ),
        matter_options=p21c.MatterOptions(
            KEEP_3D_VELOCITIES=True,
            USE_INTERPOLATION_TABLES='hmf-interpolation',
        ),
        astro_options=p21c.AstroOptions(INHOMO_RECO=True, USE_TS_FLUCT=True),
    )
    cache = p21c.OutputCache(t)

    print("=== init_box ===")
    init_box = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=True)
    print("init_box.arrays keys:", list(init_box.arrays.keys()))
    for k in init_box.arrays.keys():
        try:
            arr = init_box.get(k)
            print(f"  {k}: shape={arr.shape}")
        except Exception as e:
            print(f"  {k}: ERROR {e}")

    print()
    print("=== perturbed_field (direct call, before coeval wrapping) ===")
    pf = p21c.perturb_field(redshift=10.0, initial_conditions=init_box, inputs=inputs, cache=cache)
    print("perturbed_field.arrays keys:", list(pf.arrays.keys()))
    for k in pf.arrays.keys():
        try:
            arr = pf.get(k)
            print(f"  {k}: shape={arr.shape}")
        except Exception as e:
            print(f"  {k}: ERROR {e}")
EOF
