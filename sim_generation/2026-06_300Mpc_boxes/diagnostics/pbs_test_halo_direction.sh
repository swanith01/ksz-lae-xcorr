#!/bin/bash
#PBS -N test_halo_dir
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -l walltime=00:15:00
#PBS -q workq
#PBS -j oe
#PBS -o /user1/swanith/kSZ2_LAE_project_22Jun2026/logs/

CONDA_BASE="/user1/swanith/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate p21c_v41
export OMP_NUM_THREADS=4

python3 -u << 'EOF'
import py21cmfast as p21c
import numpy as np
import tempfile, os

with tempfile.TemporaryDirectory(dir="/user1/swanith/kSZ2_LAE_project_22Jun2026") as t:
    node_redshifts = np.array(p21c.get_logspaced_redshifts(
        min_redshift=5.0, max_redshift=20.0, z_step_factor=1.02
    ))
    inputs = p21c.InputParameters(
        node_redshifts=node_redshifts,
        random_seed=1,
        simulation_options=p21c.SimulationOptions(
            BOX_LEN=50.0, HII_DIM=25, DIM=50, N_THREADS=4,
            SAMPLER_MIN_MASS=1e8, SAMPLER_BUFFER_FACTOR=2.0, Z_HEAT_MAX=20.0,
        ),
        matter_options=p21c.MatterOptions(
            KEEP_3D_VELOCITIES=True,
            USE_INTERPOLATION_TABLES="hmf-interpolation",
        ),
        astro_options=p21c.AstroOptions(INHOMO_RECO=True, USE_TS_FLUCT=True),
    )
    cache = p21c.OutputCache(t)
    init_box = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=True)

    print("Testing LOW-z to HIGH-z chaining direction...")
    # reverse: go from lowest z to highest z
    reversed_z = sorted(node_redshifts)  # ascending = low to high
    prev_halocat = None
    for z in reversed_z[:5]:  # just test first 5 to confirm no crash
        halo_catalog = p21c.determine_halo_catalog(
            redshift=z,
            initial_conditions=init_box,
            descendant_halos=prev_halocat,
            inputs=inputs,
        )
        masses = halo_catalog.get("halo_masses")
        n_real = np.count_nonzero(masses)
        print(f"  z={z:.4f}  real_halos={n_real}")
        prev_halocat = halo_catalog

print("DONE - if all z show nonzero halos, low-to-high direction works")
EOF
