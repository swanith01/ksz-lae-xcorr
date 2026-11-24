#!/usr/bin/env python3
# =============================================================================
# run_coeval_seed.py
# kSZ2 x LAE Project  —  22 Jun 2026  (rewritten 30 Jun 2026 — two-pass halo fix)
# =============================================================================
# Runs py21cmfast v4 coeval boxes for a single seed.
# Saves per-redshift-node .npy files needed by Jahaan's LAE pipeline.
#
# 30 Jun 2026 FIX — TWO-PASS DESIGN:
#   determine_halo_catalog()'s `descendant_halos` parameter expects the
#   ALREADY-COMPUTED LOWER-z catalog (it sets desc_redshift =
#   descendant_halos.redshift internally, then walks backward to find
#   higher-z progenitors). generate_coeval() steps high-z -> low-z, which
#   is the OPPOSITE order. Calling determine_halo_catalog() inline inside
#   the generate_coeval() loop (passing the previous, higher-z catalog in
#   as "descendant") is backwards and silently produces zero halos for
#   every redshift after the first.
#
#   Verified by diagnostic job 1627635 (30 Jun 2026): chaining
#   determine_halo_catalog() low-z -> high-z (ascending z, each step's
#   result passed as descendant_halos to the next, higher-z step) gives
#   real, smoothly-varying nonzero halo counts at every redshift.
#
#   Fix: compute ALL halo catalogs first, in their own pass, low-z to
#   high-z. Save halo_coords/halo_masses to disk keyed by redshift. THEN
#   run generate_coeval() as normal (high-z -> low-z) for the field
#   quantities, and for each redshift just load the matching
#   already-computed halo catalog from pass 1 instead of calling
#   determine_halo_catalog() again.
#
# RESOLUTION NOTE:
#   BOX_LEN=300, HII_DIM=300, DIM=600 (0.5 cMpc/cell on the DIM grid,
#   1.0 cMpc/cell on the HII_DIM grid) — matches Jahaan's LAE pipeline.
#
# FIELD ACCESS NOTES (v4.1.0, verified via diagnostic jobs 29-30 Jun 2026):
#   - hires_density lives on coeval.initial_conditions, NOT
#     coeval.perturbed_field.
#   - There is no hires (DIM-grid) velocity field in this version — only
#     hires_vx/vy/vz_2LPT (2LPT correction terms, not the velocity field
#     itself) exist on initial_conditions at DIM res. The actual
#     velocity_z field only exists on coeval.perturbed_field, at HII_DIM
#     resolution. Halos remain hi-res (sampled from hires_density
#     directly), so density-halo cross-correlation keeps full DIM
#     resolution; velocity is coarser.
#   - kinetic_temperature lives on coeval.ionized_box, NOT coeval.ts_box.
#     ts_box has a related but distinct field: kinetic_temp_neutral.
#   - halo_masses/halo_coords from HaloCatalog.get() are allocated at a
#     fixed buffer size (set by HALO_CATALOG_MEM_FACTOR); only
#     nonzero-mass entries are real halos. Buffer padding is trimmed
#     before saving (verified via diagnostic job 1627548 — buffer size
#     1,000,000, only 122,462 nonzero/real for that test).
#
# OUTPUT STRUCTURE:
#   ~/kSZ2_LAE_project_22Jun2026/seed_{N}/coeval_z{z:.6f}/
#       hires_density.npy        (600, 600, 600)  float32  [dimensionless overdensity, DIM res]
#       velocity_z.npy           (300, 300, 300)  float64  [Mpc/s, comoving LoS velocity, HII_DIM res]
#       neutral_fraction.npy     (300, 300, 300)  float32  [xHI, 0-1, HII_DIM res]
#       kinetic_temperature.npy  (300, 300, 300)  float32  [K, HII_DIM res]
#       halo_coords.npy          (N_halos, 3)     float32  [cMpc]
#       halo_masses.npy          (N_halos,)       float32  [M_sun]
#       DONE                                       checkpoint marker
#
# USAGE:
#   python run_coeval_seed.py --seed 1
#
# PBS: submit via submit_all_seeds.sh
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run py21cmfast coeval boxes for one seed.")
parser.add_argument("--seed", type=int, required=True, help="Random seed (1-10)")
args = parser.parse_args()
SEED = args.seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = os.path.expanduser("~/kSZ2_LAE_project_22Jun2026")
SEED_DIR      = os.path.join(PROJECT_ROOT, f"seed_{SEED}")
CACHE_DIR     = os.path.join(SEED_DIR, "cache")
HALO_DIR      = os.path.join(SEED_DIR, "halo_catalogs")  # pass-1 output

for d in [PROJECT_ROOT, SEED_DIR, CACHE_DIR, HALO_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# py21cmfast setup
# ---------------------------------------------------------------------------

import py21cmfast as p21c
p21c.config["direc"] = CACHE_DIR
p21c.config["HALO_CATALOG_MEM_FACTOR"] = 3.0

print(f"\n{'='*70}")
print(f"kSZ2 x LAE  —  py21cmfast v{p21c.__version__}  —  seed {SEED}")
print(f"{'='*70}")
print(f"  Project root : {PROJECT_ROOT}")
print(f"  Seed dir     : {SEED_DIR}")
print(f"  Cache dir    : {CACHE_DIR}")
print(f"  Halo dir     : {HALO_DIR}")

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

Z_MIN         = 5.0
Z_MAX         = 20.0
Z_STEP_FACTOR = 1.02

node_redshifts = p21c.get_logspaced_redshifts(
    min_redshift  = Z_MIN,
    max_redshift  = Z_MAX,
    z_step_factor = Z_STEP_FACTOR,
)
node_redshifts = np.array(node_redshifts)

print(f"\n  z range      : {Z_MIN} -> {Z_MAX}  (step factor {Z_STEP_FACTOR})")
print(f"  Nodes        : {len(node_redshifts)}  "
      f"[{node_redshifts.min():.4f}, {node_redshifts.max():.4f}]")

inputs = p21c.InputParameters(
    node_redshifts = node_redshifts,
    random_seed    = SEED,

    simulation_options = p21c.SimulationOptions(
        BOX_LEN               = 300.0,
        HII_DIM               = 300,
        DIM                   = 600,
        N_THREADS             = int(os.environ.get("OMP_NUM_THREADS", 64)),
        Z_HEAT_MAX             = 20.0,
        SAMPLER_MIN_MASS      = 1e8,
        SAMPLER_BUFFER_FACTOR = 2.0,
    ),

    matter_options = p21c.MatterOptions(
        KEEP_3D_VELOCITIES       = True,
        USE_INTERPOLATION_TABLES = "hmf-interpolation",
    ),

    astro_options = p21c.AstroOptions(
        INHOMO_RECO  = True,
        USE_TS_FLUCT = True,
    ),
)

print(f"\n  BOX_LEN      : {inputs.simulation_options.BOX_LEN:.1f} cMpc")
print(f"  HII_DIM      : {inputs.simulation_options.HII_DIM}  "
      f"({inputs.simulation_options.BOX_LEN/inputs.simulation_options.HII_DIM:.3f} cMpc/cell)")
print(f"  DIM          : {inputs.simulation_options.DIM}  "
      f"({inputs.simulation_options.BOX_LEN/inputs.simulation_options.DIM:.3f} cMpc/cell)")
print(f"  N_THREADS    : {inputs.simulation_options.N_THREADS}")
print(f"  KEEP_3D_VEL  : {inputs.matter_options.KEEP_3D_VELOCITIES}")
print(f"  INHOMO_RECO  : {inputs.astro_options.INHOMO_RECO}")
print(f"  USE_TS_FLUCT : {inputs.astro_options.USE_TS_FLUCT}")
print(f"  SAMPLE_METHOD: {inputs.matter_options.SAMPLE_METHOD}")

# ---------------------------------------------------------------------------
# Initial conditions  (computed once, reused for all redshifts)
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"  Computing initial conditions ...")
t0_ic = time.time()

cache    = p21c.OutputCache(CACHE_DIR)
init_box = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=True)

print(f"  ✓ Initial conditions done  ({(time.time()-t0_ic)/60:.1f} min)")


def halo_paths(z):
    coords_path = os.path.join(HALO_DIR, f"halo_coords_z{z:.6f}.npy")
    masses_path = os.path.join(HALO_DIR, f"halo_masses_z{z:.6f}.npy")
    return coords_path, masses_path


# ---------------------------------------------------------------------------
# PASS 1 — halo catalogs, LOW-z to HIGH-z
# ---------------------------------------------------------------------------
# determine_halo_catalog()'s descendant_halos must be the already-computed
# LOWER-z catalog. So we must iterate node_redshifts ASCENDING (low z
# first), chaining each result forward as the descendant for the next
# (higher-z) step. This is the opposite order to generate_coeval().

print(f"\n{'='*70}")
print(f"  PASS 1 — halo catalogs, {len(node_redshifts)} nodes, low-z -> high-z")
print(f"{'='*70}\n")

t0_halo = time.time()
ascending_z = np.sort(node_redshifts)  # low z first
prev_halocat = None

for z in ascending_z:
    coords_path, masses_path = halo_paths(z)

    if os.path.exists(coords_path) and os.path.exists(masses_path):
        # already done — but we still need this catalog object in memory
        # to chain to the NEXT (higher-z) step, so reload it as a
        # descendant input. determine_halo_catalog needs a HaloCatalog
        # object, not just raw arrays, so we still have to recompute it
        # here cheaply via the cache (py21cmfast's own OutputCache will
        # skip recomputation if the underlying fields are unchanged).
        pass

    t_node = time.time()
    try:
        halo_catalog = p21c.determine_halo_catalog(
            redshift            = z,
            initial_conditions  = init_box,
            descendant_halos    = prev_halocat,
            inputs              = inputs,
        )
        prev_halocat = halo_catalog

        if not (os.path.exists(coords_path) and os.path.exists(masses_path)):
            coords_full = halo_catalog.get("halo_coords").astype(np.float32)
            masses_full = halo_catalog.get("halo_masses").astype(np.float32)
            valid  = masses_full > 0
            coords = coords_full[valid]
            masses = masses_full[valid]
            np.save(coords_path, coords)
            np.save(masses_path, masses)
            n_halos = len(masses)
            del coords, masses, coords_full, masses_full
        else:
            masses = np.load(masses_path)
            n_halos = len(masses)
            del masses

    except Exception as e:
        print(f"  ✗ Halo catalog failed at z={z:.4f}: {e}")
        n_halos = 0

    node_t  = time.time() - t_node
    elapsed = (time.time() - t0_halo) / 60
    print(f"  z={z:7.4f}  halos={n_halos:>10,}  "
          f"node_time={node_t:.1f}s  elapsed={elapsed:.1f}min")
    sys.stdout.flush()

halo_total_min = (time.time() - t0_halo) / 60
print(f"\n  ✓ PASS 1 complete — {halo_total_min:.1f} min total\n")

# ---------------------------------------------------------------------------
# Helper: save one coeval node's field data + matching pre-computed halos
# ---------------------------------------------------------------------------

def save_coeval(coeval, out_dir):
    """
    Extract and save all field quantities, plus load the matching
    pre-computed (pass 1) halo catalog for this redshift.
    Purges large arrays immediately after saving to keep RAM flat.
    """
    os.makedirs(out_dir, exist_ok=True)
    z = coeval.redshift

    # ---- hires density (DIM grid, from initial_conditions) -----------------
    hires_den = coeval.initial_conditions.get("hires_density").astype(np.float32)
    np.save(os.path.join(out_dir, "hires_density.npy"), hires_den)
    del hires_den

    # ---- LoS velocity (HII_DIM grid, z-axis) -- v4.1.0 has no hires velocity --
    vz = coeval.perturbed_field.get("velocity_z").astype(np.float64)
    np.save(os.path.join(out_dir, "velocity_z.npy"), vz)
    del vz

    # ---- neutral fraction (HII_DIM grid) -----------------------------------
    xHI = coeval.ionized_box.get("neutral_fraction").astype(np.float32)
    np.save(os.path.join(out_dir, "neutral_fraction.npy"), xHI)
    del xHI

    # ---- kinetic temperature (HII_DIM grid) --------------------------------
    Tk = coeval.ionized_box.get("kinetic_temperature").astype(np.float32)
    np.save(os.path.join(out_dir, "kinetic_temperature.npy"), Tk)
    del Tk

    # ---- halo catalogue — load pre-computed result from PASS 1 -------------
    coords_path, masses_path = halo_paths(z)
    if os.path.exists(coords_path) and os.path.exists(masses_path):
        coords = np.load(coords_path)
        masses = np.load(masses_path)
        np.save(os.path.join(out_dir, "halo_coords.npy"), coords)
        np.save(os.path.join(out_dir, "halo_masses.npy"), masses)
        n_halos = len(masses)
        del coords, masses
    else:
        print(f"  ⚠ no pre-computed halo catalog found for z={z:.6f}")
        n_halos = 0

    open(os.path.join(out_dir, "DONE"), "w").write("complete")
    return n_halos


# ---------------------------------------------------------------------------
# PASS 2 — coeval fields, HIGH-z to LOW-z (generate_coeval's native order)
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"  PASS 2 — coeval fields, {len(node_redshifts)} nodes, high-z -> low-z")
print(f"{'='*70}\n")

t0_total = time.time()

for coeval, in_outputs in p21c.generate_coeval(
    inputs = inputs,
    cache  = cache,
):
    z = coeval.redshift
    out_dir = os.path.join(SEED_DIR, f"coeval_z{z:.6f}")

    if os.path.exists(os.path.join(out_dir, "DONE")):
        elapsed = (time.time() - t0_total) / 60
        print(f"  z={z:7.4f}  [SKIPPED - already done]  elapsed={elapsed:.1f}min")
        sys.stdout.flush()
        del coeval
        continue

    t_node = time.time()
    n_halos = save_coeval(coeval, out_dir)

    elapsed = (time.time() - t0_total) / 60
    node_t  = time.time() - t_node
    print(f"  z={z:7.4f}  halos={n_halos:>10,}  "
          f"node_time={node_t:.1f}s  elapsed={elapsed:.1f}min  -> {out_dir}")
    sys.stdout.flush()

    del coeval

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

total_min = (time.time() - t0_total) / 60 + halo_total_min
print(f"\n{'='*70}")
print(f"  ✓  SEED {SEED} COMPLETE  —  {total_min:.1f} min total (both passes)")
print(f"  Output: {SEED_DIR}")
print(f"{'='*70}\n")
