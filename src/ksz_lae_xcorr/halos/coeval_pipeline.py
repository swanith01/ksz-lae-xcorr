"""
halos/coeval_pipeline.py
=========================
Runs py21cmfast v4 coeval boxes + halo catalogs for a single seed.

This is a direct refactor of the original run_coeval_seed.py (22 Jun 2026,
two-pass fix 30 Jun 2026) into importable, config-driven functions. The
science logic is unchanged -- only hardcoded paths/params were pulled out
into configs/fiducial.yaml.

TWO-PASS DESIGN (unchanged from the original — see git history / docstring
below for the full diagnostic writeup that justified this):
    determine_halo_catalog()'s `descendant_halos` parameter expects the
    ALREADY-COMPUTED LOWER-z catalog. generate_coeval() steps high-z -> low-z,
    the opposite order. So:
      PASS 1: compute all halo catalogs first, low-z -> high-z, chaining
              each result forward as the descendant for the next step.
              Save halo_coords/halo_masses to disk keyed by redshift.
      PASS 2: run generate_coeval() normally (high-z -> low-z) for field
              quantities; for each redshift, load the matching
              already-computed halo catalog from PASS 1 instead of calling
              determine_halo_catalog() again.

FIELD ACCESS NOTES (v4.1.0, verified via diagnostic jobs 29-30 Jun 2026):
    - hires_density lives on coeval.initial_conditions, not perturbed_field.
    - No hires (DIM-grid) velocity field exists in this version -- only
      velocity_z on coeval.perturbed_field, at HII_DIM resolution.
    - kinetic_temperature lives on coeval.ionized_box, not ts_box.
    - halo_masses/halo_coords are allocated at a fixed buffer size
      (HALO_CATALOG_MEM_FACTOR); only nonzero-mass entries are real halos.

Usage (see scripts/01_run_coeval_seed.py for the CLI):
    from ksz_lae_xcorr.halos.coeval_pipeline import run_seed
    run_seed(cfg, seed=1)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def _seed_dirs(cfg, seed: int) -> dict:
    root = cfg.paths.coeval_root
    seed_dir = os.path.join(root, f"seed_{seed}")
    return {
        "seed_dir": seed_dir,
        "cache_dir": os.path.join(seed_dir, "cache"),
        "halo_dir": os.path.join(seed_dir, "halo_catalogs"),
    }


def build_inputs(cfg, seed: int):
    """Construct the py21cmfast InputParameters object from the config."""
    import py21cmfast as p21c

    node_redshifts = np.array(
        p21c.get_logspaced_redshifts(
            min_redshift=cfg.box.z_min,
            max_redshift=cfg.box.z_max,
            z_step_factor=cfg.box.z_step_factor,
        )
    )

    opts = cfg.p21c_options
    inputs = p21c.InputParameters(
        node_redshifts=node_redshifts,
        random_seed=seed,
        simulation_options=p21c.SimulationOptions(
            BOX_LEN=cfg.box.box_len_mpc,
            HII_DIM=cfg.box.hii_dim,
            DIM=cfg.box.dim,
            N_THREADS=int(os.environ.get("OMP_NUM_THREADS", 64)),
            Z_HEAT_MAX=cfg.box.z_max,
            SAMPLER_MIN_MASS=opts.sampler_min_mass,
            SAMPLER_BUFFER_FACTOR=opts.sampler_buffer_factor,
        ),
        matter_options=p21c.MatterOptions(
            KEEP_3D_VELOCITIES=opts.keep_3d_velocities,
            USE_INTERPOLATION_TABLES=opts.use_interpolation_tables,
        ),
        astro_options=p21c.AstroOptions(
            INHOMO_RECO=opts.inhomo_reco,
            USE_TS_FLUCT=opts.use_ts_fluct,
        ),
    )
    return inputs, node_redshifts


def _halo_paths(halo_dir: str, z: float) -> tuple[str, str]:
    coords_path = os.path.join(halo_dir, f"halo_coords_z{z:.6f}.npy")
    masses_path = os.path.join(halo_dir, f"halo_masses_z{z:.6f}.npy")
    return coords_path, masses_path


def run_pass1_halo_catalogs(cfg, seed, inputs, init_box, node_redshifts, dirs, log=print):
    """PASS 1 -- halo catalogs, low-z -> high-z. Returns elapsed minutes."""
    import py21cmfast as p21c

    t0 = time.time()
    ascending_z = np.sort(node_redshifts)
    prev_halocat = None

    for z in ascending_z:
        coords_path, masses_path = _halo_paths(dirs["halo_dir"], z)
        t_node = time.time()
        try:
            halo_catalog = p21c.determine_halo_catalog(
                redshift=z,
                initial_conditions=init_box,
                descendant_halos=prev_halocat,
                inputs=inputs,
            )
            prev_halocat = halo_catalog

            if not (os.path.exists(coords_path) and os.path.exists(masses_path)):
                coords_full = halo_catalog.get("halo_coords").astype(np.float32)
                masses_full = halo_catalog.get("halo_masses").astype(np.float32)
                valid = masses_full > 0
                coords = coords_full[valid]
                masses = masses_full[valid]
                np.save(coords_path, coords)
                np.save(masses_path, masses)
                n_halos = len(masses)
            else:
                n_halos = len(np.load(masses_path))
        except Exception as e:  # noqa: BLE001 -- log and continue, matches original behaviour
            log(f"  \u2717 Halo catalog failed at z={z:.4f}: {e}")
            n_halos = 0

        node_t = time.time() - t_node
        log(f"  z={z:7.4f}  halos={n_halos:>10,}  node_time={node_t:.1f}s  "
            f"elapsed={(time.time()-t0)/60:.1f}min")

    return (time.time() - t0) / 60


def save_coeval(coeval, out_dir, halo_dir, log=print) -> int:
    """Extract and save one coeval node's field data + matching PASS-1 halos."""
    os.makedirs(out_dir, exist_ok=True)
    z = coeval.redshift

    hires_den = coeval.initial_conditions.get("hires_density").astype(np.float32)
    np.save(os.path.join(out_dir, "hires_density.npy"), hires_den)
    del hires_den

    vz = coeval.perturbed_field.get("velocity_z").astype(np.float64)
    np.save(os.path.join(out_dir, "velocity_z.npy"), vz)
    del vz

    xHI = coeval.ionized_box.get("neutral_fraction").astype(np.float32)
    np.save(os.path.join(out_dir, "neutral_fraction.npy"), xHI)
    del xHI

    Tk = coeval.ionized_box.get("kinetic_temperature").astype(np.float32)
    np.save(os.path.join(out_dir, "kinetic_temperature.npy"), Tk)
    del Tk

    coords_path, masses_path = _halo_paths(halo_dir, z)
    if os.path.exists(coords_path) and os.path.exists(masses_path):
        coords = np.load(coords_path)
        masses = np.load(masses_path)
        np.save(os.path.join(out_dir, "halo_coords.npy"), coords)
        np.save(os.path.join(out_dir, "halo_masses.npy"), masses)
        n_halos = len(masses)
    else:
        log(f"  \u26a0 no pre-computed halo catalog found for z={z:.6f}")
        n_halos = 0

    open(os.path.join(out_dir, "DONE"), "w").write("complete")
    return n_halos


def run_seed(cfg, seed: int, log=print) -> None:
    """Run the full two-pass pipeline (init conditions, PASS 1, PASS 2) for one seed."""
    import py21cmfast as p21c

    dirs = _seed_dirs(cfg, seed)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    p21c.config["direc"] = dirs["cache_dir"]
    p21c.config["HALO_CATALOG_MEM_FACTOR"] = cfg.p21c_options.halo_catalog_mem_factor

    inputs, node_redshifts = build_inputs(cfg, seed)

    log(f"Computing initial conditions for seed {seed}...")
    cache = p21c.OutputCache(dirs["cache_dir"])
    init_box = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=True)

    log(f"PASS 1 -- halo catalogs ({len(node_redshifts)} nodes, low-z -> high-z)")
    halo_min = run_pass1_halo_catalogs(cfg, seed, inputs, init_box, node_redshifts, dirs, log=log)

    log(f"PASS 2 -- coeval fields ({len(node_redshifts)} nodes, high-z -> low-z)")
    t0 = time.time()
    for coeval, _ in p21c.generate_coeval(inputs=inputs, cache=cache):
        z = coeval.redshift
        out_dir = os.path.join(dirs["seed_dir"], f"coeval_z{z:.6f}")
        if os.path.exists(os.path.join(out_dir, "DONE")):
            log(f"  z={z:7.4f}  [SKIPPED - already done]")
            continue
        n_halos = save_coeval(coeval, out_dir, dirs["halo_dir"], log=log)
        log(f"  z={z:7.4f}  halos={n_halos:>10,}  -> {out_dir}")
        del coeval

    total_min = (time.time() - t0) / 60 + halo_min
    log(f"SEED {seed} COMPLETE -- {total_min:.1f} min total. Output: {dirs['seed_dir']}")
