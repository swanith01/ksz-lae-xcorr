"""
io/loaders.py
==============
Loads the stitched 3D lightcone products (written by
ksz_lae_xcorr.lightcone.stitch) into the field_data / tracer_data dict
structure the correlation/ and snr/ modules expect. Direct refactor of
notebook Cell 2.

Only reads pre-computed products from paths.lightcone_root -- no heavy
computation happens here, consistent with the project convention that
notebooks/scripts only load finished products.
"""

from __future__ import annotations

import os

import numpy as np

from ksz_lae_xcorr.utils import constants


def load_lightcone_products(cfg, seeds: list[int]) -> tuple[dict, dict]:
    """
    Returns (field_data, tracer_data), keyed by seed:

    field_data[seed] = {
        'z_lc', 'xHI_lc', 'density_lc' (1+delta), 'velocity_kms', 'velocity_lc' (Mpc/s)
    }
    tracer_data[seed] = {
        'z_nodes', 'halo_count_lc', 'lae_count_lc', 'lbg_count_lc'
    }
    Seeds whose lightcone products aren't found on disk are silently skipped
    (matches the original notebook's "only load seeds that are ready" behaviour).
    """
    lc_root = cfg.paths.lightcone_root
    c_kms = constants.C_KMS
    c_mpc_s = constants.c_mpc_per_s()

    field_data: dict = {}
    tracer_data: dict = {}

    for seed in seeds:
        seed_dir = os.path.join(lc_root, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue

        try:
            xHI = np.load(os.path.join(seed_dir, "lc_xH.npz"))["lc"]
            density = np.load(os.path.join(seed_dir, "lc_density.npz"))["lc"]
            vz_data = np.load(os.path.join(seed_dir, "lc_vz.npz"))
            vz_kms = vz_data["lc"]
            z_lc = vz_data["z_arr"]
        except FileNotFoundError:
            continue

        density_1pdelta = 1.0 + density.astype(np.float64)
        vz_mpc_s = vz_kms.astype(np.float64) / c_kms * c_mpc_s

        field_data[seed] = {
            "z_lc": z_lc,
            "xHI_lc": xHI.astype(np.float64),
            "density_lc": density_1pdelta,
            "velocity_kms": vz_kms.astype(np.float64),
            "velocity_lc": vz_mpc_s,
        }

        try:
            halos = np.load(os.path.join(seed_dir, "lc_halos.npz"))["lc"]
            lae = np.load(os.path.join(seed_dir, "lc_lae.npz"))["lc"]
            lbg = np.load(os.path.join(seed_dir, "lc_lbg.npz"))["lc"]
        except FileNotFoundError:
            continue

        tracer_data[seed] = {
            "z_nodes": z_lc,
            "halo_count_lc": halos.astype(np.float64),
            "lae_count_lc": lae.astype(np.float64),
            "lbg_count_lc": lbg.astype(np.float64),
        }

    return field_data, tracer_data


def save_product(path: str, **arrays) -> None:
    """Thin wrapper around np.savez_compressed with directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
