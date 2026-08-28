"""
utils/external_catalogue.py
==============================
Filename convention for Jahaan's external LAE/LBG catalogue files.

CONFIRMED against real 300 Mpc data (Aug 2026): files are named
    {prefix}_z{z:.6f}_{grid}_{box_len}_s{seed}.npy
e.g. halo_ids_obs_z10.085533_300_300_s1.npy (grid=300, box_len=300 Mpc).
The old 400 Mpc/64^3 exploratory run's files use the same pattern with
_64_400_ instead, confirming [grid]_[box_len] order (verified against that
run's known parameters: grid=64, box_len=400 Mpc).

This differs from what earlier code assumed (4-decimal z, no grid/box tag)
-- centralizing it here so it's defined in exactly one place, not
independently duplicated (and able to silently drift) across
lightcone/stitch.py, tracers/type_b_grids.py, and lightcone/value_fields.py,
which is exactly what happened before this was caught.
"""

from __future__ import annotations


def external_catalogue_filename(prefix: str, z: float, cfg, seed: int) -> str:
    """Build the exact real filename Jahaan's pipeline uses for one field/z/seed."""
    grid = int(cfg.box.hii_dim)
    box_len = int(cfg.box.box_len_mpc)
    return f"{prefix}_z{z:.6f}_{grid}_{box_len}_s{seed}.npy"
