"""
utils/grid.py
==============
Small shared grid-manipulation helpers. Kept separate from any one module
so both lightcone/stitch.py (which downsamples density on-the-fly during
stitching) and any standalone per-snapshot processing script (e.g. a
density-companion generator for external pipelines) use the exact same
logic -- one implementation, not two that can silently drift apart.
"""

from __future__ import annotations

import numpy as np


def block_average_downsample(box: np.ndarray, target_n: int) -> np.ndarray:
    """
    Downsample a cubic (N, N, N) array to (target_n, target_n, target_n) by
    block-averaging, requiring N to be an exact integer multiple of target_n.

    Standard, correct way to coarse-grain a finer grid onto a coarser one:
    reshape so each output cell's contributing input cells sit on their own
    axis, then mean over those axes. Preserves the full extent of the box
    (unlike naive slicing/subsampling, which would silently drop most of
    the volume -- see the DIM-vs-HII_DIM misalignment bug this replaced,
    git history Jul 2026).
    """
    n_in = box.shape[0]
    if box.shape != (n_in, n_in, n_in):
        raise ValueError(f"block_average_downsample expects a cubic array, got {box.shape}")
    if n_in == target_n:
        return box
    if n_in % target_n != 0:
        raise ValueError(
            f"Cannot downsample: input size {n_in} is not an integer multiple "
            f"of target size {target_n}. Block-averaging requires this."
        )
    factor = n_in // target_n
    reshaped = box.reshape(target_n, factor, target_n, factor, target_n, factor)
    return reshaped.mean(axis=(1, 3, 5))


def aggregate_transverse(field_3d: np.ndarray, mode: str = "sum", axis: int = 1) -> np.ndarray:
    """
    Collapse a (Nx, Ny, Nz) lightcone cube across one transverse axis
    (default axis=1) to a 2D (remaining-transverse, Nz) map, for plotting
    or side-by-side comparison of multiple fields.

    mode='sum' for discrete tracer counts (halo/LAE/LBG), mode='mean' for
    continuous fields (xHI, density, ...). Use the SAME choice consistently
    for every field shown together in one figure -- a full-transverse-width
    *sum* of a sparse discrete field is not on the same footing as a thin
    *slice* (or a mean) of a continuous one; mixing them makes panels look
    inconsistently sparse/dense for reasons that have nothing to do with
    the physics (see ksz-lae-xcorr_HANDOFF.md's lightcone visualization
    note, and the lc_panels_combined.png vs lightcone_fields_seed1.pdf
    mismatch that motivated writing this function explicitly rather than
    re-deriving the aggregation ad hoc at each call site).
    """
    if mode == "sum":
        return field_3d.sum(axis=axis)
    if mode == "mean":
        return field_3d.mean(axis=axis)
    raise ValueError(f"Unknown aggregation mode: {mode!r} (use 'sum' or 'mean')")
