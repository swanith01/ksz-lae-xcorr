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
