"""
lightcone/wrap_cycle.py
========================
Wrap-cycle stitched lightcone: the periodicity fix developed in
ksz-pipeline (src/ksz_pipeline/ksz/stitch_from_coeval.py, commits 0c3b578 /
41157a5; handoff docs/HANDOFF_wrap_cycle.md), adapted to this repo's
interfaces. ALGORITHM copied, INTERFACES adapted -- see "What differs" below.

THE PROBLEM
-----------
A lightcone through a periodic box of side L = BOX_LEN re-visits the same
LOS slab every L of comoving distance (every `ngrid` cells).  A single FIXED
transverse rotation does NOT break this: the identical (i, j) -> (ir, jr)
map is applied on every visit, so each revisit pulls a bit-identical slab.
Those exact repeats make P_off (cross terms between LOS pixels) a pure
periodicity artifact, which then inflates P_total over the true
(Limber/direct) answer.

THE FIX
-------
  * one "wrap cycle" = one BOX_LEN of comoving distance along the LOS;
  * each cycle gets its own transverse rotation angle, an independent draw
    keyed on (wrap_cycle_seed, cycle_index);
  * the angle depends ONLY on LOS position, never on the snapshot: at every
    target redshift one angle is applied to ALL snapshots' slabs (and to all
    three fields) before interpolating across snapshots, so density, x_HI
    and v_z stay physically co-registered;
  * rotated sampling is bilinear and periodic (map_coordinates order=1)
    rather than nearest-index, which aliases 10-31% of pixels at generic
    angles.
Rotation is in the transverse plane only, about index (0, 0); v_z is a
scalar under rotation about the LOS so it needs no transformation.

WHAT DIFFERS FROM ksz-pipeline (all deliberate)
-----------------------------------------------
  1. Cosmology: build_los_z_grid takes OUR astropy cosmology
     (utils.cosmology.get_cosmology), not Planck18.  No second cosmology.
  2. Velocity: NOTHING here touches velocity units.  ksz-pipeline runs
     py21cmfast v3, whose raw lowres velocity is a Zel'dovich displacement
     needing D(z) f(z) H(z)/(1+z); this repo is v4, whose velocity_z is
     already a comoving peculiar velocity in Mpc/s (confirmed on real data,
     see io/loaders.py).  The stitcher passes values through untouched, and
     tests/test_wrap_cycle.py fails on any rescaling.  NEVER apply the v3
     conversion to our v4 boxes.
  3. Periodic edge: scipy mode='wrap' has period n-1 (a one-pixel
     distortion at every wrap); mode='grid-wrap' has period n and is
     correct for a periodic grid.  Default here is 'grid-wrap'.  Pass
     mode='wrap' to reproduce ksz-pipeline bit-for-bit.
  4. Cycle bookkeeping uses the integer slab index n = rint(d/cell):
     y_cell = n % ngrid, cycle = n // ngrid, so a slab's cycle is always
     consistent with its y_cell.  ksz-pipeline uses floor(d/BOX_LEN) for the
     cycle and round(d/cell) for y_cell, which disagree for the half-cell
     before each cycle boundary (<= 1 slab per cycle; immaterial).
  5. Streaming: only the two snapshots bracketing the current target are
     held in memory (per field), not all ~70.  Piecewise-linear
     interpolation across snapshots only ever uses the bracketing pair, so
     this is mathematically identical to interp1d over the full stack.
  6. Density: boxes are expected as RAW delta (our hires_density, block-
     averaged to HII_DIM by Stitcher.load_field_box).  The caller adds 1
     exactly once.

Pure functions plus one streaming stitcher; no I/O except through the
`load_box` callable the caller supplies.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

_VALID_MODES = ("grid-wrap", "wrap")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def build_los_z_grid(cosmo, z_min: float, z_max: float, cell_size: float,
                     z_oversample: int = 4000):
    """
    LOS redshift grid with UNIFORM COMOVING spacing == cell_size (not
    uniform in z).

    The legacy linspace(z_min, z_max, n_lc_pix) grid has dchi/dz-driven
    spacing from ~16 Mpc/pixel at z=5 to ~2.4 Mpc at z=20 for this repo's
    300 Mpc / 512-pixel setup -- i.e. up to 16 box cells per LOS pixel at low
    z, where the kSZ signal lives.  The kSZ integral is ds-weighted per
    pixel and P_diag = sum_i |theta_i|^2 scales as ds^2 per pixel, so that
    spacing mismatch alone biases P_diag (and under-samples structure)
    independent of any periodicity effect.

    Returns
    -------
    z_arr   : (n_pix,) ascending redshifts
    chi_mpc : (n_pix,) comoving distance of each pixel, EXACTLY
              chi_mpc[0] + cell_size * arange(n_pix)
    """
    z_fine = np.linspace(z_min, z_max, int(z_oversample))
    chi_fine = np.asarray(cosmo.comoving_distance(z_fine).to_value("Mpc"), dtype=float)
    n_pix = int(round((chi_fine[-1] - chi_fine[0]) / cell_size))
    chi_target = chi_fine[0] + cell_size * np.arange(n_pix)
    return np.interp(chi_target, chi_fine, z_fine), chi_target


def slab_index(d_mpc, cell_size: float) -> np.ndarray:
    """Integer LOS slab number n = rint(d / cell_size), d measured from z0."""
    return np.rint(np.asarray(d_mpc, dtype=float) / cell_size).astype(np.int64)


def wrap_cycle_index(d_mpc, cell_size: float, ngrid: int) -> np.ndarray:
    """Wrap cycle of each LOS position: n // ngrid with n = slab_index(d)."""
    return slab_index(d_mpc, cell_size) // int(ngrid)


def cycle_angle(cycle_idx: int, seed: int) -> float:
    """
    Rotation angle [degrees] for one wrap cycle: an independent uniform draw
    on [0, 360) keyed on (seed, cycle_idx).  Same value for every field and
    every snapshot at that LOS position (see module docstring).

    NOTE: numpy does not formally guarantee Generator streams across
    versions; the reference values in tests/test_wrap_cycle.py were produced
    with numpy 2.4.4 (same as ksz-pipeline's handoff).
    """
    rng = np.random.default_rng((int(seed), int(cycle_idx)))
    return float(rng.uniform(0.0, 360.0))


def rotated_coords(ngrid: int, angle_deg: float):
    """Float (ic, jc) sampling coordinates for a rotation by angle_deg:
    ic = cos(a) i - sin(a) j,  jc = sin(a) i + cos(a) j  (meshgrid 'ij')."""
    a = np.deg2rad(angle_deg)
    i, j = np.meshgrid(np.arange(ngrid), np.arange(ngrid), indexing="ij")
    return np.cos(a) * i - np.sin(a) * j, np.sin(a) * i + np.cos(a) * j


def get_slab_bilinear(box: np.ndarray, y_cell: int, coords, mode: str = "grid-wrap") -> np.ndarray:
    """Rotated, bilinearly-sampled, transversely periodic (ngrid, ngrid) slab
    of `box` at LOS index y_cell."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    return map_coordinates(box[:, :, int(y_cell)], [coords[0], coords[1]],
                           order=1, mode=mode)


# --------------------------------------------------------------------------
# streaming stitcher
# --------------------------------------------------------------------------

def stitch_wrapcycle(load_box: Callable[[float, str], np.ndarray],
                     snap_z: Sequence[float],
                     z_arr: np.ndarray,
                     chi_mpc: np.ndarray,
                     *,
                     ngrid: int,
                     cell_size: float,
                     wrap_cycle_seed: int,
                     fields: Sequence[str] = ("xH", "density", "vz"),
                     mode: str = "grid-wrap",
                     dtype=np.float32,
                     progress: Callable[[int, int], None] | None = None) -> dict:
    """
    Build lightcone cubes (ngrid, ngrid, len(z_arr)) for each field.

    Parameters
    ----------
    load_box : callable (z, field_name) -> ndarray (ngrid, ngrid, ngrid).
        Must return the box ALREADY on the HII_DIM grid, with units exactly
        as they should appear in the lightcone (we do not convert anything).
    snap_z : snapshot redshifts (any order); must have >= 2 entries.
        Targets outside [min, max] are linearly extrapolated from the end
        pair (same as scipy interp1d fill_value='extrapolate').
    z_arr, chi_mpc : target grid from build_los_z_grid (ascending in z).
        Distance from the first pixel, chi_mpc - chi_mpc[0], sets the slab
        index / wrap cycle.
    wrap_cycle_seed : seed for cycle_angle.

    Returns {field: ndarray(ngrid, ngrid, n_pix) of `dtype`}.
    """
    snap = np.asarray(sorted(float(z) for z in snap_z))
    if len(snap) < 2:
        raise ValueError("need at least 2 snapshots to interpolate across")
    z_arr = np.asarray(z_arr, dtype=float)
    if np.any(np.diff(z_arr) <= 0):
        raise ValueError("z_arr must be strictly ascending (streaming relies on it)")
    n_pix = len(z_arr)

    n = slab_index(np.asarray(chi_mpc, dtype=float) - float(chi_mpc[0]), cell_size)
    y_cells = n % ngrid
    cycles = n // ngrid

    out = {f: np.empty((ngrid, ngrid, n_pix), dtype=dtype) for f in fields}
    coords_by_cycle: dict[int, tuple] = {}
    box_cache: dict[tuple, np.ndarray] = {}

    for k in range(n_pix):
        z = z_arr[k]
        lo = int(np.clip(np.searchsorted(snap, z) - 1, 0, len(snap) - 2))
        hi = lo + 1
        t = (z - snap[lo]) / (snap[hi] - snap[lo])

        for key in [kk for kk in box_cache if kk[0] not in (lo, hi)]:
            del box_cache[key]

        c = int(cycles[k])
        if c not in coords_by_cycle:
            coords_by_cycle[c] = rotated_coords(ngrid, cycle_angle(c, wrap_cycle_seed))
        coords = coords_by_cycle[c]
        y = int(y_cells[k])

        for f in fields:
            slabs = []
            for idx in (lo, hi):
                if (idx, f) not in box_cache:
                    box = np.asarray(load_box(float(snap[idx]), f))
                    if box.shape != (ngrid,) * 3:
                        raise ValueError(
                            f"load_box({snap[idx]:.4f}, {f!r}) returned shape {box.shape}, "
                            f"expected {(ngrid,) * 3}")
                    box_cache[(idx, f)] = box
                slabs.append(get_slab_bilinear(box_cache[(idx, f)], y, coords, mode))
            out[f][:, :, k] = (1.0 - t) * slabs[0] + t * slabs[1]

        if progress is not None and (k % 250 == 0 or k == n_pix - 1):
            progress(k, n_pix)
    return out
