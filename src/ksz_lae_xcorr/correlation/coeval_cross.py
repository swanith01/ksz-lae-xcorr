"""
correlation/coeval_cross.py
============================
D_ell^cross from COEVAL boxes (no lightcone stitching, no interpolation between snapshots, no wrap-cycle rotations).

The lightcone estimator (lp_maps) projects the whole stitched lightcone into ONE kSZ map, filters it, squares it, and
cross-correlates with the window's tracer map.  Its coeval counterpart does the same thing snapshot by snapshot:

  for every snapshot s whose redshift range overlaps the window [z_lo, z_hi]
      n_s = the LOS pixels (1 Mpc each) of the window that are closest to snapshot s            (a slab of n_s Mpc)
      Theta_s = sum over n_s consecutive cells of the snapshot box of  theta_i  (same integrand and weights W_i as the
                lightcone: W_i = tau' (1+z_i)^2 exp(-tau_i) ds_i, with the pixel's own z_i)
      g_s     = sum over the same cells of  w_i b_i delta_tracer   (same top-hat-in-z weights as build_lp_galaxy_map)
      S_s     = [F Theta_s]^2                                             F = the CMB filter of scripts/28 (2D, k_perp only)
      C_s     = C_ell( S_s x g_s )
  D_ell = ell(ell+1)/2pi T_CMB^2 * sum_s C_s

i.e. the kSZ^2 field is built from each slab ALONE (a "diagonal" estimate: pairs of LOS pixels belonging to different
snapshots are dropped, pairs inside one snapshot are kept).  The lightcone map also contains the cross-snapshot pairs, so
the two agree if those are negligible -- which is the point of the comparison.  Because a slab (tens of Mpc) is thinner
than the 300 Mpc box, the estimate is averaged over all the non-overlapping slab positions inside the box (same
expectation, less variance).  The filter is applied in k_perp only (a slab is projected along the box z axis before
filtering), so the line-of-sight modes with k_par != 0 never enter, as in a real CMB map.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps


def pixel_assignment(z_pix: np.ndarray, snap_z) -> np.ndarray:
    """Index (into the sorted snapshot list) of the snapshot nearest in z to every LOS pixel."""
    snap = np.asarray(sorted(float(s) for s in snap_z))
    mid = 0.5 * (snap[1:] + snap[:-1])
    return np.searchsorted(mid, np.asarray(z_pix))


def window_groups(assign: np.ndarray, zi_lo: int, zi_hi: int) -> list:
    """[(snapshot index, pixel indices)] for the contiguous runs of the window's pixels [zi_lo, zi_hi)."""
    idx = np.arange(zi_lo, zi_hi)
    a = assign[idx]
    groups, start = [], 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or a[k] != a[start]:
            groups.append((int(a[start]), idx[start:k]))
            start = k
    return groups


def window_pixel_weights(z_pix: np.ndarray, zi_lo: int, zi_hi: int) -> np.ndarray:
    """Top-hat in z with sum 1, each pixel weighted by its own dz (exactly lp_maps.build_lp_galaxy_map)."""
    dz = np.gradient(z_pix[zi_lo:zi_hi])
    return dz / dz.sum()


def slab_offsets(n_pix: int, n_box: int) -> list:
    """Starting cells of the non-overlapping slab positions in the box (just [0] if the slab fills the box)."""
    if n_pix >= n_box:
        return [0]
    return list(range(0, n_box - n_pix + 1, n_pix))


def _take(cube: np.ndarray, o: int, n: int) -> np.ndarray:
    n_box = cube.shape[2]
    if o + n <= n_box:
        return cube[:, :, o:o + n]
    return np.take(cube, (o + np.arange(n)) % n_box, axis=2)


def snapshot_group_cross(theta_cube: np.ndarray, gal_cubes: dict, W_theta: np.ndarray, w_gal: dict, f2d: np.ndarray,
                         kg, chi_eff: float) -> dict:
    """One snapshot, one window: {tracer: D_ell (uK^2, binned on kg's k bins)} averaged over the slab positions.

    theta_cube : (n, n, n) the integrand WITHOUT W_i:  -(1+delta) x_e v/c      (z axis = the LOS of the slab)
    gal_cubes  : {name: (n, n, n) tracer overdensity}
    W_theta    : (n_pix,) the kSZ weights W_i of the slab's LOS pixels
    w_gal      : {name: (n_pix,) weights of the galaxy map (window top hat x bias)}
    """
    n_pix, n_box = len(W_theta), theta_cube.shape[2]
    offs = slab_offsets(n_pix, n_box)
    acc = {k: None for k in gal_cubes}
    ell = None
    for o in offs:
        Theta = np.tensordot(_take(theta_cube, o, n_pix), W_theta, axes=([2], [0]))
        S = kc.apply_filter(Theta, f2d) ** 2
        for name, cube in gal_cubes.items():
            g = np.tensordot(_take(cube, o, n_pix), w_gal[name], axes=([2], [0]))
            c = lp_maps.lp_cross_dell(S, g, kg, chi_eff)
            ell = c["ell"]
            acc[name] = c["D_ell"] / len(offs) if acc[name] is None else acc[name] + c["D_ell"] / len(offs)
    return {"ell": ell, "D": acc, "n_offsets": len(offs)}
