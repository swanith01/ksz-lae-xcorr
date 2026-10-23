"""
correlation/ksz_components.py
==============================
Which part of the kSZ^2 field carries the kSZ^2 x tracer cross-power: the electron (ionisation) field or the
velocity field?

The kSZ integrand of one line-of-sight (LOS) pixel i is (correlation/coherence_decomposition.compute_ksz_slices)

    theta_i = - W_i (1 + delta_i) x_e,i (v_i / c),     W_i = tau' (1+z_i)^2 exp(-tau_i) ds_i ,

and the La Plante+22 kSZ^2 field is  S = ( F * sum_i theta_i )^2  (F = the CMB filter on the 2D FFT grid).  To see what
modulates the small-scale kSZ power we build the same kind of squared field from ONE ingredient at a time:

    x : X_i = W_i x_e,i                       (the ionisation field; the velocity is not in it at all)
    v : X_i = W_i xbar_e(z_i) v_i / c         (the velocity field; x_e and delta are at their slice mean)
    d : X_i = W_i xbar_e(z_i) (1 + delta_i)   (the density field, for closure; not plotted by default)
    full : the real theta_i

and, for each of them, TWO orderings of "project" and "square":

    A  [ F sum_i X_i ]^2         project along the LOS first, then square   (what a CMB map does: includes every
                                 pair of pixels i != j along the line of sight)
    B  sum_i [ F X_i ]^2         square every slice first, then project     (diagonal: pairs i = j only; no
                                 correlation along the line of sight, i.e. a Limber / coeval-box-like estimate)

so "x_e projected, then squared" is A_x, "x_e squared, then projected" is B_x, and the same for v.  A - B is the
cross-slice part.  Every map has the same filter F (the one scripts/28 uses), the same projection weights W_i, and
only the z >= kSZ_z_min pixels, exactly like the kSZ map of lp_maps.

The maps differ in units, so the cross-power is taken with the FRACTIONAL modulation  s = S / <S> - 1  :

    D^frac_ell = ell(ell+1)/2pi  C_ell( s x delta_g ).

"A_full" is then the (dimensionless) twin of the usual D_ell^cross.  The ratio D^frac(X) / D^frac(full) says which
fraction of the kSZ^2-tracer correlation each ingredient reproduces.
"""

from __future__ import annotations

import glob
import os
import pickle
import re

import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d, make_ell, to_Cell, to_Dell
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology

COMPONENTS = ("x", "v", "d", "full")
ORDERINGS = ("A", "B")
VARIANTS = tuple(f"{o}_{c}" for c in COMPONENTS for o in ORDERINGS)     # A_x, B_x, A_v, ... , A_full, B_full
LABELS = {"x": r"$x_e$", "v": r"$v_{\rm los}$", "d": r"$1+\delta$", "full": "full kSZ"}
ORDER_LABELS = {"A": "project, then square", "B": "square, then project"}


# --------------------------------------------------------------------------- #
# Per-pixel weights (identical to compute_ksz_slices)
# --------------------------------------------------------------------------- #
def ksz_slice_weights(cfg, fd: dict) -> dict:
    """W_i = tau' (1+z_i)^2 exp(-tau_i) ds_i for the LOS pixels cfg.box.z_min <= z < cfg.box.z_max, with exactly the
    arithmetic of compute_ksz_slices.  Returns {'W','z','chi','xe_mean','zi_lo','zi_hi'}."""
    cosmo = get_cosmology(cfg)
    tau_pref = constants.tau_prefactor(cfg) * float(fd.get("ne_scale", 1.0))
    tau0 = float(fd.get("tau0", 0.0))
    z_lc = fd["z_lc"]
    zi_lo = int(np.searchsorted(z_lc, cfg.box.z_min))
    zi_hi = int(np.searchsorted(z_lc, cfg.box.z_max))
    z = np.asarray(z_lc[zi_lo:zi_hi])
    xe_mean = (1.0 - fd["xHI_lc"][:, :, zi_lo:zi_hi]).mean(axis=(0, 1))
    return {**slice_weights_from_means(cosmo, tau_pref, tau0, z, xe_mean), "zi_lo": zi_lo, "zi_hi": zi_hi}


def slice_weights_from_means(cosmo, tau_pref: float, tau0: float, z: np.ndarray, xe_mean: np.ndarray) -> dict:
    """The same weights from the slice-mean ionised fraction alone (no 3D field needed: the coeval estimator
    uses it with xbar_e interpolated from the snapshot means)."""
    d_com = cosmo.comoving_distance(z).to_value("Mpc")
    ds = np.abs(np.gradient(d_com))
    dtau = tau_pref * xe_mean * (1.0 + z) ** 2 * ds
    tau = tau0 + np.cumsum(dtau)
    W = tau_pref * (1.0 + z) ** 2 * np.exp(-tau) * ds
    return {"W": W, "z": z, "chi": d_com, "xe_mean": np.asarray(xe_mean, float), "ds": ds, "tau": tau}


# --------------------------------------------------------------------------- #
# Filter on the FFT grid
# --------------------------------------------------------------------------- #
def filter_grid(kg: KGrid, ell_grid: np.ndarray, fl: np.ndarray, chi_ref: float) -> np.ndarray:
    """f(l = k chi_ref) on the rfft2 grid (n, n//2+1); zero outside ell_grid -- the same filter as
    lp_maps.filter_and_square, just on the half-plane."""
    n = kg.n_side
    dk = 2 * np.pi / kg.box_len
    ky = np.fft.fftfreq(n) * n * dk
    kx = np.fft.rfftfreq(n) * n * dk
    ell2d = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2) * chi_ref
    return interp1d(ell_grid, fl, bounds_error=False, fill_value=0.0)(ell2d)


def apply_filter(arr: np.ndarray, f2d: np.ndarray) -> np.ndarray:
    """Filter the (n, n) map or the (n, n, K) stack (filter over the first two axes)."""
    n = arr.shape[0]
    F = np.fft.rfft2(arr, axes=(0, 1))
    F = F * (f2d if arr.ndim == 2 else f2d[:, :, None])
    return np.fft.irfft2(F, s=(n, n), axes=(0, 1))


# --------------------------------------------------------------------------- #
# The component maps
# --------------------------------------------------------------------------- #
def component_maps(cfg, fd: dict, f2d: np.ndarray, kSZ_z_min: float = lp_maps.DEFAULT_KSZ_Z_MIN,
                   chunk: int = 32, weights: dict | None = None) -> dict:
    """{variant: (n, n) float32 map} for the 8 variants (A/B x x/v/d/full) of one seed, plus 'theta_sum' (the kSZ
    map itself, as a check against lp_maps) in the returned dict under the key '_ksz_map'."""
    wt = weights or ksz_slice_weights(cfg, fd)
    W, z, xe_mean, zi_lo, zi_hi = wt["W"], wt["z"], wt["xe_mean"], wt["zi_lo"], wt["zi_hi"]
    c_mpc_s = constants.c_mpc_per_s()
    sel = np.where(z >= kSZ_z_min)[0]
    if sel.size == 0:
        raise ValueError(f"no LOS pixels at z >= {kSZ_z_min}")
    n = fd["xHI_lc"].shape[0]
    sumA = {c: np.zeros((n, n)) for c in COMPONENTS}
    sumB = {c: np.zeros((n, n)) for c in COMPONENTS}
    for s0 in range(0, sel.size, chunk):
        idx = sel[s0:s0 + chunk]
        gi = zi_lo + idx
        xe = 1.0 - fd["xHI_lc"][:, :, gi]
        dens = fd["density_lc"][:, :, gi]
        vc = fd["velocity_lc"][:, :, gi] / c_mpc_s
        w = W[idx][None, None, :]
        xm = xe_mean[idx][None, None, :]
        stacks = {"x": w * xe, "v": w * xm * vc, "d": w * xm * dens, "full": -(w * dens * xe * vc)}
        for c, X in stacks.items():
            sumA[c] += X.sum(axis=2)
            sumB[c] += (apply_filter(X, f2d) ** 2).sum(axis=2)
    out = {}
    for c in COMPONENTS:
        out[f"A_{c}"] = (apply_filter(sumA[c], f2d) ** 2).astype(np.float32)
        out[f"B_{c}"] = sumB[c].astype(np.float32)
    out["_ksz_map"] = sumA["full"].astype(np.float32)
    return out


def window_chunk_maps(cfg, fd: dict, f2d: np.ndarray, windows: list, kSZ_z_min: float = lp_maps.DEFAULT_KSZ_Z_MIN,
                      chunk: int = 32, weights: dict | None = None) -> list:
    """For every window {z_lo, z_hi, ...}: the maps built from the window's LOS pixels ALONE (z >= kSZ_z_min), to look at
    the kSZ^2 pattern of one redshift chunk next to the tracer map of the same chunk.

    Returns a list (one dict per window) with float32 (n, n) maps
      th_full, th_x, th_v        the linear projections  sum_i X_i  (full integrand, x_e only, v only)
      A_full, A_x, A_v           [F sum_i X_i]^2          (project, then square)
      B_full, B_x, B_v           sum_i [F X_i]^2          (square every slice, then project)
    and 'n_pix'.  The definitions of X_i are those of component_maps."""
    wt = weights or ksz_slice_weights(cfg, fd)
    W, xe_mean, zi_lo0, zi_hi0 = wt["W"], wt["xe_mean"], wt["zi_lo"], wt["zi_hi"]
    z_lc = fd["z_lc"]
    c_mpc_s = constants.c_mpc_per_s()
    n = fd["xHI_lc"].shape[0]
    floor = int(np.searchsorted(z_lc, kSZ_z_min))
    comps = ("full", "x", "v")
    out = []
    for w in windows:
        lo = max(int(np.searchsorted(z_lc, w["z_lo"])), zi_lo0, floor)
        hi = min(int(np.searchsorted(z_lc, w["z_hi"])), zi_hi0)
        if hi - lo < 1:
            raise ValueError(f"window z {w['z_lo']}-{w['z_hi']} has no LOS pixels in the kSZ range")
        sumA = {c: np.zeros((n, n)) for c in comps}
        sumB = {c: np.zeros((n, n)) for c in comps}
        for s0 in range(lo, hi, chunk):
            gi = np.arange(s0, min(s0 + chunk, hi))
            idx = gi - zi_lo0
            xe = 1.0 - fd["xHI_lc"][:, :, gi]
            dens = fd["density_lc"][:, :, gi]
            vc = fd["velocity_lc"][:, :, gi] / c_mpc_s
            wgt = W[idx][None, None, :]
            xm = xe_mean[idx][None, None, :]
            stacks = {"x": wgt * xe, "v": wgt * xm * vc, "full": -(wgt * dens * xe * vc)}
            for c, X in stacks.items():
                sumA[c] += X.sum(axis=2)
                sumB[c] += (apply_filter(X, f2d) ** 2).sum(axis=2)
        m = {"n_pix": hi - lo}
        for c in comps:
            m[f"th_{c}"] = sumA[c].astype(np.float32)
            m[f"A_{c}"] = (apply_filter(sumA[c], f2d) ** 2).astype(np.float32)
            m[f"B_{c}"] = sumB[c].astype(np.float32)
        out.append(m)
    return out


# --------------------------------------------------------------------------- #
# Fractional cross-power
# --------------------------------------------------------------------------- #
def frac_cross_dell(S: np.ndarray, gal_map: np.ndarray, kg: KGrid, chi_eff: float) -> dict:
    """D^frac_l = l(l+1)/2pi C_l( S/<S> - 1  x  delta_g ): the dimensionless cross-power of the fractional modulation
    of S with the (mean-subtracted) tracer map.  Returns {'ell','D_ell','n_modes'}; empty bins NaN."""
    S = S.astype(np.float64)
    m = S.mean()
    if not np.isfinite(m) or m <= 0:
        return {"ell": make_ell(kg.k_centers, chi_eff), "D_ell": np.full(kg.n_kbins, np.nan),
                "n_modes": np.ones(kg.n_kbins)}
    s = S / m - 1.0
    g = gal_map.astype(np.float64)
    g = g - g.mean()
    P, Pe, _ = cross_power_2d(s, g, kg)
    C, Ce = to_Cell(P, Pe, chi_eff)
    ell = make_ell(kg.k_centers, chi_eff)
    D, _ = to_Dell(ell, C, Ce, T_CMB_uK=None)
    dk = kg.k_bins[1:] - kg.k_bins[:-1]
    return {"ell": ell, "D_ell": D, "n_modes": np.maximum(1.0, kg.k_centers * dk * kg.area_2d / (2 * np.pi))}


def map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two maps (both mean-subtracted)."""
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def fraction_of_full(dv: np.ndarray, df: np.ndarray, x_hii: np.ndarray, xrange=(0.1, 0.7),
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Window-pooled ratio  sum_w <D_variant> / sum_w <D_full>  over the windows with xrange[0] <= x_HII <= xrange[1],
    <.> the seed mean, with a seed bootstrap.  dv, df: (n_seed, n_window) fractional D at one ell for the variant and
    for the full kSZ^2 of the same ordering.  Returns {'R','lo','hi'} (median and 16/84 percentiles of the bootstrap)."""
    sel = (x_hii >= xrange[0]) & (x_hii <= xrange[1]) & np.isfinite(dv).all(axis=0) & np.isfinite(df).all(axis=0)
    if not sel.any():
        return {"R": np.nan, "lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(seed)
    ns = dv.shape[0]
    rs = []
    for _ in range(n_boot):
        ii = rng.integers(0, ns, ns)
        den = df[ii][:, sel].mean(axis=0).sum()
        rs.append(dv[ii][:, sel].mean(axis=0).sum() / den if den != 0 else np.nan)
    rs = np.array(rs)
    return {"R": float(np.nanmedian(rs)), "lo": float(np.nanpercentile(rs, 16)), "hi": float(np.nanpercentile(rs, 84))}


# --------------------------------------------------------------------------- #
# Filter from the stored kSZ products (the same chain as lp_maps.run_lp_analysis)
# --------------------------------------------------------------------------- #
def load_lpmaps_products(in_dir: str, offset: int = 0) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(in_dir, f"seed*_wo{offset}_lpmaps.pkl"))):
        m = re.search(r"seed(\d+)_wo", os.path.basename(p))
        with open(p, "rb") as f:
            out[int(m.group(1))] = pickle.load(f)
    return out


def cl_tt_provider(cfg, cltt_file: str | None = None):
    if cltt_file:
        tab = np.loadtxt(cltt_file, delimiter=",", skiprows=1)
        ell_t, cl_t = tab[:, 0], tab[:, 1]
        return lambda ell: np.interp(ell, ell_t, cl_t)
    from ksz_lae_xcorr.snr.cmb_filter import camb_cl_tt
    return lambda ell: camb_cl_tt(cfg, ell)


def build_filter_from_products(cfg, prods: dict, cl_tt, experiment: str = "SO", filter_norm: str = "peak",
                               noise: str = "ilc"):
    """(kg, filt, chi_ref): the filter of scripts/28 (C_kSZreion = seed-median auto-power of the stored kSZ maps)."""
    kg = KGrid(cfg)
    seeds = sorted(prods)
    chi_ref = float(np.median([prods[s]["chi_eff_kSZ"] for s in seeds]))
    ell_grid = np.geomspace(cfg.snr.ell_min, cfg.snr.ell_max, cfg.snr.n_ell)
    Cl_kSZ = lp_maps.kSZ_reion_Cl_on_grid([prods[s]["kSZ_map"] for s in seeds], kg, chi_ref, ell_grid)
    filt = lp_maps.build_lp_filter(cfg, ell_grid, np.asarray(cl_tt(ell_grid), float), Cl_kSZ, experiment,
                                   normalise=filter_norm, noise=noise)
    return kg, filt, chi_ref
