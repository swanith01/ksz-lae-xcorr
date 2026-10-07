"""
correlation/lp_maps.py
=======================
MAP-BASED reproduction of the La Plante+2022 (ApJ 928, 162) estimator on our
wrap-cycle lightcones -- the "D_stitched" arm, with no bispectrum / Limber
shortcut anywhere.  Recipe (their Sec. 2-4):

  1. kSZ map     Theta(n) = sum over the LOS of the per-pixel kSZ integrand,
                 reionization era only (z >= kSZ_z_min, default 6: a documented
                 ASSUMPTION -- LP's map is their reionization-only kSZ).  The
                 e^{-tau} visibility keeps the full cumulative tau (incl. tau0
                 below z_min), because the slice sums come from
                 coherence_decomposition.compute_ksz_slices.
  2. CMB filter  f(l) = F(l) b(l),  F = C_kSZreion/(C_TT + C_kSZreion
                 + C_kSZlate + N_l).  C_kSZreion is the auto-power of THE MAP
                 BEING FILTERED (same z range, same chi), median over seeds.
  3. filter, then SQUARE the filtered 2D map (the kSZ^2 field).
  4. galaxy field delta_g = INT dz W_g(z) b_g(z) delta_m, W_g a top hat
                 with INT W dz = 1 (pixels weighted by their own dz), window
                 [z0-dz/2, z0+dz/2] floored at z=6, NO patchy clamp and NO
                 x_HI cut (LP apply none) -- b_g = Waters+16 by default.
  5. D_l^cross = l(l+1)/2pi * T_CMB^2 * C_l(kSZ_f^2 x delta_g), l = k chi_eff.

Two flat-sky conventions are inherited from the rest of the repo and are
approximations to LP's true angular maps: (i) the map lives on a comoving
(x, y) grid, so l = k * chi with ONE chi per map (chi_eff, the kSZ-power-
weighted mean for the kSZ map and the filter; the window's own chi_eff for the
cross); (ii) a 300 Mpc box has l_min ~ 190, so l = 500 is only ~2.6 k_f --
the l=500 points are very mode-starved (n_modes is reported for exactly that
reason).

Nothing here needs CAMB: the C_TT ingredient is passed in by the caller
(see scripts/28 for camb or a CSV).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.correlation.coherence_decomposition import compute_ksz_slices
from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d, make_ell, to_Cell, to_Dell
from ksz_lae_xcorr.snr.cmb_filter import build_filters, instrument_noise, kSZ_late_time
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils import constants

LP_Z_FLOOR = 6.0
DEFAULT_KSZ_Z_MIN = 6.0
DEFAULT_Z0_GRID = tuple(float(z) for z in np.arange(6.5, 13.01, 0.5))
DEFAULT_DZ_LIST = (1.0,)
LP_ELL_TARGETS = (500.0, 1000.0, 3000.0)


# --------------------------------------------------------------------------- #
# Windows / chi_eff / kSZ map (all from ONE set of precomputed slices)
# --------------------------------------------------------------------------- #
def lp_window_bounds(z0: float, dz: float, z_floor: float = LP_Z_FLOOR,
                     z_ceiling: float | None = None) -> tuple[float, float]:
    """Top-hat [z0-dz/2, z0+dz/2], floored at z_floor (LP truncate at z=6)."""
    lo = max(z0 - dz / 2.0, z_floor)
    hi = z0 + dz / 2.0
    if z_ceiling is not None:
        hi = min(hi, z_ceiling)
    if hi <= lo:
        raise ValueError(f"window z0={z0}, dz={dz} is empty after the z>={z_floor} floor")
    return float(lo), float(hi)


def chi_eff_from_slices(theta_slices: np.ndarray, chi_mpc: np.ndarray, z: np.ndarray,
                        z_lo: float, z_hi: float) -> float:
    """Same definition as roman_hls_benchmark.chi_eff_power_weighted
    (INT w^2 chi dz / INT w^2 dz, w = transverse RMS of the per-slice kSZ
    integrand) but on PRECOMPUTED slices -- no second pass of compute_ksz_slices."""
    m = (z >= z_lo) & (z <= z_hi)
    if m.sum() < 2:
        raise ValueError(f"window [{z_lo:.3f}, {z_hi:.3f}] has <2 LOS pixels")
    w = np.sqrt(np.mean(theta_slices[:, :, m] ** 2, axis=(0, 1)))
    zz, cc = z[m], chi_mpc[m]

    def trapz(y, x):
        return np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x))

    den = trapz(w ** 2, zz)
    if den == 0:
        raise ValueError("zero kSZ power in the chi_eff window (x_e = 0 everywhere?)")
    return float(trapz(w ** 2 * cc, zz) / den)


def ksz_map_from_slices(theta_slices: np.ndarray, z: np.ndarray, kSZ_z_min: float = DEFAULT_KSZ_Z_MIN):
    """Sum the per-pixel kSZ integrand over z >= kSZ_z_min -> (N, N) map of dT/T."""
    m = z >= kSZ_z_min
    if not np.any(m):
        raise ValueError(f"no LOS pixels at z >= {kSZ_z_min}")
    return theta_slices[:, :, m].sum(axis=2)


def build_lp_galaxy_map(fd: dict, zi_lo: int, zi_hi: int, bias_fn=bluetides_bias_gz) -> np.ndarray:
    """delta_g = sum_i W_i b(z_i) delta_m,i over LOS pixels [zi_lo, zi_hi), with W_i
    proportional to the pixel's own dz and sum W_i = 1 (a top hat in z with
    INT W dz = 1).  Same arithmetic as roman_hls_benchmark.
    build_bias_weighted_galaxy_field(clamp_to_patchy=False), but indexed by the
    already-floored pixel range so no window edge is recomputed in floating point."""
    z_win = fd["z_lc"][zi_lo:zi_hi]
    delta_m = fd["density_lc"][:, :, zi_lo:zi_hi] - 1.0
    dz_pix = np.gradient(z_win)
    W = dz_pix / np.sum(dz_pix)
    return np.sum(delta_m * (W * bias_fn(z_win))[None, None, :], axis=2)


def build_lp_products(cfg, fd: dict, z0_grid=DEFAULT_Z0_GRID, dz_list=DEFAULT_DZ_LIST,
                      kSZ_z_min: float = DEFAULT_KSZ_Z_MIN, bias_fn=bluetides_bias_gz) -> dict:
    """
    Everything the LP map estimator needs from ONE seed, in a small
    float32 pickle-friendly dict (a few MB):

      kSZ_map        (N, N) dT/T, z >= kSZ_z_min
      chi_eff_kSZ    power-weighted chi of that map [Mpc]
      windows[i]     {z0, dz, z_lo, z_hi, chi_eff, x_hii, x_hii_z0, gal_map}

    x_hii is the plain volume mean of 1 - x_HI over the window's LOS pixels
    (the LP Fig-5 upper axis counterpart, from OUR own simulation).
    Windows that fall outside the simulated lightcone are skipped (not an error).
    """
    theta, chi_mpc, z = compute_ksz_slices(cfg, fd)
    z_top = float(z[-1])
    kmap = ksz_map_from_slices(theta, z, kSZ_z_min)
    chi_kSZ = chi_eff_from_slices(theta, chi_mpc, z, float(max(kSZ_z_min, z[0])), z_top)

    z_lc = fd["z_lc"]
    windows = []
    for dz in dz_list:
        for z0 in z0_grid:
            try:
                lo, hi = lp_window_bounds(z0, dz, z_ceiling=z_top)
            except ValueError:
                continue
            if lo < float(z[0]):          # window dips below the simulated range
                continue
            zi_lo, zi_hi = int(np.searchsorted(z_lc, lo)), int(np.searchsorted(z_lc, hi))
            if zi_hi - zi_lo < 2:
                continue
            gal = build_lp_galaxy_map(fd, zi_lo, zi_hi, bias_fn)
            x_hii = float(np.mean(1.0 - fd["xHI_lc"][:, :, zi_lo:zi_hi]))
            iz0 = int(np.clip(np.searchsorted(z_lc, z0), 0, len(z_lc) - 1))
            windows.append({
                "z0": float(z0), "dz": float(dz), "z_lo": lo, "z_hi": hi,
                "chi_eff": chi_eff_from_slices(theta, chi_mpc, z, lo, hi),
                "x_hii": x_hii,
                "x_hii_z0": float(np.mean(1.0 - fd["xHI_lc"][:, :, iz0])),
                "gal_map": gal.astype(np.float32),
            })
    return {
        "kSZ_map": kmap.astype(np.float32), "chi_eff_kSZ": chi_kSZ, "kSZ_z_min": float(kSZ_z_min),
        "z_top": z_top, "windows": windows,
        "ne_scale": fd.get("ne_scale", 1.0), "tau0": fd.get("tau0", 0.0),
    }


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #
def kSZ_map_Cl(kmap: np.ndarray, kg: KGrid, chi_eff: float):
    """Auto C_l of a dT/T map in uK^2 (T_CMB^2 applied), l = k chi_eff.
    Returns (ell, C_l); empty bins are NaN."""
    m = kmap.astype(np.float64)
    m = m - m.mean()
    P, _, _ = cross_power_2d(m, m, kg)
    C, _ = to_Cell(P, np.zeros_like(P), chi_eff)
    return make_ell(kg.k_centers, chi_eff), C * constants.T_CMB_UK ** 2


def kSZ_reion_Cl_on_grid(maps: list, kg: KGrid, chi_eff: float, ell_grid: np.ndarray) -> np.ndarray:
    """C_l^kSZreion [uK^2] on ell_grid: median over the per-seed map auto-spectra
    (bins with no modes ignored), log-log interpolated, extrapolated at the
    edges the same way kSZ_reion_from_sim does, clipped >= 0."""
    Cs = []
    for m in maps:
        ell, C = kSZ_map_Cl(m, kg, chi_eff)
        Cs.append(C)
    C_med = np.nanmedian(np.array(Cs), axis=0)
    ok = np.isfinite(C_med) & (C_med > 0) & (ell > 10)
    if ok.sum() < 2:
        raise ValueError("fewer than 2 usable k-bins in the kSZ map auto-power")
    f = interp1d(np.log(ell[ok]), np.log(C_med[ok]), bounds_error=False, fill_value="extrapolate")
    return np.exp(f(np.log(ell_grid)))


FILTER_NORMS = ("peak", "none")
NOISE_MODELS = ("ilc", "naive")
ILC_NOISE_COMPONENT = {"SO": "SO_post_ILC_noise", "CMB-S4": "CMB-S4_post_ILC_noise",
                       "CMB-HD": "CMB-HD_post_ILC_noise"}


def ilc_noise_Cl(ell_grid, experiment: str, cfg=None) -> np.ndarray:
    """Post-ILC N_l [uK^2] for `experiment` from the digitised La Plante+22 Fig. 1 curve
    (D = l(l+1)N/2pi; log-log interpolation, clamped outside the digitised l ~ 96-10050)."""
    from ksz_lae_xcorr.io.la_plante_reference import load_fig1_components
    comp = ILC_NOISE_COMPONENT[experiment]
    ell, D = load_fig1_components(cfg)[comp]
    ell_grid = np.asarray(ell_grid, float)
    Dg = np.exp(np.interp(np.log(ell_grid), np.log(ell), np.log(D)))
    return Dg * 2 * np.pi / (ell_grid * (ell_grid + 1))


def build_lp_filter(cfg, ell_grid, Cl_TT, Cl_kSZ_reion, experiment: str = "SO",
                    normalise: str = "peak", noise: str = "ilc") -> dict:
    """f(l) = F(l) b(l) for one experiment (La Plante+22 Eq. 8, 10), via the repo's own
    snr.cmb_filter pieces.  Returns {'ell_grid','fl','fl_raw','fl_peak','Fl','bl','Nl','Cl_kSZ_late'}.

    normalise:
      'peak' (default): fl = F*b / max(F*b) over ell_grid, i.e. f peaks at 1 -- as in LP+22
                 Fig. 2 ("arbitrary normalisation ... unity where the filter is maximal").
                 Evidence that this is the amplitude convention of their cross-power: their
                 quoted RMS of the filtered kSZ^2 map is 1.6 uK^2 (Sec. 3) and the cross-power
                 peaks at ~0.02 uK^2 (Fig. 4/5).  With the raw Wiener weight (F ~ 0.02 at l=3000,
                 ~1e-4 at l=500) the squared map would be ~1e-3 uK^2 and, at l=500, the cross-power
                 could not reach their band (it is suppressed by f^2 ~ 1e-8).
      'none'   : fl = F*b exactly (the 2026-10-06 first run; amplitudes ~1/f_peak^2 too small).

    noise:
      'ilc'   (default): N_l = the post-ILC residual-foreground + instrument noise of LP+22 Fig. 1
                 (digitised from the paper's vector figure; the "ILC noise" case used for their headline
                 filters, Fig. 2).  Falls back to 'naive' for an experiment without a digitised curve.
      'naive' : N_l = Delta_N^2 b^-2 (LP+22 Eq. 11, instrument noise only; ~1.7x lower than ILC at l=3000 for SO).
    """
    if noise not in NOISE_MODELS:
        raise ValueError(f"noise must be one of {NOISE_MODELS}, got {noise!r}")
    if normalise not in FILTER_NORMS:
        raise ValueError(f"normalise must be one of {FILTER_NORMS}, got {normalise!r}")
    Cl_late = kSZ_late_time(ell_grid)
    Nl = instrument_noise(cfg, ell_grid)
    noise_used = "naive"
    if noise == "ilc" and experiment in ILC_NOISE_COMPONENT:
        Nl = dict(Nl)
        Nl[experiment] = ilc_noise_Cl(ell_grid, experiment, cfg)
        noise_used = "ilc"
    Fl, bl, fl = build_filters(cfg, ell_grid, Cl_TT, Cl_kSZ_reion, Cl_late, Nl)
    fl_raw = np.asarray(fl[experiment], float)
    peak = float(np.nanmax(fl_raw))
    fl_used = fl_raw / peak if normalise == "peak" else fl_raw
    return {"ell_grid": ell_grid, "fl": fl_used, "fl_raw": fl_raw, "fl_peak": peak, "normalise": normalise, "noise": noise_used,
            "Fl": Fl[experiment], "bl": bl[experiment], "Nl": Nl[experiment], "Cl_kSZ_late": Cl_late,
            "Cl_kSZ_reion": Cl_kSZ_reion, "Cl_TT": Cl_TT, "experiment": experiment}


def filter_and_square(kmap: np.ndarray, ell_grid: np.ndarray, fl: np.ndarray,
                      kg: KGrid, chi_ref: float) -> np.ndarray:
    """(filtered kSZ)^2: f(l) applied on the 2D FFT grid with l = k chi_ref
    (zero outside ell_grid, like snr_forecast.build_filtered_kSZ2_maps)."""
    n = kg.n_side
    dk = 2 * np.pi / kg.box_len
    kx = np.fft.fftfreq(n) * n * dk
    KX, KY = np.meshgrid(kx, kx)
    ell2d = np.sqrt(KX ** 2 + KY ** 2) * chi_ref
    f2d = interp1d(ell_grid, fl, bounds_error=False, fill_value=0.0)(ell2d)
    filt = np.real(np.fft.ifft2(np.fft.fft2(kmap.astype(np.float64)) * f2d))
    return filt ** 2


# --------------------------------------------------------------------------- #
# Cross-power and l-interpolation
# --------------------------------------------------------------------------- #
def lp_cross_dell(filtered_kSZ2: np.ndarray, gal_map: np.ndarray, kg: KGrid, chi_eff: float) -> dict:
    """D_l = l(l+1)/2pi T_CMB^2 C_l(kSZ_f^2 x delta_g), both maps mean-subtracted.
    Returns {'ell','D_ell','D_err','r','n_modes'}; empty bins NaN."""
    sig = filtered_kSZ2.astype(np.float64)
    sig = sig - sig.mean()
    g = gal_map.astype(np.float64)
    g = g - g.mean()
    P, Pe, r = cross_power_2d(sig, g, kg)
    C, Ce = to_Cell(P, Pe, chi_eff)
    ell = make_ell(kg.k_centers, chi_eff)
    D, De = to_Dell(ell, C, Ce, T_CMB_uK=constants.T_CMB_UK)
    dk = kg.k_bins[1:] - kg.k_bins[:-1]
    n_modes = np.maximum(1.0, kg.k_centers * dk * kg.area_2d / (2 * np.pi))
    return {"ell": ell, "D_ell": D, "D_err": De, "r": r, "n_modes": n_modes}


def interp_at_ell(ell: np.ndarray, y: np.ndarray, targets=LP_ELL_TARGETS) -> np.ndarray:
    """Linear-in-log(l) interpolation of y(l) onto `targets`; NaN outside the
    covered l range (no extrapolation) or when a neighbouring bin is empty."""
    ok = np.isfinite(y) & np.isfinite(ell) & (ell > 0)
    out = np.full(len(targets), np.nan)
    if ok.sum() < 2:
        return out
    lx, ly = np.log(ell[ok]), y[ok]
    order = np.argsort(lx)
    lx, ly = lx[order], ly[order]
    for i, t in enumerate(targets):
        if lx[0] <= np.log(t) <= lx[-1]:
            out[i] = np.interp(np.log(t), lx, ly)
    return out


def nearest_bin_nmodes(cross: dict, ell_target: float) -> float:
    i = int(np.nanargmin(np.abs(cross["ell"] - ell_target)))
    return float(cross["n_modes"][i])


def lp_cross_vs_z0(products_by_seed: dict, kg: KGrid, filt: dict,
                   targets=LP_ELL_TARGETS, chi_ref: float | None = None) -> dict:
    """
    Per-seed D_l(z0) at `targets`, plus the aggregate.

    products_by_seed[seed] = build_lp_products(...) output; the SAME filter
    `filt` (built from the seed-median map auto-power) is applied to every seed.
    chi_ref: l = k*chi used to evaluate the filter on the FFT grid (default: each
    seed's own chi_eff_kSZ; run_lp_analysis passes the seed-median, i.e. the same chi
    the filter's C_kSZreion was built with).  Windows are matched across seeds by
    (z0, dz).  Returns
      {'z0','dz','x_hii' (seed mean), 'ell_targets', 'seeds',
       'D' (n_seed, n_win, n_target), 'n_modes' (n_win, n_target),
       'median','mean','p16','p84','sigma_mean'  each (n_win, n_target)}
    """
    seeds = sorted(products_by_seed)
    keys = sorted({(w["dz"], w["z0"]) for s in seeds for w in products_by_seed[s]["windows"]})
    D = np.full((len(seeds), len(keys), len(targets)), np.nan)
    xh = np.full((len(seeds), len(keys)), np.nan)
    nm = np.full((len(keys), len(targets)), np.nan)
    for si, s in enumerate(seeds):
        p = products_by_seed[s]
        f2 = filter_and_square(p["kSZ_map"], filt["ell_grid"], filt["fl"], kg,
                               chi_ref if chi_ref is not None else p["chi_eff_kSZ"])
        wmap = {(w["dz"], w["z0"]): w for w in p["windows"]}
        for ki, key in enumerate(keys):
            w = wmap.get(key)
            if w is None:
                continue
            c = lp_cross_dell(f2, w["gal_map"], kg, w["chi_eff"])
            D[si, ki] = interp_at_ell(c["ell"], c["D_ell"], targets)
            xh[si, ki] = w["x_hii"]
            nm[ki] = [nearest_bin_nmodes(c, t) for t in targets]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_ok = np.sum(np.isfinite(D), axis=0)
        sd = np.nanstd(D, axis=0, ddof=1)
        out = {
            "z0": np.array([k[1] for k in keys]), "dz": np.array([k[0] for k in keys]),
            "x_hii": np.nanmean(xh, axis=0), "ell_targets": np.array(targets, float),
            "seeds": seeds, "D": D, "n_modes": nm,
            "median": np.nanmedian(D, axis=0), "mean": np.nanmean(D, axis=0),
            "p16": np.nanpercentile(D, 16, axis=0), "p84": np.nanpercentile(D, 84, axis=0),
            "sigma_mean": sd / np.sqrt(np.maximum(n_ok, 1)), "n_seeds_ok": n_ok,
        }
    return out


def compare_to_la_plante(res: dict, bands: dict) -> list[dict]:
    """
    For each ell target present in the digitized LP bands and each dz=1 window,
    the ratio of OUR seed-mean D_l to the band midpoint at that z0, and whether
    it falls inside the band.  `bands` = io.la_plante_reference.load_dell_vs_z0_bands().
    (LP bands are the spread over their three reionization histories / realizations --
    'inside' is a first-look check, not a significance test.)
    """
    rows = []
    for ti, ell in enumerate(res["ell_targets"]):
        b = bands.get(int(round(ell)))
        if b is None:
            continue
        for wi, z0 in enumerate(res["z0"]):
            if not (b["z0_lo"].min() <= z0 <= b["z0_lo"].max() and b["z0_hi"].min() <= z0 <= b["z0_hi"].max()):
                continue
            lo = float(np.interp(z0, b["z0_lo"], b["lo"]))
            hi = float(np.interp(z0, b["z0_hi"], b["hi"]))
            a, c = min(lo, hi), max(lo, hi)
            mine = float(res["mean"][wi, ti])
            rows.append({"ell": float(ell), "z0": float(z0), "dz": float(res["dz"][wi]),
                         "ours_mean": mine, "ours_median": float(res["median"][wi, ti]),
                         "lp_lo": a, "lp_hi": c, "ratio_to_mid": mine / (0.5 * (a + c)) if (a + c) else np.nan,
                         "inside": bool(a <= mine <= c)})
    return rows


def run_lp_analysis(cfg, products_by_seed: dict, cl_tt, experiment: str = "SO",
                    targets=LP_ELL_TARGETS, filter_norm: str = "peak", noise: str = "ilc") -> dict:
    """
    Full map-based LP chain from the per-seed products: filter ingredients
    (C_kSZreion = seed-median auto-power of the SAME maps; C_TT from `cl_tt`, a callable
    ell_grid -> raw C_l^TT in uK^2), the filter, filter+square, window cross-powers and
    the seed aggregate.  Returns lp_cross_vs_z0's dict plus 'filter', 'chi_ref', 'kg'.
    """
    kg = KGrid(cfg)
    seeds = sorted(products_by_seed)
    chi_ref = float(np.median([products_by_seed[s]["chi_eff_kSZ"] for s in seeds]))
    ell_grid = np.geomspace(cfg.snr.ell_min, cfg.snr.ell_max, cfg.snr.n_ell)
    Cl_kSZ = kSZ_reion_Cl_on_grid([products_by_seed[s]["kSZ_map"] for s in seeds], kg, chi_ref, ell_grid)
    filt = build_lp_filter(cfg, ell_grid, np.asarray(cl_tt(ell_grid), float), Cl_kSZ, experiment,
                           normalise=filter_norm, noise=noise)
    res = lp_cross_vs_z0(products_by_seed, kg, filt, targets, chi_ref=chi_ref)
    res.update({"filter": filt, "chi_ref": chi_ref, "kg": kg, "experiment": experiment})
    return res
