"""
snr/survey_selection.py
=========================
Realistic LAE survey selection layered on top of the "optimistic" S/N
forecast (snr/snr_forecast.py, which assumes every simulated LAE above the
halo mass cut is observed). Refactored from Cells 10a-10d and 11a-11h.

Two levels of realism, both implemented generically (one code path reused
across all named surveys, not one hand-copied block per survey as in the
original notebook):

1. "Optimistic-with-shot-noise" (Cell 10a-10c equivalent): keeps the full
   simulated LAE spatial field, but corrects the auto-power's shot-noise
   term for the survey's real (flux-cut-reduced) number density, computed
   via a duty-cycle fraction from the full-3D LAE luminosity field.

2. "Realistic" (Cell 11 equivalent): actually builds a NEW spatial field
   where only LAEs passing the survey's flux (and, for SILVERRUSH, REW)
   cut at each redshift survive -- both the clustering signal and the
   shot noise reflect the real, flux-limited sample.

Requires lc_lae_lum_3d.npz (and lc_lae_rew_3d.npz for SILVERRUSH) from
lightcone/value_fields.py -- run scripts/07_stitch_lae_value_fields.py first.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.correlation.power_spectra import (
    KGrid, cross_power_2d, make_ell, make_overdensity, to_Cell, to_Dell,
)
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology


# =============================================================================
# Flux / REW cut interpolators
# =============================================================================

def _table_interp(cfg_survey, z: float, key: str) -> float:
    """Linear interpolation of `key` (L_min or REW_min) vs z from a config table.
    NaN outside the table's z-range -- the survey's cut simply doesn't apply there."""
    z_arr = np.array([pt["z"] for pt in cfg_survey])
    v_arr = np.array([pt[key] for pt in cfg_survey])
    if z < z_arr.min() or z > z_arr.max():
        return np.nan
    return float(np.interp(z, z_arr, v_arr))


def silverrush_lmin(cfg, z: float) -> float:
    return _table_interp(cfg.silverrush_flux_cuts, z, "L_min")


def silverrush_rewmin(cfg, z: float) -> float:
    return _table_interp(cfg.silverrush_flux_cuts, z, "REW_min")


def roman_grism_lmin(cfg, cosmo, z: float) -> float:
    """
    L_Lya,min [erg/s] for Roman Grism at redshift z. NaN outside [6, 10]
    (the calibrated/simulated Roman-Grism range). Below `ceiling_z_max`:
    flat flux ceiling converted to L_min(z) via luminosity distance.
    Above: tabulated luminosity limits (Wold et al. 2023) interpolated directly.
    """
    rg = cfg.roman_grism_flux_cut
    if z < 6.0 or z > 10.0:
        return np.nan
    if z < rg.ceiling_z_max:
        d_L_cm = cosmo.luminosity_distance(z).to_value("cm")
        return 4 * np.pi * d_L_cm**2 * rg.flux_ceiling_erg_s_cm2
    z_arr = np.array([pt["z"] for pt in rg.tabulated])
    l_arr = np.array([pt["L_min"] for pt in rg.tabulated])
    return float(np.interp(z, z_arr, l_arr))


# =============================================================================
# Duty cycle / shot noise ("optimistic-with-shot-noise" level)
# =============================================================================

def load_lae_value_field(lightcone_root: str, seed: int, value_field: str):
    """Load a full-3D (lc_occ, lc_val) product from lightcone/value_fields.py."""
    path = os.path.join(lightcone_root, f"seed_{seed}", f"lc_{value_field}_3d.npz")
    d = np.load(path)
    return d["lc_occ"], d["lc_val"]


def compute_n_bar_sim(cfg, tracer_data: dict, seeds: list[int], z_edges, z_cents,
                       z_nodes, cosmo) -> dict:
    """Simulated LAE number density [sr^-1] per z-bin, median over seeds."""
    n_bar_sim = {}
    for zi, z_c in enumerate(z_cents):
        zi_lo = int(np.searchsorted(z_nodes, z_edges[zi]))
        zi_hi = int(np.searchsorted(z_nodes, z_edges[zi + 1]))
        if zi_hi <= zi_lo:
            continue
        counts = [tracer_data[s]["lae_count_lc"][:, :, zi_lo:zi_hi].sum()
                  for s in seeds if s in tracer_data and "lae_count_lc" in tracer_data[s]]
        if not counts:
            continue
        N_med = np.median(counts)
        chi_c = cosmo.comoving_distance(z_c).to_value("Mpc")
        omega_sr = (cfg.box.box_len_mpc / chi_c) ** 2
        n_bar_sim[z_c] = N_med / omega_sr
    return n_bar_sim


def compute_duty_cycle(cfg, survey_cfg, lightcone_root: str, seeds: list[int],
                        z_edges, z_cents, z_nodes, lmin_fn) -> dict:
    """
    Fraction of simulated LAEs bright enough to survive the survey's flux
    cut, per z-bin, averaged over seeds. lmin_fn(z) -> L_min or NaN.
    """
    occ_3d, lum_3d = {}, {}
    for s in seeds:
        try:
            occ_3d[s], lum_3d[s] = load_lae_value_field(lightcone_root, s, "lae_lum")
        except FileNotFoundError:
            continue

    duty_cycle = {}
    for zi, z_c in enumerate(z_cents):
        L_min = lmin_fn(z_c)
        if not np.isfinite(L_min):
            continue
        zi_lo = int(np.searchsorted(z_nodes, z_edges[zi]))
        zi_hi = int(np.searchsorted(z_nodes, z_edges[zi + 1]))
        if zi_hi <= zi_lo:
            continue

        fracs = []
        for s in seeds:
            if s not in occ_3d:
                continue
            occ_slice = occ_3d[s][:, :, zi_lo:zi_hi]
            lum_slice = lum_3d[s][:, :, zi_lo:zi_hi]
            mask = occ_slice > 0
            if mask.sum() == 0:
                continue
            lum_vals = lum_slice[mask]
            fracs.append(np.sum(lum_vals >= L_min) / len(lum_vals))
        if fracs:
            duty_cycle[z_c] = float(np.mean(fracs))
    return duty_cycle


def apply_shot_noise_correction(cl_lae_auto: dict, n_bar_sim: dict, duty_cycle: dict,
                                 has_flux_cut: bool) -> dict:
    """
    Cl_lae_auto_survey[z_c] = Cl_lae_auto[z_c] + (1/n_bar_survey - 1/n_bar_sim),
    i.e. swap the simulation's (higher-density) Poisson shot noise for the
    survey's real (flux-limited, lower-density) shot noise. No-op if the
    survey has no flux cut (e.g. spectroscopic surveys without flux-cut data).
    """
    out = {}
    for z_c, base in cl_lae_auto.items():
        delta_shot = 0.0
        if has_flux_cut and z_c in duty_cycle and z_c in n_bar_sim:
            nb_sim = n_bar_sim[z_c]
            nb_surv = duty_cycle[z_c] * nb_sim
            if nb_sim > 0 and nb_surv > 0:
                delta_shot = (1.0 / nb_surv) - (1.0 / nb_sim)
        out[z_c] = base + delta_shot
    return out


# =============================================================================
# Realistic joint-cut field ("realistic" level)
# =============================================================================

def build_survey_cut_field(lightcone_root: str, seeds: list[int], z_nodes,
                            lmin_fn, rewmin_fn=None) -> dict:
    """
    Builds a NEW full-3D LAE count field per seed, keeping only cells where
    the per-pixel-averaged luminosity (and REW, if rewmin_fn given) clears
    the survey's cut at that redshift. Caveat (inherited from the original):
    lum/rew are per-pixel AVERAGES (Type-B grids average over all LAEs
    sharing a cell), not true per-source values -- a pixel is kept/rejected
    as a whole based on its average. Best available without a true
    per-source catalogue.

    rewmin_fn=None -> luminosity-only cut (e.g. Roman-Grism).
    """
    lmin_per_pixel = np.array([lmin_fn(z) for z in z_nodes])
    in_range = np.isfinite(lmin_per_pixel)
    rewmin_per_pixel = np.array([rewmin_fn(z) for z in z_nodes]) if rewmin_fn else None

    cut_fields = {}
    for s in seeds:
        try:
            occ, lum = load_lae_value_field(lightcone_root, s, "lae_lum")
            rew = load_lae_value_field(lightcone_root, s, "lae_rew")[1] if rewmin_fn else None
        except FileNotFoundError:
            continue

        cut = np.zeros_like(occ)
        for i in np.where(in_range)[0]:
            pass_cut = (occ[:, :, i] > 0) & (lum[:, :, i] >= lmin_per_pixel[i])
            if rewmin_fn:
                pass_cut &= rew[:, :, i] >= rewmin_per_pixel[i]
            cut[:, :, i] = np.where(pass_cut, occ[:, :, i], 0.0)
        cut_fields[s] = cut
    return cut_fields


def compute_realistic_power_spectra(cfg, cut_fields: dict, filtered_kSZ2: dict,
                                     seeds: list[int], z_lo: float, z_hi: float,
                                     z_edges, z_cents, z_nodes, ell_grid) -> tuple[dict, dict]:
    """
    Cl_lae_auto and Cl_signal (per CMB experiment) from a survey's realistic
    cut field, restricted to [z_lo, z_hi] (the survey's calibrated range).
    Same estimator as correlation/cross_correlation.py, reused directly.
    """
    cosmo = get_cosmology(cfg)
    kg = KGrid(cfg)

    cl_lae_auto: dict = {}
    cl_signal = {name: {} for name in cfg.snr.experiments}

    for zi, z_c in enumerate(z_cents):
        if z_c < z_lo or z_c > z_hi:
            continue
        zi_lo = int(np.searchsorted(z_nodes, z_edges[zi]))
        zi_hi = int(np.searchsorted(z_nodes, z_edges[zi + 1]))
        if zi_hi <= zi_lo:
            continue

        chi_c = cosmo.comoving_distance(z_c).to_value("Mpc")
        ell_c = make_ell(kg.k_centers, chi_c)

        cl_seeds = []
        for s in seeds:
            if s not in cut_fields:
                continue
            proj = cut_fields[s][:, :, zi_lo:zi_hi].sum(axis=2)
            if proj.sum() == 0:
                continue
            delta_g = make_overdensity(proj)
            P, Pe, _ = cross_power_2d(delta_g - delta_g.mean(), delta_g - delta_g.mean(), kg)
            C, _ = to_Cell(P, Pe, chi_c)
            cl_seeds.append(C)
        if cl_seeds:
            Cl_med = np.nanmedian(np.array(cl_seeds), axis=0)
            valid = np.isfinite(Cl_med) & (ell_c > 10)
            if valid.sum() >= 3:
                cl_lae_auto[z_c] = interp1d(ell_c[valid], Cl_med[valid], bounds_error=False,
                                             fill_value="extrapolate")(ell_grid)

        for name in cfg.snr.experiments:
            D_seeds = []
            for s in seeds:
                if s not in cut_fields or s not in filtered_kSZ2.get(name, {}):
                    continue
                proj = cut_fields[s][:, :, zi_lo:zi_hi].sum(axis=2)
                if proj.sum() == 0:
                    continue
                delta_g = make_overdensity(proj)
                sig = filtered_kSZ2[name][s].astype(np.float64)
                sig = sig - sig.mean()
                P, Pe, _ = cross_power_2d(sig, delta_g - delta_g.mean(), kg)
                C, _ = to_Cell(P, Pe, chi_c)
                D, _ = to_Dell(ell_c, C, np.zeros_like(C), T_CMB_uK=constants.T_CMB_UK)
                D_seeds.append(D)
            if D_seeds:
                D_med = np.nanmedian(np.array(D_seeds), axis=0)
                Cl_med = D_med * 2 * np.pi / (ell_c * (ell_c + 1))
                valid = np.isfinite(Cl_med) & (ell_c > 10)
                if valid.sum() >= 3:
                    cl_signal[name][z_c] = interp1d(ell_c[valid], Cl_med[valid], bounds_error=False,
                                                     fill_value=0.0)(ell_grid)

    return cl_lae_auto, cl_signal


# =============================================================================
# S/N from a (Cl_signal, Cl_lae_auto) pair -- generic, reused at every level
# =============================================================================

def compute_survey_snr(cfg, filt: dict, cl_signal: dict, cl_lae_auto: dict, f_sky: float) -> dict:
    """(S/N)^2 per z-bin per experiment -- same discrete-ell-sum estimator as
    snr/snr_forecast.py, generalized to an arbitrary f_sky and z-bin set."""
    ell_grid = filt["ell_grid"]
    ell_int = np.arange(cfg.snr.ell_min, cfg.snr.ell_max + 1)

    sn_results = {name: {} for name in cfg.snr.experiments}
    for name in cfg.snr.experiments:
        if name not in cl_signal:
            continue
        CT2_val = filt["Cl_T2T2_f"][name][0]
        for z_c, cl_s in cl_signal[name].items():
            if z_c not in cl_lae_auto:
                continue
            Cs_int = interp1d(ell_grid, cl_s, bounds_error=False, fill_value=0.0)(ell_int)
            Cg_int = interp1d(ell_grid, cl_lae_auto[z_c], bounds_error=False, fill_value=0.0)(ell_int)
            num = Cs_int**2
            denom = CT2_val * np.abs(Cg_int) + Cs_int**2
            good = denom > 0
            SN2 = f_sky * np.sum((2 * ell_int[good] + 1) * num[good] / denom[good])
            sn_results[name][z_c] = float(np.sqrt(max(SN2, 0.0)))
    return sn_results


def total_snr(sn_per_z: dict) -> float:
    """Combine per-z-bin S/N in quadrature -- sqrt(sum of squares)."""
    vals = np.array(list(sn_per_z.values()))
    return float(np.sqrt(np.sum(vals**2))) if len(vals) else float("nan")
