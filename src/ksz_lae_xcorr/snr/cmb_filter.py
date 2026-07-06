"""
snr/cmb_filter.py
==================
CMB filter ingredients for the SNR forecast, following La Plante+2022
(Eq. 8, 10, 11, 14, 15). Direct refactor of notebook Cell 9a.

LAE-only by design: this module (and snr/snr_forecast.py) is never called
with the LBG tracer -- see configs/fiducial.yaml `snr.tracer` and the note
in correlation/cross_correlation.py.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.correlation.power_spectra import make_ell
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology, little_h


def camb_cl_tt(cfg, ell_grid: np.ndarray) -> np.ndarray:
    """Raw C_ell^TT (muK^2) from CAMB, interpolated onto ell_grid."""
    import camb

    c = cfg.cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=c.H0, ombh2=c.Ob0 * (c.H0 / 100) ** 2,
                        omch2=(c.Om0 - c.Ob0) * (c.H0 / 100) ** 2)
    pars.InitPower.set_params(ns=c.ns)
    pars.set_for_lmax(cfg.snr.camb_lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    Cl_TT_raw = powers["total"][:, 0]
    ell_camb = np.arange(len(Cl_TT_raw))
    return interp1d(ell_camb[2:], Cl_TT_raw[2:], bounds_error=False,
                     fill_value="extrapolate")(ell_grid)


def kSZ_late_time(ell_grid: np.ndarray) -> np.ndarray:
    """Park+2018 late-time kSZ: D_ell = 1.38*(ell/3000)^0.21 muK^2 -> C_ell."""
    D = 1.38 * (ell_grid / 3000.0) ** 0.21
    return D * 2 * np.pi / (ell_grid * (ell_grid + 1))


def kSZ_reion_from_sim(cfg, ell_grid, kg, auto_results_ksz: dict) -> np.ndarray:
    """
    Reionization-era kSZ (linear Delta T/T auto-power), from this
    simulation's own kSZ auto-power spectrum, median over seeds.
    auto_results_ksz[seed] = (D_ell, D_err) in muK^2, at the reference z.
    """
    cosmo = get_cosmology(cfg)
    h = little_h(cfg)
    z_ref = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    chi_ref = cosmo.comoving_distance(z_ref).to_value("Mpc")
    ell_sim = make_ell(kg.k_centers, chi_ref, h)

    D_seeds = [D for D, _ in auto_results_ksz.values()]
    D_med = np.nanmedian(np.array(D_seeds), axis=0)
    Cl_sim = D_med * 2 * np.pi / (ell_sim * (ell_sim + 1))

    valid = np.isfinite(Cl_sim) & (ell_sim > 10) & (Cl_sim > 0)
    Cl = interp1d(ell_sim[valid], Cl_sim[valid], bounds_error=False,
                   fill_value="extrapolate")(ell_grid)
    return np.clip(Cl, 0, None), ell_sim


def instrument_noise(cfg, ell_grid: np.ndarray) -> dict:
    """N_ell per experiment (Table 1, La Plante+2022)."""
    Nl = {}
    for name, exp in cfg.snr.experiments.items():
        th_rad = exp.theta_fwhm_arcmin * np.pi / (180 * 60)
        Nl[name] = (exp.delta_n_uk_arcmin * np.pi / 180 / 60) ** 2 * np.exp(
            th_rad**2 * ell_grid**2 / (8 * np.log(2))
        )
    return Nl


def build_filters(cfg, ell_grid, Cl_TT, Cl_kSZ_reion, Cl_kSZ_late, Nl) -> tuple[dict, dict, dict]:
    """F(ell), b(ell) beam, f(ell) = F*b -- Eq. 8, 10."""
    Fl, bl, fl = {}, {}, {}
    for name, exp in cfg.snr.experiments.items():
        th_rad = exp.theta_fwhm_arcmin * np.pi / (180 * 60)
        denom = Cl_TT + Cl_kSZ_reion + Cl_kSZ_late + Nl[name]
        Fl[name] = Cl_kSZ_reion / denom
        bl[name] = np.exp(-th_rad**2 * ell_grid**2 / (16 * np.log(2)))
        fl[name] = Fl[name] * bl[name]
    return Fl, bl, fl


def filtered_noise_power(cfg, ell_grid, Cl_TT, Cl_kSZ_reion, Cl_kSZ_late, Nl, fl) -> tuple[dict, dict]:
    """
    C_ell^(T-bar T-bar, f) (Eq. 15) and C_ell^(T^2 T^2, f) Gaussian approx
    (Eq. 14, ell-independent scalar per experiment).
    """
    Cl_TT_f, Cl_T2T2_f = {}, {}
    for name in cfg.snr.experiments:
        total = Cl_TT + Cl_kSZ_reion + Cl_kSZ_late + Nl[name]
        Cl_TT_f[name] = fl[name] ** 2 * total
        integrand = ell_grid * Cl_TT_f[name] ** 2 / (2 * np.pi)
        integral = np.trapz(integrand, ell_grid)
        Cl_T2T2_f[name] = 2.0 * integral * np.ones_like(ell_grid)
    return Cl_TT_f, Cl_T2T2_f


def build_cmb_filter_ingredients(cfg, kg, auto_results_ksz: dict) -> dict:
    """
    Runs the full Cell-9a pipeline and returns everything snr_forecast needs:
    {'ell_grid','Cl_TT','Cl_kSZ_reion','Cl_kSZ_late','Nl','Fl','bl','fl','Cl_TT_f','Cl_T2T2_f'}
    """
    ell_grid = np.geomspace(cfg.snr.ell_min, cfg.snr.ell_max, cfg.snr.n_ell)
    Cl_TT = camb_cl_tt(cfg, ell_grid)
    Cl_kSZ_late = kSZ_late_time(ell_grid)
    Cl_kSZ_reion, _ = kSZ_reion_from_sim(cfg, ell_grid, kg, auto_results_ksz)
    Nl = instrument_noise(cfg, ell_grid)
    Fl, bl, fl = build_filters(cfg, ell_grid, Cl_TT, Cl_kSZ_reion, Cl_kSZ_late, Nl)
    Cl_TT_f, Cl_T2T2_f = filtered_noise_power(cfg, ell_grid, Cl_TT, Cl_kSZ_reion, Cl_kSZ_late, Nl, fl)
    return {
        "ell_grid": ell_grid, "Cl_TT": Cl_TT, "Cl_kSZ_reion": Cl_kSZ_reion,
        "Cl_kSZ_late": Cl_kSZ_late, "Nl": Nl, "Fl": Fl, "bl": bl, "fl": fl,
        "Cl_TT_f": Cl_TT_f, "Cl_T2T2_f": Cl_T2T2_f,
    }
