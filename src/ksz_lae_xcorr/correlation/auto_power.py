"""
correlation/auto_power.py
===========================
Auto-power spectra D_ell for the projected 2D maps (kSZ, kSZ^2, xe^2, v^2,
v_proj), evaluated at a single reference redshift (box midpoint) rather
than per tracer-redshift-bin. Refactor of notebook Cell 5.

This exists mainly to supply the linear kSZ auto-power that
snr.cmb_filter.kSZ_reion_from_sim needs -- the reionization-era kSZ term
in the La Plante+2022 filter is calibrated directly off this simulation's
own kSZ temperature power spectrum.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d, ell_at_redshift, to_Cell, to_Dell
from ksz_lae_xcorr.utils import constants


def compute_auto_spectra(cfg, maps: dict, seeds: list[int]) -> dict:
    """
    Returns auto_results[map_name][seed] = (D_ell, D_err), evaluated at the
    box midpoint redshift, for each map_name in maps (e.g. 'kSZ', 'kSZ2', ...).
    """
    kg = KGrid(cfg)
    z_ref = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    from ksz_lae_xcorr.utils.cosmology import get_cosmology, little_h
    cosmo = get_cosmology(cfg)
    h = little_h(cfg)
    chi_ref = cosmo.comoving_distance(z_ref).to_value("Mpc")
    ell_c = ell_at_redshift(cfg, kg, z_ref)

    auto_results: dict = {}
    for map_name, smaps in maps.items():
        auto_results[map_name] = {}
        for seed in seeds:
            if seed not in smaps:
                continue
            sig = smaps[seed].astype(np.float64)
            sig = sig - sig.mean()
            T_cmb = constants.T_CMB_UK if map_name == "kSZ" else None
            P, Pe, _ = cross_power_2d(sig, sig, kg)
            C, Ce = to_Cell(P, Pe, chi_ref, h)
            D, De = to_Dell(ell_c, C, Ce, T_CMB_uK=T_cmb)
            auto_results[map_name][seed] = (D, De)
    return auto_results
