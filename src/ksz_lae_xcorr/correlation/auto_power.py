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

from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d, ell_at_redshift, make_ell, make_overdensity, to_Cell, to_Dell
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def compute_auto_spectra(cfg, maps: dict, seeds: list[int]) -> dict:
    """
    Returns auto_results[map_name][seed] = (D_ell, D_err), evaluated at the
    box midpoint redshift, for each map_name in maps (e.g. 'kSZ', 'kSZ2', ...).
    """
    kg = KGrid(cfg)
    z_ref = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    from ksz_lae_xcorr.utils.cosmology import get_cosmology
    cosmo = get_cosmology(cfg)
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
            C, Ce = to_Cell(P, Pe, chi_ref)
            D, De = to_Dell(ell_c, C, Ce, T_CMB_uK=T_cmb)
            auto_results[map_name][seed] = (D, De)
    return auto_results


def compute_tracer_auto_spectra(cfg, tracer_data: dict, seeds: list[int]) -> dict:
    """
    Auto-power D_ell for the tracer fields themselves (halo/lae/lbg count
    overdensity x itself), at EVERY z-slice cfg.correlation.dz_tracer_bin
    defines -- unlike compute_auto_spectra (single reference z), this is a
    genuine z-sweep, reusing the exact same z-slicing and tracer-overdensity
    construction as correlation.cross_correlation.compute_cross_spectra so
    the two stay consistent with each other.

    IMPORTANT: this depends ONLY on tracer_data (halo/LAE/LBG counts) --
    it does not touch kSZ, velocity, or xHI at all, so it is INDEPENDENT of
    the currently-unresolved stitched kSZ pathway bug. A smooth, sensible
    result here says nothing about whether that bug is fixed; it only
    validates the tracer count fields themselves, which were separately
    confirmed populated correctly (2026-09-09 checkpoint-cache fix).

    Returns auto_results[tracer_name][seed][z_center] = {'ell', 'D_ell', 'D_err'}.
    """
    cosmo = get_cosmology(cfg)
    kg = KGrid(cfg)

    dz = cfg.correlation.dz_tracer_bin
    z_lo, z_hi = cfg.box.z_min, cfg.box.z_max
    z_edges = np.arange(z_lo, z_hi + dz, dz)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])

    count_key = {"halo": "halo_count_lc", "lae": "lae_count_lc", "lbg": "lbg_count_lc"}
    tracers = list(count_key.keys())

    auto_results: dict = {t: {s: {} for s in seeds} for t in tracers}

    for seed in seeds:
        if seed not in tracer_data:
            continue

        available_tracers = [t for t in tracers if count_key[t] in tracer_data[seed]]
        if not available_tracers:
            continue

        z_nodes_t = tracer_data[seed]["z_nodes"]

        for zi, z_c in enumerate(z_cents):
            z_slice_lo, z_slice_hi = z_edges[zi], z_edges[zi + 1]
            zi_lo_t = int(np.searchsorted(z_nodes_t, z_slice_lo))
            zi_hi_t = int(np.searchsorted(z_nodes_t, z_slice_hi))
            if zi_hi_t <= zi_lo_t:
                continue

            tracer_proj = {
                t: tracer_data[seed][count_key[t]][:, :, zi_lo_t:zi_hi_t].sum(axis=2)
                for t in available_tracers
            }
            delta = {name: make_overdensity(proj) for name, proj in tracer_proj.items()}

            if all(delta[t].std() == 0 for t in available_tracers):
                continue

            chi_c = cosmo.comoving_distance(z_c).to_value("Mpc")
            ell_c = make_ell(kg.k_centers, chi_c)

            for t in available_tracers:
                d = delta[t]
                P, Pe, _ = cross_power_2d(d, d, kg)
                C, Ce = to_Cell(P, Pe, chi_c)
                D, De = to_Dell(ell_c, C, Ce, T_CMB_uK=None)
                auto_results[t][seed][z_c] = {"ell": ell_c, "D_ell": D, "D_err": De}

    return auto_results
