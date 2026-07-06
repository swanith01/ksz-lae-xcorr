"""
correlation/cross_correlation.py
==================================
Cross-power spectra: {kSZ2, xe2, v2, v_proj, v_proj2} x {halo, LAE, LBG},
per redshift slice. Direct refactor of notebook Cell 6.

Note: halo and LBG cross-correlations are computed here for physics
interpretation (see README) but are NOT propagated into the SNR pipeline
(ksz_lae_xcorr.snr), which is LAE-only per the fiducial config.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import (
    KGrid,
    cross_power_2d,
    make_ell,
    make_overdensity,
    to_Cell,
    to_Dell,
)
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology, little_h


def compute_cross_spectra(cfg, maps: dict, tracer_data: dict, seeds: list[int]) -> dict:
    """
    maps: output of correlation.projected_maps.build_projected_maps
    tracer_data: as in build_projected_maps

    Returns: results[tracer_name][seed][z_center][signal_name] =
        {'ell', 'D_ell', 'D_err', 'r'}
    for tracer_name in ('halo', 'lae', 'lbg').
    """
    cosmo = get_cosmology(cfg)
    h = little_h(cfg)
    kg = KGrid(cfg)

    dz = cfg.correlation.dz_tracer_bin
    z_lo, z_hi = cfg.box.z_min, cfg.box.z_max
    z_edges = np.arange(z_lo, z_hi + dz, dz)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])

    signal_maps = {name: maps[name] for name in cfg.correlation.signals}
    tracer_names = cfg.correlation.tracers  # ('halo', 'lae', 'lbg')

    results = {t: {s: {} for s in seeds} for t in tracer_names}

    for seed in seeds:
        if seed not in maps["kSZ2"] or seed not in tracer_data:
            continue

        z_nodes_t = tracer_data[seed]["z_nodes"]

        for zi, z_c in enumerate(z_cents):
            z_slice_lo, z_slice_hi = z_edges[zi], z_edges[zi + 1]
            zi_lo_t = int(np.searchsorted(z_nodes_t, z_slice_lo))
            zi_hi_t = int(np.searchsorted(z_nodes_t, z_slice_hi))
            if zi_hi_t <= zi_lo_t:
                continue

            tracer_proj = {
                "halo": tracer_data[seed]["halo_count_lc"][:, :, zi_lo_t:zi_hi_t].sum(axis=2),
                "lae": tracer_data[seed]["lae_count_lc"][:, :, zi_lo_t:zi_hi_t].sum(axis=2),
                "lbg": tracer_data[seed]["lbg_count_lc"][:, :, zi_lo_t:zi_hi_t].sum(axis=2),
            }
            delta = {name: make_overdensity(proj) for name, proj in tracer_proj.items()}

            if delta["lae"].std() == 0 and delta["lbg"].std() == 0:
                continue

            chi_c = cosmo.comoving_distance(z_c).to_value("Mpc")
            ell_c = make_ell(kg.k_centers, chi_c, h)

            for t in tracer_names:
                results[t][seed][z_c] = {}

            for signal_name, smaps in signal_maps.items():
                if seed not in smaps:
                    continue
                sig = smaps[seed].astype(np.float64)
                sig = sig - sig.mean()
                T_cmb = constants.T_CMB_UK if signal_name == "kSZ2" else None

                for t in tracer_names:
                    d = delta[t]
                    P, Pe, r = cross_power_2d(sig, d - d.mean(), kg)
                    C, Ce = to_Cell(P, Pe, chi_c, h)
                    D, De = to_Dell(ell_c, C, Ce, T_CMB_uK=T_cmb)
                    results[t][seed][z_c][signal_name] = {"ell": ell_c, "D_ell": D, "D_err": De, "r": r}

    return results
