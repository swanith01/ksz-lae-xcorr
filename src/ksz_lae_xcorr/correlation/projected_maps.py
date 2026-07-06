"""
correlation/projected_maps.py
==============================
Projects 3D lightcones (from ksz_lae_xcorr.lightcone.stitch output) down to
2D maps: the kSZ signal and its derived quantities, plus tracer overdensity
maps for halo/LAE/LBG. Direct refactor of notebook Cell 4.

kSZ integrand (see also utils.constants.tau_prefactor):
    Delta T / T = -sigma_T n_e0 INT (1+delta) x_e (v/c) (1/a^2) e^{-tau} ds
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import make_overdensity
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def build_projected_maps(cfg, field_data: dict, tracer_data: dict, seeds: list[int]) -> dict:
    """
    field_data[seed]  = {'z_lc','xHI_lc','density_lc' (1+delta),'velocity_kms','velocity_lc' (Mpc/s)}
    tracer_data[seed] = {'z_nodes','halo_count_lc','lae_count_lc','lbg_count_lc'}

    Returns a dict of dicts, one entry per map name, each keyed by seed:
    {'kSZ','kSZ2','xe2','v2','v_proj','v_proj2','halo','lae','lbg'}
    """
    cosmo = get_cosmology(cfg)
    tau_prefactor = constants.tau_prefactor(cfg)
    c_mpc_s = constants.c_mpc_per_s()

    z_lo, z_hi = cfg.box.z_min, cfg.box.z_max

    out = {k: {} for k in ("kSZ", "kSZ2", "xe2", "v2", "v_proj", "v_proj2", "halo", "lae", "lbg")}

    for seed in seeds:
        if seed not in field_data:
            continue

        z_lc = field_data[seed]["z_lc"]
        xHI_lc = field_data[seed]["xHI_lc"]
        density_lc = field_data[seed]["density_lc"]  # (1+delta)
        v_kms_lc = field_data[seed]["velocity_kms"]
        v_mpcs_lc = field_data[seed]["velocity_lc"]

        zi_lo = int(np.searchsorted(z_lc, z_lo))
        zi_hi = int(np.searchsorted(z_lc, z_hi))
        z = z_lc[zi_lo:zi_hi]
        xHI = xHI_lc[:, :, zi_lo:zi_hi]
        delta = density_lc[:, :, zi_lo:zi_hi]
        v_k = v_kms_lc[:, :, zi_lo:zi_hi]
        v_m = v_mpcs_lc[:, :, zi_lo:zi_hi]
        x_e = 1.0 - xHI

        d_com = cosmo.comoving_distance(z).to_value("Mpc")
        ds = np.abs(np.gradient(d_com))
        a_arr = 1.0 / (1.0 + z)

        x_e_mean = x_e.mean(axis=(0, 1))
        dtau = tau_prefactor * x_e_mean * (1.0 + z) ** 2 * ds
        tau_arr = np.cumsum(dtau)
        e_tau = np.exp(-tau_arr)

        v_over_c = v_m / c_mpc_s
        integrand = (tau_prefactor * delta * x_e * v_over_c
                     * (1.0 / a_arr**2)[None, None, :]
                     * e_tau[None, None, :]
                     * ds[None, None, :])
        kSZ_map = -np.sum(integrand, axis=2)
        xe2_map = np.sum(x_e**2 * ds[None, None, :], axis=2)
        v2_map = np.sum(v_k**2 * ds[None, None, :], axis=2)
        vproj_map = np.sum(v_k * ds[None, None, :], axis=2)
        vproj2_map = vproj_map**2

        if seed in tracer_data:
            z_nodes_t = tracer_data[seed]["z_nodes"]
            zi_lo_t = int(np.searchsorted(z_nodes_t, z_lo))
            zi_hi_t = int(np.searchsorted(z_nodes_t, z_hi))
            halo_proj = tracer_data[seed]["halo_count_lc"][:, :, zi_lo_t:zi_hi_t].sum(axis=2)
            lae_proj = tracer_data[seed]["lae_count_lc"][:, :, zi_lo_t:zi_hi_t].sum(axis=2)
            lbg_proj = tracer_data[seed]["lbg_count_lc"][:, :, zi_lo_t:zi_hi_t].sum(axis=2)
            halo_map = make_overdensity(halo_proj)
            lae_map = make_overdensity(lae_proj)
            lbg_map = make_overdensity(lbg_proj)
        else:
            n = cfg.box.hii_dim
            halo_map = lae_map = lbg_map = np.zeros((n, n))

        out["kSZ"][seed] = kSZ_map.astype(np.float32)
        out["kSZ2"][seed] = (kSZ_map**2).astype(np.float32)
        out["xe2"][seed] = xe2_map.astype(np.float32)
        out["v2"][seed] = v2_map.astype(np.float32)
        out["v_proj"][seed] = vproj_map.astype(np.float32)
        out["v_proj2"][seed] = vproj2_map.astype(np.float32)
        out["halo"][seed] = halo_map.astype(np.float32)
        out["lae"][seed] = lae_map.astype(np.float32)
        out["lbg"][seed] = lbg_map.astype(np.float32)

    return out
