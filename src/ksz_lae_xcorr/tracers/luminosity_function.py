"""
tracers/luminosity_function.py
=================================
The real LAE luminosity function: number density per unit log-luminosity,
Phi(L) dlogL, from the RAW per-object lya_lum_obs values -- not a grid
cell average, not derived from the count-grid tracer this repo otherwise
uses everywhere else. This is a genuinely different, independent check on
the LAE catalogue: does its actual luminosity distribution look like a
real observed LAE population, not just "are there the right number of
them in the right place" (which the count-grid tracer already checks).

Also has the analogous LBG UV luminosity function (added 2026-09-30, per
Girish's "can we have tracer LFs" -- LBG is the other real catalogue-
backed tracer, with per-object MUV_lbg values already loaded by
lightcone.stitch.Stitcher.load_lbg_grid for the count-grid tracer). UV
LFs are conventionally reported per unit ABSOLUTE MAGNITUDE (Phi(M_UV)
dM_UV, mag^-1 Mpc^-3), not per unit log-luminosity -- a different binning
convention from the LAE Lya LF above, kept separate rather than forced
into the same log-L machinery.

Reuses Stitcher only for its path/attribute plumbing (root_lae/root_lbg,
box_len, external_catalogue_filename convention) -- does NOT go through
the full 3D gridding machinery in lightcone/value_fields.py, since a
luminosity function only needs the raw per-object values, not their
positions.
"""

from __future__ import annotations

import os

import numpy as np

from ksz_lae_xcorr.utils.external_catalogue import external_catalogue_filename


def load_raw_lae_luminosities(cfg, seed: int, z: float) -> np.ndarray:
    """
    The raw, per-object Lya luminosity values (lya_lum_obs) at one
    (seed, z) -- NOT gridded, NOT averaged, just the actual catalogue
    values as delivered. Returns an empty array (not an error) if the
    file isn't there yet, matching this repo's existing convention for
    external catalogue files pending handover (see
    lightcone/stitch.py's load_lae_grid).
    """
    root_lae = cfg.paths.lae_catalogue_root
    fname = external_catalogue_filename("lya_lum_obs", z, cfg, seed)
    valpath = os.path.join(root_lae, "lya_lum_obs", fname)
    if not os.path.exists(valpath):
        return np.array([])
    return np.asarray(np.load(valpath, mmap_mode="r"))


def compute_luminosity_function(cfg, seeds: list[int], z: float, log_l_min: float = 41.0,
                                 log_l_max: float = 44.0, n_bins: int = 15) -> dict:
    """
    Phi(L) dlogL, averaged over seeds, at one redshift -- the standard
    luminosity-function convention: number density per unit log10(L),
    normalized by this box's comoving volume (each coeval snapshot IS a
    fixed-z comoving volume already, box_len_mpc^3 -- no shell/lightcone
    integration needed, unlike a real flux-limited survey).

    Returns {'log_l_centers', 'phi', 'phi_err' (seed scatter), 'n_seeds_with_data',
    'n_objects_total'}. If NO seed has data, phi is all-NaN, not zero --
    "no data" and "confirmed zero" are different things and shouldn't look
    the same on a log-scale plot.
    """
    log_l_edges = np.linspace(log_l_min, log_l_max, n_bins + 1)
    log_l_centers = 0.5 * (log_l_edges[:-1] + log_l_edges[1:])
    dlogL = log_l_edges[1] - log_l_edges[0]
    volume_mpc3 = cfg.box.box_len_mpc ** 3

    per_seed_phi = []
    n_objects_total = 0
    n_seeds_with_data = 0

    for seed in seeds:
        lum = load_raw_lae_luminosities(cfg, seed, z)
        if len(lum) == 0:
            continue
        n_seeds_with_data += 1
        n_objects_total += len(lum)
        log_lum = np.log10(lum[lum > 0])
        counts, _ = np.histogram(log_lum, bins=log_l_edges)
        phi = counts / (volume_mpc3 * dlogL)
        per_seed_phi.append(phi)

    if not per_seed_phi:
        return {"log_l_centers": log_l_centers, "phi": np.full(n_bins, np.nan),
                "phi_err": np.full(n_bins, np.nan), "n_seeds_with_data": 0, "n_objects_total": 0}

    stacked = np.array(per_seed_phi)
    phi_mean = stacked.mean(axis=0)
    phi_err = stacked.std(axis=0) if len(per_seed_phi) > 1 else np.zeros(n_bins)
    return {"log_l_centers": log_l_centers, "phi": phi_mean, "phi_err": phi_err,
            "n_seeds_with_data": n_seeds_with_data, "n_objects_total": n_objects_total}


def load_raw_lbg_muv(cfg, seed: int, z: float) -> np.ndarray:
    """
    The raw, per-object UV absolute magnitude values (MUV_lbg) at one
    (seed, z) -- the FULL catalogue, same file lightcone.stitch.Stitcher.
    load_lbg_grid loads before it applies cfg.tracers.lbg_muv_cut to
    build the count-grid tracer. Deliberately returns the UNCUT
    distribution here (a luminosity function should show the actual
    population, not just the subset already used for cross-correlation --
    lbg_muv_cut is a downstream analysis choice, not a property of the
    catalogue itself), same "raw, not gridded/cut" spirit as
    load_raw_lae_luminosities above.

    Returns an empty array (not an error) if the file isn't there yet,
    matching load_raw_lae_luminosities's convention for external
    catalogue files pending handover.
    """
    root_lbg = cfg.paths.lbg_catalogue_root
    fname = external_catalogue_filename("MUV_lbg", z, cfg, seed)
    valpath = os.path.join(root_lbg, "MUV_lbg", fname)
    if not os.path.exists(valpath):
        return np.array([])
    return np.asarray(np.load(valpath, mmap_mode="r"))


def compute_uv_luminosity_function(cfg, seeds: list[int], z: float, m_uv_min: float = -24.0,
                                    m_uv_max: float = -14.0, n_bins: int = 15) -> dict:
    """
    Phi(M_UV) dM_UV, averaged over seeds, at one redshift -- the standard
    UV-LF convention (number density PER UNIT ABSOLUTE MAGNITUDE,
    Mpc^-3 mag^-1), NOT per unit log-luminosity -- deliberately a
    separate binning convention from compute_luminosity_function's Lya
    LF above, matching how UV LFs are reported in the literature (e.g.
    Bouwens et al., Harikane et al.) so any published comparison can be
    overlaid without a unit conversion.

    m_uv_min/m_uv_max: bin edges in absolute magnitude, min < max in the
    usual sense (m_uv_min more negative = brighter). Default range
    (-24, -14) comfortably brackets cfg.tracers.lbg_muv_cut (-17.0) with
    room on both sides.

    Returns {'m_uv_centers', 'phi', 'phi_err' (seed scatter),
    'n_seeds_with_data', 'n_objects_total'}. phi is all-NaN (not zero)
    when NO seed has any data, same "no measurement" vs "confirmed zero"
    distinction as compute_luminosity_function.
    """
    m_uv_edges = np.linspace(m_uv_min, m_uv_max, n_bins + 1)
    m_uv_centers = 0.5 * (m_uv_edges[:-1] + m_uv_edges[1:])
    dM = m_uv_edges[1] - m_uv_edges[0]
    volume_mpc3 = cfg.box.box_len_mpc ** 3

    per_seed_phi = []
    n_objects_total = 0
    n_seeds_with_data = 0

    for seed in seeds:
        muv = load_raw_lbg_muv(cfg, seed, z)
        if len(muv) == 0:
            continue
        n_seeds_with_data += 1
        n_objects_total += len(muv)
        counts, _ = np.histogram(muv, bins=m_uv_edges)
        phi = counts / (volume_mpc3 * dM)
        per_seed_phi.append(phi)

    if not per_seed_phi:
        return {"m_uv_centers": m_uv_centers, "phi": np.full(n_bins, np.nan),
                "phi_err": np.full(n_bins, np.nan), "n_seeds_with_data": 0, "n_objects_total": 0}

    stacked = np.array(per_seed_phi)
    phi_mean = stacked.mean(axis=0)
    phi_err = stacked.std(axis=0) if len(per_seed_phi) > 1 else np.zeros(n_bins)
    return {"m_uv_centers": m_uv_centers, "phi": phi_mean, "phi_err": phi_err,
            "n_seeds_with_data": n_seeds_with_data, "n_objects_total": n_objects_total}
