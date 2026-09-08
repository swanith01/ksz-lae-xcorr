"""
snr/snr_forecast.py
=====================
Filters the kSZ maps per experiment, computes the filtered kSZ^2 x tracer
cross-power per redshift bin, and forecasts S/N via the discrete-ell-sum
estimator (La Plante+2022 Eq. 13). Direct refactor of notebook Cells 9b-9c.

Tracer-generic as of 2026-09-08 (was LAE-only): every function takes a
tracer_key defaulting to 'lae_count_lc', so scripts/05's production
forecast is byte-for-byte unchanged. Pass tracer_key='lbg_count_lc' for
the Stage 1 La Plante+2022 benchmark (Roman HLS Lyman-break galaxies --
that paper's actual tracer, not LAEs) -- see
scripts/10_stage1_lbg_benchmark.py.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.correlation.power_spectra import (
    KGrid, cross_power_2d, make_ell, make_overdensity, to_Cell, to_Dell,
)
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def build_filtered_kSZ2_maps(cfg, kg: KGrid, kSZ_maps: dict, filt: dict, seeds: list[int]) -> dict:
    """filtered_kSZ2[exp_name][seed] = (filtered kSZ)^2 real-space map."""
    cosmo = get_cosmology(cfg)
    z_ref = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    chi_ref = cosmo.comoving_distance(z_ref).to_value("Mpc")

    n = kg.n_side
    dk = 2 * np.pi / kg.box_len
    kx = np.fft.fftfreq(n) * n * dk
    KX, KY = np.meshgrid(kx, kx)
    k2d = np.sqrt(KX**2 + KY**2)
    ell2d_ref = make_ell(k2d, chi_ref)
    ell2d_ref[n // 2, n // 2] = 1e-6

    out = {}
    for name in cfg.snr.experiments:
        out[name] = {}
        f_2d = interp1d(filt["ell_grid"], filt["fl"][name], bounds_error=False,
                         fill_value=0.0)(ell2d_ref)
        for seed in seeds:
            if seed not in kSZ_maps:
                continue
            kSZ_map = kSZ_maps[seed].astype(np.float64)
            kSZ_k = np.fft.fft2(kSZ_map)
            kSZ_filt = np.real(np.fft.ifft2(kSZ_k * f_2d))
            out[name][seed] = kSZ_filt**2
    return out


def compute_lae_auto_power(cfg, kg: KGrid, tracer_data: dict, seeds: list[int],
                            z_edges, z_cents, ell_grid, tracer_key: str = "lae_count_lc") -> dict:
    """
    C_ell^(delta_g delta_g) per z-bin, median over seeds, interpolated onto ell_grid.

    tracer_key selects which stitched count field to use (default
    'lae_count_lc', preserving every existing call site's behavior
    unchanged). Pass 'lbg_count_lc' for the Stage 1 La Plante+2022
    benchmark (Roman HLS Lyman-break galaxies, not LAEs) -- see
    scripts/10_stage1_lbg_benchmark.py. Name kept as compute_lae_auto_power
    rather than renamed, so the many existing positional-arg call sites
    (scripts/05, scripts/08) don't need touching.
    """
    cosmo = get_cosmology(cfg)
    z_nodes_ref = tracer_data[seeds[0]]["z_nodes"]

    Cl_lae_auto = {}
    for zi, z_c in enumerate(z_cents):
        zi_lo = int(np.searchsorted(z_nodes_ref, z_edges[zi]))
        zi_hi = int(np.searchsorted(z_nodes_ref, z_edges[zi + 1]))
        if zi_hi <= zi_lo:
            continue
        chi_c = cosmo.comoving_distance(z_c).to_value("Mpc")
        ell_c = make_ell(kg.k_centers, chi_c)

        Cl_seeds = []
        for seed in seeds:
            if seed not in tracer_data or tracer_key not in tracer_data[seed]:
                continue
            lae_proj = tracer_data[seed][tracer_key][:, :, zi_lo:zi_hi].sum(axis=2)
            if lae_proj.sum() == 0:
                continue
            delta_g = make_overdensity(lae_proj)
            P, Pe, _ = cross_power_2d(delta_g - delta_g.mean(), delta_g - delta_g.mean(), kg)
            C, _ = to_Cell(P, Pe, chi_c)
            Cl_seeds.append(C)

        if not Cl_seeds:
            continue
        Cl_med = np.nanmedian(np.array(Cl_seeds), axis=0)
        valid = np.isfinite(Cl_med) & (ell_c > 10)
        if valid.sum() < 3:
            continue
        Cl_lae_auto[z_c] = interp1d(ell_c[valid], Cl_med[valid], bounds_error=False,
                                     fill_value="extrapolate")(ell_grid)
    return Cl_lae_auto


def compute_filtered_signal(cfg, kg: KGrid, filtered_kSZ2: dict, tracer_data: dict,
                             seeds: list[int], z_edges, z_cents, ell_grid,
                             tracer_key: str = "lae_count_lc") -> dict:
    """
    C_ell^(T_f^2 x tracer) per experiment per z-bin (muK^2), median over seeds.

    tracer_key: see compute_lae_auto_power above -- same default, same
    Stage-1 LBG override.
    """
    cosmo = get_cosmology(cfg)
    z_nodes_ref = tracer_data[seeds[0]]["z_nodes"]

    Cl_signal = {name: {} for name in cfg.snr.experiments}
    for name in cfg.snr.experiments:
        for zi, z_c in enumerate(z_cents):
            zi_lo = int(np.searchsorted(z_nodes_ref, z_edges[zi]))
            zi_hi = int(np.searchsorted(z_nodes_ref, z_edges[zi + 1]))
            if zi_hi <= zi_lo:
                continue
            chi_c = cosmo.comoving_distance(z_c).to_value("Mpc")
            ell_c = make_ell(kg.k_centers, chi_c)

            D_seeds = []
            for seed in seeds:
                if seed not in filtered_kSZ2[name] or seed not in tracer_data:
                    continue
                if tracer_key not in tracer_data[seed]:
                    continue
                lae_proj = tracer_data[seed][tracer_key][:, :, zi_lo:zi_hi].sum(axis=2)
                if lae_proj.sum() == 0:
                    continue
                delta_lae = make_overdensity(lae_proj)
                sig = filtered_kSZ2[name][seed].astype(np.float64)
                sig = sig - sig.mean()
                P, Pe, _ = cross_power_2d(sig, delta_lae - delta_lae.mean(), kg)
                C, _ = to_Cell(P, Pe, chi_c)
                D, _ = to_Dell(ell_c, C, np.zeros_like(C), T_CMB_uK=constants.T_CMB_UK)
                D_seeds.append(D)

            if not D_seeds:
                continue
            D_med = np.nanmedian(np.array(D_seeds), axis=0)
            Cl_med = D_med * 2 * np.pi / (ell_c * (ell_c + 1))
            valid = np.isfinite(Cl_med) & (ell_c > 10)
            if valid.sum() < 3:
                continue
            Cl_signal[name][z_c] = interp1d(ell_c[valid], Cl_med[valid], bounds_error=False,
                                             fill_value=0.0)(ell_grid)
    return Cl_signal


def compute_snr_vs_z(cfg, filt: dict, Cl_signal: dict, Cl_lae_auto: dict, z_cents) -> dict:
    """(S/N)^2 = f_sky * sum_ell (2ell+1) Cs^2 / (CT2*Cg + Cs^2)  -- Eq. 13, per z-bin."""
    ell_grid = filt["ell_grid"]
    ell_int = np.arange(cfg.snr.ell_min, cfg.snr.ell_max + 1)
    f_sky = cfg.snr.f_sky

    SN_results = {}
    for name in cfg.snr.experiments:
        SN_results[name] = {}
        CT2_val = filt["Cl_T2T2_f"][name][0]

        for z_c in z_cents:
            if z_c not in Cl_signal[name] or z_c not in Cl_lae_auto:
                continue
            Cs_int = interp1d(ell_grid, Cl_signal[name][z_c], bounds_error=False,
                               fill_value=0.0)(ell_int)
            Cg_int = interp1d(ell_grid, Cl_lae_auto[z_c], bounds_error=False,
                               fill_value=0.0)(ell_int)
            num = Cs_int**2
            denom = CT2_val * np.abs(Cg_int) + Cs_int**2
            good = denom > 0
            SN2 = f_sky * np.sum((2 * ell_int[good] + 1) * num[good] / denom[good])
            SN_results[name][z_c] = float(np.sqrt(max(SN2, 0.0)))
    return SN_results


def run_snr_pipeline(cfg, kg: KGrid, kSZ_maps: dict, tracer_data: dict, seeds: list[int],
                      auto_results_ksz: dict, tracer_key: str = "lae_count_lc") -> dict:
    """
    Full SNR pipeline: CMB filter -> filtered kSZ^2 maps -> filtered signal
    x tracer auto -> S/N vs z, per experiment.

    tracer_key defaults to 'lae_count_lc' (unchanged behavior for
    scripts/05's production LAE forecast). Pass 'lbg_count_lc' for the
    Stage 1 La Plante+2022 benchmark (Roman HLS LBGs) -- see
    scripts/10_stage1_lbg_benchmark.py.
    """
    from ksz_lae_xcorr.snr.cmb_filter import build_cmb_filter_ingredients

    dz = cfg.correlation.dz_tracer_bin
    z_edges = np.arange(cfg.box.z_min, cfg.box.z_max + dz, dz)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])

    filt = build_cmb_filter_ingredients(cfg, kg, auto_results_ksz)
    filtered_kSZ2 = build_filtered_kSZ2_maps(cfg, kg, kSZ_maps, filt, seeds)
    Cl_lae_auto = compute_lae_auto_power(cfg, kg, tracer_data, seeds, z_edges, z_cents,
                                          filt["ell_grid"], tracer_key=tracer_key)
    Cl_signal = compute_filtered_signal(cfg, kg, filtered_kSZ2, tracer_data, seeds, z_edges,
                                         z_cents, filt["ell_grid"], tracer_key=tracer_key)
    SN_results = compute_snr_vs_z(cfg, filt, Cl_signal, Cl_lae_auto, z_cents)

    return {"filt": filt, "Cl_lae_auto": Cl_lae_auto, "Cl_signal": Cl_signal,
            "SN_results": SN_results, "z_cents": z_cents}
