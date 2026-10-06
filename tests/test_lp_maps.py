"""
Tests for correlation/lp_maps.py (map-based La Plante+22 estimator) on synthetic
fields.  No CAMB, no cluster data.  The decisive one is the known-amplitude
cross: for kmap = n (1 + eps s), delta_g = s with n white and s Gaussian,
E[kmap^2 x s] = 2 eps P_s, so the estimator must return C^{kSZ2,g} = 2 eps C^{ss}.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.coherence_decomposition import compute_ksz_slices
from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d
from ksz_lae_xcorr.snr.roman_hls_benchmark import (
    build_bias_weighted_galaxy_field,
    chi_eff_power_weighted,
)
from ksz_lae_xcorr.snr.snr_forecast import build_filtered_kSZ2_maps
from ksz_lae_xcorr.utils.config import Config

N = 32
BOX = 64.0


def _cfg():
    return Config({
        "box": Config({"box_len_mpc": BOX, "hii_dim": N, "dim": 2 * N, "z_min": 5.0, "z_max": 9.0, "seeds": [1]}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 8}),
        "snr": Config({"ell_min": 100, "ell_max": 10000, "n_ell": 60, "camb_lmax": 12000, "f_sky": 0.053,
                       "experiments": Config({"SO": Config({"delta_n_uk_arcmin": 10.0, "theta_fwhm_arcmin": 1.4})})}),
    })


def _fd(seed=0, nz=80):
    rng = np.random.default_rng(seed)
    z = np.linspace(5.0, 9.0, nz)
    xhi = np.clip(np.linspace(0.0, 1.0, nz)[None, None, :] + 0.05 * rng.normal(size=(N, N, nz)), 0, 1)
    return {
        "z_lc": z, "xHI_lc": xhi,
        "density_lc": 1.0 + 0.3 * rng.normal(size=(N, N, nz)),
        "velocity_lc": 1e-16 * rng.normal(size=(N, N, nz)),
        "ne_scale": 0.82, "tau0": 0.03,
    }


def test_window_bounds_floor_ceiling_and_empty():
    assert lp_maps.lp_window_bounds(6.5, 1.0) == (6.0, 7.0)
    assert lp_maps.lp_window_bounds(6.2, 1.0) == (6.0, 6.7)          # floored at z=6
    assert lp_maps.lp_window_bounds(12.5, 2.0, z_ceiling=13.0) == (11.5, 13.0)
    with pytest.raises(ValueError):
        lp_maps.lp_window_bounds(5.0, 1.0)


def test_galaxy_map_normalisation_and_agreement_with_existing_builder():
    cfg, fd = _cfg(), _fd()
    # INT W dz = 1: constant delta_m = c with b = 1  ->  delta_g = c
    fd_c = dict(fd, density_lc=np.full_like(fd["density_lc"], 1.7))
    zi = (int(np.searchsorted(fd["z_lc"], 6.5)), int(np.searchsorted(fd["z_lc"], 7.5)))
    g = lp_maps.build_lp_galaxy_map(fd_c, *zi, bias_fn=lambda z: np.ones_like(z))
    np.testing.assert_allclose(g, 0.7, rtol=1e-12)
    # and it IS the existing builder for an interior (unclamped) window
    ref = build_bias_weighted_galaxy_field(cfg, fd, 7.0, 1.0, clamp_to_patchy=False)
    np.testing.assert_allclose(lp_maps.build_lp_galaxy_map(fd, *zi), ref, rtol=1e-12)


def test_chi_eff_and_kSZ_map_from_slices_match_existing_code():
    cfg, fd = _cfg(), _fd()
    theta, chi, z = compute_ksz_slices(cfg, fd)
    a = lp_maps.chi_eff_from_slices(theta, chi, z, 6.0, 7.0)
    b = chi_eff_power_weighted(cfg, fd, 6.0, 7.0)
    assert a == pytest.approx(b, rel=1e-12)
    np.testing.assert_allclose(lp_maps.ksz_map_from_slices(theta, z, 0.0), theta.sum(axis=2))
    np.testing.assert_allclose(lp_maps.ksz_map_from_slices(theta, z, 6.0), theta[:, :, z >= 6.0].sum(axis=2))


def test_known_amplitude_cross_through_filter_square_and_cross():
    cfg = _cfg()
    kg = KGrid(cfg)
    rng = np.random.default_rng(3)
    eps = 0.3
    # smooth-ish Gaussian s (white noise lightly smoothed in Fourier space)
    k = np.fft.fftfreq(N)
    K2 = k[:, None] ** 2 + k[None, :] ** 2
    ests = []
    for _ in range(40):
        s = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(N, N))) * np.exp(-K2 * 30)))
        s = s / s.std()
        n = rng.normal(size=(N, N))
        kmap = n * (1 + eps * s)
        f2 = lp_maps.filter_and_square(kmap, np.geomspace(1.0, 1e7, 50), np.ones(50), kg, chi_ref=1000.0)
        np.testing.assert_allclose(f2, (kmap - kmap.mean()) ** 2, rtol=1e-6, atol=1e-9)  # identity filter (DC mode is zeroed, as in production)
        P_sg, _, _ = cross_power_2d(f2 - f2.mean(), s - s.mean(), kg)
        P_ss, _, _ = cross_power_2d(s - s.mean(), s - s.mean(), kg)
        ests.append(P_sg / P_ss)
    ratio = np.nanmean(np.array(ests), axis=0)
    # lowest-k bins carry the signal (s is smooth); expect 2*eps (+O(eps^2) from <n^2 s^2> -> 0 cross)
    good = np.isfinite(ratio)
    assert np.nanmedian(ratio[good][:4]) == pytest.approx(2 * eps, rel=0.15)


def test_cross_dell_units_and_interp():
    cfg = _cfg()
    kg = KGrid(cfg)
    rng = np.random.default_rng(5)
    a = rng.normal(size=(N, N))
    c = lp_maps.lp_cross_dell(a, a, kg, chi_eff=8000.0)
    # auto of a map with itself -> positive; n_modes >= 1; ell = k chi
    assert np.all(c["D_ell"][np.isfinite(c["D_ell"])] > 0)
    assert np.all(c["n_modes"] >= 1)
    np.testing.assert_allclose(c["ell"], kg.k_centers * 8000.0)
    ell = np.array([100.0, 1000.0, 10000.0])
    y = np.array([1.0, 2.0, 3.0])
    out = lp_maps.interp_at_ell(ell, y, targets=(50.0, 1000.0, 3162.2776601683795, 20000.0))
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert out[1] == pytest.approx(2.0) and out[2] == pytest.approx(2.5, rel=1e-6)


def test_build_filtered_kSZ2_maps_chi_ref_default_is_legacy():
    cfg = _cfg()
    kg = KGrid(cfg)
    rng = np.random.default_rng(7)
    maps = {1: rng.normal(size=(N, N))}
    ell = np.geomspace(100, 10000, 40)
    filt = {"ell_grid": ell, "fl": {"SO": 1.0 / (1.0 + (ell / 1500.0) ** 2)}}
    from ksz_lae_xcorr.utils.cosmology import get_cosmology
    chi_mid = get_cosmology(cfg).comoving_distance(7.0).to_value("Mpc")
    legacy = build_filtered_kSZ2_maps(cfg, kg, maps, filt, [1])["SO"][1]
    same = build_filtered_kSZ2_maps(cfg, kg, maps, filt, [1], chi_ref=chi_mid)["SO"][1]
    other = build_filtered_kSZ2_maps(cfg, kg, maps, filt, [1], chi_ref=0.5 * chi_mid)["SO"][1]
    np.testing.assert_allclose(legacy, same)
    assert not np.allclose(legacy, other)
    # and lp_maps.filter_and_square is the same operation
    np.testing.assert_allclose(lp_maps.filter_and_square(maps[1], ell, filt["fl"]["SO"], kg, chi_mid), legacy)


def test_end_to_end_products_filter_and_aggregate():
    cfg = _cfg()
    prods = {s: lp_maps.build_lp_products(cfg, _fd(seed=s), z0_grid=(6.5, 7.5, 8.5), dz_list=(1.0,),
                                          kSZ_z_min=6.0) for s in (1, 2, 3)}
    p = prods[1]
    assert p["kSZ_map"].shape == (N, N) and p["kSZ_map"].dtype == np.float32
    assert [w["z0"] for w in p["windows"]] == [6.5, 7.5, 8.5]
    assert all(w["z_lo"] >= 6.0 for w in p["windows"])
    assert all(0.0 <= w["x_hii"] <= 1.0 for w in p["windows"])
    assert p["windows"][0]["x_hii"] > p["windows"][-1]["x_hii"]            # synthetic xHI rises with z

    cl_tt = lambda ell: 1e4 * (ell / 100.0) ** -2.0 * 2 * np.pi / (ell * (ell + 1)) * 100 ** 2 / 10
    res = lp_maps.run_lp_analysis(cfg, prods, cl_tt, "SO", targets=(1500.0, 3000.0, 5000.0))  # box is 64 Mpc: l_f ~ 900
    assert res["D"].shape == (3, 3, 3)
    assert res["mean"].shape == (3, 3) and np.all(np.isfinite(res["mean"]))
    assert np.all(res["n_modes"] >= 1) and list(res["seeds"]) == [1, 2, 3]
    f = res["filter"]
    assert np.all((f["fl"] >= 0) & (f["fl"] <= 1.0 + 1e-12))
    # filter is F*b with F = Ckz/(CTT+Ckz+Clate+N): recompute independently
    den = f["Cl_TT"] + f["Cl_kSZ_reion"] + f["Cl_kSZ_late"] + f["Nl"]
    np.testing.assert_allclose(f["Fl"], f["Cl_kSZ_reion"] / den)
    # no-band rows still computable (bands dict only has ell=300 here)
    bands = {1500: {"z0_lo": np.array([6.0, 9.0]), "lo": np.array([0.0, 0.0]),
                   "z0_hi": np.array([6.0, 9.0]), "hi": np.array([1e3, 1e3])}}
    rows = lp_maps.compare_to_la_plante(res, bands)
    assert all(r["ell"] == 1500.0 for r in rows)


def test_products_do_not_depend_on_unused_lightcone_tail():
    """Windows above the simulated range are skipped, not an error."""
    cfg = _cfg()
    p = lp_maps.build_lp_products(cfg, _fd(), z0_grid=(8.9, 12.0), dz_list=(1.0,))
    assert len(p["windows"]) <= 1


def test_integration_wrapcycle_lightcone_to_lp_products(tmp_path):
    """Synthetic coeval tree -> wrap-cycle lightcone -> LP products -> aggregate."""
    import logging
    from test_ksz_auto_wrapcycle import _cfg as wc_cfg, _write_tree
    from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import build_wrapcycle_field_data, normalisation_keys
    from ksz_lae_xcorr.lightcone.stitch import Stitcher

    for sd in (1, 2):
        _write_tree(tmp_path, seed=sd)
    cfg = wc_cfg(tmp_path)
    cfg.snr = Config({"ell_min": 100, "ell_max": 10000, "n_ell": 60, "camb_lmax": 12000, "f_sky": 0.053,
                      "experiments": Config({"SO": Config({"delta_n_uk_arcmin": 10.0, "theta_fwhm_arcmin": 1.4})})})
    prods = {}
    for sd in (1, 2):
        fd = build_wrapcycle_field_data(cfg, sd, 7 + sd, Stitcher(cfg), logging.getLogger("t_lp"),
                                        norm=normalisation_keys(cfg))
        prods[sd] = lp_maps.build_lp_products(cfg, fd, z0_grid=(6.5, 7.0), dz_list=(1.0,), kSZ_z_min=6.0)
    assert all(len(p["windows"]) >= 1 for p in prods.values())
    cl_tt = lambda ell: 3e3 / (ell * (ell + 1)) * 2 * np.pi
    res = lp_maps.run_lp_analysis(cfg, prods, cl_tt, "SO", targets=(3000.0,))
    assert res["D"].shape[0] == 2 and res["D"].shape[2] == 1
