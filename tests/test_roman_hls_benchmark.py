"""
Tests for snr/roman_hls_benchmark.py -- run against synthetic density
fields to confirm the Eq. 6-7 (La Plante+2022) implementation is actually
correct: window normalization, bias curve anchor points, and end-to-end
cross-power against a filtered-kSZ2-shaped map.

Run with:
    pytest tests/test_roman_hls_benchmark.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.snr.roman_hls_benchmark import (
    XHI_MAX_PATCHY,
    XHI_MIN_PATCHY,
    Z_FLOOR_REIONIZATION,
    build_bias_weighted_galaxy_field,
    bluetides_bias_gz,
    chi_eff_power_weighted,
    clamp_window_to_patchy_regime,
    compute_bias_weighted_cross_power,
    compute_volume_averaged_xHI,
)
from ksz_lae_xcorr.utils.config import Config

NGRID = 10
NPIX = 30
Z_MIN, Z_MAX = 5.0, 12.0


def _cfg():
    return Config({
        "box": Config({"box_len_mpc": 40.0, "hii_dim": NGRID, "z_min": Z_MIN, "z_max": Z_MAX}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 6}),
    })


def test_bias_anchor_points_match_paper():
    """Waters+2016 Sec 5/Fig 14: bg(z) = 2.1(1+z) - 5.3, calibrated to
    bg=13.4 at z=8. Cross-check against La Plante+2022's own quoted
    anchors (bg~9 at z=6, bg~20 at z=12) -- both land within the ~9/~20
    the paper states, confirming this is the same curve they cite."""
    assert bluetides_bias_gz(8.0) == pytest.approx(13.4, abs=0.5)  # formula vs quoted point, within their +/-1.8
    assert bluetides_bias_gz(6.0) == pytest.approx(9.4)
    assert bluetides_bias_gz(12.0) == pytest.approx(22.0)


def test_bias_clipped_below_z6():
    """Waters+2016 calibrate this fit for the WFIRST HLS focus (z=8-15);
    below z=6 it's extrapolating past where it was ever fit, so this
    repo clips flat there instead of silently extrapolating."""
    assert bluetides_bias_gz(4.0) == bluetides_bias_gz(6.0)
    assert bluetides_bias_gz(6.0) == pytest.approx(9.4)


def test_bias_clipped_above_z12():
    """Neither Waters+2016 nor La Plante+2022 discuss this bias beyond
    z~12 -- left unclipped, the linear formula runs away unboundedly
    (this was a real bug: it produced an unphysical 'peak' at z~17-18,
    x_HI~1.0 in scripts/13's D_ell-vs-z sweep before this fix)."""
    assert bluetides_bias_gz(15.0) == bluetides_bias_gz(12.0)
    assert bluetides_bias_gz(20.0) == bluetides_bias_gz(12.0)
    assert bluetides_bias_gz(12.0) == pytest.approx(22.0)


def test_bias_monotonic_increasing_in_range():
    z = np.linspace(6, 12, 20)
    bg = bluetides_bias_gz(z)
    assert np.all(np.diff(bg) >= 0)


def _fake_field_data(seed=0):
    rng = np.random.default_rng(seed)
    z_lc = np.linspace(Z_MIN, Z_MAX, NPIX)
    density_lc = 1.0 + rng.normal(0, 0.3, size=(NGRID, NGRID, NPIX))
    xHI_lc = np.clip(0.05 + 0.9 * (z_lc - Z_MIN) / (Z_MAX - Z_MIN), 0.02, 0.98)
    xHI_lc = np.broadcast_to(xHI_lc[None, None, :], (NGRID, NGRID, NPIX)).copy()
    velocity_lc = rng.normal(0, 1e-3, size=(NGRID, NGRID, NPIX))  # Mpc/s, needed by
                                                                   # chi_eff_power_weighted
                                                                   # (via compute_ksz_slices)
    return {"z_lc": z_lc, "density_lc": density_lc, "xHI_lc": xHI_lc, "velocity_lc": velocity_lc}


def test_uniform_bias_and_window_reduces_to_plain_mean():
    """With bias_fn=1 (constant) the Eq 6-7 field is just the top-hat-
    windowed MEAN of delta_m over the window -- check this reduces
    correctly to a sanity-checkable baseline before trusting the
    bias-weighted version."""
    fd = _fake_field_data()
    z0, dz = 8.0, 2.0
    delta_g = build_bias_weighted_galaxy_field(_cfg(), fd, z0, dz, bias_fn=lambda z: np.ones_like(z))

    z_lc = fd["z_lc"]
    mask = (z_lc >= z0 - dz / 2) & (z_lc < z0 + dz / 2)
    expected = np.mean(fd["density_lc"][:, :, mask] - 1.0, axis=2)
    np.testing.assert_allclose(delta_g, expected, rtol=1e-6)


def test_bias_weighting_actually_changes_result():
    """The whole point of this module: a redshift-dependent bias must
    give a DIFFERENT field than uniform bias, given a density field that
    varies with z (otherwise bias_fn is being ignored)."""
    fd = _fake_field_data(seed=1)
    z0, dz = 9.0, 4.0  # wide window so bias varies noticeably across it

    delta_g_uniform = build_bias_weighted_galaxy_field(_cfg(), fd, z0, dz, bias_fn=lambda z: np.ones_like(z))
    delta_g_biased = build_bias_weighted_galaxy_field(_cfg(), fd, z0, dz, bias_fn=bluetides_bias_gz)

    assert not np.allclose(delta_g_uniform, delta_g_biased)


def test_empty_window_raises():
    fd = _fake_field_data()
    with pytest.raises(ValueError, match="patchy regime|no LOS pixels"):
        build_bias_weighted_galaxy_field(_cfg(), fd, z0=100.0, dz=0.1)


def test_compute_bias_weighted_cross_power_runs_end_to_end():
    cfg = _cfg()
    kg = KGrid(cfg)
    fd = _fake_field_data(seed=2)
    rng = np.random.default_rng(3)
    filtered_kSZ2 = rng.normal(0, 1, size=(NGRID, NGRID)) ** 2

    result = compute_bias_weighted_cross_power(cfg, kg, filtered_kSZ2, fd, z0=9.0, dz=1.0)

    assert result["ell"].shape == result["D_ell"].shape
    assert len(result["ell"]) > 0
    assert np.all(np.isfinite(result["D_ell"]))
    assert result["z0"] == 9.0 and result["dz"] == 1.0


def test_compute_volume_averaged_xHI_matches_direct_mean():
    fd = _fake_field_data(seed=4)
    z0, dz = 8.0, 2.0
    x_hi = compute_volume_averaged_xHI(fd, z0, dz)

    z_lc = fd["z_lc"]
    mask = (z_lc >= z0 - dz / 2) & (z_lc < z0 + dz / 2)
    expected = np.mean(fd["xHI_lc"][:, :, mask])
    assert x_hi == pytest.approx(expected)


def test_compute_volume_averaged_xHI_empty_window_raises():
    fd = _fake_field_data()
    with pytest.raises(ValueError, match="patchy regime|no LOS pixels"):
        compute_volume_averaged_xHI(fd, z0=100.0, dz=0.1)


# --- clamp_window_to_patchy_regime ---

def test_clamp_floors_at_z6():
    """La Plante+2022 Fig 9: window must never dip below z=6, even if
    the requested (z0, dz) would otherwise include lower z."""
    z_lc = np.linspace(4.0, 12.0, 40)
    xHI_lc = np.broadcast_to(np.full_like(z_lc, 0.3), (5, 5, 40)).copy()  # uniformly in patchy range
    z_lo, z_hi = clamp_window_to_patchy_regime(z0=6.0, dz=4.0, z_lc=z_lc, xHI_lc=xHI_lc)
    assert z_lo >= Z_FLOOR_REIONIZATION - 1e-9
    assert z_lo == pytest.approx(Z_FLOOR_REIONIZATION, abs=0.3)  # snaps to a real grid point near 6


def test_clamp_excludes_fully_neutral_high_z():
    """
    THE regression test: reproduces the actual bug found in scripts/13
    (peak signal at z~17-18, x_HI~1.00, fully neutral -- physically
    backwards). A window reaching into fully-neutral territory (x_HI > 
    XHI_MAX_PATCHY) must have that part excluded.
    """
    z_lc = np.linspace(6.0, 20.0, 60)
    # Ramps from ~0.1 (ionized) to ~0.999 by z=18, then PLATEAUS at fully
    # neutral (x_HI=1.0) for z>=18 -- matching the real bug's shape (a
    # genuine fully-neutral plateau, not an asymptote only reached at the
    # very last grid point).
    xHI_1d = np.where(
        z_lc < 18.0,
        np.clip(0.1 + 0.9 * (z_lc - 6.0) / (18.0 - 6.0), 0.05, 0.999),
        1.0,
    )
    xHI_lc = np.broadcast_to(xHI_1d, (4, 4, 60)).copy()

    z_lo, z_hi = clamp_window_to_patchy_regime(z0=17.0, dz=6.0, z_lc=z_lc, xHI_lc=xHI_lc)
    # The clamped window's upper edge must NOT reach the fully-neutral
    # plateau (z>=18) -- i.e. z_hi must stay below 18.
    assert z_hi < 18.0, f"Clamped window still reaches into the fully-neutral plateau (z_hi={z_hi})"


def test_clamp_raises_when_nothing_survives():
    z_lc = np.linspace(6.0, 12.0, 30)
    xHI_lc = np.broadcast_to(np.full_like(z_lc, 0.3), (4, 4, 30)).copy()
    with pytest.raises(ValueError, match="patchy regime"):
        clamp_window_to_patchy_regime(z0=100.0, dz=0.1, z_lc=z_lc, xHI_lc=xHI_lc)


def test_clamp_deterministic_same_inputs_same_output():
    """Calling twice with identical inputs (as build_bias_weighted_galaxy_field
    and compute_volume_averaged_xHI each do independently) must give
    identical bounds, so the two stay consistent without threading state."""
    z_lc = np.linspace(6.0, 14.0, 40)
    xHI_lc = np.broadcast_to(np.clip(0.1 + 0.05 * (z_lc - 6.0), 0.05, 0.95), (4, 4, 40)).copy()
    r1 = clamp_window_to_patchy_regime(9.0, 2.0, z_lc, xHI_lc)
    r2 = clamp_window_to_patchy_regime(9.0, 2.0, z_lc, xHI_lc)
    assert r1 == r2


# --- chi_eff_power_weighted ---

def test_chi_eff_lies_within_window_bounds():
    """A power-weighted mean of chi(z) over a window must itself lie
    between chi(z_lo) and chi(z_hi) -- a basic sanity bound any correctly
    computed weighted average satisfies."""
    cfg = _cfg()
    fd = _fake_field_data(seed=7)

    z_lo, z_hi = 7.0, 9.0
    chi_eff = chi_eff_power_weighted(cfg, fd, z_lo, z_hi)

    from ksz_lae_xcorr.utils.cosmology import get_cosmology
    cosmo = get_cosmology(cfg)
    chi_lo = cosmo.comoving_distance(z_lo).to_value("Mpc")
    chi_hi = cosmo.comoving_distance(z_hi).to_value("Mpc")
    assert min(chi_lo, chi_hi) <= chi_eff <= max(chi_lo, chi_hi)


def test_chi_eff_raises_on_too_narrow_window():
    cfg = _cfg()
    fd = _fake_field_data(seed=8)
    with pytest.raises(ValueError, match="LOS pixels|Zero total weight"):
        chi_eff_power_weighted(cfg, fd, z_lo=100.0, z_hi=100.001)


def test_cross_power_use_chi_eff_actually_changes_result():
    """The whole point of wiring this in: use_chi_eff=True must give a
    DIFFERENT chi_c (hence different ell grid) than use_chi_eff=False --
    if this ever passes with identical output, the flag is being ignored."""
    cfg = _cfg()
    kg = KGrid(cfg)
    fd = _fake_field_data(seed=9)
    rng = np.random.default_rng(10)
    filtered_kSZ2 = rng.normal(0, 1, size=(NGRID, NGRID)) ** 2

    r_eff = compute_bias_weighted_cross_power(cfg, kg, filtered_kSZ2, fd, z0=9.0, dz=1.0,
                                               use_chi_eff=True)
    r_naive = compute_bias_weighted_cross_power(cfg, kg, filtered_kSZ2, fd, z0=9.0, dz=1.0,
                                                 use_chi_eff=False)
    assert r_eff["chi_eff_used"] is True
    assert r_naive["chi_eff_used"] is False
    assert r_eff["chi_c"] != pytest.approx(r_naive["chi_c"])
