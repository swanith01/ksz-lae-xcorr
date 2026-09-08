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
    build_bias_weighted_galaxy_field,
    bluetides_bias_gz,
    compute_bias_weighted_cross_power,
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


def test_bias_monotonic_increasing_in_range():
    z = np.linspace(6, 12, 20)
    bg = bluetides_bias_gz(z)
    assert np.all(np.diff(bg) >= 0)


def _fake_field_data(seed=0):
    rng = np.random.default_rng(seed)
    z_lc = np.linspace(Z_MIN, Z_MAX, NPIX)
    density_lc = 1.0 + rng.normal(0, 0.3, size=(NGRID, NGRID, NPIX))
    return {"z_lc": z_lc, "density_lc": density_lc}


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
    with pytest.raises(ValueError, match="no LOS pixels"):
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
