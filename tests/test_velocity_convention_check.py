"""
Tests for correlation/velocity_convention_check.py -- the load-bearing
test builds a velocity field that EXACTLY satisfies the continuity
equation relative to a known density field (both in Fourier space, from
first principles, not from py21cmfast), and confirms the check recovers
a correction factor of 1.0 (i.e. correctly identifies that no correction
is needed when none actually is).

Run with:
    pytest tests/test_velocity_convention_check.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.velocity_convention_check import (
    check_velocity_convention,
    radial_power_spectrum_3d,
    theoretical_velocity_over_density_ratio,
)
from ksz_lae_xcorr.lightcone.stitch import _growth_rate_linder
from ksz_lae_xcorr.utils.config import Config
from ksz_lae_xcorr.utils.cosmology import get_cosmology

N = 48
BOX_LEN = 100.0
Z_TEST = 9.0


def _cfg():
    return Config({"cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665})})


def _build_consistent_density_and_velocity(cfg, z, seed=0):
    """
    Build delta(x) as random Gaussian-ish noise in Fourier space, then
    construct v_los(k) = i*(k_los/k^2)*f*a*H*delta(k) EXACTLY (the
    continuity equation, in km/s), inverse-FFT both back to real space.
    This is a synthetic field constructed to exactly satisfy the physics
    being tested -- not real cosmological structure, just a controlled
    check that the ESTIMATOR correctly recovers factor=1 when the input
    truly has no unit error.
    """
    rng = np.random.default_rng(seed)
    delta_k = (rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N)))

    kfreq = np.fft.fftfreq(N, d=BOX_LEN / N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmag[0, 0, 0] = 1.0  # avoid div-by-zero at k=0; DC mode set to 0 power below
    delta_k[0, 0, 0] = 0.0

    # Give delta_k a k^-2-ish falloff so power is concentrated at low k
    # (more realistic-shaped spectrum, not that it matters for this test).
    delta_k = delta_k / kmag

    cosmo = get_cosmology(cfg)
    a = 1.0 / (1.0 + z)
    H = cosmo.H(z).to_value("km/s/Mpc")
    f = _growth_rate_linder(cosmo, z)

    k_los = KZ  # line-of-sight = z-axis, arbitrary choice
    v_k_kms = 1j * (k_los / kmag**2) * f * a * H * delta_k
    v_k_kms[0, 0, 0] = 0.0

    delta_x = np.real(np.fft.ifftn(delta_k))
    v_x_kms = np.real(np.fft.ifftn(v_k_kms))

    density_1plus_delta = 1.0 + delta_x * 0.1  # keep delta small/physical
    v_x_kms = v_x_kms * 0.1  # scale consistently with the density scaling above

    return density_1plus_delta, v_x_kms


def test_radial_power_spectrum_shape_and_positivity():
    rng = np.random.default_rng(1)
    field = rng.normal(0, 1, size=(N, N, N))
    result = radial_power_spectrum_3d(field, BOX_LEN, n_kbins=10)
    assert result["k_centers"].shape == (10,)
    valid = np.isfinite(result["P"])
    assert np.all(result["P"][valid] >= 0)


def test_theoretical_ratio_decreases_with_k():
    """P_v/P_delta ~ 1/k^2 -- must decrease monotonically with k."""
    cfg = _cfg()
    k = np.geomspace(0.01, 1.0, 10)
    ratio = theoretical_velocity_over_density_ratio(cfg, k, Z_TEST)
    assert np.all(np.diff(ratio) < 0)


def test_correction_factor_recovers_unity_for_consistent_fields():
    """
    THE load-bearing test: if raw_velocity_z is fed in ALREADY in km/s
    and ALREADY exactly satisfying the continuity equation relative to
    the density field, the recovered correction_factor_kms must be ~1.0
    at every k -- confirming the estimator correctly identifies 'no
    correction needed' when none is actually needed, rather than always
    reporting some spurious factor regardless of input.
    """
    cfg = _cfg()
    density, v_kms_exact = _build_consistent_density_and_velocity(cfg, Z_TEST, seed=2)

    result = check_velocity_convention(cfg, density, v_kms_exact, BOX_LEN, Z_TEST, n_kbins=8)

    valid = np.isfinite(result["correction_factor_kms"]) & (result["k_centers"] < 0.3)
    factors = result["correction_factor_kms"][valid]
    assert len(factors) >= 3, "Not enough valid k-bins to test meaningfully"
    print(f"\n  Recovered correction factors (should be ~1.0): {factors}")
    np.testing.assert_allclose(factors, 1.0, rtol=0.35)


def test_correction_factor_detects_a_real_wrong_scaling():
    """
    Sanity check in the other direction: if the velocity field is
    wrong by a KNOWN factor (e.g. 1000x too large), the recovered
    correction_factor_kms should be ~1/1000, not ~1 -- confirming the
    check actually detects a real error rather than always returning
    something close to 1 regardless of input.
    """
    cfg = _cfg()
    density, v_kms_exact = _build_consistent_density_and_velocity(cfg, Z_TEST, seed=3)
    v_wrong = v_kms_exact * 1000.0

    result = check_velocity_convention(cfg, density, v_wrong, BOX_LEN, Z_TEST, n_kbins=8)
    valid = np.isfinite(result["correction_factor_kms"]) & (result["k_centers"] < 0.3)
    factors = result["correction_factor_kms"][valid]
    print(f"\n  Recovered correction factors (should be ~0.001): {factors}")
    np.testing.assert_allclose(factors, 1.0 / 1000.0, rtol=0.35)
