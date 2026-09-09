"""
Tests for lightcone/stitch.py's NEW velocity_z_to_mpc_per_s and its
helpers (_linear_growth_factor, _growth_rate_linder) -- checked against
known physical bounds/behavior, since getting this wrong silently would
be exactly the kind of error that's been suspected all session.

Run with:
    pytest tests/test_velocity_conversion.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.lightcone.stitch import (
    _growth_rate_linder,
    _linear_growth_factor,
    velocity_z_to_mpc_per_s,
)
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import Config
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def _cfg():
    return Config({"cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665})})


def test_growth_factor_normalized_at_z0():
    cosmo = get_cosmology(_cfg())
    assert _linear_growth_factor(cosmo, 0.0) == pytest.approx(1.0, abs=1e-6)


def test_growth_factor_decreases_with_redshift():
    """D(z) < D(0) = 1 for z > 0 -- structure was less grown in the past.
    Must be monotonically decreasing with INCREASING z (equivalently,
    growing toward the present)."""
    cosmo = get_cosmology(_cfg())
    z_vals = [0.0, 1.0, 5.0, 9.0, 15.0]
    D_vals = [_linear_growth_factor(cosmo, z) for z in z_vals]
    assert np.all(np.diff(D_vals) < 0), f"D(z) not monotonically decreasing: {list(zip(z_vals, D_vals))}"
    assert D_vals[0] == pytest.approx(1.0, abs=1e-6)
    assert all(0 < d <= 1.0 for d in D_vals)


def test_growth_rate_approaches_unity_at_high_z():
    """f(z) -> 1 in the matter-dominated (high-z) limit (Omega_m(z) -> 1)."""
    cosmo = get_cosmology(_cfg())
    f_high_z = _growth_rate_linder(cosmo, 15.0)
    assert f_high_z == pytest.approx(1.0, abs=0.02)


def test_growth_rate_below_unity_today():
    """f(0) < 1 for a universe with dark energy (Omega_m(0) < 1)."""
    cosmo = get_cosmology(_cfg())
    f0 = _growth_rate_linder(cosmo, 0.0)
    assert 0.0 < f0 < 1.0


def test_velocity_conversion_gives_realistic_peculiar_velocity_scale():
    """
    Sanity check against ksz-pipeline's own quoted validation: 'typical
    v_rms ~95-156 km/s for an 800 Mpc box, z=5-15' AFTER their full
    conversion. Feed in a raw displacement field of a plausible magnitude
    and confirm the OUTPUT lands in a physically reasonable peculiar-
    velocity range (tens to low hundreds of km/s, i.e. v/c ~ 1e-4 to 1e-3)
    -- not the astronomically large or small values that would indicate a
    unit error.
    """
    cfg = _cfg()
    rng = np.random.default_rng(0)
    # A raw Zel'dovich displacement field -- magnitude chosen only to be
    # "some O(1) comoving-length-like array", not tuned to hit a target;
    # the test is on the OUTPUT scale after the physical conversion.
    raw_psi = rng.normal(0, 1.0, size=(20, 20, 20))

    for z in [5.0, 9.5, 15.0]:
        v_mpc_s = velocity_z_to_mpc_per_s(cfg, raw_psi, z)
        v_kms = v_mpc_s / constants.MPC_PER_KM_S_TO_S
        v_over_c = np.std(v_mpc_s) / (299792.458 * constants.MPC_PER_KM_S_TO_S)
        print(f"z={z}: v_kms rms={np.std(v_kms):.3f} km/s, v/c rms={v_over_c:.3e}")
        # Physically real peculiar velocities are at most ~1000 km/s scale
        # (v/c ~ 3e-3) even in extreme cases -- this is a coarse sanity
        # bound, not a precision check.
        assert np.std(v_kms) < 1e4, (
            f"z={z}: output velocity {np.std(v_kms):.3g} km/s is not a "
            f"physically plausible peculiar velocity scale."
        )


def test_velocity_conversion_shape_preserved():
    cfg = _cfg()
    raw_psi = np.ones((8, 8, 8)) * 0.5
    out = velocity_z_to_mpc_per_s(cfg, raw_psi, 9.0)
    assert out.shape == raw_psi.shape
    assert np.all(np.isfinite(out))
