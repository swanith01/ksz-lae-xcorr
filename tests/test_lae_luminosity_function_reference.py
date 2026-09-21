"""
Tests for io/lae_luminosity_function_reference.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.io.lae_luminosity_function_reference import (
    KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED,
    nearest_konno2018_redshift,
    schechter_phi_per_dex,
)


def test_schechter_phi_monotonically_declines_for_alpha_below_minus_one():
    """For alpha=-1.5 (alpha+1 < 0), the per-dex Schechter function has
    NO interior peak -- it is monotonically declining for all L>0 (the
    power-law term itself is already declining; the exponential cutoff
    only steepens that decline, it doesn't create a hump). A peak only
    exists in this convention when alpha > -1."""
    alpha, log_l_star, log_phi_star = -1.5, 43.0, -3.25
    log_l = np.linspace(40.0, 44.0, 200)
    phi = schechter_phi_per_dex(log_l, alpha, log_l_star, log_phi_star)
    assert np.all(np.diff(phi) < 0)


def test_schechter_phi_declines_at_high_luminosity():
    alpha, log_l_star, log_phi_star = -1.5, 43.0, -3.25
    phi_at_lstar = schechter_phi_per_dex(43.0, alpha, log_l_star, log_phi_star)
    phi_above = schechter_phi_per_dex(44.0, alpha, log_l_star, log_phi_star)
    assert phi_above < phi_at_lstar


def test_schechter_phi_positive_in_realistic_lae_range():
    """Realistic LAE luminosity range only (up to ~10^44.5 erg/s) --
    the function is mathematically positive everywhere, but floating
    point underflows to exact 0.0 many dex beyond L*, which is not a
    real code bug, just outside any range this function is ever
    actually used for."""
    log_l = np.linspace(40.0, 44.5, 50)
    phi = schechter_phi_per_dex(log_l, -1.5, 43.0, -3.25)
    assert np.all(phi > 0)


def test_konno2018_params_exist_at_expected_redshifts():
    assert 5.7 in KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED
    assert 6.6 in KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED


def test_nearest_konno2018_redshift():
    assert nearest_konno2018_redshift(5.5) == 5.7
    assert nearest_konno2018_redshift(6.9) == 6.6
    assert nearest_konno2018_redshift(6.15) in (5.7, 6.6)  # exactly midway, either is defensible
