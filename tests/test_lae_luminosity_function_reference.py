"""
Tests for io/lae_luminosity_function_reference.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.io.lae_luminosity_function_reference import (
    KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED,
    nearest_kageura2025_redshift,
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


def test_kageura2025_schechter_params_five_bins():
    from ksz_lae_xcorr.io.lae_luminosity_function_reference import KAGEURA2025_SCHECHTER_PARAMS
    assert len(KAGEURA2025_SCHECHTER_PARAMS) == 5
    for z, (alpha, log_l_star, log_phi_star) in KAGEURA2025_SCHECHTER_PARAMS.items():
        assert 4.5 < z < 15.0
        assert -3.0 < alpha < -1.0
        assert 42.0 < log_l_star < 44.0
        assert -7.0 < log_phi_star < -2.0


def test_kageura2025_lf_points_consistent_array_lengths():
    from ksz_lae_xcorr.io.lae_luminosity_function_reference import KAGEURA2025_LF_POINTS
    assert len(KAGEURA2025_LF_POINTS) == 5
    for z, d in KAGEURA2025_LF_POINTS.items():
        n = len(d["log_l"])
        assert len(d["log_phi"]) == n
        assert len(d["log_phi_err_lo"]) == n
        assert len(d["log_phi_err_hi"]) == n


def test_kageura2025_lf_points_number_density_decreases_with_redshift():
    """Physical sanity check: at a fixed luminosity present in every bin
    (log L=42.3, the only one common to all five z bins), Phi should
    decrease toward higher z -- fewer bright LAEs further from us in
    time, consistent with the paper's own headline ~3 dex decline."""
    from ksz_lae_xcorr.io.lae_luminosity_function_reference import KAGEURA2025_LF_POINTS
    zs_sorted = sorted(KAGEURA2025_LF_POINTS.keys())
    log_phi_at_42_3 = [KAGEURA2025_LF_POINTS[z]["log_phi"][0] for z in zs_sorted]
    assert all(np.diff(log_phi_at_42_3) < 0)


def test_nearest_kageura2025_redshift():
    assert nearest_kageura2025_redshift(5.0) == 5.01
    assert nearest_kageura2025_redshift(6.0) == 5.90
    assert nearest_kageura2025_redshift(12.0) == 11.00
