"""
Tests for io/la_plante_reference.py -- checks the loaders parse the real
committed CSVs correctly (shape, sortedness, sane value ranges), not
synthetic data, since these files ARE the reference data itself.

Run with:
    pytest tests/test_la_plante_reference.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.io.la_plante_reference import (
    load_dell_vs_ell_band,
    load_dell_vs_z0_bands,
    load_reionization_histories,
)

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "la_plante_2022")


def test_dell_vs_z0_bands_has_three_ell_values():
    data = load_dell_vs_z0_bands(root=ROOT)
    assert set(data.keys()) == {500, 1000, 3000}


def test_dell_vs_z0_bands_z0_in_expected_range():
    data = load_dell_vs_z0_bands(root=ROOT)
    for ell, d in data.items():
        assert d["z0_lo"].min() > 5.0 and d["z0_lo"].max() < 14.0
        assert d["z0_hi"].min() > 5.0 and d["z0_hi"].max() < 14.0


def test_dell_vs_z0_bands_sorted_by_z0():
    data = load_dell_vs_z0_bands(root=ROOT)
    for ell, d in data.items():
        assert np.all(np.diff(d["z0_lo"]) >= 0)
        assert np.all(np.diff(d["z0_hi"]) >= 0)


def test_dell_vs_z0_hi_generally_above_lo_at_overlapping_z0():
    """Upper band edge should sit at or above the lower edge, checked at
    z0 values common to both (via interpolation, since native sampling
    differs between the lo/hi digitizations)."""
    data = load_dell_vs_z0_bands(root=ROOT)
    for ell, d in data.items():
        z_common = np.linspace(max(d["z0_lo"].min(), d["z0_hi"].min()),
                                min(d["z0_lo"].max(), d["z0_hi"].max()), 20)
        lo_interp = np.interp(z_common, d["z0_lo"], d["lo"])
        hi_interp = np.interp(z_common, d["z0_hi"], d["hi"])
        assert np.mean(hi_interp >= lo_interp) > 0.8, f"ell={ell}: hi/lo mostly inverted"


def test_dell_vs_ell_band_covers_expected_ell_range():
    data = load_dell_vs_ell_band(root=ROOT)
    assert data["ell_lo"].min() > 10 and data["ell_lo"].max() < 50000
    assert np.all(np.diff(data["ell_lo"]) >= 0)


def test_reionization_histories_has_three_scenarios():
    data = load_reionization_histories(root=ROOT)
    assert set(data.keys()) == {"Fiducial", "Early", "Short"}


def test_reionization_histories_xhii_in_valid_range():
    data = load_reionization_histories(root=ROOT)
    for scenario, d in data.items():
        assert d["x_HII"].min() >= -0.05 and d["x_HII"].max() <= 1.05  # small digitization slack
        assert np.all(np.diff(d["z"]) >= 0)


def test_reionization_histories_early_ionizes_faster_than_fiducial():
    """Sanity on the SCIENCE, not just the file format: x_HII is the
    IONIZED fraction (high = more ionized, not more neutral). 'Early'
    reionization means the universe reionizes at HIGHER z (earlier
    cosmic time) than 'Fiducial' -- so at a FIXED z, 'Early' should
    already be FURTHER ALONG, i.e. HIGHER x_HII, than 'Fiducial'."""
    data = load_reionization_histories(root=ROOT)
    z_test = 9.0
    x_fiducial = np.interp(z_test, data["Fiducial"]["z"], data["Fiducial"]["x_HII"])
    x_early = np.interp(z_test, data["Early"]["z"], data["Early"]["x_HII"])
    assert x_early > x_fiducial, (
        f"At z={z_test}, Early ({x_early:.3f}) should be MORE ionized "
        f"(higher x_HII, having started earlier) than Fiducial "
        f"({x_fiducial:.3f}) -- scenario semantics may be backwards."
    )
