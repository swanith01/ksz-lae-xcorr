"""
Tests for tracers/luminosity_function.py -- the LAE luminosity function
built from raw lya_lum_obs catalogue values.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.tracers.luminosity_function import (
    compute_luminosity_function,
    load_raw_lae_luminosities,
)
from ksz_lae_xcorr.utils.config import Config
from ksz_lae_xcorr.utils.external_catalogue import external_catalogue_filename


def _cfg(tmp_path, box_len_mpc=40.0, hii_dim=16):
    root = str(tmp_path / "lae_catalogue")
    os.makedirs(os.path.join(root, "lya_lum_obs"), exist_ok=True)
    return Config({
        "box": Config({"box_len_mpc": box_len_mpc, "hii_dim": hii_dim}),
        "paths": Config({"lae_catalogue_root": root}),
    })


def _write_fake_luminosities(cfg, seed, z, values):
    fname = external_catalogue_filename("lya_lum_obs", z, cfg, seed)
    path = os.path.join(cfg.paths.lae_catalogue_root, "lya_lum_obs", fname)
    np.save(path, np.array(values, dtype=np.float64))


def test_load_raw_lae_luminosities_missing_file_returns_empty(tmp_path):
    cfg = _cfg(tmp_path)
    result = load_raw_lae_luminosities(cfg, seed=1, z=9.0)
    assert len(result) == 0


def test_load_raw_lae_luminosities_returns_exact_values(tmp_path):
    cfg = _cfg(tmp_path)
    values = [1e42, 2e42, 5e42]
    _write_fake_luminosities(cfg, seed=1, z=9.0, values=values)
    result = load_raw_lae_luminosities(cfg, seed=1, z=9.0)
    np.testing.assert_allclose(sorted(result), sorted(values))


def test_compute_luminosity_function_no_data_gives_nan_not_zero(tmp_path):
    """A bin with NO seed having any data at all must be NaN -- 'no
    measurement' and 'confirmed zero objects' are different claims and
    must not look identical on a log-scale plot."""
    cfg = _cfg(tmp_path)
    result = compute_luminosity_function(cfg, seeds=[1, 2], z=9.0)
    assert result["n_seeds_with_data"] == 0
    assert np.all(np.isnan(result["phi"]))


def test_compute_luminosity_function_matches_hand_computed_value(tmp_path):
    """Exact numeric check: 10 objects all at log10(L)=42.5 (dead center
    of one bin), box_len=40 Mpc -> phi in that bin = 10 / (40^3 * dlogL)."""
    cfg = _cfg(tmp_path, box_len_mpc=40.0)
    log_l_min, log_l_max, n_bins = 41.0, 44.0, 15
    dlogL = (log_l_max - log_l_min) / n_bins
    edges = np.linspace(log_l_min, log_l_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    target_log_l = centers[5]
    lum_value = 10 ** target_log_l

    _write_fake_luminosities(cfg, seed=1, z=9.0, values=[lum_value] * 10)
    result = compute_luminosity_function(cfg, seeds=[1], z=9.0,
                                          log_l_min=log_l_min, log_l_max=log_l_max, n_bins=n_bins)

    expected_phi = 10 / (40.0 ** 3 * dlogL)
    assert result["n_objects_total"] == 10
    assert result["phi"][5] == pytest.approx(expected_phi, rel=1e-6)
    other_bins = [i for i in range(n_bins) if i != 5]
    assert np.all(result["phi"][other_bins] == 0.0)


def test_compute_luminosity_function_averages_and_scatters_across_seeds(tmp_path):
    cfg = _cfg(tmp_path, box_len_mpc=40.0)
    edges = np.linspace(41.0, 44.0, 16)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lum_value = 10 ** centers[5]

    _write_fake_luminosities(cfg, seed=1, z=9.0, values=[lum_value] * 10)
    _write_fake_luminosities(cfg, seed=2, z=9.0, values=[lum_value] * 20)

    result = compute_luminosity_function(cfg, seeds=[1, 2], z=9.0)
    assert result["n_seeds_with_data"] == 2
    assert result["n_objects_total"] == 30
    dlogL = (44.0 - 41.0) / 15
    phi_seed1 = 10 / (40.0**3 * dlogL)
    phi_seed2 = 20 / (40.0**3 * dlogL)
    assert result["phi"][5] == pytest.approx((phi_seed1 + phi_seed2) / 2, rel=1e-6)
    assert result["phi_err"][5] > 0
