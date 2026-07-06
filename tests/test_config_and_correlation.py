"""
Smoke tests that don't need py21cmfast, CAMB, or cluster data -- just
config loading and the pure-numpy correlation math. Run with:
    pytest tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.power_spectra import (
    KGrid, cross_power_2d, make_overdensity, to_Cell, to_Dell,
)
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "fiducial.yaml")


def test_load_fiducial_config():
    cfg = load_config(CONFIG_PATH)
    assert cfg.box.box_len_mpc == 300.0
    assert cfg.box.hii_dim == 300
    assert cfg.box.dim == 600
    assert cfg.cosmology.H0 == 67.77
    assert cfg.snr.tracer == "lae"


def test_path_reference_resolution():
    cfg = load_config(CONFIG_PATH)
    # coeval_root should have resolved off project_root, not contain '${'
    assert "${" not in cfg.paths.coeval_root
    assert cfg.paths.project_root in cfg.paths.coeval_root


def test_cosmology_matches_config():
    cfg = load_config(CONFIG_PATH)
    cosmo = get_cosmology(cfg)
    assert abs(cosmo.H0.value - 67.77) < 1e-6
    assert abs(cosmo.Om0 - 0.3086) < 1e-6


def test_make_overdensity_zero_mean():
    field = np.random.default_rng(0).uniform(1, 10, size=(16, 16))
    delta = make_overdensity(field)
    assert abs(delta.mean()) < 1e-8


def test_make_overdensity_handles_zero_mean_field():
    field = np.zeros((8, 8))
    delta = make_overdensity(field)
    assert np.all(delta == 0)


def test_cross_power_2d_autopower_is_nonnegative():
    cfg = load_config(CONFIG_PATH)
    kg = KGrid(cfg)
    rng = np.random.default_rng(1)
    field = rng.normal(size=(kg.n_side, kg.n_side))
    P, Perr, r = cross_power_2d(field, field, kg)
    valid = np.isfinite(P)
    assert valid.any()
    assert np.all(P[valid] >= -1e-8)  # auto-power should be non-negative
    assert np.all(np.isnan(r[valid]) | (np.abs(r[valid] - 1) < 1e-6))  # r=1 for autopower


def test_to_cell_to_dell_roundtrip_shapes():
    ell = np.linspace(100, 1000, 10)
    C = np.ones_like(ell)
    Cerr = 0.1 * np.ones_like(ell)
    D, Derr = to_Dell(ell, C, Cerr)
    assert D.shape == ell.shape
    assert np.all(D > 0)
