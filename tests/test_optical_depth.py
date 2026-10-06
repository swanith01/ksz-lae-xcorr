"""Tests for utils/optical_depth.py and the optional 'ne_scale'/'tau0' keys of
compute_ksz_slices (legacy behaviour must be untouched when they are absent)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.coherence_decomposition import compute_ksz_slices
from ksz_lae_xcorr.utils.config import Config
from ksz_lae_xcorr.utils.optical_depth import analytic_tau_below, ne_scale_helium


def _cfg(z_min=5.0, z_max=8.0):
    return Config({
        "box": Config({"z_min": z_min, "z_max": z_max, "box_len_mpc": 30.0, "hii_dim": 8}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 5, "tracers": []}),
    })


def test_ne_scale_helium_values():
    assert ne_scale_helium(0.24) == pytest.approx(0.82)
    assert ne_scale_helium(0.0) == 1.0
    with pytest.raises(ValueError):
        ne_scale_helium(1.0)


def test_analytic_tau_below_is_about_half_of_planck_tau():
    cfg = _cfg()
    tau5 = analytic_tau_below(cfg, 5.0)
    tau6 = analytic_tau_below(cfg, 6.0)
    assert 0.025 < tau5 < 0.05          # ksz-pipeline docs: ~0.03-0.04 for z_min~5-6
    assert tau6 > tau5 > 0
    assert analytic_tau_below(cfg, 0.0) == 0.0


def _fd(rng, n=8, npix=40, z_min=5.0, z_max=8.0):
    z = np.linspace(z_min, z_max, npix)
    xHI = np.broadcast_to(np.clip(0.1 + 0.8 * (z - z_min) / (z_max - z_min), 0.1, 0.9),
                          (n, n, npix)).copy()
    return {"z_lc": z, "xHI_lc": xHI,
            "density_lc": 1.0 + rng.normal(0, 0.2, (n, n, npix)),
            "velocity_lc": rng.normal(0, 1e-16, (n, n, npix))}


def test_legacy_behaviour_untouched_without_keys():
    cfg, rng = _cfg(), np.random.default_rng(0)
    fd = _fd(rng)
    t0, _, _ = compute_ksz_slices(cfg, fd)
    t1, _, _ = compute_ksz_slices(cfg, {**fd, "ne_scale": 1.0, "tau0": 0.0})
    np.testing.assert_array_equal(t0, t1)


def test_tau0_attenuates_uniformly_and_ne_scale_changes_amplitude():
    cfg, rng = _cfg(), np.random.default_rng(1)
    fd = _fd(rng)
    t0, _, _ = compute_ksz_slices(cfg, fd)
    tau0 = 0.035
    t1, _, _ = compute_ksz_slices(cfg, {**fd, "tau0": tau0})
    np.testing.assert_allclose(t1, t0 * np.exp(-tau0), rtol=1e-12)   # constant e^{-tau0}
    t2, _, _ = compute_ksz_slices(cfg, {**fd, "ne_scale": 0.82})
    ratio = np.abs(t2).sum() / np.abs(t0).sum()
    assert 0.78 < ratio < 0.86          # ~0.82, nudged up by weaker e^{-tau}
