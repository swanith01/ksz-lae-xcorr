"""
Synthetic-data tests for plotting/lightcone_panels.py -- in particular for
_rgba_overlay_log and plot_xhi_tracer_overlay, which replaced the old
scatter-based overlay. These actually call matplotlib and write files
(tmp_path), not just parse the module, per this project's own
'recurring mistake' rule about trusting new plotting code without running it.

Run with:
    pytest tests/test_lightcone_panels.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.plotting.lightcone_panels import (
    TRACER_COLORS,
    _rgba_overlay_log,
    plot_four_field_panels,
    plot_xhi_tracer_overlay,
)
from ksz_lae_xcorr.utils.config import Config


def _fake_cfg():
    return Config({"box": Config({"box_len_mpc": 40.0, "hii_dim": 10, "z_min": 6.0, "z_max": 10.0})})


def test_rgba_overlay_zero_count_is_fully_transparent():
    count = np.zeros((8, 8))
    rgba = _rgba_overlay_log(count, TRACER_COLORS["halo"])
    assert rgba.shape == (8, 8, 4)
    assert np.all(rgba[..., 3] == 0.0), "count==0 must be fully transparent -- no invented density"


def test_rgba_overlay_alpha_monotonic_in_count():
    """Higher count -> higher (or equal) alpha, everywhere -- this is the
    exact-faithfulness property the scatter version didn't have (marker
    size/alpha there depended on plot layout, not just the array value)."""
    count = np.array([[0, 1, 2, 5, 50]])
    rgba = _rgba_overlay_log(count, (1.0, 0.0, 0.0))
    alphas = rgba[0, :, 3]
    assert np.all(np.diff(alphas) >= 0)
    assert alphas[0] == 0.0
    assert alphas[-1] == np.max(alphas)


def test_rgba_overlay_all_zero_does_not_divide_by_zero():
    count = np.zeros((4, 4))
    rgba = _rgba_overlay_log(count, (0, 1, 0))
    assert np.all(np.isfinite(rgba))


def test_rgba_overlay_color_channels_fixed_regardless_of_count():
    count = np.array([[0, 3, 30]])
    color = (0.2, 0.4, 0.6)
    rgba = _rgba_overlay_log(count, color)
    np.testing.assert_allclose(rgba[..., 0], color[0])
    np.testing.assert_allclose(rgba[..., 1], color[1])
    np.testing.assert_allclose(rgba[..., 2], color[2])


def test_plot_xhi_tracer_overlay_single_tracer_runs_and_writes_file(tmp_path):
    cfg = _fake_cfg()
    rng = np.random.default_rng(0)
    n_z = 24
    xHI = np.clip(rng.uniform(0, 1, size=(10, n_z)), 0, 1)
    halo_count = rng.poisson(0.3, size=(10, n_z)).astype(float)
    z_arr = np.linspace(6.0, 10.0, n_z)

    plot_xhi_tracer_overlay(cfg, xHI, {"halo": halo_count}, z_arr, str(tmp_path), seed=1)

    out_file = tmp_path / "lc_tracer_overlay_seed1.pdf"
    assert out_file.exists() and out_file.stat().st_size > 0


def test_plot_xhi_tracer_overlay_multi_tracer_runs_and_writes_file(tmp_path):
    cfg = _fake_cfg()
    rng = np.random.default_rng(1)
    n_z = 24
    xHI = np.clip(rng.uniform(0, 1, size=(10, n_z)), 0, 1)
    tracers = {
        "halo": rng.poisson(0.3, size=(10, n_z)).astype(float),
        "lae": rng.poisson(0.02, size=(10, n_z)).astype(float),
        "lbg": rng.poisson(0.1, size=(10, n_z)).astype(float),
    }
    z_arr = np.linspace(6.0, 10.0, n_z)

    plot_xhi_tracer_overlay(cfg, xHI, tracers, z_arr, str(tmp_path), seed=2)

    out_file = tmp_path / "lc_tracer_overlay_seed2.pdf"
    assert out_file.exists() and out_file.stat().st_size > 0


def test_plot_four_field_panels_still_runs_and_writes_file(tmp_path):
    """plot_four_field_panels itself is unchanged -- this just guards
    against an accidental regression while editing the rest of the file."""
    cfg = _fake_cfg()
    rng = np.random.default_rng(2)
    n_z = 24
    xHI = np.clip(rng.uniform(0, 1, size=(10, n_z)), 0, 1)
    halo = rng.poisson(0.3, size=(10, n_z)).astype(float)
    lae = rng.poisson(0.02, size=(10, n_z)).astype(float)
    lbg = rng.poisson(0.1, size=(10, n_z)).astype(float)
    z_arr = np.linspace(6.0, 10.0, n_z)

    plot_four_field_panels(cfg, xHI, halo, lae, lbg, z_arr, str(tmp_path), seed=3)

    out_file = tmp_path / "lightcone_fields_seed3.pdf"
    assert out_file.exists() and out_file.stat().st_size > 0
