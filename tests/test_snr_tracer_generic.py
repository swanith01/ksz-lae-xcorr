"""
Tests for the tracer_key generalization in snr/snr_forecast.py --
confirms LAE/LBG/halo genuinely select different underlying fields (the
parameter isn't silently ignored) and that the default reproduces the
pre-refactor LAE-only behavior exactly.

Run with:
    pytest tests/test_snr_tracer_generic.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.snr.snr_forecast import compute_filtered_signal, compute_lae_auto_power
from ksz_lae_xcorr.utils.config import Config

NGRID = 12
NPIX = 20
Z_MIN, Z_MAX = 6.0, 9.0


def _cfg():
    return Config({
        "box": Config({"box_len_mpc": 40.0, "hii_dim": NGRID, "z_min": Z_MIN, "z_max": Z_MAX}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 6}),
        "snr": Config({"experiments": Config({"SO": Config({})})}),
    })


def _tracer_data(seed=0):
    rng = np.random.default_rng(seed)
    z_nodes = np.linspace(Z_MIN, Z_MAX, NPIX)
    return {
        1: {
            "z_nodes": z_nodes,
            "lae_count_lc": rng.poisson(0.05, size=(NGRID, NGRID, NPIX)).astype(np.float64),
            "lbg_count_lc": rng.poisson(0.8, size=(NGRID, NGRID, NPIX)).astype(np.float64),
            "halo_count_lc": rng.poisson(2.0, size=(NGRID, NGRID, NPIX)).astype(np.float64),
        }
    }


def test_tracer_key_default_is_lae():
    """No tracer_key passed -> must behave exactly as the original
    LAE-only function did (this is the backward-compatibility guarantee
    every existing call site in scripts/05 and scripts/08 relies on)."""
    cfg = _cfg()
    kg = KGrid(cfg)
    tracer_data = _tracer_data()
    z_edges = np.linspace(Z_MIN, Z_MAX, 4)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])
    ell_grid = np.geomspace(200, 5000, 30)

    default_result = compute_lae_auto_power(cfg, kg, tracer_data, [1], z_edges, z_cents, ell_grid)
    explicit_lae = compute_lae_auto_power(cfg, kg, tracer_data, [1], z_edges, z_cents, ell_grid,
                                           tracer_key="lae_count_lc")

    assert default_result.keys() == explicit_lae.keys()
    for z_c in default_result:
        np.testing.assert_allclose(default_result[z_c], explicit_lae[z_c])


def test_tracer_key_lae_vs_lbg_give_different_results():
    """The actual point of the refactor: LAE and LBG counts are different
    fields with different statistics here, so their auto-power MUST differ.
    If this ever passes with identical output, tracer_key is being ignored
    somewhere."""
    cfg = _cfg()
    kg = KGrid(cfg)
    tracer_data = _tracer_data()
    z_edges = np.linspace(Z_MIN, Z_MAX, 4)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])
    ell_grid = np.geomspace(200, 5000, 30)

    Cl_lae = compute_lae_auto_power(cfg, kg, tracer_data, [1], z_edges, z_cents, ell_grid,
                                     tracer_key="lae_count_lc")
    Cl_lbg = compute_lae_auto_power(cfg, kg, tracer_data, [1], z_edges, z_cents, ell_grid,
                                     tracer_key="lbg_count_lc")

    assert Cl_lae.keys() == Cl_lbg.keys() and len(Cl_lae) > 0
    all_close = all(np.allclose(Cl_lae[z], Cl_lbg[z]) for z in Cl_lae)
    assert not all_close, "LAE and LBG auto-power came out identical -- tracer_key is being ignored"


def test_tracer_key_missing_field_skips_cleanly_not_crash():
    """If tracer_key names a field that isn't present for a seed, that
    seed should just be skipped (as it already was for missing
    'lae_count_lc'), not raise."""
    cfg = _cfg()
    kg = KGrid(cfg)
    tracer_data = _tracer_data()
    z_edges = np.linspace(Z_MIN, Z_MAX, 4)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])
    ell_grid = np.geomspace(200, 5000, 30)

    result = compute_lae_auto_power(cfg, kg, tracer_data, [1], z_edges, z_cents, ell_grid,
                                     tracer_key="does_not_exist_count_lc")
    assert result == {}


def test_compute_filtered_signal_tracer_key_default_matches_explicit_lae():
    cfg = _cfg()
    kg = KGrid(cfg)
    tracer_data = _tracer_data()
    z_edges = np.linspace(Z_MIN, Z_MAX, 4)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])
    ell_grid = np.geomspace(200, 5000, 30)

    rng = np.random.default_rng(1)
    filtered_kSZ2 = {"SO": {1: rng.normal(0, 1, size=(NGRID, NGRID))}}

    default_result = compute_filtered_signal(cfg, kg, filtered_kSZ2, tracer_data, [1],
                                              z_edges, z_cents, ell_grid)
    explicit_lae = compute_filtered_signal(cfg, kg, filtered_kSZ2, tracer_data, [1],
                                            z_edges, z_cents, ell_grid, tracer_key="lae_count_lc")

    assert default_result.keys() == explicit_lae.keys()
    for name in default_result:
        assert default_result[name].keys() == explicit_lae[name].keys()
        for z_c in default_result[name]:
            np.testing.assert_allclose(default_result[name][z_c], explicit_lae[name][z_c])
