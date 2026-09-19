"""
Tests for correlation/auto_power.py's compute_tracer_auto_spectra --
the tracer-only (halo/lae/lbg) auto-power z-sweep, added 2026-09-18 to
support checking whether these auto-spectra are smooth, independent of
the still-broken stitched kSZ pathway.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.auto_power import compute_tracer_auto_spectra
from ksz_lae_xcorr.utils.config import Config

NGRID = 8
Z_MIN, Z_MAX = 6.0, 10.0


def _cfg(dz_tracer_bin=1.0):
    return Config({
        "box": Config({"box_len_mpc": 40.0, "hii_dim": NGRID, "z_min": Z_MIN, "z_max": Z_MAX}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 4, "dz_tracer_bin": dz_tracer_bin}),
    })


def _fake_tracer_data(seed_list, n_z_nodes=20, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    z_nodes = np.linspace(Z_MIN, Z_MAX, n_z_nodes)
    tracer_data = {}
    for seed in seed_list:
        tracer_data[seed] = {
            "z_nodes": z_nodes,
            "halo_count_lc": rng.poisson(5, size=(NGRID, NGRID, n_z_nodes)).astype(np.float32),
            "lae_count_lc": rng.poisson(1, size=(NGRID, NGRID, n_z_nodes)).astype(np.float32),
        }
    return tracer_data


def test_compute_tracer_auto_spectra_runs_end_to_end():
    cfg = _cfg()
    tracer_data = _fake_tracer_data([1])
    result = compute_tracer_auto_spectra(cfg, tracer_data, seeds=[1])

    assert "halo" in result and "lae" in result and "lbg" in result
    assert 1 in result["halo"]
    assert len(result["halo"][1]) > 0, "expected at least one z-slice with data"
    for z_c, entry in result["halo"][1].items():
        assert "ell" in entry and "D_ell" in entry and "D_err" in entry
        assert len(entry["ell"]) == len(entry["D_ell"])


def test_compute_tracer_auto_spectra_auto_power_is_nonnegative():
    """Auto-power (a field cross-correlated with itself) must be >= 0
    (up to noise floor) -- unlike a cross-power, which can be negative."""
    cfg = _cfg()
    tracer_data = _fake_tracer_data([1])
    result = compute_tracer_auto_spectra(cfg, tracer_data, seeds=[1])

    for z_c, entry in result["halo"][1].items():
        assert np.all(entry["D_ell"] > -1e-6), (
            f"auto-power should be non-negative (up to noise), got {entry['D_ell']} at z={z_c}"
        )


def test_compute_tracer_auto_spectra_missing_tracer_omitted_not_crash():
    """lbg_count_lc is absent from the fake fixture -- must be skipped
    cleanly, not raise, matching compute_cross_spectra's own convention
    for tracers pending external catalogue handover."""
    cfg = _cfg()
    tracer_data = _fake_tracer_data([1])  # no lbg_count_lc key
    result = compute_tracer_auto_spectra(cfg, tracer_data, seeds=[1])

    assert result["lbg"][1] == {}, "lbg should be present as a key but empty, not crash"


def test_compute_tracer_auto_spectra_seed_not_in_tracer_data_skipped():
    cfg = _cfg()
    tracer_data = _fake_tracer_data([1])
    result = compute_tracer_auto_spectra(cfg, tracer_data, seeds=[1, 2])
    assert result["halo"][2] == {}


def test_compute_tracer_auto_spectra_multiple_z_slices_with_finer_bin():
    cfg = _cfg(dz_tracer_bin=0.5)
    tracer_data = _fake_tracer_data([1])
    result = compute_tracer_auto_spectra(cfg, tracer_data, seeds=[1])
    # dz=0.5 over a 4-unit z range should give ~8 slices (vs ~4 for dz=1.0)
    assert len(result["halo"][1]) >= 5
