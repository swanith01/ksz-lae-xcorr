"""
Tests for correlation.seed_stats.aggregate_coherence_over_seeds and
snr.cmb_filter.kSZ_reion_from_sim_diag -- both new, both need to actually
run against synthetic data shaped like the real scripts/09 output, not
just parse.

Run with:
    pytest tests/test_coherence_seed_aggregation.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.seed_stats import aggregate_coherence_over_seeds
from ksz_lae_xcorr.snr.cmb_filter import kSZ_reion_from_sim_diag


def _fake_coherence_results(n_seeds=5, n_ell=20, seed_offset=0.0, rng_seed=0):
    """Shaped exactly like scripts/09_coherence_decomposition.py's saved
    results[seed] = {'ell', 'D_total', 'D_diag', 'D_off', ...}."""
    rng = np.random.default_rng(rng_seed)
    ell = np.logspace(2, 4, n_ell)
    results = {}
    for s in range(1, n_seeds + 1):
        base = 1e-2 * (ell / 3000.0) ** -1.0
        noise = rng.normal(0, 0.05 * base)
        D_diag = base + noise + seed_offset
        D_off = rng.normal(0, 0.2 * base)  # can be negative, that's expected
        results[s] = {
            "ell": ell, "D_total": D_diag + D_off, "D_diag": D_diag, "D_off": D_off,
            "chi_eff": 8000.0, "frac_off_at_ell3000": 0.1,
        }
    return results


def test_aggregate_coherence_over_seeds_shapes_and_values():
    results = _fake_coherence_results(n_seeds=6, n_ell=15)
    agg = aggregate_coherence_over_seeds(results, seeds=list(results.keys()), field="D_diag")
    assert agg["n_seeds"] == 6
    assert agg["ell"].shape == (15,)
    assert agg["median"].shape == (15,)
    np.testing.assert_allclose(agg["lower"], agg["median"] - agg["sigma"])
    np.testing.assert_allclose(agg["upper"], agg["median"] + agg["sigma"])


def test_aggregate_coherence_requires_at_least_two_seeds():
    results = _fake_coherence_results(n_seeds=1)
    with pytest.raises(ValueError, match="at least 2 seeds"):
        aggregate_coherence_over_seeds(results, seeds=[1], field="D_diag")


def test_aggregate_coherence_rejects_mismatched_ell_grids():
    results = _fake_coherence_results(n_seeds=3, n_ell=15)
    # Corrupt one seed's ell grid to simulate a different-config run.
    results[2] = dict(results[2])
    results[2]["ell"] = results[2]["ell"] * 1.001
    with pytest.raises(ValueError, match="doesn't match"):
        aggregate_coherence_over_seeds(results, seeds=list(results.keys()), field="D_diag")


def test_aggregate_coherence_field_selection_differs_for_diag_vs_off():
    results = _fake_coherence_results(n_seeds=5)
    agg_diag = aggregate_coherence_over_seeds(results, seeds=list(results.keys()), field="D_diag")
    agg_off = aggregate_coherence_over_seeds(results, seeds=list(results.keys()), field="D_off")
    # D_diag is built strictly positive-ish here, D_off is noise around 0 --
    # medians should clearly differ, confirming 'field' actually selects
    # different data rather than silently always reading the same array.
    assert not np.allclose(agg_diag["median"], agg_off["median"])


class _FakeCfg:
    """Minimal stand-in -- kSZ_reion_from_sim_diag doesn't actually touch
    cfg directly (it's threaded through only for interface parity with
    kSZ_reion_from_sim), but keep the signature honest."""
    pass


def test_kSZ_reion_from_sim_diag_runs_on_raw_per_seed_dict():
    results = _fake_coherence_results(n_seeds=5, n_ell=30)
    ell_grid = np.logspace(2.2, 3.8, 50)
    Cl, ell_sim = kSZ_reion_from_sim_diag(_FakeCfg(), ell_grid, results)
    assert Cl.shape == ell_grid.shape
    assert np.all(Cl >= 0), "kSZ_reion_from_sim_diag must clip to non-negative, like kSZ_reion_from_sim does"
    assert np.all(np.isfinite(Cl))


def test_kSZ_reion_from_sim_diag_runs_on_pre_aggregated_dict():
    results = _fake_coherence_results(n_seeds=5, n_ell=30)
    agg = aggregate_coherence_over_seeds(results, seeds=list(results.keys()), field="D_diag")
    ell_grid = np.logspace(2.2, 3.8, 50)
    Cl, ell_sim = kSZ_reion_from_sim_diag(_FakeCfg(), ell_grid, agg)
    assert Cl.shape == ell_grid.shape
    assert np.all(Cl >= 0)


def test_kSZ_reion_from_sim_diag_matches_manual_aggregation():
    """Calling with the raw per-seed dict should give the identical result
    to pre-aggregating and passing that in -- two code paths, one answer."""
    results = _fake_coherence_results(n_seeds=5, n_ell=30, rng_seed=3)
    ell_grid = np.logspace(2.2, 3.8, 40)

    Cl_raw, _ = kSZ_reion_from_sim_diag(_FakeCfg(), ell_grid, results)
    agg = aggregate_coherence_over_seeds(results, seeds=list(results.keys()), field="D_diag")
    Cl_pre, _ = kSZ_reion_from_sim_diag(_FakeCfg(), ell_grid, agg)

    np.testing.assert_allclose(Cl_raw, Cl_pre)
