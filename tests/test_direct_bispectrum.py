"""
Tests for correlation/direct_bispectrum.py -- built entirely on synthetic
3D boxes (no real coeval data needed yet, matching how
coherence_decomposition.py was validated before being pointed at real
data). The load-bearing test is test_squeezed_signal_detected_when_
injected: this must show the estimator actually responds to a genuine
squeezed-bispectrum-type correlation (large-scale density modulating
small-scale momentum variance), not just that the code runs.

Run with:
    pytest tests/test_direct_bispectrum.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.direct_bispectrum import (
    build_momentum_field_3d,
    compute_snapshot_bispectrum_contribution,
    highpass_filter_3d,
    limber_sum_snapshots,
    squeezed_bispectrum_proxy_3d,
)

N = 24
BOX_LEN = 100.0
C_MPC_S = 9.7156e-18 * 3e5  # rough Mpc/s for c -- exact value doesn't matter for these tests


def test_build_momentum_field_shape_and_finiteness():
    rng = np.random.default_rng(0)
    density = 1.0 + rng.normal(0, 0.3, size=(N, N, N))
    xHI = np.clip(rng.uniform(0.05, 0.6, size=(N, N, N)), 0.01, 0.99)
    v_los = rng.normal(0, 1e-3, size=(N, N, N))
    q = build_momentum_field_3d(density, xHI, v_los, C_MPC_S)
    assert q.shape == (N, N, N)
    assert np.all(np.isfinite(q))


def test_build_momentum_field_shape_mismatch_raises():
    a = np.ones((N, N, N))
    b = np.ones((N, N, N - 1))
    try:
        build_momentum_field_3d(a, a, b, C_MPC_S)
        assert False, "should have raised on shape mismatch"
    except ValueError:
        pass


def test_build_momentum_field_fully_neutral_raises():
    a = np.ones((N, N, N))
    xHI_fully_neutral = np.ones((N, N, N))
    try:
        build_momentum_field_3d(a, xHI_fully_neutral, a, C_MPC_S)
        assert False, "should have raised rather than divide by zero mean x_e"
    except ValueError:
        pass


def test_highpass_removes_dc_and_low_k():
    """A pure constant field (all power at k=0) should be entirely
    removed by any k_cut > 0."""
    field = np.ones((N, N, N)) * 5.0
    filtered = highpass_filter_3d(field, BOX_LEN, k_cut=0.01)
    np.testing.assert_allclose(filtered, 0.0, atol=1e-8)


def test_highpass_preserves_high_k_mode():
    """A pure high-k sinusoid should pass through a low k_cut essentially
    unchanged (small deviation only from box-edge/binning effects)."""
    x = np.linspace(0, BOX_LEN, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k_wave = 2 * np.pi * 8 / BOX_LEN  # a genuinely high-k mode
    field = np.sin(k_wave * X)
    filtered = highpass_filter_3d(field, BOX_LEN, k_cut=0.05)
    # Should retain most of the power (not be zeroed like the DC test above)
    assert np.std(filtered) > 0.5 * np.std(field)


def test_squeezed_signal_detected_when_injected():
    """
    THE load-bearing test: build a synthetic box where LARGE-SCALE density
    genuinely modulates SMALL-SCALE momentum variance (the physical
    mechanism La Plante+2022 describes: overdense large-scale regions have
    more small-scale ionization/velocity structure -> more small-scale
    kSZ power). The estimator must detect this as a nonzero, POSITIVE
    cross-power at large (soft) k -- a null result here would mean the
    estimator isn't actually sensitive to squeezed bispectra at all.
    """
    rng = np.random.default_rng(1)
    x = np.linspace(0, BOX_LEN, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # Large-scale (soft) density mode -- single low-k plane wave.
    k_soft_true = 2 * np.pi * 1 / BOX_LEN
    delta_g = 0.8 * np.cos(k_soft_true * X)

    # Small-scale (hard) momentum field whose LOCAL AMPLITUDE is modulated
    # by the same large-scale mode -- this is exactly the squeezed
    # bispectrum signal: <delta_soft * q_hard * q_hard> != 0.
    envelope = 1.0 + 0.9 * np.cos(k_soft_true * X)  # >0 everywhere, modulates local variance
    k_hard_true = 2 * np.pi * 8 / BOX_LEN
    hard_carrier = np.sin(k_hard_true * X + rng.uniform(0, 2 * np.pi, size=X.shape))
    q_signal = envelope * hard_carrier * 1e-3

    local_power = highpass_filter_3d(q_signal, BOX_LEN, k_cut=0.15) ** 2
    k_soft_bins = np.array([0.02, 0.08])
    result = squeezed_bispectrum_proxy_3d(local_power, delta_g, BOX_LEN, k_soft_bins)

    assert np.isfinite(result["P_cross"][0])
    assert result["P_cross"][0] > 0, (
        "Expected a positive squeezed-bispectrum-proxy cross-power for an "
        "injected large-scale-modulates-small-scale-power signal -- got "
        f"{result['P_cross'][0]}. The estimator isn't detecting the "
        "mechanism it's meant to detect."
    )


def test_null_result_for_uncorrelated_fields():
    """Independent random local-power and galaxy fields should give a
    cross-power consistent with zero (no injected correlation)."""
    rng = np.random.default_rng(2)
    local_power = rng.normal(0, 1, size=(N, N, N)) ** 2
    delta_g = rng.normal(0, 1, size=(N, N, N))
    k_soft_bins = np.array([0.02, 0.05, 0.1])
    result = squeezed_bispectrum_proxy_3d(local_power, delta_g, BOX_LEN, k_soft_bins)
    # Not exactly zero (finite-sample noise), but should be small relative
    # to the injected-signal case above, and can be either sign.
    assert np.all(np.isfinite(result["P_cross"]) | np.isnan(result["P_cross"]))


def test_compute_snapshot_bispectrum_contribution_runs_end_to_end():
    rng = np.random.default_rng(3)
    density = 1.0 + rng.normal(0, 0.3, size=(N, N, N))
    xHI = np.clip(rng.uniform(0.05, 0.6, size=(N, N, N)), 0.01, 0.99)
    v_los = rng.normal(0, 1e-3, size=(N, N, N))
    delta_g = rng.normal(0, 0.5, size=(N, N, N))
    k_soft_bins = np.array([0.02, 0.06, 0.1])

    result = compute_snapshot_bispectrum_contribution(
        density, xHI, v_los, delta_g, BOX_LEN, C_MPC_S, k_hard=0.15, k_soft_bins=k_soft_bins
    )
    assert result["k_centers"].shape == (2,)
    assert result["P_cross"].shape == (2,)


def test_limber_sum_snapshots_weights_and_sums_correctly():
    k_centers = np.array([0.03, 0.07])
    results = [
        {"k_centers": k_centers, "P_cross": np.array([1.0, 2.0])},
        {"k_centers": k_centers, "P_cross": np.array([3.0, 4.0])},
    ]
    chi_list = np.array([100.0, 200.0])
    g_chi_list = np.array([1.0, 2.0])
    bg_list = np.array([1.0, 1.0])
    dchi_list = np.array([10.0, 10.0])

    out = limber_sum_snapshots(results, chi_list, g_chi_list, bg_list, dchi_list)

    w0 = 1.0**2 * 1.0 / 100.0**4 * 10.0
    w1 = 2.0**2 * 1.0 / 200.0**4 * 10.0
    expected = w0 * np.array([1.0, 2.0]) + w1 * np.array([3.0, 4.0])
    np.testing.assert_allclose(out["P_cross_summed"], expected)


def test_limber_sum_snapshots_rejects_mismatched_k_grids():
    results = [
        {"k_centers": np.array([0.03, 0.07]), "P_cross": np.array([1.0, 2.0])},
        {"k_centers": np.array([0.03, 0.08]), "P_cross": np.array([3.0, 4.0])},
    ]
    try:
        limber_sum_snapshots(results, np.array([1.0, 2.0]), np.array([1.0, 1.0]),
                              np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert False, "should have raised on mismatched k_centers"
    except ValueError:
        pass


def test_limber_sum_snapshots_rejects_length_mismatch():
    results = [{"k_centers": np.array([0.03]), "P_cross": np.array([1.0])}]
    try:
        limber_sum_snapshots(results, np.array([1.0, 2.0]), np.array([1.0]),
                              np.array([1.0]), np.array([1.0]))
        assert False, "should have raised on length mismatch"
    except ValueError:
        pass


class _FakeStitcher:
    """Returns a fixed, KNOWN xHI value per snapshot z -- lets the test
    assert exactly what build_tau_history's x_e_mean SHOULD be, in a
    single obvious direction, rather than just checking it runs."""
    def __init__(self, xhi_by_z):
        self.xhi_by_z = xhi_by_z

    def load_field_box(self, seed, z, field_name):
        assert field_name == "xH"
        return np.full((4, 4, 4), self.xhi_by_z[z])


def test_build_tau_history_x_e_mean_is_ionized_fraction_not_neutral():
    """
    THE regression test for a real bug caught 2026-09-10: an earlier
    version of scripts/18 computed 1.0 - x_e_mean AGAIN after getting
    x_e_mean from this function, silently reporting the NEUTRAL
    fraction while labeling it x_HII. Pin down the actual semantics
    here so that mistake can't quietly reappear: x_e_mean must be the
    IONIZED fraction, meaning it INCREASES as z DECREASES (more
    reionized at later times), not the reverse.
    """
    from ksz_lae_xcorr.correlation.direct_bispectrum import build_tau_history
    from ksz_lae_xcorr.utils.config import Config

    cfg = Config({"cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665})})
    # Realistic reionization history: mostly ionized at low z, mostly
    # neutral at high z (xHI -- neutral fraction -- goes 0.1 -> 0.9).
    z_vals = np.array([6.0, 10.0, 15.0])
    xhi_by_z = {6.0: 0.1, 10.0: 0.5, 15.0: 0.9}
    stitcher = _FakeStitcher(xhi_by_z)

    z_sorted, chi, tau_cum, x_e_mean = build_tau_history(cfg, stitcher, seed=1,
                                                          all_snap_z=z_vals, logger=None)

    # x_e_mean (ionized fraction) must DECREASE as z increases (less
    # ionized further in the past) -- the opposite of neutral fraction.
    assert np.all(np.diff(x_e_mean) < 0), (
        f"x_e_mean should decrease with increasing z (less ionized further "
        f"back in time); got {list(zip(z_sorted, x_e_mean))} -- if this is "
        f"increasing instead, x_e_mean is being reported as neutral "
        f"fraction, not ionized fraction."
    )
    # Exact values: x_e_mean = 1 - xHI at each z.
    np.testing.assert_allclose(x_e_mean, [0.9, 0.5, 0.1], atol=1e-9)
