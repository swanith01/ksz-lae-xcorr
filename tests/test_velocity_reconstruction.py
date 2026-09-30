"""
Synthetic-data tests for correlation/velocity_reconstruction.py -- the
linear/continuity-equation ("not ML") LOS velocity reconstruction, per
Girish's 2026-09-30 request (arXiv:2609.36355 Eqs. 11-12, adapted to a
real-space coeval box -- see the module's own docstring).

No cluster data or py21cmfast needed -- pure numpy, small fake fields.

Run with:
    pytest tests/test_velocity_reconstruction.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.velocity_reconstruction import (
    _k_grid_3d_components,
    compare_reconstructed_to_native,
    measure_large_scale_bias,
    reconstruct_velocity_los,
)
from ksz_lae_xcorr.utils.config import Config
from ksz_lae_xcorr.utils.cosmology import get_cosmology

NGRID = 16
BOX_LEN = 40.0
Z = 8.0


def _make_cfg():
    return Config({
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
    })


def _random_delta(seed: int, ngrid: int = NGRID) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = rng.normal(0, 0.3, size=(ngrid, ngrid, ngrid))
    return field - field.mean()


def _exact_linear_theory_v_los(cfg, delta, box_len_mpc, z, los_axis=2):
    """Hand-computed reference: literally the same formula, built
    independently of reconstruct_velocity_los's own code, to check the
    module's FFT machinery against the documented physics rather than
    against itself."""
    from ksz_lae_xcorr.lightcone.stitch import _growth_rate_linder
    from ksz_lae_xcorr.utils import constants

    n = delta.shape[0]
    cosmo = get_cosmology(cfg)
    a = 1.0 / (1.0 + z)
    H = cosmo.H(z).to_value("km/s/Mpc")
    f = _growth_rate_linder(cosmo, z)

    KX, KY, KZ, kmag = _k_grid_3d_components(n, box_len_mpc)
    K_LOS = (KX, KY, KZ)[los_axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = 1j * K_LOS / kmag**2
    kernel[kmag == 0] = 0.0

    delta_k = np.fft.fftn(delta)
    v_kms = a * H * f * np.real(np.fft.ifftn(kernel * delta_k))
    return v_kms * constants.MPC_PER_KM_S_TO_S


def test_reconstruct_velocity_los_shape_and_finite():
    cfg = _make_cfg()
    delta = _random_delta(seed=1)
    v = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z)
    assert v.shape == delta.shape
    assert np.all(np.isfinite(v))


def test_reconstruct_velocity_los_rejects_noncubic():
    cfg = _make_cfg()
    delta = np.zeros((8, 8, 12))
    try:
        reconstruct_velocity_los(cfg, delta, BOX_LEN, Z)
        assert False, "expected ValueError for non-cubic input"
    except ValueError:
        pass


def test_reconstruct_velocity_los_rejects_zero_bias():
    cfg = _make_cfg()
    delta = _random_delta(seed=2)
    try:
        reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, bias=0.0)
        assert False, "expected ValueError for bias=0"
    except ValueError:
        pass


def test_reconstruct_velocity_los_matches_hand_derived_formula():
    """The load-bearing correctness check: reconstruct_velocity_los's FFT
    kernel must exactly reproduce the continuity-equation formula
    documented in the module (and independently reimplemented here),
    not just be internally self-consistent."""
    cfg = _make_cfg()
    delta = _random_delta(seed=3)

    v_module = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z)
    v_hand = _exact_linear_theory_v_los(cfg, delta, BOX_LEN, Z)

    np.testing.assert_allclose(v_module, v_hand, rtol=1e-8, atol=1e-30)


def test_compare_reconstructed_to_native_perfect_recovery_gives_r_near_one():
    """If the 'native' field IS exactly the linear-theory prediction for
    this same delta (by construction), the reconstruction should recover
    it almost perfectly: r(k) ~ 1, transfer(k) ~ 1, pixel Pearson r ~ 1."""
    cfg = _make_cfg()
    delta = _random_delta(seed=4)

    v_rec = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z)
    v_native = _exact_linear_theory_v_los(cfg, delta, BOX_LEN, Z)

    result = compare_reconstructed_to_native(v_rec, v_native, BOX_LEN, n_kbins=8)
    finite = np.isfinite(result["r_k"])
    assert finite.sum() > 0
    np.testing.assert_allclose(result["r_k"][finite], 1.0, atol=1e-6)
    np.testing.assert_allclose(result["transfer_k"][finite], 1.0, atol=1e-6)
    assert result["pixel_pearson_r"] == pytest.approx(1.0, abs=1e-6)


def test_compare_uncorrelated_fields_gives_low_correlation():
    """Reconstructing from one random field and comparing against a
    SEPARATE, independent random 'native' field should show no
    meaningful correlation -- guards against a bug that spuriously
    correlates anything with anything (e.g. a constant-offset artifact)."""
    cfg = _make_cfg()
    delta_a = _random_delta(seed=10)
    v_rec = reconstruct_velocity_los(cfg, delta_a, BOX_LEN, Z)

    rng = np.random.default_rng(11)
    v_independent = rng.normal(0, 1e-3, size=v_rec.shape)

    result = compare_reconstructed_to_native(v_rec, v_independent, BOX_LEN, n_kbins=8)
    finite = np.isfinite(result["r_k"])
    assert finite.sum() > 0
    assert np.mean(np.abs(result["r_k"][finite])) < 0.5, (
        "Uncorrelated fields should not show strong |r(k)| on average"
    )
    assert abs(result["pixel_pearson_r"]) < 0.3


def test_bias_scaling_is_inverse():
    """v_los(k) has an explicit 1/bias factor -- doubling bias should
    exactly halve the reconstructed field everywhere."""
    cfg = _make_cfg()
    delta = _random_delta(seed=5)

    v_bias1 = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, bias=1.0)
    v_bias2 = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, bias=2.0)

    np.testing.assert_allclose(v_bias2, v_bias1 / 2.0, rtol=1e-10)


def test_smoothing_suppresses_high_k_power():
    cfg = _make_cfg()
    delta = _random_delta(seed=6)

    v_unsmoothed = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, r_smooth_mpc=None)
    v_smoothed = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, r_smooth_mpc=8.0)

    from ksz_lae_xcorr.correlation.velocity_convention_check import radial_power_spectrum_3d
    p_un = radial_power_spectrum_3d(v_unsmoothed, BOX_LEN, n_kbins=8)
    p_sm = radial_power_spectrum_3d(v_smoothed, BOX_LEN, n_kbins=8)

    # Compare at the highest-k bin with finite power in both -- smoothing
    # should suppress it there.
    valid = np.isfinite(p_un["P"]) & np.isfinite(p_sm["P"]) & (p_un["P"] > 0)
    assert valid.any()
    i_hi = np.where(valid)[0][-1]
    assert p_sm["P"][i_hi] < p_un["P"][i_hi]


def test_los_axis_selection_changes_result_on_anisotropic_field():
    """A density field that varies only along one Cartesian axis should
    reconstruct differently depending on which axis is declared the LOS
    -- confirms los_axis is actually wired into the kernel, not ignored."""
    cfg = _make_cfg()
    n = NGRID
    rng = np.random.default_rng(7)
    profile = rng.normal(0, 0.3, size=n)
    profile -= profile.mean()
    # Varies only along axis 0; constant along axes 1, 2.
    delta = np.broadcast_to(profile[:, None, None], (n, n, n)).copy()

    v_los0 = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, los_axis=0)
    v_los2 = reconstruct_velocity_los(cfg, delta, BOX_LEN, Z, los_axis=2)

    # v is in Mpc/s, so physical values are ~1e-18 -- np.allclose's default
    # atol=1e-8 would trivially call any two such tiny numbers "equal", so
    # comparisons here must be scale-aware, not default-tolerance.
    #
    # A field with structure ONLY along axis 0 has zero Fourier support
    # off axis 0 (k_y=k_z=0 plane only) -- so k_z/k^2 * delta(k) is
    # identically zero everywhere on that support, and the los_axis=2
    # reconstruction should vanish (up to floating point), while
    # los_axis=0 should show real, macroscopic-for-its-units signal.
    v0_scale = np.max(np.abs(v_los0))
    assert v0_scale > 1e-22, f"los_axis=0 reconstruction unexpectedly near zero: {v0_scale:.3e}"
    assert np.max(np.abs(v_los2)) < 1e-12 * v0_scale


def test_measure_large_scale_bias_exact_recovery_noiseless():
    """delta_tracer = b_true * delta_matter EXACTLY (no shot noise) -> the
    cross/auto power ratio must recover b_true exactly at every k (it's a
    linear rescaling of the same field, so there's no sample-variance
    scatter to average away, unlike a real discrete tracer)."""
    delta_matter = _random_delta(seed=20)
    b_true = 2.35
    delta_tracer = b_true * delta_matter

    result = measure_large_scale_bias(delta_tracer, delta_matter, BOX_LEN, n_kbins=8)
    assert result["b_eff"] == pytest.approx(b_true, rel=1e-8)
    assert result["b_eff_std"] < 1e-6
    finite = np.isfinite(result["b_of_k"])
    np.testing.assert_allclose(result["b_of_k"][finite], b_true, rtol=1e-6)


def test_measure_large_scale_bias_recovers_true_value_with_shot_noise():
    """A more realistic case: delta_tracer = b_true * delta_matter PLUS
    independent shot-noise-like scatter -- the large-scale-only average
    should still land close to b_true, since shot noise is white (flat in
    k) while the signal grows toward low k for a clustered field, so the
    lowest-k bins are the least shot-noise-dominated."""
    rng = np.random.default_rng(21)
    delta_matter = _random_delta(seed=22, ngrid=32)
    b_true = 1.8
    shot_noise = rng.normal(0, 0.05, size=delta_matter.shape)
    delta_tracer = b_true * delta_matter + shot_noise
    delta_tracer -= delta_tracer.mean()

    result = measure_large_scale_bias(delta_tracer, delta_matter, BOX_LEN, n_kbins=10,
                                       n_large_scale_bins=3)
    assert result["b_eff"] == pytest.approx(b_true, rel=0.15)


def test_measure_large_scale_bias_rejects_shape_mismatch():
    delta_matter = _random_delta(seed=23)
    delta_tracer = np.zeros((8, 8, 8))
    try:
        measure_large_scale_bias(delta_tracer, delta_matter, BOX_LEN)
        assert False, "expected ValueError for shape mismatch"
    except ValueError:
        pass
