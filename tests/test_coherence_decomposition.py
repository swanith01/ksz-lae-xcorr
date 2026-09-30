"""
Synthetic-data tests for correlation/coherence_decomposition.py. No cluster
data, py21cmfast, or CAMB needed -- pure numpy, small fake fields with a
DELIBERATELY INJECTED periodicity, so we can check the decomposition
actually detects the thing it's meant to detect (not just that it runs).

Run with:
    pytest tests/test_coherence_decomposition.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.coherence_decomposition import (
    compute_ksz_slices,
    compute_patchy_window_diag_power,
    cross_power_by_dchi,
    decompose_p_total_diag_off,
    group_slices_by_snapshot,
    random_shift_slices,
)
from ksz_lae_xcorr.correlation.projected_maps import build_projected_maps
from ksz_lae_xcorr.utils.config import Config

NGRID = 12
NPIX = 60
PERIOD = 15          # box "repeats" every 15 LOS pixels -- like a stitched
                      # box of comoving depth = (z-range's chi span)/NPIX*PERIOD
Z_MIN, Z_MAX = 5.0, 12.0


def _make_cfg():
    return Config({
        "box": Config({"z_min": Z_MIN, "z_max": Z_MAX, "box_len_mpc": 30.0, "hii_dim": NGRID}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 8, "tracers": []}),
    })


def _xHI_mean(z):
    # Monotonic, bounded away from 0/1 so x_e and e^-tau stay well-behaved.
    return np.clip(0.05 + 0.9 * (z - Z_MIN) / (Z_MAX - Z_MIN), 0.05, 0.95)


def _make_field_data(periodic: bool, seed: int = 0):
    """
    periodic=True:  density/velocity transverse patterns repeat exactly
                    every PERIOD LOS pixels (simulates a stitched box
                    reused periodically along the line of sight).
    periodic=False: every LOS pixel gets an independent random pattern
                    (the idealized uncorrelated-slice / Limber assumption).
    """
    rng = np.random.default_rng(seed)
    z_lc = np.linspace(Z_MIN, Z_MAX, NPIX)

    if periodic:
        base_density = 1.0 + rng.normal(0, 0.3, size=(PERIOD, NGRID, NGRID))
        base_velocity = rng.normal(0, 1e-3, size=(PERIOD, NGRID, NGRID))
        density_lc = np.stack([base_density[i % PERIOD] for i in range(NPIX)], axis=-1)
        velocity_lc = np.stack([base_velocity[i % PERIOD] for i in range(NPIX)], axis=-1)
    else:
        density_lc = 1.0 + rng.normal(0, 0.3, size=(NGRID, NGRID, NPIX))
        velocity_lc = rng.normal(0, 1e-3, size=(NGRID, NGRID, NPIX))

    xHI_lc = np.broadcast_to(_xHI_mean(z_lc)[None, None, :], (NGRID, NGRID, NPIX)).copy()

    return {
        "z_lc": z_lc,
        "xHI_lc": xHI_lc,
        "density_lc": density_lc,
        "velocity_lc": velocity_lc,       # Mpc/s -- what compute_ksz_slices uses
        "velocity_kms": velocity_lc.copy(),  # dummy, only v2/vproj diagnostics use this
    }


def test_theta_slices_sum_matches_build_projected_maps():
    """compute_ksz_slices must reproduce build_projected_maps's kSZ map
    exactly when summed over the LOS axis -- this is the load-bearing
    correctness check for the whole module: if this ever fails, the two
    formulas have drifted apart and nothing downstream can be trusted."""
    cfg = _make_cfg()
    fd = _make_field_data(periodic=True)

    theta_slices, chi_mpc, z = compute_ksz_slices(cfg, fd)
    assert theta_slices.shape == (NGRID, NGRID, len(z))

    maps = build_projected_maps(cfg, {1: fd}, {}, [1])
    kSZ_expected = maps["kSZ"][1]

    reconstructed = theta_slices.sum(axis=2)
    assert reconstructed.shape == kSZ_expected.shape
    # build_projected_maps stores kSZ_map as float32 (by design, for storage
    # size); cast our float64 reconstruction the same way before comparing,
    # so this checks the FORMULA matches, not float32-vs-float64 rounding.
    np.testing.assert_allclose(reconstructed.astype(np.float32), kSZ_expected, rtol=1e-6, atol=1e-30)


def test_p_total_equals_p_diag_plus_p_off():
    """Algebraic identity that must hold regardless of periodicity --
    P_total = P_diag + P_off is true by construction (before any binning,
    pointwise), and binning is linear, so it must survive to D_ell too."""
    cfg = _make_cfg()
    fd = _make_field_data(periodic=True)
    theta_slices, chi_mpc, z = compute_ksz_slices(cfg, fd)

    ell, D_total, D_diag, D_off = decompose_p_total_diag_off(cfg, theta_slices, float(chi_mpc.mean()))
    assert len(ell) > 0
    np.testing.assert_allclose(D_total, D_diag + D_off, rtol=1e-6, atol=1e-25)


def test_periodicity_is_detected_as_p_off():
    """The actual point of this module: a map built from a periodically-
    repeating box should show much more P_off (relative to P_diag) than
    one built from fully independent slices."""
    cfg = _make_cfg()

    fd_periodic = _make_field_data(periodic=True, seed=1)
    fd_indep = _make_field_data(periodic=False, seed=1)

    theta_p, chi_p, _ = compute_ksz_slices(cfg, fd_periodic)
    theta_i, chi_i, _ = compute_ksz_slices(cfg, fd_indep)

    _, _, D_diag_p, D_off_p = decompose_p_total_diag_off(cfg, theta_p, float(chi_p.mean()))
    _, _, D_diag_i, D_off_i = decompose_p_total_diag_off(cfg, theta_i, float(chi_i.mean()))

    off_p = np.sum(np.abs(D_off_p))
    off_i = np.sum(np.abs(D_off_i))
    print(f"\n  sum|D_off| periodic={off_p:.3e}  independent={off_i:.3e}")

    assert off_p > 3 * off_i, (
        "Periodic-box synthetic case should show far more |P_off| than the "
        "independent-slice control -- if this fails, the decomposition isn't "
        "actually sensitive to periodicity."
    )


def test_random_shift_removes_periodicity_but_preserves_p_diag():
    """random_shift_slices should (a) leave P_diag EXACTLY unchanged
    (translation is a pure Fourier phase rotation, magnitude-preserving)
    and (b) substantially reduce |P_off| by destroying the fixed spatial
    alignment between periodic replicas."""
    cfg = _make_cfg()
    fd = _make_field_data(periodic=True, seed=2)
    theta_slices, chi_mpc, _ = compute_ksz_slices(cfg, fd)
    chi_eff = float(chi_mpc.mean())

    _, _, D_diag_before, D_off_before = decompose_p_total_diag_off(cfg, theta_slices, chi_eff)

    shifted = random_shift_slices(theta_slices, seed=123)
    _, _, D_diag_after, D_off_after = decompose_p_total_diag_off(cfg, shifted, chi_eff)

    np.testing.assert_allclose(D_diag_before, D_diag_after, rtol=1e-6, atol=1e-25)

    off_before = np.sum(np.abs(D_off_before))
    off_after = np.sum(np.abs(D_off_after))
    print(f"\n  sum|D_off| before shift={off_before:.3e}  after shift={off_after:.3e}")
    assert off_after < 0.5 * off_before


def test_group_slices_by_snapshot_conserves_los_sum():
    cfg = _make_cfg()
    fd = _make_field_data(periodic=True, seed=3)
    theta_slices, chi_mpc, z = compute_ksz_slices(cfg, fd)

    z_snapshots = [5.0, 7.0, 9.0, 12.0]
    grouped, chi_grouped = group_slices_by_snapshot(cfg, theta_slices, chi_mpc, z_snapshots)

    assert grouped.shape[-1] == len(z_snapshots)
    assert chi_grouped.shape == (len(z_snapshots),)
    # Regrouping only re-sums subsets of the same LOS axis -- total must
    # be conserved exactly regardless of how it's bucketed.
    np.testing.assert_allclose(grouped.sum(axis=2), theta_slices.sum(axis=2), rtol=1e-10)


def _make_cfg_wide_patchy():
    # z_min=4.0 is deliberately BELOW Z_FLOOR_REIONIZATION (6.0), so the
    # floor clamp is actually exercised, not just the x_HI-range clamp.
    return Config({
        "box": Config({"z_min": 4.0, "z_max": 14.0, "box_len_mpc": 30.0, "hii_dim": NGRID}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 8, "tracers": []}),
    })


def _xHI_mean_patchy(z):
    """Saturates to EXACTLY 0 (fully ionized) at/below z=6 and EXACTLY 1
    (fully neutral) at/above z=13, with a patchy ramp in between -- so
    clamp_window_to_patchy_regime has real degenerate tails to trim on
    both ends, not just the reionization floor."""
    raw = (z - 6.0) / (13.0 - 6.0)
    return np.clip(raw, 0.0, 1.0)


def _make_field_data_patchy(seed: int = 0, n_pix: int = 100, period: int = 10):
    rng = np.random.default_rng(seed)
    z_lc = np.linspace(4.0, 14.0, n_pix)

    base_density = 1.0 + rng.normal(0, 0.3, size=(period, NGRID, NGRID))
    base_velocity = rng.normal(0, 1e-3, size=(period, NGRID, NGRID))
    density_lc = np.stack([base_density[i % period] for i in range(n_pix)], axis=-1)
    velocity_lc = np.stack([base_velocity[i % period] for i in range(n_pix)], axis=-1)

    xHI_lc = np.broadcast_to(_xHI_mean_patchy(z_lc)[None, None, :], (NGRID, NGRID, n_pix)).copy()

    return {
        "z_lc": z_lc,
        "xHI_lc": xHI_lc,
        "density_lc": density_lc,
        "velocity_lc": velocity_lc,
        "velocity_kms": velocity_lc.copy(),
    }


def test_compute_patchy_window_diag_power_restricts_to_real_patchy_range():
    """The whole point of this function: z_lo/z_hi in the output should be
    a genuine SUBSET of the requested box.z_min-z_max range (4-14), pulled
    in on both ends -- the low end by the z>=6 reionization floor, the
    high end by excluding the exactly-fully-neutral tail (z>=13)."""
    cfg = _make_cfg_wide_patchy()
    fd = _make_field_data_patchy(seed=10)

    result = compute_patchy_window_diag_power(cfg, fd)

    assert result["z_lo"] >= 6.0 - 1e-6, (
        f"z_lo={result['z_lo']} should respect the z>=6 reionization floor"
    )
    assert result["z_hi"] < 13.0 + 1e-3, (
        f"z_hi={result['z_hi']} should exclude the fully-neutral (x_HI=1) tail at z>=13"
    )
    assert result["z_lo"] > cfg.box.z_min
    assert result["z_hi"] < cfg.box.z_max
    assert len(result["ell"]) > 0
    np.testing.assert_allclose(
        result["D_total"], result["D_diag"] + result["D_off"], rtol=1e-6, atol=1e-25
    )


def test_compute_patchy_window_diag_power_matches_manual_composition():
    """Correctness of the composition itself: manually slicing theta_slices
    to the same clamped window and calling decompose_p_total_diag_off
    directly should reproduce compute_patchy_window_diag_power's output
    exactly -- this function should just be wiring, not new math."""
    from ksz_lae_xcorr.snr.roman_hls_benchmark import chi_eff_power_weighted, clamp_window_to_patchy_regime

    cfg = _make_cfg_wide_patchy()
    fd = _make_field_data_patchy(seed=11)

    result = compute_patchy_window_diag_power(cfg, fd)

    theta_full, chi_full, z_full = compute_ksz_slices(cfg, fd)
    z0 = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    dz = cfg.box.z_max - cfg.box.z_min
    z_lo, z_hi = clamp_window_to_patchy_regime(z0, dz, fd["z_lc"], fd["xHI_lc"])
    mask = (z_full >= z_lo) & (z_full < z_hi)
    chi_eff_expected = chi_eff_power_weighted(cfg, fd, z_lo, z_hi)
    ell_exp, D_total_exp, D_diag_exp, D_off_exp = decompose_p_total_diag_off(
        cfg, theta_full[:, :, mask], chi_eff_expected
    )

    assert result["z_lo"] == z_lo
    assert result["z_hi"] == z_hi
    assert result["chi_eff"] == chi_eff_expected
    np.testing.assert_allclose(result["ell"], ell_exp, rtol=1e-10)
    np.testing.assert_allclose(result["D_diag"], D_diag_exp, rtol=1e-10)


def test_compute_patchy_window_diag_power_chi_eff_differs_from_box_midpoint():
    """chi_eff should be the power-weighted mean over the ACTUAL patchy
    window, not a naive box-midpoint chi -- guards against silently
    falling back to the old z=12.5-style arbitrary reference."""
    cfg = _make_cfg_wide_patchy()
    fd = _make_field_data_patchy(seed=12)

    result = compute_patchy_window_diag_power(cfg, fd)

    from ksz_lae_xcorr.utils.cosmology import get_cosmology
    cosmo = get_cosmology(cfg)
    z_box_mid = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    chi_box_mid = cosmo.comoving_distance(z_box_mid).to_value("Mpc")

    assert result["chi_eff"] != chi_box_mid


def test_cross_power_by_dchi_detects_periodicity():
    """Same periodic-vs-independent contrast as test_periodicity_is_detected_
    as_p_off, but via the real-space Delta-chi cross-power (the literal
    'Delta-chi/L_box overlay' diagnostic) instead of the Fourier P_off."""
    cfg = _make_cfg()
    fd_periodic = _make_field_data(periodic=True, seed=4)
    fd_indep = _make_field_data(periodic=False, seed=4)

    theta_p, chi_p, _ = compute_ksz_slices(cfg, fd_periodic)
    theta_i, chi_i, _ = compute_ksz_slices(cfg, fd_indep)

    centers_p, mean_p, std_p, n_p = cross_power_by_dchi(theta_p, chi_p, cfg.box.box_len_mpc, n_dchi_bins=10)
    centers_i, mean_i, std_i, n_i = cross_power_by_dchi(theta_i, chi_i, cfg.box.box_len_mpc, n_dchi_bins=10)

    assert centers_p.shape == (10,)
    total_p = np.nansum(np.abs(mean_p))
    total_i = np.nansum(np.abs(mean_i))
    print(f"\n  sum|cross_by_dchi| periodic={total_p:.3e}  independent={total_i:.3e}")
    assert total_p > 3 * total_i
