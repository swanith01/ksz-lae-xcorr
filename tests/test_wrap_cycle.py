"""
Tests for lightcone/wrap_cycle.py (wrap-cycle stitching ported from
ksz-pipeline).  Pure numpy/scipy, synthetic boxes, no cluster data.

Run with:  pytest tests/test_wrap_cycle.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.lightcone.wrap_cycle import (
    build_los_z_grid,
    cycle_angle,
    get_slab_bilinear,
    rotated_coords,
    slab_index,
    stitch_wrapcycle,
    wrap_cycle_index,
)
from ksz_lae_xcorr.utils.config import Config
from ksz_lae_xcorr.utils.cosmology import get_cosmology

N = 16


def _cosmo():
    return get_cosmology(Config({"cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489})}))


# ---- geometry --------------------------------------------------------------

def test_los_grid_is_uniform_in_comoving_distance():
    z, chi = build_los_z_grid(_cosmo(), 5.0, 7.0, cell_size=1.0)
    assert np.all(np.diff(z) > 0)
    np.testing.assert_allclose(np.diff(chi), 1.0, atol=1e-9)
    assert z[0] == pytest.approx(5.0)
    # z really maps to chi (inversion accuracy << 1 cell)
    chi_check = _cosmo().comoving_distance(z).to_value("Mpc")
    np.testing.assert_allclose(chi_check, chi, atol=5e-3)


def test_cycle_and_slab_bookkeeping_consistent():
    d = np.arange(0, 5 * N + 3) * 1.0
    n = slab_index(d, 1.0)
    cyc = wrap_cycle_index(d, 1.0, N)
    np.testing.assert_array_equal(n % N + cyc * N, n)
    assert cyc[0] == 0 and cyc[N - 1] == 0 and cyc[N] == 1 and cyc[2 * N] == 2
    # y_cell wraps to 0 exactly when a new cycle starts
    assert np.all((n % N == 0) == (np.arange(len(d)) % N == 0))


# reference values from ksz-pipeline docs/HANDOFF_wrap_cycle.md (numpy 2.4.4)
@pytest.mark.parametrize("seed,expected", [
    (123, [245.6467, 114.6296, 131.6559, 151.2682, 61.4123, 192.6411]),
    (101, [339.6717, 14.5908, 51.8010, 60.4586, 214.4531, 138.8351]),
])
def test_cycle_angle_matches_ksz_pipeline_reference(seed, expected):
    got = [cycle_angle(c, seed) for c in range(6)]
    np.testing.assert_allclose(got, expected, atol=1e-3,
                               err_msg="angle stream differs from the validated ksz-pipeline one "
                                       "(different numpy Generator stream?)")


def test_cycle_angle_deterministic_independent_and_in_range():
    a = [cycle_angle(c, 7) for c in range(50)]
    assert a == [cycle_angle(c, 7) for c in range(50)]
    assert len(set(a)) == 50
    assert all(0.0 <= x < 360.0 for x in a)
    assert cycle_angle(3, 7) != cycle_angle(3, 8)


def test_rotated_coords_identity_and_90deg_exact():
    ic, jc = rotated_coords(N, 0.0)
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    np.testing.assert_allclose(ic, i, atol=1e-12)
    np.testing.assert_allclose(jc, j, atol=1e-12)
    ic, jc = rotated_coords(N, 90.0)
    np.testing.assert_allclose(ic, -j, atol=1e-12)
    np.testing.assert_allclose(jc, i, atol=1e-12)


def test_slab_angle0_is_exact_and_90deg_is_exact_permutation():
    rng = np.random.default_rng(0)
    box = rng.normal(size=(N, N, N))
    s0 = get_slab_bilinear(box, 5, rotated_coords(N, 0.0))
    np.testing.assert_allclose(s0, box[:, :, 5], atol=1e-12)
    s90 = get_slab_bilinear(box, 5, rotated_coords(N, 90.0))
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    np.testing.assert_allclose(s90, box[(-j) % N, i % N, 5], atol=1e-10)


def test_grid_wrap_has_period_n_not_n_minus_1():
    # slab[i, j] = i: sampling at i = N - 0.5 straddles the seam between the
    # last index (value N-1) and index 0 (value 0) -> period-N interpolation
    # gives (N-1)/2.  scipy mode='wrap' (period N-1) gives something else.
    box = np.zeros((N, N, 1))
    box[:, :, 0] = np.arange(N)[:, None]
    coords = (np.full((1, 1), N - 0.5), np.zeros((1, 1)))
    gw = get_slab_bilinear(box, 0, coords, mode="grid-wrap")[0, 0]
    assert gw == pytest.approx((N - 1) / 2.0)
    w = get_slab_bilinear(box, 0, coords, mode="wrap")[0, 0]
    assert abs(w - gw) > 1.0
    with pytest.raises(ValueError):
        get_slab_bilinear(box, 0, coords, mode="nearest")


# ---- stitching -------------------------------------------------------------

def _stitch(boxes_fn, snap_z, z_arr, chi, seed=3, fields=("xH", "density", "vz"), **kw):
    calls = []

    def load(z, f):
        calls.append((round(z, 6), f))
        return boxes_fn(z, f)

    out = stitch_wrapcycle(load, snap_z, z_arr, chi, ngrid=N, cell_size=1.0,
                           wrap_cycle_seed=seed, fields=fields, **kw)
    return out, calls


def _grid(n_pix):
    z = np.linspace(5.0, 5.5, n_pix)
    return z, np.arange(n_pix) * 1.0


def test_wrap_cycle_breaks_bit_identical_revisits_fixed_angle_does_not():
    """Headline property: the same slab y_cell visited in two different wrap
    cycles must NOT be bit-identical (a fixed angle always is)."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(N, N, N)).astype(np.float32)
    snaps = [4.9, 5.6]
    z, chi = _grid(3 * N)
    out, _ = _stitch(lambda zz, f: base, snaps, z, chi, fields=("density",))
    lc = out["density"]
    assert not np.allclose(lc[:, :, 2], lc[:, :, 2 + N])
    assert not np.allclose(lc[:, :, 2], lc[:, :, 2 + 2 * N])
    # same cycle, same box -> consecutive-cycle comparison is the only freedom;
    # a fixed-angle reference (same angle every cycle) repeats exactly:
    fixed = get_slab_bilinear(base, 2, rotated_coords(N, 10.0))
    np.testing.assert_array_equal(fixed, get_slab_bilinear(base, (2 + N) % N, rotated_coords(N, 10.0)))


def test_fields_share_the_rotation():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(N, N, N)).astype(np.float32)
    boxes = {"xH": a, "density": 3.0 * a, "vz": -2.0 * a}
    z, chi = _grid(2 * N)
    out, _ = _stitch(lambda zz, f: boxes[f], [4.9, 5.6], z, chi)
    np.testing.assert_allclose(out["density"], 3.0 * out["xH"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out["vz"], -2.0 * out["xH"], rtol=1e-5, atol=1e-6)


def test_rotation_depends_on_los_position_not_snapshot():
    """Two different snapshots with identical content give the same lightcone
    as one: the angle is not a function of which snapshot is sampled."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(N, N, N)).astype(np.float32)
    z, chi = _grid(2 * N)
    one, _ = _stitch(lambda zz, f: base, [4.9, 5.6], z, chi, fields=("xH",))
    many, _ = _stitch(lambda zz, f: base, [4.9, 5.1, 5.3, 5.6], z, chi, fields=("xH",))
    np.testing.assert_allclose(one["xH"], many["xH"], atol=1e-5)


def test_linear_interpolation_across_snapshots_and_extrapolation():
    z, chi = _grid(N)
    lo, hi = 5.0, 5.4
    boxes = {5.0: np.full((N, N, N), 1.0), 5.4: np.full((N, N, N), 3.0)}
    out, _ = _stitch(lambda zz, f: boxes[round(zz, 6)], [lo, hi], z, chi, fields=("xH",))
    expected = 1.0 + 2.0 * (z - lo) / (hi - lo)       # includes z > hi: extrapolation
    np.testing.assert_allclose(out["xH"][3, 4, :], expected, atol=1e-5)


def test_velocity_is_passed_through_unscaled():
    """v4 velocity_z is already Mpc/s -- the stitcher must not touch units.
    A constant 3.3e-16 field must come out as exactly 3.3e-16 (guards against
    anyone 'fixing' it with the v3 Zel'dovich D f H/(1+z) conversion)."""
    z, chi = _grid(N)
    v = np.full((N, N, N), 3.3e-16, dtype=np.float64)
    out, _ = _stitch(lambda zz, f: v, [4.9, 5.6], z, chi, fields=("vz",), dtype=np.float64)
    np.testing.assert_allclose(out["vz"], 3.3e-16, rtol=1e-12)


def test_streaming_loads_each_snapshot_field_once():
    z = np.linspace(5.0, 6.0, 4 * N)
    chi = np.arange(len(z)) * 1.0
    snaps = np.linspace(4.9, 6.1, 7)
    rng = np.random.default_rng(4)
    out, calls = _stitch(lambda zz, f: rng.normal(size=(N, N, N)), snaps, z, chi)
    assert len(calls) == len(set(calls)) == len(snaps) * 3
    assert out["xH"].shape == (N, N, len(z))


def test_bad_inputs_rejected():
    z, chi = _grid(N)
    with pytest.raises(ValueError, match="at least 2"):
        _stitch(lambda zz, f: np.zeros((N, N, N)), [5.0], z, chi)
    with pytest.raises(ValueError, match="ascending"):
        _stitch(lambda zz, f: np.zeros((N, N, N)), [4.9, 5.6], z[::-1], chi)
    with pytest.raises(ValueError, match="shape"):
        _stitch(lambda zz, f: np.zeros((N + 1,) * 3), [4.9, 5.6], z, chi)
