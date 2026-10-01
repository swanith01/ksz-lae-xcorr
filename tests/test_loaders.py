"""
Tests for io/loaders.load_lightcone_products -- specifically the velocity
unit handling, fixed 2026-10-01 after real-data diagnosis (check_vz_units.py)
showed lc_vz.npz's raw array is ALREADY Mpc/s, not km/s as the old code
assumed (same bug class as the earlier lightcone/stitch.py velocity_z fix,
reintroduced here since this loader is a separate code path).

Builds tiny synthetic lc_*.npz products on disk (no cluster data needed)
shaped exactly like lightcone.stitch's real output, then loads them back
through load_lightcone_products.

Run with:
    pytest tests/test_loaders.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import Config

NGRID = 8
NPIX = 12


def _make_cfg(lc_root):
    return Config({
        "paths": Config({"lightcone_root": lc_root}),
    })


def _write_fake_lightcone(tmp_path, seed, vz_raw):
    seed_dir = tmp_path / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    z_arr = np.linspace(6.0, 15.0, NPIX)
    rng = np.random.default_rng(seed)
    xHI = rng.uniform(0, 1, size=(NGRID, NGRID, NPIX))
    density = rng.normal(0, 0.2, size=(NGRID, NGRID, NPIX))

    np.savez_compressed(seed_dir / "lc_xH.npz", lc=xHI)
    np.savez_compressed(seed_dir / "lc_density.npz", lc=density)
    np.savez_compressed(seed_dir / "lc_vz.npz", lc=vz_raw, z_arr=z_arr)
    return seed_dir


def test_velocity_lc_is_raw_array_unconverted(tmp_path):
    """The headline fix: 'velocity_lc' (Mpc/s, feeds the kSZ integrand)
    must now be the raw lc_vz.npz array AS-IS -- no division by c_kms/c_mpc_s
    -- since that array is already Mpc/s on disk (confirmed against real
    data). The old code's extra conversion crushed it by ~3e19x."""
    rng = np.random.default_rng(0)
    # Realistic Mpc/s-scale peculiar velocities (this session's own
    # reconstructed/native v_z fields land in the ~1e-17 to 1e-14 range).
    vz_raw = rng.normal(0, 5e-17, size=(NGRID, NGRID, NPIX))
    _write_fake_lightcone(tmp_path, seed=1, vz_raw=vz_raw)

    cfg = _make_cfg(str(tmp_path))
    field_data, _ = load_lightcone_products(cfg, seeds=[1])

    np.testing.assert_allclose(field_data[1]["velocity_lc"], vz_raw, rtol=1e-12)


def test_velocity_kms_is_properly_converted_from_mpc_s(tmp_path):
    """'velocity_kms' must be DERIVED from the (already-Mpc/s) raw array by
    an actual Mpc/s -> km/s conversion, not just the untouched raw array
    under a misleading name (the old behaviour)."""
    rng = np.random.default_rng(1)
    vz_raw = rng.normal(0, 5e-17, size=(NGRID, NGRID, NPIX))
    _write_fake_lightcone(tmp_path, seed=1, vz_raw=vz_raw)

    cfg = _make_cfg(str(tmp_path))
    field_data, _ = load_lightcone_products(cfg, seeds=[1])

    c_kms = constants.C_KMS
    c_mpc_s = constants.c_mpc_per_s()
    expected_kms = vz_raw * c_kms / c_mpc_s  # Mpc/s -> km/s, correct direction

    np.testing.assert_allclose(field_data[1]["velocity_kms"], expected_kms, rtol=1e-10)
    # Sanity: for Mpc/s-scale raw input, the km/s representation should be
    # many orders of magnitude LARGER (km/s is a "bigger" unit of speed per
    # count than Mpc/s), not smaller -- guards against silently swapping
    # the conversion direction back.
    assert np.abs(field_data[1]["velocity_kms"]).mean() > np.abs(field_data[1]["velocity_lc"]).mean()


def test_velocity_kms_and_velocity_lc_describe_the_same_physical_velocity(tmp_path):
    """Round-trip check: converting velocity_kms back to Mpc/s must recover
    velocity_lc exactly -- they're two unit representations of ONE physical
    field, not two independently-scaled arrays."""
    rng = np.random.default_rng(2)
    vz_raw = rng.normal(0, 3e-16, size=(NGRID, NGRID, NPIX))
    _write_fake_lightcone(tmp_path, seed=1, vz_raw=vz_raw)

    cfg = _make_cfg(str(tmp_path))
    field_data, _ = load_lightcone_products(cfg, seeds=[1])

    c_kms = constants.C_KMS
    c_mpc_s = constants.c_mpc_per_s()
    roundtrip_mpc_s = field_data[1]["velocity_kms"] / c_kms * c_mpc_s

    np.testing.assert_allclose(roundtrip_mpc_s, field_data[1]["velocity_lc"], rtol=1e-10)


def test_missing_seed_dir_is_silently_skipped(tmp_path):
    cfg = _make_cfg(str(tmp_path))
    field_data, tracer_data = load_lightcone_products(cfg, seeds=[99])
    assert field_data == {}
    assert tracer_data == {}


def test_tracer_products_loaded_when_present(tmp_path):
    rng = np.random.default_rng(3)
    vz_raw = rng.normal(0, 1e-16, size=(NGRID, NGRID, NPIX))
    seed_dir = _write_fake_lightcone(tmp_path, seed=1, vz_raw=vz_raw)

    halo_counts = rng.integers(0, 5, size=(NGRID, NGRID, NPIX)).astype(np.float64)
    np.savez_compressed(seed_dir / "lc_halos.npz", lc=halo_counts)

    cfg = _make_cfg(str(tmp_path))
    _, tracer_data = load_lightcone_products(cfg, seeds=[1])

    assert "halo_count_lc" in tracer_data[1]
    np.testing.assert_allclose(tracer_data[1]["halo_count_lc"], halo_counts)
    # LAE/LBG weren't written -- must be absent, not zero-filled.
    assert "lae_count_lc" not in tracer_data[1]
    assert "lbg_count_lc" not in tracer_data[1]
