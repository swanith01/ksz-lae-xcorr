"""Stitcher.load_field_box field_convention option (legacy default, physical opt-in)."""

import logging
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_ksz_auto_wrapcycle import _cfg, _write_tree  # noqa: E402

from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import (  # noqa: E402
    build_wrapcycle_field_data, normalisation_keys)
from ksz_lae_xcorr.lightcone import stitch  # noqa: E402
from ksz_lae_xcorr.lightcone.stitch import Stitcher  # noqa: E402
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio  # noqa: E402

Z = 6.2


def _st(tmp_path, **kw):
    _write_tree(tmp_path)
    return Stitcher(_cfg(tmp_path), **kw)


def _ic(tmp_path, field):
    name = {"vz": "velocity_z.npy", "density": "hires_density.npy"}[field]
    return np.load(tmp_path / "seed_1" / f"coeval_z{Z:.6f}" / name)


def test_default_is_legacy_raw_and_warns_once(tmp_path, monkeypatch):
    monkeypatch.delenv(stitch.FIELD_CONVENTION_ENV, raising=False)
    monkeypatch.setattr(stitch, "_LEGACY_WARNED", False)
    st = _st(tmp_path)
    assert st.field_convention == "legacy"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = st.load_field_box(1, Z, "vz")
        st.load_field_box(1, Z, "xH")
    assert sum("RAW py21cmfast fields" in str(x.message) for x in w) == 1
    np.testing.assert_array_equal(v, _ic(tmp_path, "vz"))


def test_physical_scalings(tmp_path):
    st = _st(tmp_path, field_convention="physical")
    v = st.load_field_box(1, Z, "vz")
    np.testing.assert_allclose(v, _ic(tmp_path, "vz") / (1 + Z), rtol=1e-6)
    d = st.load_field_box(1, Z, "density")
    legacy = Stitcher(_cfg(tmp_path), field_convention="legacy").load_field_box(1, Z, "density")
    D = float(growth_factor_ratio(0.3086, Z))
    assert 0.05 < D < 0.3
    np.testing.assert_allclose(d, legacy * D, rtol=1e-5, atol=1e-12)
    # xH untouched
    np.testing.assert_array_equal(st.load_field_box(1, Z, "xH"),
                                  Stitcher(_cfg(tmp_path), field_convention="legacy").load_field_box(1, Z, "xH"))


def test_precedence_explicit_env_cfg(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.delenv(stitch.FIELD_CONVENTION_ENV, raising=False)
    cfg.lightcone["field_convention"] = "physical"
    assert Stitcher(cfg).field_convention == "physical"            # cfg key
    monkeypatch.setenv(stitch.FIELD_CONVENTION_ENV, "legacy")
    assert Stitcher(cfg).field_convention == "legacy"              # env beats cfg
    assert Stitcher(cfg, field_convention="physical").field_convention == "physical"  # arg beats env
    monkeypatch.setenv(stitch.FIELD_CONVENTION_ENV, "bogus")
    with pytest.raises(ValueError):
        Stitcher(cfg)
    monkeypatch.delenv(stitch.FIELD_CONVENTION_ENV)
    with pytest.raises(ValueError):
        Stitcher(cfg).load_field_box(1, Z, "vz", convention="bogus")


def test_per_call_override_and_wrapcycle_single_code_path(tmp_path, monkeypatch):
    """wrapcycle passes its own convention per call, independent of the Stitcher default
    (no double application): a 'physical' Stitcher + wrapcycle 'legacy' gives raw fields."""
    monkeypatch.delenv(stitch.FIELD_CONVENTION_ENV, raising=False)
    st = _st(tmp_path, field_convention="physical")
    raw = st.load_field_box(1, Z, "vz", convention="legacy")
    np.testing.assert_array_equal(raw, _ic(tmp_path, "vz"))
    lg = logging.getLogger("t_conv")
    cfg = _cfg(tmp_path)
    fd_phys = build_wrapcycle_field_data(cfg, 1, 7, st, lg, norm=normalisation_keys(cfg), field_convention="physical")
    fd_leg = build_wrapcycle_field_data(cfg, 1, 7, st, lg, norm=normalisation_keys(cfg), field_convention="legacy")
    # per-pixel ratios are ill-defined where interpolated snapshots cancel; the rms ratio must sit
    # inside (1/(1+z_hi), 1/(1+z_lo)) of the lightcone range z=6-7.4 (-> ~0.12-0.14)
    rms = np.sqrt(np.mean(fd_phys["velocity_lc"] ** 2) / np.mean(fd_leg["velocity_lc"] ** 2))
    assert 1 / (1 + 7.5) * 0.97 < rms < 1 / (1 + 6.0) * 1.03
    assert fd_phys["field_convention"] == "physical" and fd_leg["field_convention"] == "legacy"

