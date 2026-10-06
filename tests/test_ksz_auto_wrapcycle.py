"""
End-to-end test of correlation/ksz_auto_wrapcycle.py on a tiny synthetic
coeval tree written to disk in the layout lightcone.stitch.Stitcher reads
(coeval_z{z:.6f}/{neutral_fraction,hires_density,velocity_z}.npy; density on
the 2x finer DIM grid).  No cluster data, no py21cmfast.
"""

import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import (
    build_wrapcycle_field_data,
    compute_wrapcycle_auto_power,
    normalisation_keys,
    summarise_at_ell,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher
from ksz_lae_xcorr.utils.config import Config

N, DIM = 16, 32
Z_MIN, Z_MAX = 6.0, 7.4


def _cfg(root):
    return Config({
        "box": Config({"box_len_mpc": 16.0, "hii_dim": N, "dim": DIM,
                       "z_min": Z_MIN, "z_max": Z_MAX, "seeds": [1]}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "lightcone": Config({"n_lc_pix": 64, "angle_deg": 10.0,
                             "fields": Config({"continuous": ["xH", "density", "vz"], "discrete": []})}),
        "tracers": Config({"halo_mass_cut_msun": 1e10, "lae_lbg_mass_cut_msun": 3e9, "lbg_muv_cut": -17.0}),
        "correlation": Config({"n_kbins": 6, "tracers": []}),
        "paths": Config({"coeval_root": str(root), "halo_root": str(root), "lae_catalogue_root": str(root),
                         "lbg_catalogue_root": str(root), "lightcone_root": str(root / "lc"),
                         "products_root": str(root / "prod")}),
    })


def _write_tree(root, vz_scale=1.0, seed=1):
    rng = np.random.default_rng(42)
    for z in np.arange(5.6, 8.01, 0.2):
        d = root / f"seed_{seed}" / f"coeval_z{z:.6f}"
        d.mkdir(parents=True)
        xh_mean = np.clip(0.1 + 0.8 * (z - Z_MIN) / (Z_MAX - Z_MIN), 0.1, 0.9)
        np.save(d / "neutral_fraction.npy",
                np.clip(xh_mean + 0.05 * rng.normal(size=(N,) * 3), 0, 1).astype(np.float32))
        np.save(d / "hires_density.npy", (0.3 * rng.normal(size=(DIM,) * 3)).astype(np.float32))
        np.save(d / "velocity_z.npy", (vz_scale * 1e-16 * rng.normal(size=(N,) * 3)).astype(np.float32))


def _run(root, wrap_seed=7, **norm_kw):
    cfg = _cfg(root)
    st = Stitcher(cfg)
    lg = logging.getLogger("t_wc")
    fd = build_wrapcycle_field_data(cfg, 1, wrap_seed, st, lg, norm=normalisation_keys(cfg, **norm_kw))
    return cfg, fd, compute_wrapcycle_auto_power(cfg, fd)


def test_end_to_end_shapes_sum_rule_and_density_convention(tmp_path):
    _write_tree(tmp_path)
    cfg, fd, res = _run(tmp_path)

    # uniform 1-Mpc LOS spacing, several wrap cycles, density is 1+delta ONCE
    assert fd["xHI_lc"].shape[:2] == (N, N) and fd["xHI_lc"].shape[2] == len(fd["z_lc"])
    assert fd["n_cycles"] >= 3
    assert abs(fd["density_lc"].mean() - 1.0) < 0.05

    for key in ("patchy", "full"):
        r = res[key]
        assert np.all(np.isfinite(r["D_total"])) and np.all(np.isfinite(r["D_diag"]))
        np.testing.assert_allclose(r["D_total"], r["D_diag"] + r["D_off"], rtol=1e-6, atol=1e-30)
        assert np.all(r["D_diag"] >= 0)
    assert res["patchy"]["z_lo"] >= 6.0
    s = summarise_at_ell(res["patchy"], 3000.0)
    assert set(s) == {"ell", "D_total", "D_diag", "D_off", "D_off_over_total"}


def test_D_scales_as_velocity_squared_no_hidden_unit_conversion(tmp_path):
    """x10 on v_z must give exactly x100 on D (D ~ v^2): proves the pipeline
    applies NO velocity rescaling (v4 velocity_z is already Mpc/s)."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_tree(a, vz_scale=1.0)
    _write_tree(b, vz_scale=10.0)
    _, _, r1 = _run(a)
    _, _, r2 = _run(b)
    np.testing.assert_allclose(r2["patchy"]["D_total"], 100.0 * r1["patchy"]["D_total"], rtol=1e-4)
    np.testing.assert_allclose(r2["patchy"]["D_diag"], 100.0 * r1["patchy"]["D_diag"], rtol=1e-4)


def test_wrap_seed_changes_result_and_same_seed_reproduces(tmp_path):
    _write_tree(tmp_path)
    _, _, r_a = _run(tmp_path, wrap_seed=7)
    _, _, r_a2 = _run(tmp_path, wrap_seed=7)
    _, _, r_b = _run(tmp_path, wrap_seed=8)
    np.testing.assert_array_equal(r_a["patchy"]["D_total"], r_a2["patchy"]["D_total"])
    assert not np.allclose(r_a["patchy"]["D_total"], r_b["patchy"]["D_total"], rtol=1e-3)
    # the incoherent part is rotation-invariant to first order; only P_off is
    # sensitive to the wrap seed
    np.testing.assert_allclose(r_a["patchy"]["D_diag"], r_b["patchy"]["D_diag"], rtol=0.3)


def test_normalisation_conventions(tmp_path):
    cfg = _cfg(tmp_path)
    leg = normalisation_keys(cfg, ne_convention="legacy", tau0_mode="none")
    assert leg == {"ne_scale": 1.0, "tau0": 0.0}
    new = normalisation_keys(cfg)
    assert new["ne_scale"] == pytest.approx(0.82)
    assert 0.02 < new["tau0"] < 0.05
    with pytest.raises(ValueError):
        normalisation_keys(cfg, ne_convention="bogus")
