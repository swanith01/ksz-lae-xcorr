"""
Tests for correlation/coeval_cross.py and scripts/44 (+ scripts/41 end to end) on a tiny synthetic coeval tree.
"""
import importlib.util
import logging
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ksz_lae_xcorr.correlation import coeval_cross as cc
from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.utils.config import Config


def test_pixel_assignment_and_groups():
    snap = [6.0, 6.2, 6.4, 6.8]
    z = np.array([5.95, 6.05, 6.09, 6.11, 6.29, 6.31, 6.5, 6.59, 6.61, 7.0])
    a = cc.pixel_assignment(z, snap)
    assert list(a) == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]          # nearest snapshot, midpoints at 6.1, 6.3, 6.6
    g = cc.window_groups(a, 2, 9)
    assert [(s, list(i)) for s, i in g] == [(0, [2]), (1, [3, 4]), (2, [5, 6, 7]), (3, [8])]
    zp = np.linspace(6.0, 7.0, 101)
    w = cc.window_pixel_weights(zp, 10, 60)
    assert w.sum() == pytest.approx(1.0) and np.allclose(w, w[0])
    assert cc.slab_offsets(50, 300) == [0, 50, 100, 150, 200, 250]
    assert cc.slab_offsets(400, 300) == [0] and cc.slab_offsets(300, 300) == [0]


def _setup(n=32, box=64.0):
    cfg = Config({"box": Config({"box_len_mpc": box, "hii_dim": n, "dim": 2 * n}), "correlation": Config({"n_kbins": 8})})
    kg = KGrid(cfg)
    ell = np.geomspace(100, 10000, 60)
    fl = np.exp(-0.5 * ((np.log(ell) - np.log(3000)) / 0.8) ** 2)
    return cfg, kg, kc.filter_grid(kg, ell, fl, 9000.0), ell, fl


def test_snapshot_group_cross_equals_direct_formula():
    n = 32
    cfg, kg, f2d, ell, fl = _setup(n)
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(n, n, n)).astype(np.float32)
    gal = {"a": rng.normal(size=(n, n, n)).astype(np.float32), "b": rng.normal(size=(n, n, n)).astype(np.float32)}
    npix = 8
    W = rng.uniform(0.5, 1.5, npix)
    wg = {"a": np.full(npix, 1 / npix), "b": rng.uniform(0, 1, npix)}
    res = cc.snapshot_group_cross(theta, gal, W, wg, f2d, kg, 9000.0)
    offs = cc.slab_offsets(npix, n)
    assert res["n_offsets"] == len(offs) == 4
    for name in gal:
        acc = 0
        for o in offs:
            Theta = np.einsum("xyk,k->xy", theta[:, :, o:o + npix].astype(np.float64), W)
            S = lp_maps.filter_and_square(Theta, ell, fl, kg, 9000.0)
            g = np.einsum("xyk,k->xy", gal[name][:, :, o:o + npix].astype(np.float64), wg[name])
            acc = acc + lp_maps.lp_cross_dell(S, g, kg, 9000.0)["D_ell"] / len(offs)
        np.testing.assert_allclose(res["D"][name], acc, rtol=1e-4, atol=1e-30, equal_nan=True)
    # a slab longer than the box wraps instead of failing
    res2 = cc.snapshot_group_cross(theta, {"a": gal["a"]}, np.ones(40), {"a": np.full(40, 1 / 40)}, f2d, kg, 9000.0)
    assert res2["n_offsets"] == 1 and np.isfinite(res2["D"]["a"]).any()


def test_coeval_equals_lightcone_estimator_for_a_single_slab():
    """If the kSZ lightcone map is ONE slab's projection and the galaxy map is the same slab, both estimators coincide."""
    n = 32
    cfg, kg, f2d, ell, fl = _setup(n)
    rng = np.random.default_rng(4)
    theta = rng.normal(size=(n, n, n)).astype(np.float32)
    gal = rng.normal(size=(n, n, n)).astype(np.float32)
    W = rng.uniform(0.5, 1.5, n)
    wg = np.full(n, 1 / n)
    res = cc.snapshot_group_cross(theta, {"g": gal}, W, {"g": wg}, f2d, kg, 9000.0)       # slab = the whole box -> one offset
    Theta = np.einsum("xyk,k->xy", theta.astype(np.float64), W)
    ref = lp_maps.lp_cross_dell(lp_maps.filter_and_square(Theta, ell, fl, kg, 9000.0),
                                np.einsum("xyk,k->xy", gal.astype(np.float64), wg), kg, 9000.0)["D_ell"]
    np.testing.assert_allclose(res["D"]["g"], ref, rtol=1e-4, atol=1e-30, equal_nan=True)


def test_planted_modulation_is_recovered_by_the_coeval_estimator():
    """Small-scale velocity power modulated by a soft tracer field -> positive D, and ~ twice the bias scaling."""
    n = 64
    cfg, kg, f2d, ell, fl = _setup(n, 128.0)
    rng = np.random.default_rng(8)
    kx = np.fft.fftfreq(n)[:, None]; ky = np.fft.fftfreq(n)[None, :]
    s2 = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-(kx ** 2 + ky ** 2) / 0.004)))
    s2 /= s2.std()
    gal = np.repeat(s2[:, :, None], n, axis=2).astype(np.float32)                     # soft tracer, same in every cell
    ds = []
    for amp in (0.3, 0.6):
        theta = ((1.0 + amp * s2[:, :, None]) * rng.normal(size=(n, n, n))).astype(np.float32)
        d = cc.snapshot_group_cross(theta, {"g": gal}, np.ones(n), {"g": np.full(n, 1 / n)}, f2d, kg, 9000.0)["D"]["g"]
        ds.append(np.nanmean(d[np.isfinite(d)][:4]))
    assert ds[0] > 0 and ds[1] > 1.5 * ds[0], ds


# ----------------------------------------------------------------------------------------------------------------------
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_scripts_41_and_44_end_to_end_on_a_tiny_tree(tmp_path, monkeypatch):
    from test_ksz_auto_wrapcycle import _cfg as wc_cfg, _write_tree
    from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import build_wrapcycle_field_data, normalisation_keys
    from ksz_lae_xcorr.lightcone.stitch import Stitcher

    for sd in (1, 2):
        _write_tree(tmp_path, seed=sd)
    cfg = wc_cfg(tmp_path)
    cfg.snr = Config({"ell_min": 100, "ell_max": 10000, "n_ell": 60, "camb_lmax": 12000, "f_sky": 0.053,
                      "experiments": Config({"SO": Config({"delta_n_uk_arcmin": 10.0, "theta_fwhm_arcmin": 1.4})})})
    cfg.box.box_len_mpc = 64.0                              # a box big enough for ell ~ 500-3000 modes (tree is 16^3 cells -> 4 Mpc cells)
    prod_dir = tmp_path / "prod" / "ksz_auto_wrapcycle"
    prod_dir.mkdir(parents=True)
    for sd in (1, 2):
        fd = build_wrapcycle_field_data(cfg, sd, sd, Stitcher(cfg), logging.getLogger("t44"), norm=normalisation_keys(cfg))
        prod = lp_maps.build_lp_products(cfg, fd, z0_grid=(6.5, 7.0), dz_list=(1.0,), kSZ_z_min=6.0)
        with open(prod_dir / f"seed{sd}_wo0_lpmaps.pkl", "wb") as f:
            pickle.dump(prod, f)
    ell_t = np.geomspace(50, 2e4, 40)
    np.savetxt(tmp_path / "cltt.csv", np.column_stack([ell_t, 3e3 / (ell_t * (ell_t + 1)) * 2 * np.pi]), delimiter=",",
               header="ell,Cl_uK2", comments="")

    m41 = _load("s41", "41_ksz_component_maps.py")
    monkeypatch.setattr(m41, "load_config", lambda p: cfg)
    m41.main(["--seed", "1", "--cltt-file", str(tmp_path / "cltt.csv"), "--out-dir", str(tmp_path / "comp")])
    with open(tmp_path / "comp" / "seed1_wo0_components.pkl", "rb") as f:
        comp = pickle.load(f)
    assert comp["check"]["ksz_map_rel"] < 1e-3 and comp["check"]["A_full_rel"] < 1e-3
    assert set(kc.VARIANTS) == set(comp["maps"])

    m44 = _load("s44", "44_coeval_cross.py")
    monkeypatch.setattr(m44, "load_config", lambda p: cfg)
    m44.main(["--seed", "1", "--specs", "none", "--cltt-file", str(tmp_path / "cltt.csv"), "--out-dir", str(tmp_path / "coeval")])
    with open(tmp_path / "coeval" / "seed1_wo0_coeval.pkl", "rb") as f:
        out = pickle.load(f)
    assert out["names"] == ["matter_Waters"]
    Dc, Dl = out["D_coeval"]["matter_Waters"], out["D_lightcone"]["matter_Waters"]
    assert Dc.shape == Dl.shape == (len(out["windows"]), 3)
    assert np.isfinite(Dl).any() and np.isfinite(Dc).any()
    assert all(0 <= w["x_hii_coeval"] <= 1 for w in out["windows"])
    # the x_HII of the window agrees between the two constructions (interpolated snapshot means vs lightcone pixels)
    for w in out["windows"]:
        assert abs(w["x_hii_coeval"] - w["x_hii_lightcone"]) < 0.05


def test_script45_aggregates_and_overlays(tmp_path, capsys):
    rng = np.random.default_rng(3)
    names = ["matter_Waters", "halo_M10.0", "lae_L42.5", "lbg_MUV-19.0"]
    z0 = np.arange(6.5, 10.01, 0.5)
    x = np.linspace(0.9, 0.1, len(z0))
    base = {n: 0.02 * (1 + i) * np.exp(-((x - 0.4) / 0.3) ** 2) for i, n in enumerate(names)}
    for seed in range(1, 6):
        d = {"seed": seed, "ell_targets": np.array([500.0, 1000.0, 3000.0]), "names": names, "chi_ref": 9000.0,
             "windows": [{"z0": float(a), "dz": 1.0, "z_lo": a - 0.5, "z_hi": a + 0.5, "chi_eff": 9000.0,
                          "x_hii_coeval": float(b), "x_hii_lightcone": float(b)} for a, b in zip(z0, x)],
             "D_coeval": {}, "D_lightcone": {}}
        for n in names:
            lc = base[n][:, None] * np.array([1.0, 0.8, 0.5])[None, :] + rng.normal(0, 0.003, (len(z0), 3))
            d["D_lightcone"][n] = lc
            d["D_coeval"][n] = lc * 0.9 + rng.normal(0, 0.003, (len(z0), 3))        # coeval = 90% of the lightcone
        with open(tmp_path / f"seed{seed}_wo0_coeval.pkl", "wb") as f:
            pickle.dump(d, f)
    m45 = _load("s45", "45_coeval_vs_lightcone.py")
    out = tmp_path / "plots"
    m45.main(["--coeval-dir", str(tmp_path), "--out-dir", str(out), "--xrange", "0.1", "0.9"])
    assert (out / "coeval_vs_lightcone.png").exists() and (out / "coeval_matter_Waters.csv").exists()
    rows = np.genfromtxt(out / "coeval_vs_lightcone_summary.csv", delimiter=",", names=True, dtype=None, encoding=None)
    r = [q for q in rows if q["tracer"] == "halo_M10.0" and q["ell"] == 500][0]
    assert 0.8 < r["ratio_coeval_over_lightcone"] < 1.0 and r["lo16"] <= r["ratio_coeval_over_lightcone"] <= r["hi84"]
    assert "matter_Waters" in capsys.readouterr().out
