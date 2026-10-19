"""
Tests for correlation/ksz_components.py and scripts/41-43 (component decomposition of kSZ^2 x tracer) on synthetic fields.
"""
import importlib.util
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.coherence_decomposition import compute_ksz_slices
from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import Config

N = 32
BOX = 64.0


def _cfg():
    return Config({
        "box": Config({"box_len_mpc": BOX, "hii_dim": N, "dim": 2 * N, "z_min": 5.0, "z_max": 9.0, "seeds": [1]}),
        "cosmology": Config({"H0": 67.77, "Om0": 0.3086, "Ob0": 0.0489, "ns": 0.9665}),
        "correlation": Config({"n_kbins": 8}),
        "snr": Config({"ell_min": 100, "ell_max": 10000, "n_ell": 60, "camb_lmax": 12000, "f_sky": 0.053,
                       "experiments": Config({"SO": Config({"delta_n_uk_arcmin": 10.0, "theta_fwhm_arcmin": 1.4})})}),
    })


def _fd(seed=0, nz=80):
    rng = np.random.default_rng(seed)
    z = np.linspace(5.0, 9.0, nz)
    xhi = np.clip(np.linspace(0.0, 1.0, nz)[None, None, :] + 0.05 * rng.normal(size=(N, N, nz)), 0, 1)
    return {"z_lc": z, "xHI_lc": xhi, "density_lc": 1.0 + 0.3 * rng.normal(size=(N, N, nz)),
            "velocity_lc": 1e-16 * rng.normal(size=(N, N, nz)), "ne_scale": 0.82, "tau0": 0.03}


def _filter(cfg):
    kg = KGrid(cfg)
    ell = np.geomspace(100, 10000, 60)
    fl = np.exp(-0.5 * ((np.log(ell) - np.log(3000)) / 0.8) ** 2)
    return kg, ell, fl


def test_weights_reproduce_compute_ksz_slices():
    cfg, fd = _cfg(), _fd()
    theta, chi, z = compute_ksz_slices(cfg, fd)
    wt = kc.ksz_slice_weights(cfg, fd)
    np.testing.assert_allclose(wt["z"], z)
    np.testing.assert_allclose(wt["chi"], chi)
    sl = slice(wt["zi_lo"], wt["zi_hi"])
    xe = 1.0 - fd["xHI_lc"][:, :, sl]
    mine = -wt["W"][None, None, :] * fd["density_lc"][:, :, sl] * xe * fd["velocity_lc"][:, :, sl] / constants.c_mpc_per_s()
    np.testing.assert_allclose(mine, theta, rtol=1e-10, atol=1e-30)


def test_rfft_filter_equals_filter_and_square():
    cfg = _cfg()
    kg, ell, fl = _filter(cfg)
    rng = np.random.default_rng(3)
    m = rng.normal(size=(N, N))
    chi = 9000.0
    ref = lp_maps.filter_and_square(m, ell, fl, kg, chi)
    f2d = kc.filter_grid(kg, ell, fl, chi)
    np.testing.assert_allclose(kc.apply_filter(m, f2d) ** 2, ref, rtol=1e-9, atol=1e-12)
    stack = rng.normal(size=(N, N, 3))
    out = kc.apply_filter(stack, f2d)
    np.testing.assert_allclose(out[:, :, 1] ** 2, lp_maps.filter_and_square(stack[:, :, 1], ell, fl, kg, chi), rtol=1e-9, atol=1e-12)


def test_component_maps_consistency():
    cfg, fd = _cfg(), _fd()
    kg, ell, fl = _filter(cfg)
    chi = 9000.0
    f2d = kc.filter_grid(kg, ell, fl, chi)
    theta, chi_mpc, z = compute_ksz_slices(cfg, fd)
    maps = kc.component_maps(cfg, fd, f2d, kSZ_z_min=6.0, chunk=7)
    kmap = lp_maps.ksz_map_from_slices(theta, z, 6.0)
    np.testing.assert_allclose(maps["_ksz_map"], kmap, rtol=1e-4, atol=1e-30)
    ref = lp_maps.filter_and_square(kmap, ell, fl, kg, chi)
    np.testing.assert_allclose(maps["A_full"], ref, rtol=2e-4, atol=1e-6 * ref.max())
    assert set(kc.VARIANTS) <= set(maps)
    # white-in-z fields: no cross-slice correlation, so project-then-square and square-then-project have the same mean
    for c in ("v", "full"):
        assert abs(maps[f"A_{c}"].mean() / maps[f"B_{c}"].mean() - 1.0) < 0.3
    # a single contributing pixel: the two orderings coincide exactly
    one = kc.component_maps(cfg, fd, f2d, kSZ_z_min=float(z[-1]) - 1e-9)
    for c in kc.COMPONENTS:
        np.testing.assert_allclose(one[f"A_{c}"], one[f"B_{c}"], rtol=1e-4, atol=1e-30)


def test_ionisation_modulation_is_picked_up_by_x_but_not_v():
    """If the small-scale power of x_e is modulated by a tracer field and v is not, only x correlates with the tracer."""
    cfg = _cfg()
    kg, ell, fl = _filter(cfg)
    rng = np.random.default_rng(11)
    nz = 40
    z = np.linspace(6.0, 7.0, nz)
    cfgb = Config({**cfg.__dict__}) if False else cfg
    # soft modulating field s(x, y), smooth: a few low-k modes
    kx = np.fft.fftfreq(N)[:, None]; ky = np.fft.fftfreq(N)[None, :]
    s = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(N, N))) * np.exp(-(kx ** 2 + ky ** 2) / 0.01)))
    s = s / s.std()
    xe = 0.5 + 0.2 * (1 + 0.8 * s[:, :, None]) * rng.normal(size=(N, N, nz))        # small-scale x_e power follows s
    fd = {"z_lc": z, "xHI_lc": 1.0 - xe, "density_lc": np.ones((N, N, nz)), "velocity_lc": 1e-16 * rng.normal(size=(N, N, nz)),
          "ne_scale": 1.0, "tau0": 0.0}
    cfg2 = _cfg(); cfg2.box.z_min, cfg2.box.z_max = 6.0, 7.5
    f2d = kc.filter_grid(kg, ell, fl, 9000.0)
    maps = kc.component_maps(cfg2, fd, f2d, kSZ_z_min=6.0)
    rx = abs(kc.map_correlation(maps["B_x"], s)); rv = abs(kc.map_correlation(maps["B_v"], s))
    assert rx > 0.4 and rv < 0.2, (rx, rv)


def test_frac_cross_is_dimensionless_cross_power():
    cfg = _cfg()
    kg = KGrid(cfg)
    rng = np.random.default_rng(5)
    s = rng.normal(size=(N, N)); s -= s.mean()
    S = 3.0 * (1.0 + 0.1 * s)
    c = kc.frac_cross_dell(S, s, kg, 9000.0)
    Sf = S / S.mean() - 1.0
    P, _, _ = cross_power_2d(Sf, s - s.mean(), kg)
    ell = c["ell"]
    np.testing.assert_allclose(c["D_ell"], ell * (ell + 1) * P / 9000.0 ** 2 / (2 * np.pi), rtol=1e-10, equal_nan=True)
    # invariant to the overall amplitude of S
    c2 = kc.frac_cross_dell(1e6 * S, s, kg, 9000.0)
    np.testing.assert_allclose(c["D_ell"], c2["D_ell"], rtol=1e-8, equal_nan=True)
    assert np.all(np.isnan(kc.frac_cross_dell(np.zeros((N, N)), s, kg, 9000.0)["D_ell"]))


def test_fraction_of_full_bootstrap():
    rng = np.random.default_rng(1)
    df = 1.0 + 0.1 * rng.normal(size=(10, 6))
    x = np.linspace(0.9, 0.05, 6)
    r = kc.fraction_of_full(0.5 * df, df, x, (0.1, 0.7))
    assert r["R"] == pytest.approx(0.5) and r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5)
    r2 = kc.fraction_of_full(0.5 * df + 0.05 * rng.normal(size=df.shape), df, x, (0.1, 0.7))
    assert r2["lo"] < r2["R"] < r2["hi"] and abs(r2["R"] - 0.5) < 0.1
    assert np.isnan(kc.fraction_of_full(df, df, x, (2.0, 3.0))["R"])


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_scripts_42_and_43_on_synthetic_pickles(tmp_path, capsys):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "variants", "quicktest.yaml")))
    cfgp = tmp_path / "cfg.yaml"; cfgp.write_text(yaml.safe_dump(cfg))
    from ksz_lae_xcorr.utils.config import load_config
    kg = KGrid(load_config(str(cfgp)))
    n = kg.n_side
    rng = np.random.default_rng(2)
    comp_dir, maps_root = tmp_path / "comp", tmp_path / "maps"
    names = ("halo_M10.0", "halo_M11.0", "halo_M12.0", "lae_L42.5", "lbg_MUV-19.0")
    z0s = [6.5, 7.0, 7.5, 8.0]
    for seed in (1, 2, 3):
        comp_dir.mkdir(exist_ok=True)
        maps = {v: (1.0 + 0.3 * rng.normal(size=(n, n))).astype(np.float32) ** 2 for v in kc.VARIANTS}
        with open(comp_dir / f"seed{seed}_wo0_components.pkl", "wb") as f:
            pickle.dump({"maps": maps, "seed": seed, "chi_ref": 9000.0, "rho_with_A_full": {v: 0.5 for v in kc.VARIANTS},
                         "check": {}, "kSZ_z_min": 6.0}, f)
        for nm in names:
            (maps_root / nm).mkdir(parents=True, exist_ok=True)
            wins = [{"z0": z0, "dz": 1.0, "z_lo": z0 - 0.5, "z_hi": z0 + 0.5, "chi_eff": 9000.0, "x_hii": 1.0 - 0.1 * (z0 - 6),
                     "gal_map": rng.normal(size=(n, n)).astype(np.float32)} for z0 in z0s]
            with open(maps_root / nm / f"seed{seed}_wo0_lpmaps.pkl", "wb") as f:
                pickle.dump({"windows": wins, "kSZ_map": maps["A_full"], "chi_eff_kSZ": 9000.0}, f)
    m42 = _load("s42", "42_ksz_component_cross.py")
    out = tmp_path / "out"
    m42.main(["--config", str(cfgp), "--maps-root", str(maps_root), "--comp-dir", str(comp_dir), "--base-dir", "none", "--out-dir", str(out)])
    z = np.load(out / "components_halo_M10.0.npz")
    assert z["D"].shape == (3, len(kc.VARIANTS), len(z0s), 3) and list(z["variants"]) == list(kc.VARIANTS)
    assert (out / "components_halo_M10.0.csv").exists() and (out / "components_rho.csv").exists()
    m43 = _load("s43", "43_ksz_component_plots.py")
    m43.main(["--root", str(out), "--ell", "1000", "--xrange", "0.0", "1.0"])
    assert (out / "components_halo_l1000.png").exists() and (out / "components_fraction_l1000.png").exists()
    assert (out / "components_fraction_l1000.csv").exists()
    assert "halo_M10.0" in capsys.readouterr().out


def test_script41_compiles():
    compile(open(os.path.join(ROOT, "scripts", "41_ksz_component_maps.py")).read(), "41", "exec")
