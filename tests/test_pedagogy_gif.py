"""Tests for scripts/51_pedagogy_gif.py on synthetic maps."""
import importlib.util
import os
import pickle
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ksz_lae_xcorr.utils.config import Config


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S51 = _load("s51", "51_pedagogy_gif.py")


def _blob(n, r0, c0, w=3.0):
    y, x = np.mgrid[0:n, 0:n]
    dy, dx = (y - r0 + n / 2) % n - n / 2, (x - c0 + n / 2) % n - n / 2
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * w ** 2))


def test_local_product_mean_is_pearson_and_arrow_finds_the_shared_blob():
    n = 64
    rng = np.random.default_rng(0)
    shared = _blob(n, 40, 20, 4.0)
    a = S51.smooth_zero_mean(rng.normal(0, 0.1, (n, n)) + 3 * shared, 1.5)
    b = S51.smooth_zero_mean(rng.normal(0, 0.1, (n, n)) + 3 * shared, 1.5)
    p = S51.local_product(a, b)
    assert abs(p.mean() - np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 1e-9
    r, c = S51.arrow_pixel(p, 2.0)
    assert abs(r - 40) <= 3 and abs(c - 20) <= 3
    assert np.all(S51.local_product(np.zeros((4, 4)), np.ones((4, 4))) == 0)


def test_pick_peak_respects_the_x_range():
    x = [0.02, 0.2, 0.5, 0.95]
    D = [9.0, 1.0, 3.0, 8.0]
    assert S51.pick_peak(x, D, (0.05, 0.9)) == 2
    assert S51.pick_peak(x, D, (0.0, 1.0)) == 0
    assert S51.pick_peak(x, [np.nan] * 4, (0.0, 1.0)) is None


def test_script51_end_to_end_from_a_chunks_pickle(tmp_path, monkeypatch):
    n, box = 32, 64.0
    cfg = Config({"box": Config({"box_len_mpc": box, "hii_dim": n, "dim": 2 * n}), "correlation": Config({"n_kbins": 8})})
    rng = np.random.default_rng(1)
    S = 1.0 + 0.3 * rng.normal(size=(n, n)) ** 2 + 2.0 * _blob(n, 10, 22, 2.5)
    z0s = [12.0, 10.0, 8.0, 7.0]
    xs = [0.03, 0.12, 0.45, 0.8]
    gains = [0.0, 1.0, 3.0, 0.5]                                  # the tracer follows the kSZ^2 blob most strongly at z0=8
    gal = [rng.normal(size=(n, n)) * 0.2 + g * _blob(n, 10, 22, 2.5) for g in gains]
    d = {"seed": 1, "chi_ref": 9000.0, "S_global": S.astype(np.float32),
         "windows": [{"z0": z, "dz": 1.0, "z_lo": z - 0.5, "z_hi": z + 0.5, "chi_eff": 9000.0, "x_hii": x, "maps": {}, "n_pix": 10}
                     for z, x in zip(z0s, xs)],
         "gal": {"lae_L42.5": gal}}
    pk = tmp_path / "seed1_wo0_chunks.pkl"
    with open(pk, "wb") as f:
        pickle.dump(d, f)
    monkeypatch.setattr(S51, "load_config", lambda p: cfg)
    out = tmp_path / "out"
    S51.main(["--chunks", str(pk), "--out-dir", str(out), "--ell", "1500", "--flashes", "1", "--fps", "50"])
    gif = out / "cross_correlation_lae_L42.5.gif"
    assert gif.exists()
    im = Image.open(gif)
    assert im.n_frames == 3 + 2 + 1 and im.size[0] > 1000                 # three panels now                  # 3 ordinary frames + (2 flash + 1 hold) for the peak window
    rows = np.genfromtxt(out / "cross_correlation_lae_L42.5_windows.csv", delimiter=",", names=True)
    peak = rows[rows["is_peak"] == 1]
    assert len(peak) == 1
    assert abs(peak["arrow_row"][0] - 10) <= 4 and abs(peak["arrow_col"][0] - 22) <= 4


def test_two_panel_option_and_curve_panel_width(tmp_path, monkeypatch):
    n, box = 32, 64.0
    cfg = Config({"box": Config({"box_len_mpc": box, "hii_dim": n, "dim": 2 * n}), "correlation": Config({"n_kbins": 8})})
    rng = np.random.default_rng(2)
    S = 1.0 + 0.3 * rng.normal(size=(n, n)) ** 2
    d = {"seed": 1, "chi_ref": 9000.0, "S_global": S.astype(np.float32),
         "windows": [{"z0": z, "dz": 1.0, "z_lo": z - 0.5, "z_hi": z + 0.5, "chi_eff": 9000.0, "x_hii": x, "maps": {}, "n_pix": 10}
                     for z, x in ((11.0, 0.05), (9.0, 0.3))],
         "gal": {"lae_L42.5": [rng.normal(size=(n, n)), rng.normal(size=(n, n))]}}
    pk = tmp_path / "c.pkl"
    with open(pk, "wb") as f:
        pickle.dump(d, f)
    monkeypatch.setattr(S51, "load_config", lambda p: cfg)
    widths = {}
    for tag, extra in (("three", []), ("two", ["--no-curve"])):
        S51.main(["--chunks", str(pk), "--out-dir", str(tmp_path / tag), "--ell", "1500", "--flashes", "1"] + extra)
        widths[tag] = Image.open(tmp_path / tag / "cross_correlation_lae_L42.5.gif").size[0]
    assert widths["three"] > 1.4 * widths["two"]
