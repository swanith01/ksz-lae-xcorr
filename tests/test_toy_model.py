"""Tests for correlation/toy_model.py and scripts/50_toy_model.py."""
import csv
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from ksz_lae_xcorr.correlation import toy_model as tm
from ksz_lae_xcorr.utils.config import load_config

CFG = load_config(os.path.join(ROOT, "configs", "fiducial.yaml"))
Z = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
X = np.array([0.95, 0.8, 0.45, 0.23, 0.12, 0.07, 0.04])


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_history_is_monotone_bounded_and_decays_above_the_table():
    f = tm.history_fn(Z, X[::-1] * 0 + np.array([0.04, 0.07, 0.3, 0.2, 0.45, 0.8, 0.95]))        # noisy, non-monotonic input
    zz = np.linspace(3, 20, 200)
    x = f(zz)
    assert np.all(np.diff(x) <= 1e-12) and x.min() >= 0 and x.max() <= 1
    assert f(2.0) == 1.0 and 0 < f(14.0) < f(12.0)


def test_pieces_follow_the_two_limits():
    xb = tm.history_fn(Z, X)
    hom = tm.Toy(CFG, xb, eps=0.0)
    s, rd, ri = hom.pieces(0.1)
    assert np.allclose(rd, 2.0 * s)                                   # density response of a homogeneous slice is 2 s
    pat = tm.Toy(CFG, xb, eps=50.0)
    c = pat.cross(8.0, 1.0, 1000.0)
    assert c["D_ion"] > 0 and abs(c["D"] - c["D_dens"] - c["D_ion"]) < 1e-12 * abs(c["D"])
    assert 0.2 < c["x_hii"] < 0.7


def test_patchy_toy_falls_at_late_times_homogeneous_does_not():
    xb = tm.history_fn(Z, X)
    z0 = np.arange(6.5, 13.1, 0.5)
    for eps, late_falls in ((0.0, False), (10.0, True)):
        c = tm.Toy(CFG, xb, eps=eps).curve(z0, 1.0, 1000.0)
        o = np.argsort(c["x_hii"])
        x, d = c["x_hii"][o], c["D"][o]
        late, mid = d[x > 0.8].mean(), d[(x > 0.15) & (x < 0.35)].mean()
        assert (late < 0.3 * mid) == late_falls


def test_fit_amplitude():
    m = np.array([1.0, 2.0, 3.0])
    assert abs(tm.fit_amplitude(m, 4.0 * m) - 4.0) < 1e-12


def test_script50_on_a_synthetic_measurement(tmp_path):
    s50 = _load("s50", "50_toy_model.py")
    xb = tm.history_fn(Z, X)
    z0 = np.linspace(13, 6.5, 12)
    toy = tm.Toy(CFG, tm.history_fn(z0, xb(z0)), eps=10.0)
    c = toy.curve(z0, 1.0, 1000.0)
    d = tmp_path / "ours" / "halo_M10.0"
    os.makedirs(d)
    cols = {"z0": z0, "dz": np.ones(12), "x_hii": xb(z0), "mean_l1000": 3.0 * c["D"], "sigma_mean_l1000": 0.05 * 3.0 * c["D"] + 1e-9}
    with open(d / s50.TAG, "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(cols)); w.writerows(zip(*cols.values()))
    out = tmp_path / "out"
    s50.main(["--ours-root", str(tmp_path / "ours"), "--out-dir", str(out), "--eps", "10"])
    assert (out / "toy_vs_laplante.png").exists() and (out / "toy_vs_ours.png").exists()
