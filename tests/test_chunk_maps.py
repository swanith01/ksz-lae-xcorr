"""
Tests for ksz_components.window_chunk_maps and scripts/48-49 (kSZ^2 chunk maps next to tracer maps) on synthetic fields.
"""
import importlib.util
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from ksz_lae_xcorr.correlation import ksz_components as kc


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("t_ksz_components", os.path.join(ROOT, "tests", "test_ksz_components.py"))


def test_chunk_maps_add_up_to_the_whole_lightcone():
    cfg, fd = T._cfg(), T._fd()
    kg, ell, fl = T._filter(cfg)
    f2d = kc.filter_grid(kg, ell, fl, 9000.0)
    full = kc.component_maps(cfg, fd, f2d, kSZ_z_min=6.0, chunk=7)
    z = fd["z_lc"]
    whole = {"z_lo": 6.0, "z_hi": 9.0}
    mid = float(z[len(z) // 2])
    lo_half, hi_half = {"z_lo": 6.0, "z_hi": mid}, {"z_lo": mid, "z_hi": 9.0}
    cw, c1, c2 = kc.window_chunk_maps(cfg, fd, f2d, [whole, lo_half, hi_half], kSZ_z_min=6.0, chunk=5)
    # the window covering everything is the whole-lightcone map (A and B, every ingredient)
    for c in ("full", "x", "v"):
        np.testing.assert_allclose(cw[f"A_{c}"], full[f"A_{c}"], rtol=1e-4, atol=1e-6 * full[f"A_{c}"].max())
        np.testing.assert_allclose(cw[f"B_{c}"], full[f"B_{c}"], rtol=1e-4, atol=1e-6 * full[f"B_{c}"].max())
    # B is a sum over slices, the linear projection is too: the two halves add up
    np.testing.assert_allclose(c1["B_full"] + c2["B_full"], cw["B_full"], rtol=1e-4, atol=1e-6 * cw["B_full"].max())
    np.testing.assert_allclose(c1["th_full"] + c2["th_full"], cw["th_full"], rtol=1e-4, atol=1e-6 * np.abs(cw["th_full"]).max())
    np.testing.assert_allclose(cw["th_full"], full["_ksz_map"], rtol=1e-4, atol=1e-30)
    assert c1["n_pix"] + c2["n_pix"] == cw["n_pix"]


def test_script49_on_synthetic_pickle(tmp_path):
    rng = np.random.default_rng(1)
    n = 24
    def m(): return rng.normal(size=(n, n)).astype(np.float32)
    wins = []
    for z0 in (10.0, 8.0):
        maps = {f"{o}_{c}": (1.0 + 0.3 * np.abs(m())) for o in "AB" for c in ("full", "x", "v")}
        wins.append({"z0": z0, "dz": 1.0, "z_lo": z0 - 0.5, "z_hi": z0 + 0.5, "chi_eff": 9000.0, "x_hii": 0.3, "maps": maps, "n_pix": 100})
    d = {"seed": 1, "chi_ref": 9000.0, "S_global": 1.0 + 0.3 * np.abs(m()),
         "windows": wins, "gal": {"matter_Waters": [m(), m()], "lae_L42.5": [m(), m()]}}
    p = tmp_path / "seed1_wo0_chunks.pkl"
    with open(p, "wb") as f:
        pickle.dump(d, f)
    s49 = _load("s49", os.path.join(ROOT, "scripts", "49_chunk_map_figures.py"))
    out = tmp_path / "out"
    s49.main(["--chunks", str(p), "--out-dir", str(out), "--cell", "4.0"])
    assert (out / "maps_estimator.png").exists() and (out / "maps_chunk_lae_L42.5.png").exists()
    r = np.genfromtxt(out / "maps_correlations.csv", delimiter=",", names=True, dtype=None, encoding=None)
    assert np.all(np.abs(r["pearson"]) <= 1.0 + 1e-9)


def test_pearson_and_smoothing():
    s49 = _load("s49b", os.path.join(ROOT, "scripts", "49_chunk_map_figures.py"))
    a = np.random.default_rng(0).normal(size=(16, 16))
    assert abs(s49.pearson(a, 3 * a + 2) - 1.0) < 1e-12
    assert abs(s49.pearson(a, -a) + 1.0) < 1e-12


def test_script48_helpers_compile_and_pick_windows():
    s48 = _load("s48", os.path.join(ROOT, "scripts", "48_chunk_maps.py"))
    prod = {"windows": [{"z0": z, "dz": dz} for dz in (0.5, 1.0) for z in (7.0, 8.0, 9.0, 10.0)]}
    idx = s48.pick_windows(prod, [10.2, 7.9, 7.1])
    assert [prod["windows"][i]["z0"] for i in idx] == [7.0, 8.0, 10.0]
    assert all(prod["windows"][i]["dz"] == 1.0 for i in idx)
