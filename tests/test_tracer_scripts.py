"""scripts/35 (bias suite) end-to-end on a mock seed folder with halo + LAE + LBG catalogues."""
import importlib.util
import os
import sys

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build_mock(tmp_path, zs=(6.0, 7.0, 8.0, 9.0, 10.0), n=32, dim=64, L=50.0, b_true=3.0, seed=1, nh=40000):
    """coeval boxes + halo catalogue whose positions trace (1 + b_true * delta) + Jahaan-style LAE/LBG id/value files."""
    rng = np.random.default_rng(5)
    ic = rng.normal(0, 2.5, size=(dim, dim, dim)).astype(np.float32)
    ic_c = ic.reshape(n, dim // n, n, dim // n, n, dim // n).mean((1, 3, 5))
    root, lae, lbg = tmp_path / "coeval", tmp_path / "obs_prop", tmp_path / "int_prop"
    cfgd = yaml.safe_load(open(os.path.join(ROOT, "configs", "variants", "quicktest.yaml")))
    Om0 = cfgd["cosmology"]["Om0"]
    cell = L / n
    for z in zs:
        d = root / f"seed_{seed}" / f"coeval_z{z:.6f}"; d.mkdir(parents=True)
        np.save(d / "hires_density.npy", ic)
        np.save(d / "velocity_z.npy", np.zeros((n, n, n), np.float32))
        np.save(d / "neutral_fraction.npy", np.ones((n, n, n), np.float32))
        delta = float(growth_factor_ratio(Om0, z)) * ic_c
        w = np.clip(1.0 + b_true * delta, 0.02, None).ravel(); w /= w.sum()
        cells = rng.choice(w.size, nh, p=w)
        ijk = np.stack(np.unravel_index(cells, (n, n, n)), 1)
        coords = ((ijk + rng.uniform(0, 1, (nh, 3))) * cell).astype(np.float32)
        masses = (10 ** rng.uniform(8, 12, nh)).astype(np.float32)
        hd = root / f"seed_{seed}" / "halo_catalogs"; hd.mkdir(exist_ok=True)
        np.save(hd / f"halo_coords_z{z:.6f}.npy", coords); np.save(hd / f"halo_masses_z{z:.6f}.npy", masses)
        grid, bl = int(n), int(L)
        tag = f"_z{z:.6f}_{grid}_{bl}_s{seed}.npy"
        ids = np.sort(rng.choice(nh, 8000, replace=False))
        for root_, p, arr in ((lae, "halo_ids_obs", ids), (lae, "lya_lum_obs", 10 ** rng.uniform(41, 44, 8000)),
                              (lbg, "halo_ids_lbg", ids), (lbg, "MUV_lbg", rng.uniform(-22, -15, 8000))):
            (root_ / p).mkdir(parents=True, exist_ok=True)
            np.save(root_ / p / (p + tag), arr)
    cfgd["paths"].update(coeval_root=str(root), halo_root=str(root), lightcone_root=str(tmp_path / "lc"),
                         lae_catalogue_root=str(lae), lbg_catalogue_root=str(lbg))
    cfgd["box"]["z_min"], cfgd["box"]["z_max"] = min(zs), max(zs)
    cfgp = tmp_path / "cfg.yaml"; cfgp.write_text(yaml.safe_dump(cfgd))
    return cfgp, cfgd


def test_script35_end_to_end(tmp_path, capsys):
    cfgp, _ = build_mock(tmp_path)
    m = _load("s35", "35_tracer_bias_suite.py")
    out = tmp_path / "bias"
    m.main(["--config", str(cfgp), "--seed", "1", "--out-dir", str(out), "--kmax", "0.4",
            "--specs", "halo:1e8,halo:1e10,halo:1e11,lae:1e42,lbg:-19"])
    rows = np.genfromtxt(out / "tracer_bias_seed1.csv", delimiter=",", names=True, dtype=None, encoding=None)
    names = set(rows["name"])
    assert names == {"halo_M8.0", "halo_M10.0", "halo_M11.0", "lae_L42.0", "lbg_MUV-19.0"}
    r8 = rows[(rows["name"] == "halo_M8.0") & (np.isclose(rows["z"], 8.0))][0]
    assert r8["n_obj"] == 40000 and 2.0 < r8["b_lin"] < 4.0                      # tracer built with b_true = 3
    n_by_cut = [rows[(rows["name"] == f"halo_M{c}.0") & (np.isclose(rows["z"], 8.0))]["n_obj"][0] for c in (8, 10, 11)]
    assert n_by_cut[0] > n_by_cut[1] > n_by_cut[2] > 0                    # higher cut -> fewer objects
    assert np.all(np.isfinite(rows["b_ST"]))
    assert (out / "tracer_bias_seed1.npz").exists()
    assert "b_Waters" in capsys.readouterr().out


def test_script36_plots_from_two_seed_csvs(tmp_path, capsys):
    import shutil
    cfgp, _ = build_mock(tmp_path)
    m35 = _load("s35b", "35_tracer_bias_suite.py")
    out = tmp_path / "bias"
    m35.main(["--config", str(cfgp), "--seed", "1", "--out-dir", str(out), "--kmax", "0.4", "--specs",
              "halo:8;halo:10;halo:11;lae:42;lae:43;lbg:-18;lbg:-20"])
    shutil.copy(out / "tracer_bias_seed1.csv", out / "tracer_bias_seed2.csv")
    m36 = _load("s36", "36_tracer_bias_plot.py")
    m36.main(["--in-dir", str(out), "--out-dir", str(tmp_path / "plots"), "--box-len", "50"])
    for f in ("tracer_bias_vs_z", "tracer_number_density", "tracer_bias_vs_number"):
        assert (tmp_path / "plots" / f"{f}.png").exists()
    assert (tmp_path / "plots" / "tracer_bias_summary.csv").exists()
    txt = capsys.readouterr().out
    assert "La Plante / Waters+16 bias" in txt and "vol. factor" in txt


def test_script37_tracer_window_maps_follow_matter(tmp_path):
    """Tracer built with bias 3 on the mock boxes -> its projected window map correlates with, and is ~3x, the matter map
    built with the SAME wrap-cycle geometry and window weights."""
    import pickle
    from ksz_lae_xcorr.correlation import lp_maps
    from ksz_lae_xcorr.lightcone.stitch import Stitcher
    from ksz_lae_xcorr.lightcone.wrap_cycle import build_los_z_grid, stitch_wrapcycle
    from ksz_lae_xcorr.utils.config import load_config
    from ksz_lae_xcorr.utils.cosmology import get_cosmology

    cfgp, _ = build_mock(tmp_path, zs=(6.0, 7.0, 8.0, 9.0, 10.0), nh=600000)   # many haloes: shot noise small
    n = 32
    prod = {"kSZ_map": np.random.default_rng(0).normal(size=(n, n)).astype(np.float32), "chi_eff_kSZ": 9000.0,
            "kSZ_z_min": 6.0, "z_top": 10.0, "ne_scale": 1.0, "tau0": 0.0,
            "windows": [dict(z0=7.5, dz=1.0, z_lo=7.0, z_hi=8.0, chi_eff=9000.0, x_hii=0.5, x_hii_z0=0.5,
                             gal_map=np.zeros((n, n), np.float32)),
                        dict(z0=8.5, dz=1.0, z_lo=8.0, z_hi=9.0, chi_eff=9000.0, x_hii=0.7, x_hii_z0=0.7,
                             gal_map=np.zeros((n, n), np.float32))]}
    pdir = tmp_path / "prod"; pdir.mkdir()
    pickle.dump(prod, open(pdir / "seed1_wo0_lpmaps.pkl", "wb"))
    m37 = _load("s37", "37_tracer_lightcone_maps.py")
    out = tmp_path / "maps"
    m37.main(["--config", str(cfgp), "--seed", "1", "--in-dir", str(pdir), "--out-root", str(out),
              "--specs", "halo:8;halo:11;lae:42"])
    assert {p.name for p in out.iterdir()} == {"halo_M8.0", "halo_M11.0", "lae_L42.0"}
    res = pickle.load(open(out / "halo_M8.0" / "seed1_wo0_lpmaps.pkl", "rb"))
    assert res["tracer"]["name"] == "halo_M8.0" and len(res["windows"]) == 2
    np.testing.assert_allclose(res["kSZ_map"], prod["kSZ_map"])                   # kSZ products untouched

    # matter reference on the identical geometry
    cfg = load_config(str(cfgp))
    st = Stitcher(cfg, field_convention="physical")
    cell = 50.0 / n
    z_arr, chi = build_los_z_grid(get_cosmology(cfg), cfg.box.z_min, cfg.box.z_max, cell)
    z_arr, chi = z_arr[:int(np.searchsorted(z_arr, 9.0 + 0.2)) + 2], chi[:int(np.searchsorted(z_arr, 9.0 + 0.2)) + 2]
    lc = stitch_wrapcycle(lambda z, f: np.asarray(st.load_field_box(1, z, "density", convention="physical"), np.float32),
                          [6.0, 7.0, 8.0, 9.0, 10.0], z_arr, chi, ngrid=n, cell_size=cell, wrap_cycle_seed=1,
                          fields=["density"])
    fd = {"z_lc": z_arr, "density_lc": 1.0 + lc["density"]}
    for i, w in enumerate(res["windows"]):
        lo, hi = int(np.searchsorted(z_arr, w["z_lo"])), int(np.searchsorted(z_arr, w["z_hi"]))
        mm = lp_maps.build_lp_galaxy_map(fd, lo, hi, bias_fn=lambda z: np.ones_like(z))
        g = w["gal_map"]
        assert g.shape == (n, n) and abs(g.mean()) < 0.05
        assert np.corrcoef(g.ravel(), mm.ravel())[0, 1] > 0.8
        slope = np.polyfit(mm.ravel(), g.ravel(), 1)[0]
        assert 2.7 < slope < 3.4, slope                                          # b_true = 3


def test_script38_compares_tracer_cross_csvs(tmp_path, capsys):
    """scripts/38 on mock scripts/28 CSVs (3 tracers + reference): figure, summary csv, table."""
    m = _load("s38", "38_tracer_cross_compare.py")
    x = np.linspace(0.9, 0.05, 12)
    z0 = np.linspace(6, 13, 12)

    def write(path, amp):
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = {"z0": z0, "dz": np.ones(12), "x_hii": x}
        for ell in (1000, 3000):
            cols[f"mean_l{ell}"] = amp * np.exp(-((x - 0.4) / 0.2) ** 2) * (ell / 1000)
            cols[f"sigma_mean_l{ell}"] = np.full(12, 0.01)
        np.savetxt(path, np.column_stack(list(cols.values())), delimiter=",", header=",".join(cols), comments="")

    root = tmp_path / "cross"
    for name, amp in (("halo_M10.0", 0.3), ("lae_L42.5", 0.1), ("lbg_MUV-19.0", 0.2)):
        write(root / name / "lp_cross_vs_z0_SO_wo0.csv", amp)
    write(tmp_path / "ref.csv", 0.25)
    (root / "not_a_tracer").mkdir()
    rc = m.main(["--root", str(root), "--ref-csv", str(tmp_path / "ref.csv"), "--panels"])
    assert rc is None or rc == 0
    out = capsys.readouterr().out
    assert "3 tracers" in out and "halo_M10.0" in out and "reference" in out
    assert (root / "tracer_cross_vs_xhii.png").exists() and (root / "tracer_cross_summary.csv").exists()
    assert (root / "panels" / "cross_lae_l1000.png").exists() and len(list((root / "panels").glob("*.png"))) == 6
    assert m.main(["--root", str(root), "--yscale", "linear", "--out-dir", str(tmp_path / "lin")]) is None
    rows = np.genfromtxt(root / "tracer_cross_summary.csv", delimiter=",", names=True, dtype=None, encoding=None)
    r = [q for q in rows if q["name"] == "halo_M10.0" and q["ell"] == 1000][0]
    assert abs(r["peak"] - 0.3) < 0.02 and 0.3 < r["x_hii_at_peak"] < 0.5


def test_script39_and_40_lf_pipeline(tmp_path, capsys):
    """scripts/39 on the mock seed (halo mass function vs ST, UV LF, Lya LF) and scripts/40 plotting from two copies."""
    import shutil
    cfgp, _ = build_mock(tmp_path)
    m39 = _load("s39", "39_tracer_luminosity_functions.py")
    out = tmp_path / "lf"
    m39.main(["--config", str(cfgp), "--seed", "1", "--out-dir", str(out)])
    rows = np.genfromtxt(out / "lf_seed1.csv", delimiter=",", names=True, dtype=None, encoding=None)
    kinds = set(rows["kind"])
    assert kinds == {"halo_mf", "uvlf", "lae_lf"}
    r = rows[rows["kind"] == "halo_mf"]
    z8 = r[np.isclose(r["z"], 8.0)]
    assert z8["n"].sum() == 40000                              # all mock haloes (masses uniform in log M over 8..12)
    assert np.all(np.isfinite(z8["phi_ST"])) and np.all(z8["phi_ST"] > 0)
    u = rows[(rows["kind"] == "uvlf") & np.isclose(rows["z"], 8.0)]
    assert u["n"].sum() == 8000 and np.isfinite(u["median_log10M"][u["n"] > 50]).all()
    # Lya LF: 8000 mock LAEs with log L uniform in 41..44 -> phi per dex flat ~ 8000/3/vol
    l = rows[(rows["kind"] == "lae_lf") & np.isclose(rows["z"], 8.0) & (rows["x"] > 41.2) & (rows["x"] < 43.8)]
    assert np.allclose(l["phi"], 8000 / 3.0 / 50.0 ** 3, rtol=0.35)
    shutil.copy(out / "lf_seed1.csv", out / "lf_seed2.csv")
    m40 = _load("s40", "40_tracer_lf_plots.py")
    m40.main(["--in-dir", str(out), "--out-dir", str(tmp_path / "lfplots")])
    for f in ("halo_mass_function_vs_ST", "uv_luminosity_function_vs_lit", "lya_luminosity_function_vs_lit"):
        assert (tmp_path / "lfplots" / f"{f}.png").exists()
    assert (tmp_path / "lfplots" / "lf_summary.csv").exists()
    assert "cat/obs" in capsys.readouterr().out
