"""zreion/convert.py and scripts/34 on a mock py21cmfast seed folder (small grids)."""
import importlib.util
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ksz_lae_xcorr.zreion.convert import (binned_spectra, write_zreion_coeval_root, xhi_from_zre,  # noqa: E402
                                          xhi_quantile_matched, zre_from_ic)


def test_xhi_quantile_matched_hits_target_fraction():
    rng = np.random.default_rng(0)
    zre = rng.normal(8, 1.2, size=(20, 20, 20)).astype(np.float32)
    for x in (0.0, 0.1, 0.43, 0.9, 1.0):
        assert abs(1.0 - xhi_quantile_matched(zre, x).mean() - x) < 2e-3
    # earliest-reionised cells are the ionised ones
    m = xhi_quantile_matched(zre, 0.3)
    assert zre[m == 0].min() > zre[m == 1].max() - 1e-6


def test_zre_from_ic_has_zmean_and_responds_to_density():
    rng = np.random.default_rng(1)
    n, L = 24, 80.0
    ic = rng.normal(0, 1.0, size=(n, n, n)).astype(np.float32)
    for lpt in ("za", "linear"):
        zre = zre_from_ic(ic, L, 0.31, 0.68, zmean=8.0, lpt=lpt)
        assert abs(zre.mean() - 8.0) < 0.1
        # overdense regions reionise EARLIER (larger z_re)
        sm = np.fft.irfftn(np.fft.rfftn(ic) * np.exp(-0.5 * (np.fft.rfftfreq(n) ** 2)[None, None, :]), s=(n, n, n), axes=(0, 1, 2))
        assert np.corrcoef(zre.ravel(), sm.ravel())[0, 1] > 0.3


def test_binned_spectra_identical_fields_r_is_one():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(16, 16, 16))
    sp = binned_spectra(a, a, 50.0, nbins=8)
    ok = np.isfinite(sp["r"])
    assert np.allclose(sp["r"][ok], 1.0) and np.allclose(sp["Paa"][ok], sp["Pbb"][ok])


def _mock_root(tmp_path, zs=(6.0, 7.0, 8.0, 9.0, 10.0), n=32, dim=64, L=50.0):
    rng = np.random.default_rng(3)
    ic = rng.normal(0, 1.0, size=(dim, dim, dim)).astype(np.float32)
    ic_c = ic.reshape(n, dim // n, n, dim // n, n, dim // n).mean((1, 3, 5))
    root = tmp_path / "coeval"
    for z in zs:
        d = root / "seed_1" / f"coeval_z{z:.6f}"; d.mkdir(parents=True)
        np.save(d / "hires_density.npy", ic)
        np.save(d / "velocity_z.npy", np.zeros((n, n, n), np.float32))
        thr = np.quantile(ic_c, 1 - 1 / (1 + np.exp((z - 8.0) / 1.0)))              # ionised where ic_c high
        np.save(d / "neutral_fraction.npy", (ic_c < thr).astype(np.float32))
    return root


def test_write_root_symlinks_and_replaces_only_xh(tmp_path):
    root = _mock_root(tmp_path)
    dst = tmp_path / "zre_root"
    zs = write_zreion_coeval_root(str(root), str(dst), 1, lambda z: np.zeros((32, 32, 32), np.float32), log=lambda *_: None)
    assert len(zs) == 5
    d = dst / "seed_1" / "coeval_z8.000000"
    assert os.path.islink(d / "hires_density.npy") and os.path.islink(d / "velocity_z.npy")
    assert not os.path.islink(d / "neutral_fraction.npy") and np.load(d / "neutral_fraction.npy").sum() == 0
    with pytest.raises(FileExistsError):
        write_zreion_coeval_root(str(root), str(dst), 1, lambda z: np.zeros((32, 32, 32)), log=lambda *_: None)


def test_script34_end_to_end_on_mock_seed(tmp_path, capsys):
    root = _mock_root(tmp_path)
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "variants", "quicktest.yaml")))
    base["paths"]["coeval_root"] = str(root); base["paths"]["halo_root"] = str(root)
    base["box"]["z_min"] = 6.0; base["box"]["z_max"] = 10.0
    cfgp = tmp_path / "cfg.yaml"; cfgp.write_text(yaml.safe_dump(base))
    spec = importlib.util.spec_from_file_location("s34", os.path.join(ROOT, "scripts", "34_zreion_on_21cmfast.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    out = tmp_path / "out"
    rc = m.main(["--config", str(cfgp), "--seed", "1", "--out-dir", str(out), "--write-root", str(tmp_path / "zroot"),
                 "--match", "quantile", "--coarsen", "2"])
    assert rc == 0
    txt = capsys.readouterr().out
    assert "x_HII(21cmFAST)" in txt and "Morphology at EQUAL mean x_HII" in txt and "hires_density identical" in txt
    # quantile-matched boxes reproduce the 21cmFAST mean x_HII at every snapshot
    for z in (6.0, 8.0, 10.0):
        a = np.load(root / "seed_1" / f"coeval_z{z:.6f}" / "neutral_fraction.npy").mean()
        b = np.load(tmp_path / "zroot" / "seed_1" / f"coeval_z{z:.6f}" / "neutral_fraction.npy").mean()
        assert abs(a - b) < 2e-3
    assert any(f.endswith(".png") for f in os.listdir(out))
