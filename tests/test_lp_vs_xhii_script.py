"""Smoke test: scripts/31 reads a scripts/28-style CSV and draws the D vs x_HII figure."""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load():
    spec = importlib.util.spec_from_file_location("s31", os.path.join(ROOT, "scripts", "31_lp_cross_vs_xhii.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_make_figure_and_main(tmp_path, monkeypatch, capsys):
    m = _load()
    z0 = np.arange(6.5, 13.01, 0.5)
    names = ["z0", "dz", "x_hii"] + [f"{k}_l{l}" for l in (500, 1000, 3000) for k in ("mean", "sigma_mean")]
    cols = [z0, np.ones_like(z0), np.clip(1.2 - 0.1 * z0, 0.0, 1.0)]
    for l in (500, 1000, 3000):
        cols += [0.01 * np.ones_like(z0), 0.004 * np.ones_like(z0)]
    csv = tmp_path / "lp_cross_vs_z0_SO_wo0.csv"
    np.savetxt(csv, np.column_stack(cols), delimiter=",", header=",".join(names), comments="")
    c = m.load_cross_csv(str(csv))
    assert set(names) <= set(c)
    fig = m.make_figure(c, m.load_fig7_vs_xhii())
    assert len(fig.axes) == 3 and fig.axes[0].get_xlim()[0] > fig.axes[0].get_xlim()[1]   # x axis reversed
    monkeypatch.setattr(sys, "argv", ["31", "--csv", str(csv), "--out-dir", str(tmp_path)])
    m.main()
    assert (tmp_path / "lp_cross_vs_xhii_SO_wo0.png").exists() and (tmp_path / "lp_cross_vs_xhii_SO_wo0.pdf").exists()
    assert "paper peak" in capsys.readouterr().out
