"""Smoke test: scripts/27 make_figure draws the simple 3-panel figure (default) and the old 2x2 one (diagnostics)."""
import importlib.util
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load():
    spec = importlib.util.spec_from_file_location("s27", os.path.join(ROOT, "scripts", "27_ksz_auto_power_aggregate.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _fake(seeds, ell):
    rng = np.random.default_rng(0)
    out, res = {}, {}
    for key in ("patchy", "full"):
        agg = {}
        for f in ("D_total", "D_diag", "D_off"):
            st = np.abs(rng.normal(1.0, 0.1, (len(seeds), len(ell))))
            agg[f] = {"ell": ell, "seeds": seeds, "stack": st, "median": np.median(st, 0), "p16": st.min(0), "p84": st.max(0),
                      "min": st.min(0), "max": st.max(0), "sigma": st.std(0)}
        out[key] = {"_agg": agg}
    z = np.linspace(5, 20, 40)
    for s in seeds:
        res[s] = {"z_lc": z, "xHI_mean": 1 / (1 + np.exp(-(z - 8))),
                  "patchy": {"z_lo": 6.0, "z_hi": 19.8}, "full": {"z_lo": 5.0, "z_hi": 20.0}}
    return out, res


def test_simple_and_diagnostic_figures():
    m = _load()
    seeds = [1, 2, 3]; ell = np.logspace(2.3, 4.5, 12)
    out, res = _fake(seeds, ell)
    fig = m.make_figure(out, res, seeds, "grid-wrap", "helium", "analytic")
    assert len(fig.axes) == 3
    labels = [t.get_text() for ax in fig.axes for t in ax.get_legend().get_texts()]
    assert not any("diag" in t.lower() for t in labels)
    assert "mean over seeds" in labels and "median over seeds" in labels
    fig2 = m.make_figure(out, res, seeds, "grid-wrap", "helium", "analytic", diagnostics=True)
    assert len(fig2.axes) == 4
