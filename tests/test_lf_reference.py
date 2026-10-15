"""Compiled observational luminosity functions (data/reference/luminosity_functions) + the analytic helpers."""
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ksz_lae_xcorr.io import luminosity_functions as lf  # noqa: E402


def test_csvs_load_and_have_sources():
    for load in (lf.uvlf_points, lf.uvlf_fits, lf.lae_lf_points, lf.lae_lf_fits):
        t = load()
        assert len(t["z"]) > 5 and len(set(t["ref_key"])) >= 3
    src = open(os.path.join(lf.lf_dir(), "SOURCES.md")).read()
    for key in set(lf.uvlf_points()["ref_key"]) | set(lf.lae_lf_points()["ref_key"]) | set(lf.lae_lf_fits()["ref_key"]):
        assert key in src, f"{key} missing from SOURCES.md"


def test_points_are_physical():
    u = lf.uvlf_points()
    assert np.all(u["phi_Mpc3_mag"] > 0) and np.all(u["phi_Mpc3_mag"] < 0.1)
    assert np.all((u["M_UV"] > -25) & (u["M_UV"] < -14))
    a = lf.lae_lf_points()
    assert np.all((a["log_L"] > 41.5) & (a["log_L"] < 45.5)) and np.all((a["log_phi"] < -1) & (a["log_phi"] > -8))


def test_schechter_cumulative_matches_incomplete_gamma():
    fits = {"form": np.array(["Schechter"], dtype=object), "Mstar": np.array([-21.0]), "phistar_Mpc3": np.array([1e-3]),
            "alpha": np.array([-1.7]), "beta": np.array([np.nan])}
    M_cut = -19.0
    x = 10 ** (0.4 * (-21.0 - M_cut))
    # n(<M) = phi* Gamma(alpha+1, x): use the numerical integral of the integrand as the reference
    xs = np.linspace(x, 80, 400000)
    ref = 1e-3 * lf._trapz(xs ** (-1.7) * np.exp(-xs), xs)
    assert math.isclose(lf.uv_cumulative(fits, 0, M_cut, M_bright=-30.0), ref, rel_tol=2e-3)


def test_dpl_and_lya_forms_are_finite_and_decreasing_bright_end():
    M = np.linspace(-23, -17, 25)
    p = lf.dpl_uv(M, -20.0, 1e-4, -2.0, -4.0)
    assert np.all(np.isfinite(p)) and p[0] < p[12] < p[-1]
    L = np.linspace(42.0, 44.0, 20)
    s = lf.schechter_lya_dex(L, 43.0, -4.0, -1.8)
    assert np.all(s > 0) and s[-1] < s[10]
    g = lf.lae_lf_fits()
    i = int(np.argmin(np.abs(g["z"] - 6.6) + (g["ref_key"] != "Umeda25") * 10))
    n = lf.lya_cumulative(g, i, 42.5)
    assert 1e-5 < n < 1e-2                                   # ~1e-4 Mpc^-3 observed at z=6.6
