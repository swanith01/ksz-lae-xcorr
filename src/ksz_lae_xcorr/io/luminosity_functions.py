"""
io/luminosity_functions.py
============================
Loaders + analytic helpers for the observational UV and Lya luminosity functions compiled in
data/reference/luminosity_functions/ (see SOURCES.md there for the paper, table and arXiv id of every row).

Units (everywhere): phi_UV in Mpc^-3 mag^-1 (comoving), M_UV in AB mag; phi_Lya in Mpc^-3 dex^-1, L in erg/s.
"""
from __future__ import annotations

import csv
import os

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz


def lf_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "reference", "luminosity_functions"))


def _load(name: str, root: str | None = None) -> dict:
    path = os.path.join(root or lf_dir(), name)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(c) if c != "" else np.nan for c in col])
        except ValueError:
            out[k] = np.array(col, dtype=object)
    return out


def uvlf_points(root=None) -> dict:
    return _load("uvlf_points.csv", root)


def uvlf_fits(root=None) -> dict:
    return _load("uvlf_fits.csv", root)


def lae_lf_points(root=None) -> dict:
    return _load("lae_lf_points.csv", root)


def lae_lf_fits(root=None) -> dict:
    return _load("lae_lf_fits.csv", root)


def select(table: dict, mask) -> dict:
    return {k: v[mask] for k, v in table.items()}


def near_z(table: dict, z: float, tol: float = 0.6) -> np.ndarray:
    """boolean mask of rows whose z lies within `tol` of z."""
    return np.abs(np.asarray(table["z"], dtype=float) - z) <= tol


# ---------------------------------------------------------------- UV: analytic forms
def schechter_uv(M, Mstar, phistar, alpha):
    """phi(M) per mag: 0.4 ln10 phi* 10^{-0.4 (M-M*)(alpha+1)} exp(-10^{-0.4 (M-M*)})."""
    x = 10.0 ** (-0.4 * (np.asarray(M, dtype=float) - Mstar))
    return 0.4 * np.log(10.0) * phistar * x ** (alpha + 1.0) * np.exp(-x)


def dpl_uv(M, Mstar, phistar, alpha, beta):
    """double power law (Donnan+24 / McLeod+24): phi* / (10^{0.4 (alpha+1)(M-M*)} + 10^{0.4 (beta+1)(M-M*)})."""
    d = np.asarray(M, dtype=float) - Mstar
    return phistar / (10.0 ** (0.4 * (alpha + 1.0) * d) + 10.0 ** (0.4 * (beta + 1.0) * d))


def uv_phi_of_row(fits: dict, i: int, M):
    if str(fits["form"][i]) == "DPL":
        return dpl_uv(M, fits["Mstar"][i], fits["phistar_Mpc3"][i], fits["alpha"][i], fits["beta"][i])
    return schechter_uv(M, fits["Mstar"][i], fits["phistar_Mpc3"][i], fits["alpha"][i])


def uv_cumulative(fits: dict, i: int, M_cut: float, M_bright: float = -26.0, n: int = 4000) -> float:
    """n(< M_cut) [Mpc^-3]: the fit integrated from M_bright to M_cut."""
    M = np.linspace(M_bright, M_cut, n)
    return float(_trapz(uv_phi_of_row(fits, i, M), M))


# ---------------------------------------------------------------- Lya: analytic forms
def schechter_lya_dex(logL, logLstar, logphistar, alpha):
    """phi per dex: ln10 phi* (L/L*)^{alpha+1} exp(-L/L*)."""
    r = 10.0 ** (np.asarray(logL, dtype=float) - logLstar)
    return np.log(10.0) * 10.0 ** logphistar * r ** (alpha + 1.0) * np.exp(-r)


def lya_cumulative(fits: dict, i: int, logL_cut: float, logL_max: float = 45.5, n: int = 4000) -> float:
    """n(> L_cut) [Mpc^-3]: integral of phi per dex over log L."""
    x = np.linspace(logL_cut, logL_max, n)
    return float(_trapz(schechter_lya_dex(x, fits["log_Lstar"][i], fits["log_phistar"][i], fits["alpha"][i]), x))
