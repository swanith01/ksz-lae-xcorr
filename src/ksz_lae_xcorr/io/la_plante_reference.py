"""
io/la_plante_reference.py
============================
Loaders for the digitized La Plante et al. 2022 reference data in
data/reference/la_plante_2022/ -- see that directory's README.md for
exact provenance of each file.
"""

import csv
import os

import numpy as np


def _default_root(cfg=None):
    if cfg is not None and hasattr(cfg, "paths") and hasattr(cfg.paths, "project_root"):
        return os.path.join(cfg.paths.project_root, "data", "reference", "la_plante_2022")
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "reference", "la_plante_2022")


def load_dell_vs_z0_bands(cfg=None, root=None) -> dict:
    """
    Returns {ell: {'z0': array, 'lo': array, 'hi': array}} for ell in
    (500, 1000, 3000). 'lo'/'hi' are the digitized band edges at their
    OWN native z0 sampling (irregular, from wherever points were
    clicked) -- interpolate onto a common grid at the call site if
    needed, not resampled here to avoid baking in interpolation choices
    upstream of where they're actually used.
    """
    root = root or _default_root(cfg)
    path = os.path.join(root, "dell_vs_z0_bands_digitized.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ell = int(row["ell"])
            bound = row["bound"]
            out.setdefault(ell, {"z0_lo": [], "lo": [], "z0_hi": [], "hi": []})
            out[ell][f"z0_{bound}"].append(float(row["z0"]))
            out[ell][bound].append(float(row["D_ell_uK2"]))
    for ell in out:
        for k in list(out[ell].keys()):
            out[ell][k] = np.array(out[ell][k])
        # sort each bound by its own z0 for clean interpolation downstream
        order_lo = np.argsort(out[ell]["z0_lo"])
        order_hi = np.argsort(out[ell]["z0_hi"])
        out[ell]["z0_lo"], out[ell]["lo"] = out[ell]["z0_lo"][order_lo], out[ell]["lo"][order_lo]
        out[ell]["z0_hi"], out[ell]["hi"] = out[ell]["z0_hi"][order_hi], out[ell]["hi"][order_hi]
    return out


def load_dell_vs_ell_band(cfg=None, root=None) -> dict:
    """Returns {'ell_lo','lo','ell_hi','hi'} -- single band at x_HII~0.43,
    since the individual z-window curves overlap within it in the
    original figure (see README)."""
    root = root or _default_root(cfg)
    path = os.path.join(root, "dell_vs_ell_xhii043_band_digitized.csv")
    out = {"ell_lo": [], "lo": [], "ell_hi": [], "hi": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            bound = row["bound"]
            out[f"ell_{bound}"].append(float(row["ell"]))
            out[bound].append(float(row["D_ell_uK2"]))
    for k in out:
        out[k] = np.array(out[k])
    order_lo = np.argsort(out["ell_lo"])
    order_hi = np.argsort(out["ell_hi"])
    out["ell_lo"], out["lo"] = out["ell_lo"][order_lo], out["lo"][order_lo]
    out["ell_hi"], out["hi"] = out["ell_hi"][order_hi], out["hi"][order_hi]
    return out


def load_reionization_histories(cfg=None, root=None) -> dict:
    """Returns {scenario: {'z': array, 'x_HII': array}} for
    'Fiducial', 'Early', 'Short'."""
    root = root or _default_root(cfg)
    path = os.path.join(root, "reionization_histories_digitized.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            scenario = row["scenario"]
            out.setdefault(scenario, {"z": [], "x_HII": []})
            out[scenario]["z"].append(float(row["z"]))
            out[scenario]["x_HII"].append(float(row["x_HII"]))
    for scenario in out:
        order = np.argsort(out[scenario]["z"])
        out[scenario]["z"] = np.array(out[scenario]["z"])[order]
        out[scenario]["x_HII"] = np.array(out[scenario]["x_HII"])[order]
    return out
