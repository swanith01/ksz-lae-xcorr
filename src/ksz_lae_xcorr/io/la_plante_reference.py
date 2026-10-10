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


def _repo_root_dir():
    """data/reference/la_plante_2022 inside this repository checkout."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                         "data", "reference", "la_plante_2022"))


def _project_root_dir(cfg=None):
    if cfg is not None and hasattr(cfg, "paths") and hasattr(cfg.paths, "project_root"):
        return os.path.join(cfg.paths.project_root, "data", "reference", "la_plante_2022")
    return None


def _resolve(filename, cfg=None, root=None):
    """Path of a reference CSV.  An explicit `root` always wins.  Otherwise the copy that ships
    with the repository is used (it is the single source of truth and is always current);
    cfg.paths.project_root/data/reference/... is only a fallback.  (Before 2026-10-10 the
    project_root copy came first: on the cluster project_root is the older
    kSZ2_LAE_project_22Jun2026 folder, which lacks newer CSVs -> FileNotFoundError.)"""
    if root:
        return os.path.join(root, filename)
    cands = [_repo_root_dir()]
    pr = _project_root_dir(cfg)
    if pr:
        cands.append(pr)
    for d in cands:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{filename} not found in any of: {cands}")


def load_dell_vs_z0_bands(cfg=None, root=None) -> dict:
    """
    Returns {ell: {'z0': array, 'lo': array, 'hi': array}} for ell in
    (500, 1000, 3000). 'lo'/'hi' are the digitized band edges at their
    OWN native z0 sampling (irregular, from wherever points were
    clicked) -- interpolate onto a common grid at the call site if
    needed, not resampled here to avoid baking in interpolation choices
    upstream of where they're actually used.
    """
    path = _resolve("dell_vs_z0_bands_digitized.csv", cfg, root)
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
    path = _resolve("dell_vs_ell_xhii043_band_digitized.csv", cfg, root)
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
    path = _resolve("reionization_histories_digitized.csv", cfg, root)
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


def load_fig1_components(cfg=None, root=None) -> dict:
    """La Plante+22 Fig. 1 curves (vector-extracted from the PDF, see the README):
    {component: (ell, D_ell_uK2)} with D = l(l+1)C_l/2pi.  Components: 'SO_post_ILC_noise',
    'CMB-S4_post_ILC_noise', 'CMB-HD_post_ILC_noise', 'lensed_primary_CMB', 'kSZ_reion_30sim_mean',
    'kSZ_late_Park18'."""
    path = _resolve("fig1_filter_components_digitized.csv", cfg, root)
    tmp = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tmp.setdefault(row["component"], []).append((float(row["ell"]), float(row["D_ell_uK2"])))
    out = {}
    for k, v in tmp.items():
        a = np.array(sorted(v))
        out[k] = (a[:, 0], a[:, 1])
    return out


def load_fig7_vs_xhii(cfg=None, root=None) -> dict:
    """La Plante+22 Fig. 7 (right panel), FIDUCIAL scenario only: {ell: (x_HII, D_ell_uK2)}
    for ell = 500, 1000, 3000, x_HII ascending (0 = fully neutral ... 1 = fully ionised;
    the paper plots the axis reversed, 1 on the left).  Vector-traced from the PDF, see the README."""
    path = _resolve("fig7_dell_vs_xhii_fiducial_digitized.csv", cfg, root)
    tmp = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tmp.setdefault(int(row["ell"]), []).append((float(row["x_HII"]), float(row["D_ell_uK2"])))
    out = {}
    for ell, v in tmp.items():
        a = np.array(sorted(v))
        out[ell] = (a[:, 0], a[:, 1])
    return out
