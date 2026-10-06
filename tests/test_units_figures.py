"""Smoke test: scripts/30 figure functions run on synthetic inputs."""
import importlib.util
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load():
    spec = importlib.util.spec_from_file_location("s30", os.path.join(ROOT, "scripts", "30_units_fixed_figures.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_figures_run(tmp_path):
    m = _load()
    k = np.geomspace(0.03, 1.5, 12); zs = np.array([6.0, 9.0])
    P = np.array([1e4 * k ** -2 for _ in zs])
    np.savez(tmp_path / "u.npz", k=k, z=zs, P_vz_raw=P * 1e-40, P_vz_pec_theory=P * 1e-42, P_d=P,
             P_lin_z=P * 0.5, P_lin_0=P)
    assert all(os.path.exists(p) for p in m.fig_velocity_density(str(tmp_path / "u.npz"), 1, str(tmp_path), 0.3086))
    ell = np.geomspace(200, 1e4, 10)
    d = {"ell": ell, "D_total": ell / 1e3, "D_diag": ell / 3e3}
    for n in ("a", "b"):
        pickle.dump({"patchy": d}, open(tmp_path / f"{n}.pkl", "wb"))
    paths, notes = m.fig_ksz(str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl"), 1, str(tmp_path))
    assert all(os.path.exists(p) for p in paths) and notes == []
    paths, notes = m.fig_ksz(str(tmp_path / "a.pkl"), str(tmp_path / "missing.pkl"), 1, str(tmp_path))
    assert len(notes) == 1
