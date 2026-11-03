"""
snr/sim_snr.py
================
S/N of filtered-kSZ^2 x tracer from the SIMULATION's own maps (the 300 Mpc wrap-cycle lightcones, physical field units),
i.e. the old exploratory notebook's Figs. 4-5 redone on the fiducial box.  Two halves:

  measure  (swarm, scripts/53): from the per-seed  seed*_wo*_lpmaps.pkl  products (scripts/26 --save-maps for matter x
           Waters+16 bias; scripts/37 for each tracer), filter + square the kSZ map for every CMB experiment, and store the
           BINNED spectra  Cs(l) = C_l(kSZ_f^2 x delta_g) [uK^2]  and  Cg(l) = C_l(delta_g x delta_g)  per seed and window,
           plus the filtered-map noise C_T2T2,f [uK^4] of each experiment.  Small .npz, one per tracer.
  forecast (desktop, scripts/54): La Plante+22 Eq. 13 on those bins,

           (S/N)^2 = f_sky * sum_bins (l_hi^2 - l_lo^2) * Cs^2 / (C_T2T2,f * Cg + Cs^2)

           (sum_{l in bin} (2l+1) = l_hi^2 - l_lo^2), per window, cumulative in l_max, total over NON-overlapping windows.

The filter is snr.feasibility.build_filter (f = F b normalised to peak 1, the same convention as scripts/28 and 52), with
C_kSZ,reion = the seed-median auto-power of the simulated kSZ map (as in scripts/28 and the old notebook), or La Plante's.

Noise of the measured signal: Cs is a 10-seed mean of a ~4 deg^2 patch, so Cs^2 is biased high by its own scatter.  Both
the raw mean^2 and the debiased  max(mean^2 - sem^2, 0)  are reported.

Survey version (survey_cg): the tracer's clustering from the simulation, rescaled to the survey's bias and shell depth,
plus the survey's shot noise:   Cg = r^2 (Cg_sim - N_sim) dchi_win / dchi_shell + 1/n_2D,   Cs -> r Cs,   r = b_survey/b_sim.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import cross_power_2d, to_Cell
from ksz_lae_xcorr.utils import constants


# --------------------------------------------------------------------------------------------------------------------- #
# measurement (maps -> binned spectra)
# --------------------------------------------------------------------------------------------------------------------- #
def binned_cl(map_a: np.ndarray, map_b: np.ndarray, kg, chi: float) -> np.ndarray:
    """C_l of two 2D maps on kg's k-bins with l = k chi (both mean-subtracted).  Units = product of the map units."""
    a = map_a.astype(np.float64)
    b = map_b.astype(np.float64)
    P, Pe, _ = cross_power_2d(a - a.mean(), b - b.mean(), kg)
    C, _ = to_Cell(P, Pe, chi)
    return C


def ell_edges(kg, chi: float) -> np.ndarray:
    return np.asarray(kg.k_bins, float) * chi


def window_shot_noise(n_of_z, z_lo: float, z_hi: float, chi_of_z, chi_eff: float, n_sub: int = 64) -> float:
    """Poisson C_l [sr] of a top-hat (in chi) projected overdensity with W = 1/dchi:
    C = (1/dchi^2) INT dchi / n(chi) / chi_eff^2.  n_of_z: callable z -> n_3D [Mpc^-3]."""
    z = np.linspace(z_lo, z_hi, n_sub)
    chi = np.asarray(chi_of_z(z), float)
    n = np.asarray(n_of_z(z), float)
    if np.any(n <= 0):
        return np.inf
    dchi = chi[-1] - chi[0]
    inv = 1.0 / n
    integral = float(np.sum(0.5 * (inv[1:] + inv[:-1]) * np.diff(chi)))
    return integral / dchi ** 2 / chi_eff ** 2


# --------------------------------------------------------------------------------------------------------------------- #
# forecast
# --------------------------------------------------------------------------------------------------------------------- #
def modes_in_range(edges: np.ndarray, ell_min: float, ell_max: float) -> np.ndarray:
    """sum_{l in bin and in [ell_min, ell_max]} (2l+1)  ~=  l_hi^2 - l_lo^2  with the bin clipped to the range."""
    lo = np.clip(edges[:-1], ell_min, ell_max)
    hi = np.clip(edges[1:], ell_min, ell_max)
    return np.maximum(hi ** 2 - lo ** 2, 0.0)


def signal_sq(Cs_seeds: np.ndarray, debias: bool) -> np.ndarray:
    """Cs^2 from per-seed Cs (n_seed, n_bin): mean^2, or max(mean^2 - sem^2, 0) when debias."""
    Cs_seeds = np.atleast_2d(Cs_seeds)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(Cs_seeds, axis=0)
        n = np.sum(np.isfinite(Cs_seeds), axis=0)
        sem2 = np.where(n > 1, np.nanvar(Cs_seeds, axis=0, ddof=1) / np.maximum(n, 1), 0.0)
    s2 = m ** 2
    if debias:
        s2 = np.maximum(s2 - sem2, 0.0)
    return np.where(np.isfinite(s2), s2, 0.0)


def snr_terms(Cs2: np.ndarray, Cg: np.ndarray, CT2T2: float, edges: np.ndarray, f_sky: float,
              ell_min: float = 100.0, ell_max: float = 10000.0) -> np.ndarray:
    """Per-bin contribution to (S/N)^2 (La Plante+22 Eq. 13, bin-summed).  Empty / non-finite bins contribute 0."""
    m = modes_in_range(edges, ell_min, ell_max)
    den = CT2T2 * np.abs(Cg) + Cs2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = f_sky * m * Cs2 / den
    return np.where(np.isfinite(t) & (den > 0), t, 0.0)


def cumulative_snr(terms_by_window: list, edges_by_window: list, ell_max_grid: np.ndarray, ell_min: float = 100.0) -> np.ndarray:
    """sqrt( sum over windows and bins of the terms up to l_max ), with partial bins at the l_max edge
    (each bin's term is spread over its modes ~ uniformly in l^2)."""
    out = np.zeros(len(ell_max_grid))
    for t, e in zip(terms_by_window, edges_by_window):
        full = modes_in_range(e, ell_min, np.inf)
        for i, L in enumerate(ell_max_grid):
            part = modes_in_range(e, ell_min, L)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(full > 0, part / full, 0.0)
            out[i] += float(np.sum(t * frac))
    return np.sqrt(out)


def non_overlapping(z0: np.ndarray, dz: np.ndarray, start: float | None = None) -> np.ndarray:
    """Indices of a set of windows that tile z without overlap: z0 = start, start + dz, ... (dz from the first window)."""
    z0 = np.asarray(z0, float)
    d = float(np.asarray(dz, float)[0])
    s = float(np.min(z0)) if start is None else start
    keep = []
    for i, z in enumerate(z0):
        k = (z - s) / d
        if k > -1e-6 and abs(k - round(k)) < 1e-6:
            keep.append(i)
    return np.array(keep, int)


def survey_cg(Cg_sim: np.ndarray, shot_sim: float, r: float, dchi_win: float, dchi_shell: float, n2d_per_sr: float) -> np.ndarray:
    """Tracer auto-power for a survey: the simulated clustering (shot noise removed, floored at 0), rescaled by the bias
    ratio squared and by the depth (Limber: C_clust ~ 1/dchi), plus the survey's Poisson term."""
    clust = np.maximum(np.asarray(Cg_sim, float) - (0.0 if not np.isfinite(shot_sim) else shot_sim), 0.0)
    return r ** 2 * clust * (dchi_win / dchi_shell) + 1.0 / n2d_per_sr


T_CMB_UK2 = constants.T_CMB_UK ** 2


# --------------------------------------------------------------------------------------------------------------------- #
# survey helpers (scripts/55)
# --------------------------------------------------------------------------------------------------------------------- #
def load_bias_table(path):
    """{tracer name: (z array, b array)} from tracer_bias_summary.csv (scripts/36)."""
    import csv
    import os
    if not path or not os.path.exists(path):
        return {}
    tab = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            tab.setdefault(r["name"], []).append((float(r["z"]), float(r["b"])))
    return {k: (np.array([a for a, _ in sorted(v)]), np.array([b for _, b in sorted(v)])) for k, v in tab.items()}


def sim_bias(name, z, btab):
    """Large-scale bias of the simulated tracer at z: Waters+16 for the matter run, else the measured one (scripts/35-36),
    linear in z, held constant outside the measured range."""
    if name.startswith("matter"):
        from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
        return float(bluetides_bias_gz(np.array([z]))[0])
    if name not in btab:
        return np.nan
    zz, bb = btab[name]
    return float(np.interp(z, zz, bb))


def shell_window(sp, z):
    """Index of the simulated window (from a scripts/53 .npz) that contains z, nearest z0; None if none does."""
    inside = [i for i in range(len(sp["z0"])) if sp["z_lo"][i] <= z <= sp["z_hi"][i]]
    return min(inside, key=lambda i: abs(sp["z0"][i] - z)) if inside else None


def survey_shell_snr2(sp, ei, wi, r, f_sky, n2d_per_sr, dchi_win, dchi_shell, ell_min, ell_max, debias=True):
    """(S/N)^2 of one survey shell from the simulated window wi, signal and clustering rescaled by r = b_survey / b_sim."""
    Cg = survey_cg(np.nanmean(sp["Cg"][:, wi, :], axis=0), float(sp["shot_sim"][wi]), r, dchi_win, dchi_shell, n2d_per_sr)
    Cs2 = signal_sq(sp["Cs"][:, ei, wi, :] * r, debias)
    return float(snr_terms(Cs2, Cg, float(sp["CT2T2"][ei]), sp["edges"][wi], f_sky, ell_min, ell_max).sum())
