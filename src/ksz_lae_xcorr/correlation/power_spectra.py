"""
correlation/power_spectra.py
=============================
2D FFT cross-power estimator and Limber-approximation ell<->k conversion.
Direct refactor of notebook Cell 3. All box/cosmology parameters come from
Config; nothing here should be hardcoded a second time.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.utils.cosmology import get_cosmology, little_h


class KGrid:
    """2D Fourier k-grid + log-spaced k-bins for a square box of side `box_len`."""

    def __init__(self, cfg):
        self.box_len = cfg.box.box_len_mpc
        self.n_side = cfg.box.hii_dim
        self.pix_size_mpc = self.box_len / self.n_side
        self.area_2d = self.box_len**2
        self.n_kbins = cfg.correlation.n_kbins

        dk = 2 * np.pi / self.box_len
        kx = np.fft.fftshift(np.fft.fftfreq(self.n_side)) * self.n_side * dk
        ky = kx.copy()
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.kgrid = np.sqrt(KX**2 + KY**2)

        k_min = dk
        k_max = self.kgrid.max() * 0.95
        self.k_bins = np.logspace(np.log10(k_min), np.log10(k_max), self.n_kbins + 1)
        self.k_centers = 0.5 * (self.k_bins[:-1] + self.k_bins[1:])


def make_overdensity(field_2d: np.ndarray) -> np.ndarray:
    mean = field_2d.mean()
    if mean <= 0:
        return np.zeros_like(field_2d)
    return (field_2d - mean) / mean


def cross_power_2d(field_a: np.ndarray, field_b: np.ndarray, kg: KGrid):
    """FFT cross-power of two 2D maps, binned onto kg.k_bins. Returns (P, P_err, r)."""
    fft_a = np.fft.fftshift(np.fft.fft2(field_a))
    fft_b = np.fft.fftshift(np.fft.fft2(field_b))
    norm = (kg.pix_size_mpc / kg.n_side) ** 2
    cross_2d = np.real(np.conj(fft_a) * fft_b) * norm
    auto_a_2d = np.abs(fft_a) ** 2 * norm
    auto_b_2d = np.abs(fft_b) ** 2 * norm

    n_kbins = kg.n_kbins
    P_cross = np.full(n_kbins, np.nan)
    P_err = np.full(n_kbins, np.nan)
    r_cross = np.full(n_kbins, np.nan)

    for j in range(n_kbins):
        mask = (kg.kgrid >= kg.k_bins[j]) & (kg.kgrid < kg.k_bins[j + 1])
        if mask.sum() == 0:
            continue
        cv = cross_2d[mask]
        PA = np.mean(auto_a_2d[mask])
        PB = np.mean(auto_b_2d[mask])
        P_cross[j] = np.mean(cv)
        dk_bin = kg.k_bins[j + 1] - kg.k_bins[j]
        n_modes = max(1, kg.k_centers[j] * dk_bin * kg.area_2d / (2 * np.pi))
        P_err[j] = np.sqrt(PA * PB + P_cross[j] ** 2) / np.sqrt(n_modes)
        denom = np.sqrt(PA * PB)
        if denom > 0:
            r_cross[j] = P_cross[j] / denom

    return P_cross, P_err, r_cross


def make_ell(k_centers: np.ndarray, chi_mpc: float, h: float) -> np.ndarray:
    """Limber approximation: ell = k * chi / h."""
    return k_centers * chi_mpc / h


def to_Cell(P_cross, P_err, chi_mpc: float, h: float):
    factor = h**2 / chi_mpc**2
    return P_cross * factor, P_err * factor


def to_Dell(ell, C_ell, C_ell_err, T_CMB_uK: float | None = None):
    D_ell = ell * (ell + 1) * C_ell / (2 * np.pi)
    D_ell_err = ell * (ell + 1) * C_ell_err / (2 * np.pi)
    if T_CMB_uK is not None:
        D_ell *= T_CMB_uK**2
        D_ell_err *= T_CMB_uK**2
    return D_ell, D_ell_err


def ell_at_redshift(cfg, kg: KGrid, z: float) -> np.ndarray:
    """Convenience: Limber ell grid at redshift z for this config's cosmology."""
    cosmo = get_cosmology(cfg)
    chi = cosmo.comoving_distance(z).to_value("Mpc")
    return make_ell(kg.k_centers, chi, little_h(cfg))
