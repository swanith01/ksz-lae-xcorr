"""
utils/optical_depth.py
=======================
Two small normalisation helpers for the kSZ line-of-sight integrand, both
ported in spirit from ksz-pipeline (src/ksz_pipeline/ksz/optical_depth.py)
but built on THIS repo's own cosmology (utils.cosmology.get_cosmology) and
constants -- no second cosmology is introduced.

1. ne_scale_helium(Y_He)
   utils.constants.tau_prefactor(cfg) computes Ob0*rho_crit/m_p and calls it
   "n_H0". That is the number of PROTON MASSES per unit volume, not the
   hydrogen (or electron) density: with helium mass fraction Y_He the true
   free-electron density for fully ionised H + singly-ionised He is

       n_e = (rho_b / m_p) * X_H * (1 + y),   X_H = 1 - Y_He,  y = Y_He/(4 X_H)
           = (rho_b / m_p) * (1 - 3/4 Y_He)            (= 0.82 for Y_He=0.24)

   so the legacy prefactor is 1/0.82 = 1.22x too large in n_e, i.e. ~1.49x
   too large in any kSZ power (D_ell ~ n_e^2).  ksz-pipeline's validated
   ne0_cgs() (2.064e-7 cm^-3) is the helium-inclusive value.  ne_scale_helium
   is the multiplicative correction; compute_ksz_slices applies it when
   field_data_seed["ne_scale"] is present.  Legacy callers (absent key) are
   untouched.

2. analytic_tau_below(cfg, z_min)
   Thomson optical depth from z=0 to z_min assuming a fully ionised IGM
   below the lowest simulated redshift.  compute_ksz_slices' cumulative tau
   starts at 0 at z_min, which drops ~0.03-0.04 of real optical depth
   (roughly half the Planck tau) and inflates D_ell by ~6-8% through the
   e^{-tau} visibility factor (ksz-pipeline documented the same omission).
   Consumed via field_data_seed["tau0"].
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology

_C_CM_S = 2.99792458e10
_CM_PER_KM = 1.0e5


def ne_scale_helium(Y_He: float = 0.24) -> float:
    """Factor n_e / (rho_b/m_p) = X_H (1 + y) = 1 - 3/4 Y_He. 0.82 at Y_He=0.24."""
    if not 0.0 <= Y_He < 1.0:
        raise ValueError(f"Y_He must be in [0, 1), got {Y_He}")
    return 1.0 - 0.75 * Y_He


def analytic_tau_below(cfg, z_min: float, Y_He: float = 0.24,
                        z_HeIII: float = 3.5, dz_HeIII: float = 0.5,
                        n_z: int = 4001) -> float:
    """
    Thomson optical depth accumulated over 0 <= z <= z_min for a fully
    ionised IGM (H + HeII everywhere; HeIII following a smooth tanh
    transition centred at z_HeIII, CAMB's fiducial timing -- not something
    the simulation constrains).

        tau = sigma_T * INT_0^z_min n_e(z) (1+z)^2 c / H(z) dz,
        n_e(z) = n_H0 [1 + y (1 + f_HeIII(z))] (1+z)^3 / (1+z)  -> (1+z)^2 below

    n_H0 = X_H Ob0 rho_crit0 / m_p, built from THIS repo's constants
    (same rho_crit convention as constants.tau_prefactor, so the
    in-simulation and below-z_min pieces share one baryon normalisation).
    Deterministic trapezoid on n_z points (no scipy.quad dependency, fast
    enough to call per seed).
    """
    if z_min <= 0.0:
        return 0.0
    cosmo = get_cosmology(cfg)
    h = cfg.cosmology.H0 / 100.0
    X_H = 1.0 - Y_He
    y = Y_He / (4.0 * X_H)
    rho_crit_cgs = 1.88e-29 * h ** 2
    n_H0 = X_H * cfg.cosmology.Ob0 * rho_crit_cgs / constants.M_P_G

    z = np.linspace(0.0, float(z_min), n_z)
    f_HeIII = 0.5 * (1.0 + np.tanh((z_HeIII - z) / dz_HeIII))
    n_e = n_H0 * (1.0 + y * (1.0 + f_HeIII)) * (1.0 + z) ** 2
    H_per_s = cosmo.H(z).to_value("km/s/Mpc") * _CM_PER_KM / (3.0856775814913673e24)
    integrand = n_e * _C_CM_S / H_per_s
    tau = constants.SIGMA_T_CM2 * np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(z))
    return float(tau)
