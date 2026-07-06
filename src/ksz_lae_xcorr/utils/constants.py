"""
utils/constants.py
===================
Physical constants shared by the correlation and snr modules. Values that
depend on cosmology (e.g. the kSZ tau prefactor, which needs Ob0 and h) are
computed from the Config rather than hardcoded twice.
"""

from __future__ import annotations

C_KMS = 299792.458                  # km/s
CM_PER_MPC = 3.08568e24
MPC_PER_KM_S_TO_S = 1.0 / (3.08567758e19)  # km -> Mpc-equivalent factor used
                                            # for velocity unit conversions
SIGMA_T_CM2 = 6.6524e-25            # Thomson cross-section
M_P_G = 1.6726e-24                  # proton mass, grams
T_CMB_UK = 2.7255e6                 # muK


def c_mpc_per_s() -> float:
    return C_KMS / 3.08567758e19


def tau_prefactor(cfg) -> float:
    """n_H0 * sigma_T * (cm per Mpc) -- prefactor for the kSZ optical depth."""
    h = cfg.cosmology.H0 / 100.0
    Ob0 = cfg.cosmology.Ob0
    rho_crit_cgs = 1.88e-29 * h**2
    n_H0_cm3 = Ob0 * rho_crit_cgs / M_P_G
    return n_H0_cm3 * SIGMA_T_CM2 * CM_PER_MPC
