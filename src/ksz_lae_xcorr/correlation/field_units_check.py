"""
correlation/field_units_check.py
==================================
INDEPENDENT test of what the raw py21cmfast coeval fields are, against linear
theory with an external P_lin(k, z) (CAMB) -- unlike correlation/
velocity_convention_check.py, which divided P_v(raw) by P_delta(raw
hires_density) and therefore could not see that BOTH are mis-scaled (their
errors cancel: obs/theory = (D/a)^2, reported as 'correction 0.78-0.90' = a/D).

For a snapshot at redshift z (flat LCDM, f = growth rate, H in s^-1) the PECULIAR-velocity
prediction is  P_v,pec(k) = (1/3) (a f H / k)^2 P_lin(k, z)   (angle average <k_z^2/k^2> = 1/3), and

  R_v(k) = P_vz,raw(k) / P_v,pec(k)  =  1         if raw velocity_z is v_pec  [Mpc/s]
                                      =  (1+z)^2  if raw is the comoving-coordinate rate dx/dt = v_pec/a

  R_d(k)  = P_hires(k) / P_lin(k, 0) =  1  if hires_density is the z=0-normalised linear IC field
  R_d'(k) = P_hires(k) / P_lin(k, z) =  1  if it is already delta(z)   (then R_d = D(z)^-2... i.e. R_d' != 1 otherwise)

Scale dependence: both ratios should be FLAT in k over the linear, unsmoothed range
(k ~ 0.04-0.4 Mpc^-1 for a 300 Mpc box at 1 Mpc cells); a k-dependent ratio would point to
something other than a pure normalisation/units factor.

The estimator takes P_lin as a callable, so it is unit-tested without CAMB.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.velocity_convention_check import radial_power_spectrum_3d

KM_PER_MPC = 3.0856775814913673e19


def compare_to_linear_theory(delta_ic, vz_raw, box_len_mpc: float, z: float,
                             Pk_lin, H_kms_mpc: float, f: float, n_kbins: int = 15) -> dict:
    """
    Pk_lin(k, z) -> P_lin in Mpc^3 (k in Mpc^-1).  vz_raw in Mpc/s (as saved), delta_ic = the
    saved hires_density block-averaged to the vz grid.  Returns per-k-bin arrays plus the
    predictions of each hypothesis.
    """
    a = 1.0 / (1.0 + z)
    pv = radial_power_spectrum_3d(vz_raw, box_len_mpc, n_kbins)
    pd = radial_power_spectrum_3d(delta_ic, box_len_mpc, n_kbins)
    k = pv["k_centers"]
    H_s = H_kms_mpc / KM_PER_MPC                                  # 1/s
    Pv_pec = (1.0 / 3.0) * (a * f * H_s / k) ** 2 * Pk_lin(k, z)   # Mpc^2/s^2 * Mpc^3
    return {
        "k": k, "z": z,
        "R_v": pv["P"] / Pv_pec,               # 1 => raw is v_pec ; (1+z)^2 => raw is v_pec/a
        "R_d_z0": pd["P"] / Pk_lin(k, 0.0),    # 1 => hires_density is the z=0-normalised linear field
        "R_d_z": pd["P"] / Pk_lin(k, z),       # 1 => already delta(z)
        "pred_R_v_peculiar": 1.0, "pred_R_v_comoving_rate": (1.0 + z) ** 2,
    }


def band_median(res: dict, key: str, kmin: float = 0.04, kmax: float = 0.4):
    m = (res["k"] >= kmin) & (res["k"] <= kmax) & np.isfinite(res[key])
    return (float(np.median(res[key][m])), float(np.std(np.log(res[key][m])))) if m.any() else (np.nan, np.nan)
