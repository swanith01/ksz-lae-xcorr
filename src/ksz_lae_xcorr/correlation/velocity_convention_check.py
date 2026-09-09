"""
correlation/velocity_convention_check.py
===========================================
DEFINITIVE check of what units py21cmfast's raw velocity_z is actually
in -- not from documentation (searched extensively, found no explicit
statement: the autoapi docs have literally no docstring on this specific
attribute), but from first-principles linear theory, checked directly
against real data.

THE PHYSICS: in linear theory, the continuity equation relates the
line-of-sight peculiar velocity to the density field via

    v_los(k, z) = i (k_los / k^2) * f(z) * a(z) * H(z) * delta(k, z)

(e.g. Zel'dovich 1970; see also the standard derivation in any
large-scale-structure review, or Iliev+2007's kSZ radiative-transfer
paper for an equivalent form). Averaging over angles (<k_los^2/k^2> = 1/3
for an isotropic field), this gives an exact, testable relationship
between the two fields' power spectra:

    P_v(k, z) = (1/3) * (f(z) a(z) H(z) / k)^2 * P_delta(k, z)

This is checkable directly on real data: measure BOTH power spectra from
the SAME snapshot's raw fields (density AND raw, unconverted velocity_z),
take their ratio, and compare to the theory prediction. Whatever
constant multiplicative factor is needed to make the two match IS the
real conversion factor -- determined empirically from physics, not
guessed from a formula that might be for the wrong py21cmfast version.

If the ratio-of-ratios (observed / theory) is close to 1 with NO
conversion applied to the raw velocity_z at all, that confirms py21cmfast
v4's coeval velocity_z is already a genuine physical velocity. If it's
close to 1 only after applying some other specific factor, THAT is the
real correction -- read off empirically, not assumed.
"""

from __future__ import annotations

import numpy as np


def radial_power_spectrum_3d(field: np.ndarray, box_len_mpc: float, n_kbins: int = 15):
    """
    Generic isotropic 3D power spectrum estimator, standard convention
    P(k) = (V/N^6) |FFT(field)|^2, radially binned. field should already
    be mean-subtracted (or will be here) -- same convention used
    throughout this repo (see correlation/direct_bispectrum.py).

    Returns {'k_centers', 'P'} with log-spaced bins from the box's own
    fundamental mode to its Nyquist frequency.
    """
    n = field.shape[0]
    field = field - field.mean()
    field_k = np.fft.fftn(field)
    vol = box_len_mpc**3
    power_3d = np.abs(field_k) ** 2 * vol / n**6

    kfreq = np.fft.fftfreq(n, d=box_len_mpc / n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_fund = 2 * np.pi / box_len_mpc
    k_nyq = np.pi * n / box_len_mpc
    k_edges = np.geomspace(k_fund, k_nyq, n_kbins + 1)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])

    P = np.full(n_kbins, np.nan)
    for j in range(n_kbins):
        mask = (kmag >= k_edges[j]) & (kmag < k_edges[j + 1])
        if np.any(mask):
            P[j] = np.mean(power_3d[mask])
    return {"k_centers": k_centers, "P": P}


def theoretical_velocity_over_density_ratio(cfg, k: np.ndarray, z: float) -> np.ndarray:
    """
    (1/3) * (f(z) a(z) H(z) / k)^2 -- the linear-theory prediction for
    P_v(k)/P_delta(k), continuity equation, angle-averaged. H(z) in
    km/s/Mpc (astropy convention via get_cosmology), so this predicts the
    ratio assuming v is measured in km/s and k in Mpc^-1. If checking
    against v in Mpc/s instead, divide this prediction by
    constants.MPC_PER_KM_S_TO_S**2 (the km/s -> Mpc/s conversion applied
    to velocity, squared for a power spectrum) -- see the module-level
    check function below, which handles this explicitly.
    """
    from ksz_lae_xcorr.lightcone.stitch import _growth_rate_linder
    from ksz_lae_xcorr.utils.cosmology import get_cosmology

    cosmo = get_cosmology(cfg)
    a = 1.0 / (1.0 + z)
    H = cosmo.H(z).to_value("km/s/Mpc")
    f = _growth_rate_linder(cosmo, z)
    return (1.0 / 3.0) * (f * a * H / k) ** 2


def check_velocity_convention(cfg, density_1plus_delta: np.ndarray, raw_velocity_z: np.ndarray,
                               box_len_mpc: float, z: float, n_kbins: int = 15) -> dict:
    """
    THE definitive check. Computes P_v_raw(k) and P_delta(k) from the
    SAME real snapshot, and the theoretical P_v/P_delta ratio (in km/s
    convention). Reports, per k bin, the empirical correction factor
    (in km/s) that would need to be applied to raw_velocity_z to match
    linear theory -- sqrt(P_v_theory / P_v_raw) -- so you can see directly
    whether it's close to 1 (raw is already correct, in SOME unit -- check
    which by comparing to km/s vs Mpc/s) at every k (a real physical
    velocity field) or wildly off / k-dependent (something else is wrong).

    Returns a dict with per-k-bin arrays: 'k_centers', 'P_v_raw',
    'P_delta', 'ratio_observed', 'ratio_theory_kms', 'correction_factor_kms'
    (multiply raw_velocity_z by this to get km/s, if constant across k),
    'correction_factor_mpc_s' (same, but interpreting raw as already
    being scaled to give Mpc/s -- i.e. correction_factor_kms /
    MPC_PER_KM_S_TO_S, for comparing to a hypothesis that raw is already
    close to Mpc/s directly).
    """
    from ksz_lae_xcorr.utils import constants

    delta = density_1plus_delta - 1.0
    pv = radial_power_spectrum_3d(raw_velocity_z, box_len_mpc, n_kbins)
    pd = radial_power_spectrum_3d(delta, box_len_mpc, n_kbins)

    k = pv["k_centers"]
    ratio_observed = pv["P"] / pd["P"]
    ratio_theory_kms = theoretical_velocity_over_density_ratio(cfg, k, z)

    # sqrt(theory/observed) = the multiplicative factor needed on
    # raw_velocity_z (in whatever units it's currently in) to make its
    # power spectrum match the km/s theory prediction exactly.
    correction_factor_kms = np.sqrt(ratio_theory_kms / ratio_observed)
    correction_factor_mpc_s = correction_factor_kms * constants.MPC_PER_KM_S_TO_S

    return {
        "k_centers": k,
        "P_v_raw": pv["P"],
        "P_delta": pd["P"],
        "ratio_observed": ratio_observed,
        "ratio_theory_kms": ratio_theory_kms,
        "correction_factor_kms": correction_factor_kms,
        "correction_factor_mpc_s": correction_factor_mpc_s,
    }
