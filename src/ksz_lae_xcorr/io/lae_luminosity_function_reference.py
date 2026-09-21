"""
io/lae_luminosity_function_reference.py
==========================================
Published LAE luminosity function data, for comparing against our own
simulated LAE catalogue's luminosity distribution
(tracers/luminosity_function.py). Two sources:

1. KAGEURA2025 -- Kageura, Y., Ouchi, M., Nakane, M., et al. 2025, ApJS,
   278, 33 ("Census of Ly-alpha Emission from ~600 Galaxies at z=5-14").
   JWST/NIRSpec spectroscopic sample. Both the discrete data points
   (their Table 2, log-Phi at three fixed log-L values, five redshift
   bins spanning z~5 to z~10-14) AND the best-fit Schechter parameters
   (their Table 3) are literal numbers quoted directly in the paper's
   text/tables -- no digitization involved, no pixel-reading
   uncertainty. In their Table 3, L_star and alpha are themselves FIXED
   to values from Umeda et al. 2024b at each redshift; only phi_star is
   fit freely by Kageura et al.

2. KONNO2018 -- Konno et al. 2018, PASJ, 70, S16 (SILVERRUSH), kept as a
   second, independent, ground-based (Subaru narrow-band) cross-check at
   z=5.7/6.6, overlapping Kageura's z~5/6 bins.

NOT YET INCLUDED: Umeda et al. 2025's own Lya LF/clustering measurement
(Subaru narrow-band survey) -- referenced as a comparison dataset in
Kageura et al. 2025's own Figure 6, but this repo does not yet have its
precise tabulated numbers (found only as comparison points in someone
else's figure, not as directly-quoted values) -- follow up separately
if a precise version is needed.

Schechter function, standard form:
    Phi(L) dL = phi_star * (L/L_star)^alpha * exp(-L/L_star) * dL/L_star

Converted to per-dex (dlogL) convention, matching
tracers.luminosity_function.compute_luminosity_function's own convention:
    Phi(L) dlogL = ln(10) * phi_star * (L/L_star)^(alpha+1) * exp(-L/L_star)
"""

from __future__ import annotations

import numpy as np

# {redshift: (alpha, log10(L_star [erg/s]), log10(phi_star [Mpc^-3]))}
KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED = {
    5.7: (-1.5, 43.06, -3.25),
    6.6: (-1.5, 43.12, -3.73),
}

# Kageura et al. 2025, Table 3 -- best-fit Schechter parameters.
# {mean redshift: (alpha [fixed to Umeda+2024b], log10(L_star) [fixed], log10(phi_star) [free, fit by Kageura])}
KAGEURA2025_SCHECHTER_PARAMS = {
    5.01: (-1.61, 42.69, -2.50),
    5.90: (-1.39, 42.75, -3.03),
    6.96: (-2.49, 43.23, -4.45),
    8.41: (-2.56, 43.03, -4.98),
    11.00: (-2.56, 43.03, -6.33),
}

# Kageura et al. 2025, Table 2 -- discrete measured log10(Phi) [Mpc^-3 dex^-1]
# at three fixed log10(L [erg/s]) values, with asymmetric 1-sigma errors.
# Missing entries (z~8-9 and z~10-14 have no log(L)=43.3 measurement) are
# simply absent -- not zero, not interpolated.
KAGEURA2025_LF_POINTS = {
    5.01: {
        "log_l": [42.3, 42.8, 43.3],
        "log_phi": [-2.45, -2.80, -3.49],
        "log_phi_err_lo": [0.12, 0.10, 0.15],
        "log_phi_err_hi": [0.11, 0.09, 0.15],
    },
    5.90: {
        "log_l": [42.3, 42.8, 43.3],
        "log_phi": [-2.80, -3.25, -4.00],
        "log_phi_err_lo": [0.13, 0.15, 0.26],
        "log_phi_err_hi": [0.08, 0.14, 0.06],
    },
    6.96: {
        "log_l": [42.3, 42.8, 43.3],
        "log_phi": [-2.91, -3.87, -4.47],
        "log_phi_err_lo": [0.31, 0.24, 0.14],
        "log_phi_err_hi": [0.20, 0.19, 0.29],
    },
    8.41: {
        "log_l": [42.3, 42.8],
        "log_phi": [-3.59, -4.38],
        "log_phi_err_lo": [0.48, 0.74],
        "log_phi_err_hi": [0.36, 0.11],
    },
    11.00: {
        "log_l": [42.3, 42.8],
        "log_phi": [-4.57, -6.07],
        "log_phi_err_lo": [0.83, 0.59],
        "log_phi_err_hi": [0.52, 0.65],
    },
}


def schechter_phi_per_dex(log_l: np.ndarray, alpha: float, log_l_star: float, log_phi_star: float) -> np.ndarray:
    """Phi(L) dlogL at the given log10(L) values, for a Schechter function
    with the given parameters. Returns Mpc^-3 dex^-1."""
    log_l = np.asarray(log_l, dtype=float)
    L_over_Lstar = 10 ** (log_l - log_l_star)
    phi_star = 10 ** log_phi_star
    return np.log(10) * phi_star * L_over_Lstar ** (alpha + 1) * np.exp(-L_over_Lstar)


def nearest_konno2018_redshift(z: float) -> float:
    """The closer of the two available reference redshifts (5.7 or 6.6) to z."""
    available = list(KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED.keys())
    return min(available, key=lambda z_ref: abs(z_ref - z))


def nearest_kageura2025_redshift(z: float) -> float:
    """The closest of Kageura+2025's five reference redshift bins to z."""
    available = list(KAGEURA2025_SCHECHTER_PARAMS.keys())
    return min(available, key=lambda z_ref: abs(z_ref - z))
