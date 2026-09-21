"""
io/lae_luminosity_function_reference.py
==========================================
Published LAE luminosity function Schechter parameters, for comparing
against our own simulated LAE catalogue's luminosity distribution
(tracers/luminosity_function.py). Unlike the La Plante+2022 reference
(digitized from figures), these are literal best-fit numbers quoted
directly in the source paper's text/tables -- no digitization involved,
so no pixel-reading uncertainty.

Source: Konno, A., Ouchi, M., Shibuya, T., et al. 2018, PASJ, 70, S16
(the SILVERRUSH survey) -- Table 4, the alpha=-1.5 (fixed) fits, chosen
because that slope is the common reference value used across most LAE LF
studies in this redshift range, making it the most broadly comparable
choice (the free-alpha fits have much larger uncertainties and differ
in convention paper to paper).

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
