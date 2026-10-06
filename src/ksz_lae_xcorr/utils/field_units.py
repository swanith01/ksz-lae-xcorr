"""
utils/field_units.py
=====================
Physical-field conversions for the raw py21cmfast v4 coeval outputs this repo
saves (halos/coeval_pipeline.py).  Both conversions were found MISSING on
2026-10-06 and are what made the first wrap-cycle kSZ auto-power ~700x too big
(D_total ~ 700-950 uK^2 at l~2800 instead of O(1)):

1. velocity_z  (coeval.perturbed_field) is the COMOVING-COORDINATE rate
       dx_com/dt = (dD/dt) * psi_0           [Mpc/s]
   (py21cmfast/src/PerturbedField.c:  velocity_k *= (dDdt/D) * i k / k^2 on the
   linear density at z), NOT the peculiar velocity.  The physical peculiar
   velocity (what the kSZ integrand needs) is  v_pec = a * dx_com/dt, i.e.
       v_pec = velocity_z / (1+z).
   Evidence on real data (seed 1, z=9.04): raw std = 1135 km/s per component;
   linear theory at z=9 gives ~110-150 km/s, i.e. a factor (1+z)=10 smaller.

2. hires_density  is saved from coeval.INITIAL_conditions, not from the
   perturbed field: it is the LINEAR density extrapolated to z=0 (growth
   normalised to D(0)=1), identical for every snapshot of a seed (std 4.5, min
   -22 at 0.5 Mpc cells -- impossible for delta(z) >= -1).  The linear density
   at z is  delta(z) = D(z)/D(0) * hires_density.

The old velocity-convention check (correlation/velocity_convention_check.py)
compared P_v(raw) with (a H f/k)^2 P_delta(raw hires_density): the two errors
cancel almost exactly (obs/theory = (D/a)^2 = 1.62 at z=9 -> reported
"correction factor 0.78-0.90" = a/D = 0.784), so it could not see either.

Caveat of fix 2: delta(z) here is LINEAR (the nonlinear/Zel'dovich-displaced
density is not saved); at 1 Mpc cells sigma(delta) ~ 0.3-0.6 over z=6-9 so it is
a reasonable first approximation, but the proper fix is to save
perturbed_field.density per snapshot.  1+delta is floored at 0 by callers.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def _flat_lcdm_E(a: np.ndarray, Om0: float) -> np.ndarray:
    return np.sqrt(Om0 * a ** -3 + (1.0 - Om0))


def growth_factor_ratio(Om0: float, z) -> np.ndarray:
    """D(z)/D(0) for flat LCDM without radiation (the Heath/Carroll+92 integral that
    21cmFAST's dicke() uses):  D(a) ~ E(a) INT_0^a da' / (a' E(a'))^3."""
    la = np.linspace(np.log(1e-8), 0.0, 40001)
    a = np.exp(la)
    I = cumulative_trapezoid(a ** -2 * _flat_lcdm_E(a, Om0) ** -3, la, initial=0.0)
    D = _flat_lcdm_E(a, Om0) * I
    D = D / D[-1]
    z = np.asarray(z, dtype=np.float64)
    return np.interp(np.log(1.0 / (1.0 + z)), la, D)


def peculiar_velocity_from_raw(vz_raw, z):
    """v_pec [Mpc/s] = raw velocity_z [Mpc/s, comoving-coordinate rate] / (1+z)."""
    return vz_raw / (1.0 + z)


def delta_at_z_from_ic(delta_ic, Om0: float, z):
    """Linear delta(z) = D(z)/D(0) * delta_IC  (delta_IC = hires_density, z=0-normalised)."""
    return delta_ic * float(growth_factor_ratio(Om0, z))
