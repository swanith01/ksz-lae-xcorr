"""
Lya IGM attenuation of an intrinsic LAE catalogue on our 300 Mpc boxes  (24 Nov 2026)
======================================================================================
Port of the optical-depth and observed-property stages of Jahaan Thakkar's current LAE pipeline
(reionization_robustness, branch grid_generation: tau.py and obs_prop/observed_prop.py), adapted to our boxes:

  * sightlines run along the box z axis (the only velocity component we saved is velocity_z), starting AT the halo
    cell and going forward n_fwd cells with periodic wrap (Jahaan: forward half of a centred LOS, n_fwd = n // 2);
    the halo cell index is floor(x / dx), which removes the off-by-one of extractLOS' astype(int) truncation;
  * density = evolved delta (density_evolved.npy from sim_generation/.../repair), floored at 1 + delta >= 0;
  * peculiar velocity = velocity_z [Mpc/s, comoving-coordinate rate] * (km per Mpc) / (1 + z);
  * cosmology = astropy Planck18, which is the py21cmfast 4.1.0 default the boxes were run with;
  * cell length dx in cMpc (our box is 300 cMpc, not cMpc/h).

Physics is unchanged from Jahaan's code: Voigt profile (wofz) with thermal width b = sqrt(2 k T / (m_p + m_e)),
hydrogen fraction 0.76, comoving gas density x path length with the (1 + z)^2 factor, Hubble flow + peculiar velocity
relative to the halo cell, Gaussian emission line of width 88 km/s centred at voff, transmission averaged over
+/-5000 km/s with Simpson's rule, observed cuts L > 1e42 erg/s and REW > 0.
"""
import numpy as np
from astropy import units as u
from astropy.constants import c, k_B, m_e, m_p
from astropy.cosmology import Planck18
from scipy.integrate import simpson
from scipy.special import wofz

COSMO = Planck18
MPC_CM = u.Mpc.to(u.cm)
MPC_KM = u.Mpc.to(u.km)
L_ALPHA_CM = 1215.6e-8
H_FRAC = 0.76
OSC = 0.4164
GAMMA_LYA = 6.262e8          # s^-1
THOMSON = 6.65245e-25        # cm^2
SIGMA_V = 88.0               # km/s, emission-line width
WINDOW = 5000.0              # km/s, transmission window
N_B_COMOVING_CM3 = (COSMO.critical_density0.to_value(u.kg / u.m**3) * COSMO.Ob0 / m_p.value) / 1e6


def op2(z):
    """Lya cross-section prefactor (cgs), including the (1 + z)^2 of comoving density x comoving path."""
    return (H_FRAC * OSC * np.pi * L_ALPHA_CM * np.sqrt(3 * THOMSON / (8 * np.pi))
            * c.to_value(u.cm / u.s) * (1 + z) ** 2 / np.sqrt(np.pi))


def cell_velocity(z, dx_mpc):
    """Hubble-flow velocity across one comoving cell [km/s]."""
    return COSMO.H(z).to_value(u.km / u.s / u.Mpc) * dx_mpc / (1 + z)


def spectral_grid(z, dx_mpc, n, mode="cell", dv_fine=10.0):
    """Photon velocity offsets [km/s] inside +/-WINDOW.  'cell': Jahaan's n+1 samples at the cell Hubble spacing
    (identical results to his code inside the window); 'fine': uniform dv_fine spacing."""
    if mode == "cell":
        v = (np.arange(n + 1) - n // 2) * cell_velocity(z, dx_mpc)
    elif mode == "fine":
        v = np.arange(-WINDOW, WINDOW + 0.5 * dv_fine, dv_fine)
    else:
        raise ValueError(mode)
    return v[np.abs(v) <= WINDOW]


def halo_cells(coords_cmpc, dx_mpc, n):
    """Integer cell indices of positions in cMpc (floor, periodic)."""
    return np.floor(np.asarray(coords_cmpc, dtype=np.float64) / dx_mpc).astype(np.int64) % n


def sightlines(box, ijk, n_fwd):
    """Values of `box` along +z from each halo cell: shape (N, n_fwd); column 0 is the halo cell."""
    n = box.shape[2]
    k = (ijk[:, 2, None] + np.arange(n_fwd)[None, :]) % n
    return box[ijk[:, 0, None], ijk[:, 1, None], k]


def optical_depth(z, delta, x_hi, temp, v_pec, v_spec, dx_mpc):
    """tau(v_spec) for each sightline.  delta, x_hi, temp [K], v_pec [km/s]: arrays (N, n_fwd) starting at the halo
    cell.  Positive v_spec is redward.  Returns (N, len(v_spec))."""
    n_fwd = delta.shape[1]
    gas_v = np.arange(n_fwd) * cell_velocity(z, dx_mpc)
    v_arr = v_spec[:, None] + gas_v[None, :]                                        # (S, n_fwd)
    dens = np.clip(1.0 + np.asarray(delta, dtype=np.float64), 0.0, None) * N_B_COMOVING_CM3
    x_hi = np.asarray(x_hi, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    v_pec = np.asarray(v_pec, dtype=np.float64)
    va = (v_arr[None, :, :] - v_pec[:, 0, None, None] + v_pec[:, None, :]) * 1e5   # cm/s, (N, S, n_fwd)
    b = np.sqrt(2 * k_B.value * temp[:, None, :] / (m_p.value + m_e.value)) * 1e2  # cm/s
    aa = GAMMA_LYA * L_ALPHA_CM / (4 * np.pi * b)
    profile = np.real(wofz(1j * aa + va / b))
    op1 = dens[:, None, :] * x_hi[:, None, :] * profile * dx_mpc * MPC_CM / b
    return op1.sum(axis=2) * op2(z)


def transmission(tau, voff, v_spec, sigma_v=SIGMA_V):
    """Line-profile-weighted transmission (Jahaan's igm_transmission), on the same grid as tau."""
    log_prof = -0.5 * ((v_spec[None, :] - np.asarray(voff)[:, None]) / sigma_v) ** 2
    prof = np.exp(log_prof - log_prof.max(axis=1, keepdims=True))
    return simpson(np.exp(-tau) * prof, x=v_spec, axis=1) / simpson(prof, x=v_spec, axis=1)


def attenuate(z, fields, coords_cmpc, voff, dx_mpc, n_fwd, spec="cell", batch=2000):
    """Transmission for every LAE.  fields: dict delta, x_hi, temp [K], vz_raw [Mpc/s, as saved] on the n^3 grid."""
    n = fields["delta"].shape[0]
    v_spec = spectral_grid(z, dx_mpc, n, mode=spec)
    vfac = MPC_KM / (1 + z)
    ijk = halo_cells(coords_cmpc, dx_mpc, n)
    out = np.empty(len(voff))
    for s in range(0, len(voff), batch):
        e = min(s + batch, len(voff))
        sl = ijk[s:e]
        tau = optical_depth(z, sightlines(fields["delta"], sl, n_fwd), sightlines(fields["x_hi"], sl, n_fwd),
                            sightlines(fields["temp"], sl, n_fwd), sightlines(fields["vz_raw"], sl, n_fwd) * vfac,
                            v_spec, dx_mpc)
        out[s:e] = transmission(tau, voff[s:e], v_spec)
    return out
