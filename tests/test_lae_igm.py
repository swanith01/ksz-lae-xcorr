"""Check src/ksz_lae_xcorr/lae/igm.py against a verbatim reference copy of Jahaan's current tau / transmission code
(reionization_robustness, grid_generation: tau.py lya_optical_depth + vel_axis, obs_prop/observed_prop.py
igm_transmission), on a synthetic box where both conventions can be compared exactly."""
import numpy as np
from astropy import units as u
from astropy.constants import c, k_B, m_e, m_p
from scipy.integrate import simpson
from scipy.special import wofz

from ksz_lae_xcorr.lae import igm

# ---------------- reference: Jahaan's code (cosmology swapped to igm.COSMO so the comparison is exact) -------------
cosmo = igm.COSMO
MPC = u.Mpc.to(u.m)
smallh = cosmo.h


def j_op2(z):
    return (0.76 * 0.4164 * np.pi * 1215.6e-8 * np.sqrt(3 * 6.65245e-25 / (8 * np.pi))
            * c.to_value(u.cm / u.s) * (1 + z) ** 2 / np.sqrt(np.pi))


def j_vel_axis(z, l, n):
    dx = l / n
    dv = cosmo.H(z).value * dx / (smallh * (1 + z))
    v = (np.arange(n + 1) - n // 2) * dv
    gas_v = np.arange(n - n // 2) * dv
    return dx, v[:, None] + gas_v[None, :], v


def j_tau(z, D, X, T, V, v_arr, dx):
    half = D.shape[1] // 2
    density, neutral, temperature, velocity = D[:, half:], X[:, half:], T[:, half:], V[:, half:]
    va = (v_arr[None, :, :] - velocity[:, 0, None, None] + velocity[:, None, :]) * 1e5
    b = np.sqrt(2 * k_B.value * temperature[:, None, :] / (m_p.value + m_e.value)) * 1e2
    aa = 6.262e8 * 1215.6e-8 / (4 * np.pi * b)
    profile = np.real(wofz(1j * aa + va / b))
    op1 = density[:, None, :] * neutral[:, None, :] / 1e6 * profile * dx * MPC * 1e2 / (smallh * b)
    return np.sum(op1, axis=2) * j_op2(z)


def j_transmission(tau, voff, velocity):
    use = (velocity >= -5000) & (velocity <= 5000)
    v = velocity[use]
    lp = -0.5 * ((v[None, :] - voff[:, None]) / 88) ** 2
    prof = np.exp(lp - lp.max(axis=1, keepdims=True))
    return simpson(np.exp(-tau[:, use]) * prof, x=v, axis=1) / simpson(prof, x=v, axis=1)


def j_extract(box, xyz_cells, n):          # Jahaan's extractLOS for one axis, with an exact (floor) halo cell
    rho_b = cosmo.critical_density(0).to_value(u.kg / u.m**3) * cosmo.Ob0
    out = []
    for name in ("D", "X", "T", "V"):
        arr = box[name]
        if name == "D":
            arr = (arr + 1) * rho_b / m_p.value
        rows = []
        for (i, j, k) in xyz_cells:
            col = arr[i, j, :]                   # our sightline axis is the box z axis
            rows.append(np.roll(col, n // 2 - k))
        out.append(np.array(rows))
    return out


def test_matches_reference():
    rng = np.random.default_rng(3)
    n, L, z = 40, 40.0, 7.0
    box = dict(D=rng.normal(0, 0.4, (n, n, n)).clip(-0.9), X=rng.uniform(0, 1, (n, n, n)),
               T=rng.uniform(1e3, 2e4, (n, n, n)), V=rng.normal(0, 4e-18, (n, n, n)) * (1 + z))
    coords = rng.uniform(0, L, (25, 3))
    ijk = igm.halo_cells(coords, L / n, n)
    voff = rng.uniform(0, 400, 25)
    # ours
    fields = dict(delta=box["D"], x_hi=box["X"], temp=box["T"], vz_raw=box["V"])
    T_ours = igm.attenuate(z, fields, coords, voff, L / n, n // 2, spec="cell")
    # reference (Jahaan's box length is cMpc/h)
    D, X, T, V = j_extract(box, ijk, n)
    V = V * MPC / (1e3 * (1 + z))
    dx_h, v_arr, v = j_vel_axis(z, L * smallh, n)
    T_ref = j_transmission(j_tau(z, D, X, T, V, v_arr, dx_h), voff, v)
    assert np.allclose(T_ours, T_ref, rtol=1e-10, atol=1e-12), np.max(np.abs(T_ours - T_ref))


def test_halo_cell_floor():
    n, L = 300, 300.0
    x = np.array([[10.3, 0.0, 0.0], [50.7, 0.0, 0.0], [299.99, 0.0, 0.0], [300.0, 0.0, 0.0]])
    assert list(igm.halo_cells(x, L / n, n)[:, 0]) == [10, 50, 299, 0]


def test_neutral_free_is_transparent():
    n, z = 20, 6.0
    f = dict(delta=np.zeros((n, n, n)), x_hi=np.zeros((n, n, n)), temp=np.full((n, n, n), 1e4), vz_raw=np.zeros((n, n, n)))
    T = igm.attenuate(z, f, np.array([[5.5, 5.5, 5.5]]), np.array([200.0]), 1.0, n // 2)
    assert np.allclose(T, 1.0)
