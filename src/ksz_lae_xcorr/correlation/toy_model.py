"""
correlation/toy_model.py
=========================
An analytic TOY for the shape of D_ell^cross(x_HII) = ell(ell+1)/2pi C_ell( [F Theta]^2  x  delta_g,window ), the kSZ^2 x galaxy
cross-power of La Plante+22.  It is a squeezed-limit response picture, not a bispectrum calculation:

  * the small-scale kSZ power of the line-of-sight slice at z, s(z), is modulated by the large-scale environment delta_L(z) of
    that slice:  s -> s (1 + R(z) delta_L);  the galaxy map is  g = int w b delta.   A Limber integral gives
        C_ell = (1/dz) int dz  s'(z) R(z) b_g(z) P_mm(ell/chi, z) / chi^2          (galaxy window = top hat of width dz in z)
  * the slice power (the "diagonal", slice-by-slice picture: scripts/41 ordering B) is
        s'(z) = K(z)^2 v(z)^2 [ xbar^2 + eps xbar (1 - xbar) ],    K = tau_pref (1+z)^2 exp(-tau),  v ~ a H f D
    the first term is the kSZ of the mean ionised fraction with the velocity field, the second the patchy term (the variance of a
    binary ionisation field is xbar (1 - xbar)); eps = (patchy power) / (homogeneous power) at the filter scale is the ONE free
    shape parameter of the toy.
  * the response to a long-wavelength density mode has two parts
        density/velocity:  (1+delta)^2 -> 2 delta_L (+ r_v delta_L from the velocity variance; r_v = 0 by default)
        ionisation:        zreion: z_re -> z_re + (1+z) b_zm(k) delta_L(z_mean) shifts the ionised fraction by
                           dxbar/ddelta = (1+z) b_zm (D(z_mean)/D(z)) p(z),  p = -dxbar/dz
    so   s' R = K^2 v^2 { [xbar^2 + eps xbar(1-xbar)] (2 + r_v)  +  [2 xbar + eps (1 - 2 xbar)] dxbar/ddelta }.
  The overall amplitude (T_CMB^2, the filter's share of the slice power, v_rms) is one multiplicative constant A, fitted.

What the toy is for: which ingredient (density/velocity response vs ionisation modulation) sets WHERE the curve peaks in x_HII
and how fast it falls at late times, and whether that already reproduces La Plante's Fig. 7 without any simulation.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio
from ksz_lae_xcorr.zreion.box import B0, bias_zm, linear_pk


def _trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))        # trapezoid (numpy 1 and 2)


def history_fn(z_pts, x_pts, smooth_dz: float = 0.3, tail_dz: float = 1.0):
    """Mean ionised fraction xbar(z): a tabulated history made monotonic (non-increasing with z) and smoothed with a Gaussian of
    width `smooth_dz` in z (its derivative p = -dxbar/dz is what the ionisation response needs; digitised curves are noisy), 1 below
    the table; above it the last value decays as exp(-(z - z_max)/tail_dz) (a hard cut to 0 would put a spike into
    p = -dxbar/dz at the end of a digitised table)."""
    from scipy.ndimage import gaussian_filter1d
    z_pts, x_pts = np.asarray(z_pts, float), np.asarray(x_pts, float)
    o = np.argsort(z_pts)
    z_pts, x_pts = z_pts[o], np.minimum.accumulate(np.clip(x_pts[o], 0.0, 1.0))
    zf = np.linspace(min(z_pts.min(), 4.0) - 1.0, max(z_pts.max(), 20.0) + 1.0, 6000)
    xf = np.interp(zf, z_pts, x_pts, left=1.0, right=0.0)
    hi = zf > z_pts.max()
    xf[hi] = x_pts[-1] * np.exp(-(zf[hi] - z_pts.max()) / tail_dz)
    if smooth_dz > 0:
        xf = np.minimum.accumulate(gaussian_filter1d(xf, smooth_dz / (zf[1] - zf[0]), mode="nearest"))

    def fn(z):
        return np.interp(np.asarray(z, float), zf, xf, left=1.0, right=0.0)

    return fn


class Toy:
    """Pre-computes the redshift functions on a fine grid; `cross` returns the toy D_ell(z0) (up to the amplitude A)."""

    def __init__(self, cfg, xbar, z_grid=None, eps: float = 1.0, r_v: float = 0.0, zmean: float = 8.0, bias_fn=bluetides_bias_gz,
                 pk_backend: str = "eh", use_zm_bias: bool = True):
        self.cfg, self.xbar_fn, self.eps, self.r_v, self.zmean, self.bias_fn = cfg, xbar, eps, r_v, zmean, bias_fn
        self.use_zm_bias = use_zm_bias
        self.cosmo = get_cosmology(cfg)
        self.z = np.linspace(0.05, 22.0, 4400) if z_grid is None else np.asarray(z_grid, float)
        z = self.z
        self.chi = self.cosmo.comoving_distance(z).to_value("Mpc")
        ds = np.gradient(self.chi)
        tau_pref = constants.tau_prefactor(cfg)
        x = self.xbar_fn(z)
        self.tau = np.cumsum(tau_pref * x * (1 + z) ** 2 * ds)                    # optical depth from z = 0 (xbar = 1 below the history)
        self.K = tau_pref * (1 + z) ** 2 * np.exp(-self.tau)
        om = self.cosmo.Om0
        Ez = self.cosmo.efunc(z)
        omz = om * (1 + z) ** 3 / Ez ** 2
        self.D = growth_factor_ratio(om, z)
        self.v2 = (Ez * omz ** 0.55 * self.D / (1 + z)) ** 2                       # (a H f D)^2, H in units of H0
        self.x = x
        self.p = -np.gradient(x, z)                                                # -dxbar/dz >= 0
        self.pk = linear_pk(backend=pk_backend)
        self.Dm = float(growth_factor_ratio(om, zmean))

    # ---- the pieces of s'(z) R(z) b(z) (before the P_mm / chi^2 factor) ---------------------------------------------------
    def pieces(self, k):
        """(s', density/velocity response, ionisation response) on the z grid, each times K^2 v^2 (arbitrary units)."""
        z, x, e = self.z, self.x, self.eps
        base = self.K ** 2 * self.v2
        s = base * (x ** 2 + e * x * (1 - x))
        bz = bias_zm(k) if self.use_zm_bias else B0
        x_delta = (1 + z) * bz * (self.Dm / self.D) * self.p
        r_dens = s * (2.0 + self.r_v)
        r_ion = base * (2 * x + e * (1 - 2 * x)) * x_delta
        return s, r_dens, r_ion

    def cross(self, z0: float, dz: float, ell: float) -> dict:
        """Toy cross-power of the window [z0 - dz/2, z0 + dz/2] at multipole ell: {'D', 'D_dens', 'D_ion', 'x_hii'} (A = 1)."""
        z = self.z
        m = (z >= z0 - 0.5 * dz) & (z <= z0 + 0.5 * dz)
        k = ell / self.chi
        _, r_dens, r_ion = self.pieces(k)
        P = self.D ** 2 * self.pk(k)
        w = self.bias_fn(z) * P / self.chi ** 2
        pref = ell * (ell + 1) / (2 * np.pi) / dz
        zz = z[m]
        out = {}
        for name, r in (("D_dens", r_dens), ("D_ion", r_ion)):
            out[name] = float(pref * _trap(r[m] * w[m], zz))
        out["D"] = out["D_dens"] + out["D_ion"]
        out["x_hii"] = float(np.mean(self.x[m]))
        return out

    def curve(self, z0_list, dz: float, ell: float) -> dict:
        rows = [self.cross(z0, dz, ell) for z0 in z0_list]
        return {k: np.array([r[k] for r in rows]) for k in rows[0]} | {"z0": np.asarray(z0_list, float)}


def fit_amplitude(model, target, weights=None) -> float:
    """Least-squares A in  target ~ A * model  (both arrays on the same x_HII points)."""
    model, target = np.asarray(model, float), np.asarray(target, float)
    w = np.ones_like(model) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(model) & np.isfinite(target)
    den = float(np.sum(w[ok] * model[ok] ** 2))
    return float(np.sum(w[ok] * model[ok] * target[ok]) / den) if den > 0 else float("nan")
