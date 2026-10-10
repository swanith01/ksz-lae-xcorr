"""
zreion/box.py
================
A self-contained, memory-aware implementation of the semi-numerical reionisation model used by
La Plante, Sipple & Lidz (2022, ApJ 928, 162), Sec. 2.1 (zreion: Battaglia+13):

  1. matter density delta_m(r) at the midpoint z_mean  (they: 2LPT particles + TSC deposit;
     here: Gaussian linear field -> Zel'dovich (1LPT) displacement -> TSC deposit, or linear only),
  2. delta_z(k) = b_zm(k) delta_m(k),  b_zm = b0 / (1 + k/k0)^alpha,  b0 = 1/delta_c = 0.593   (Eqs. 2-3),
  3. 1 + z_re(r) = (1 + z_mean) (1 + delta_z(r))                                                 (Eq. 1),
  4. x_i(r, z) = 1 where z_re(r) > z, else 0.

Units: comoving Mpc (NOT h^-1 Mpc), k in Mpc^-1, except k0 which is quoted in h Mpc^-1 (converted with h).
Cosmology default = La Plante+22: Om=0.316, Ob=0.049, h=0.673, sigma8=0.812, ns=0.966.

The 2LPT -> Zel'dovich simplification changes the density PDF at the percent level at z~8 on ~2 h^-1 Mpc
cells; the quantity to check is the volume-averaged history x_HII(z) (paper Fig. 6: x_HII(z=8)=0.43 for the
fiducial run), which is sensitive to the skewness of delta_m.

Memory: n^3 float32 = 4 GB at n = 1024.  The displacement/deposit step is done in slabs; the FFT fields are
the peak (about 8-10 n^3 floats).  Use scripts/32_zreion_minitest.py to measure before scaling up.
"""
from __future__ import annotations

import numpy as np

LP22_COSMO = dict(Om=0.316, Ob=0.049, h=0.673, sigma8=0.812, ns=0.966)
DELTA_C = 1.686
B0 = 1.0 / DELTA_C                      # 0.593


# ------------------------------------------------------------------ linear power spectrum
def _eh_nowiggle_T(k_mpc, Om, Ob, h):
    """Eisenstein & Hu (1998) zero-baryon-oscillation transfer function (k in Mpc^-1)."""
    om_h2, ob_h2 = Om * h * h, Ob * h * h
    theta = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / om_h2) / np.sqrt(1 + 10 * ob_h2 ** 0.75)
    alpha_g = 1 - 0.328 * np.log(431 * om_h2) * Ob / Om + 0.38 * np.log(22.3 * om_h2) * (Ob / Om) ** 2
    k = np.asarray(k_mpc, dtype=np.float64)
    gamma = Om * h * (alpha_g + (1 - alpha_g) / (1 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / gamma
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def sigma_R(pk_func, R_mpc: float) -> float:
    """rms of the linear field smoothed with a top hat of radius R [Mpc]."""
    k = np.logspace(-5, 2, 4000)
    x = k * R_mpc
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    integrand = k ** 3 * pk_func(k) * W ** 2 / (2 * np.pi ** 2)
    y, x_ = integrand, np.log(k)
    return float(np.sqrt(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x_))))    # trapezoid (works on numpy 1 and 2)


def linear_pk(cosmo: dict | None = None, backend: str = "auto"):
    """Returns pk(k) [Mpc^3] at z = 0, k in Mpc^-1, normalised to cosmo['sigma8'].
    backend 'camb' (needs camb), 'eh' (Eisenstein-Hu no-wiggle, ~few % in shape) or 'auto' (camb if importable)."""
    c = dict(LP22_COSMO, **(cosmo or {}))
    use_camb = backend == "camb"
    if backend == "auto":
        try:
            import camb  # noqa: F401
            use_camb = True
        except Exception:
            use_camb = False
    if use_camb:
        import camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100 * c["h"], ombh2=c["Ob"] * c["h"] ** 2, omch2=(c["Om"] - c["Ob"]) * c["h"] ** 2)
        pars.InitPower.set_params(As=2.1e-9, ns=c["ns"])
        pars.set_matter_power(redshifts=[0.0], kmax=200.0)
        pars.NonLinear = camb.model.NonLinear_none
        res = camb.get_results(pars)
        kh, _, pkh = res.get_matter_power_spectrum(minkh=1e-5, maxkh=150.0, npoints=800)
        k_tab, p_tab = kh[:] * c["h"], pkh[0] / c["h"] ** 3
        raw = lambda k: np.exp(np.interp(np.log(k), np.log(k_tab), np.log(p_tab)))   # noqa: E731
    else:
        raw = lambda k: np.asarray(k, float) ** c["ns"] * _eh_nowiggle_T(k, c["Om"], c["Ob"], c["h"]) ** 2   # noqa: E731
    s8 = sigma_R(raw, 8.0 / c["h"])
    amp = (c["sigma8"] / s8) ** 2
    return lambda k: amp * raw(np.asarray(k, float))


# ------------------------------------------------------------------ fields
def _kgrid(n, boxsize):
    kf = 2 * np.pi / boxsize
    k1 = np.fft.fftfreq(n, d=1.0 / n) * kf
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kf
    return k1, kz


def gaussian_delta(n: int, boxsize: float, pk_func, seed: int = 1, dtype=np.float32):
    """Real-space Gaussian field with <delta_k delta_k*> = V P(k)  (z=0 linear).  Also returns delta_k (rfft)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n, n), dtype=dtype)
    wk = np.fft.rfftn(w)
    del w
    k1, kz = _kgrid(n, boxsize)
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    kk[0, 0, 0] = 1.0
    amp = np.sqrt(pk_func(kk) * n ** 3 / boxsize ** 3).astype(dtype)
    amp[0, 0, 0] = 0.0
    wk *= amp
    return np.fft.irfftn(wk, s=(n, n, n), axes=(0, 1, 2)).astype(dtype), wk


def tsc_deposit(pos, n: int, out=None):
    """Triangular-shaped-cloud deposit of unit-mass particles; pos in grid units (cell centres at integers),
    periodic.  Adds into `out` (n^3 float64 flat) and returns it."""
    if out is None:
        out = np.zeros(n ** 3, dtype=np.float64)
    i0 = np.floor(pos + 0.5)
    d = pos - i0                                           # in [-0.5, 0.5)
    w = (0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2)
    idx0 = i0.astype(np.int64)
    for sx in (-1, 0, 1):
        ix = (idx0[:, 0] + sx) % n
        wx = w[sx + 1][:, 0]
        for sy in (-1, 0, 1):
            iy = (idx0[:, 1] + sy) % n
            wxy = wx * w[sy + 1][:, 1]
            base = (ix * n + iy) * n
            for sz in (-1, 0, 1):
                iz = (idx0[:, 2] + sz) % n
                out += np.bincount(base + iz, weights=wxy * w[sz + 1][:, 2], minlength=n ** 3)
    return out


def zeldovich_density(delta_k_z, n: int, boxsize: float, slab: int = 8) -> np.ndarray:
    """Zel'dovich (1LPT) matter overdensity delta_m = rho/rhobar - 1 from the linear field delta_k_z (rfft of the
    linear overdensity AT the target redshift): displace one particle per cell, deposit with TSC.  Processes
    `slab` lattice planes at a time to bound memory.  Returns float32 (n,n,n)."""
    k1, kz = _kgrid(n, boxsize)
    kk2 = k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2
    kk2[0, 0, 0] = 1.0
    dx = boxsize / n
    psi = []
    for ax in range(3):
        kax = (k1[:, None, None], k1[None, :, None], kz[None, None, :])[ax]
        psik = 1j * kax * delta_k_z / kk2                 # grad.psi = -delta
        psik[0, 0, 0] = 0.0
        psi.append(np.fft.irfftn(psik, s=(n, n, n), axes=(0, 1, 2)).astype(np.float32) / dx)   # displacement in grid units
        del psik
    rho = np.zeros(n ** 3, dtype=np.float64)
    j = np.arange(n, dtype=np.float32)
    qy, qz = np.meshgrid(j, j, indexing="ij")
    for i0 in range(0, n, slab):
        i1 = min(n, i0 + slab)
        m = (i1 - i0) * n * n
        pos = np.empty((m, 3), dtype=np.float32)
        qx = np.repeat(np.arange(i0, i1, dtype=np.float32), n * n)
        pos[:, 0] = qx + psi[0][i0:i1].reshape(-1)
        pos[:, 1] = np.tile(qy.reshape(-1), i1 - i0) + psi[1][i0:i1].reshape(-1)
        pos[:, 2] = np.tile(qz.reshape(-1), i1 - i0) + psi[2][i0:i1].reshape(-1)
        tsc_deposit(pos, n, out=rho)
    return (rho.reshape(n, n, n) - 1.0).astype(np.float32)


# ------------------------------------------------------------------ zreion
def bias_zm(k_mpc, alpha: float = 0.2, k0_hmpc: float = 0.9, b0: float = B0, h: float = LP22_COSMO["h"]):
    """b_zm(k) = b0 / (1 + k/k0)^alpha   (La Plante+22 Eq. 3); k in Mpc^-1, k0 in h Mpc^-1."""
    return b0 / (1.0 + np.asarray(k_mpc, float) / (k0_hmpc * h)) ** alpha


def _tophat(x):
    """Fourier transform of a spherical top hat, W(kR) with x = kR (Taylor series for small x)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    out[...] = 3 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3
    out[small] = 1 - x[small] ** 2 / 10.0
    return out


def _sinc(x):
    x = np.asarray(x, dtype=np.float64)
    xs = np.where(np.abs(x) < 1e-6, 1.0, x)
    return np.where(np.abs(x) < 1e-6, 1 - x ** 2 / 6.0, np.sin(xs) / xs)


def apply_zreion(delta_m, boxsize: float, zmean: float = 8.0, alpha: float = 0.2, k0_hmpc: float = 0.9,
                 b0: float = B0, h: float = LP22_COSMO["h"], rsmooth_hmpc: float = 0.0,
                 deconvolve: bool = False) -> np.ndarray:
    """z_re(r) from the matter overdensity at z = zmean (Eqs. 1-3).  Returns float32 z_re field.

    Mirrors the public package `zreion` (P. La Plante, MIT; zreion.apply_zreion):
        delta_z(k) = delta_m(k) * b0 / (1 + k/k0)^alpha * W_tophat(k R_smooth) / W_CIC(k)
        z_re = (1 + z_mean) delta_z + z_mean
    rsmooth_hmpc : top-hat smoothing radius [h^-1 Mpc]  (package default 1.0; here default 0 = off so older
                   numbers reproduce -- the package default is the 'faithful' setting)
    deconvolve   : divide by the CIC window prod_i sinc^2(k_i dx / 2)  (package default True; only meaningful if
                   delta_m came from a CIC deposit -- ours is TSC, so a mismatch; test both)
    boxsize [Mpc]; k0 [h Mpc^-1] is converted with h; rsmooth [h^-1 Mpc] likewise."""
    n = delta_m.shape[0]
    dk = np.fft.rfftn(delta_m.astype(np.float32))
    k1, kz = _kgrid(n, boxsize)
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    fac = bias_zm(kk, alpha, k0_hmpc, b0, h)
    if rsmooth_hmpc > 0:
        fac = fac * _tophat(kk * (rsmooth_hmpc / h))
    if deconvolve:
        dx = boxsize / n
        w = (_sinc(k1[:, None, None] * dx / 2) * _sinc(k1[None, :, None] * dx / 2) * _sinc(kz[None, None, :] * dx / 2)) ** 2
        fac = fac / w
    dk *= fac.astype(np.float32)
    delta_z = np.fft.irfftn(dk, s=(n, n, n), axes=(0, 1, 2)).astype(np.float32)
    return ((1.0 + zmean) * delta_z + zmean).astype(np.float32)


def xhii_history(z_re, z_grid) -> np.ndarray:
    """Volume-averaged ionised fraction at each z: fraction of cells with z_re > z."""
    zr = np.sort(z_re.reshape(-1))
    return 1.0 - np.searchsorted(zr, np.asarray(z_grid, float), side="right") / zr.size
