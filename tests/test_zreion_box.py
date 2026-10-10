"""Tests for zreion/box.py on small boxes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio
from ksz_lae_xcorr.zreion import (LP22_COSMO, apply_zreion, bias_zm, gaussian_delta, linear_pk, sigma_R,
                                  tsc_deposit, xhii_history, zeldovich_density)

N, L = 48, 300.0


def test_sigma8_normalisation():
    pk = linear_pk(backend="eh")
    assert abs(sigma_R(pk, 8.0 / LP22_COSMO["h"]) - LP22_COSMO["sigma8"]) < 2e-3


def test_gaussian_field_power_matches_input():
    pk = linear_pk(backend="eh")
    d, dk = gaussian_delta(N, L, pk, seed=3)
    assert abs(d.mean()) < 1e-6
    # variance on the grid = (1/V) sum_k P(k)  (checked loosely: statistical)
    k1 = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / L
    kz = np.fft.rfftfreq(N, d=1.0 / N) * 2 * np.pi / L
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2); kk[0, 0, 0] = 1
    wt = np.full(kk.shape, 2.0); wt[..., 0] = 1.0; wt[..., -1] = 1.0
    pred = float(np.sum(wt * pk(kk)) - pk(kk[0, 0, 0])) / L ** 3
    assert 0.8 < d.var() / pred < 1.2


def test_tsc_conserves_mass_and_is_periodic():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-3, 20, size=(5000, 3))
    rho = tsc_deposit(pos, 16)
    assert abs(rho.sum() - 5000) < 1e-6 and rho.min() >= 0


def test_zeldovich_small_amplitude_matches_linear():
    pk = linear_pk(backend="eh")
    d, dk = gaussian_delta(N, L, pk, seed=5)
    dz = 0.02                                       # tiny growth factor -> still linear
    dm = zeldovich_density(dk * dz, N, L, slab=6)
    assert abs(dm.mean()) < 1e-5
    # compare on large scales only (one particle per cell + TSC is not faithful near the grid scale)
    k1 = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / L
    kz = np.fft.rfftfreq(N, d=1.0 / N) * 2 * np.pi / L
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    lowpass = lambda f: np.fft.irfftn(np.fft.rfftn(f) * (kk < 0.25 * np.pi * N / L),   # noqa: E731
                                      s=(N, N, N), axes=(0, 1, 2))
    lin = lowpass(d * dz).ravel(); nl = lowpass(dm).ravel()
    r = np.corrcoef(lin, nl)[0, 1]
    assert r > 0.98 and 0.9 < nl.std() / lin.std() < 1.05


def test_zeldovich_nonlinear_is_positively_skewed():
    pk = linear_pk(backend="eh")
    d, dk = gaussian_delta(N, L, pk, seed=7)
    D8 = float(growth_factor_ratio(LP22_COSMO["Om"], 8.0))
    dm = zeldovich_density(dk * D8, N, L)
    s = ((dm - dm.mean()) ** 3).mean() / dm.std() ** 3
    assert s > 0.0 and dm.min() > -1.0 - 1e-6


def test_bias_limits():
    assert abs(bias_zm(1e-6) - 0.5933) < 1e-3
    kk = 0.9 * LP22_COSMO["h"]
    assert abs(bias_zm(kk) - 0.5933 / 2 ** 0.2) < 1e-3


def test_zreion_history_is_monotonic_and_mean_redshift_is_zmean():
    pk = linear_pk(backend="eh")
    d, dk = gaussian_delta(N, L, pk, seed=9)
    D8 = float(growth_factor_ratio(LP22_COSMO["Om"], 8.0))
    dm = d * D8                                                    # linear density at z=8
    zre = apply_zreion(dm, L, zmean=8.0)
    assert abs(zre.mean() - 8.0) < 0.05
    zg = np.linspace(2, 20, 40)
    x = xhii_history(zre, zg)
    assert np.all(np.diff(x) <= 1e-12) and x[0] > 0.99 and x[-1] < 0.01
    assert 0.45 < xhii_history(zre, [8.0])[0] < 0.55              # Gaussian field: median at the mean


def test_larger_alpha_shortens_reionisation():
    pk = linear_pk(backend="eh")
    d, dk = gaussian_delta(N, L, pk, seed=11)
    dm = d * float(growth_factor_ratio(LP22_COSMO["Om"], 8.0))
    zg = np.array([6.0, 10.0])
    x_lo = xhii_history(apply_zreion(dm, L, alpha=0.2), zg)
    x_hi = xhii_history(apply_zreion(dm, L, alpha=1.0), zg)
    # larger alpha -> less small-scale power in delta_z -> narrower z_re distribution -> SHORTER reionisation,
    # i.e. x_HII drops by more between two fixed redshifts
    assert (x_hi[0] - x_hi[1]) > (x_lo[0] - x_lo[1])


def _reference_loops(delta, L, zmean, alpha, k0, h, R, deconv, b0=1 / 1.686):
    """Direct transcription of the cython kernel in the public `zreion` package (zreion.pyx), small grids only."""
    n = delta.shape[0]
    dk = np.fft.fftn(delta)
    out = np.empty_like(dk)
    Lh = L * h                                     # Mpc/h
    sinc = lambda x: 1 - x ** 2 / 6 if abs(x) < 1e-6 else np.sin(x) / x   # noqa: E731
    th = lambda x: 1 - x ** 2 / 10 if abs(x) < 1e-6 else 3 * (np.sin(x) - x * np.cos(x)) / x ** 3   # noqa: E731
    fr = np.fft.fftfreq(n, d=1.0 / n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                kx, ky, kz_ = (2 * np.pi / n * fr[i], 2 * np.pi / n * fr[j], 2 * np.pi / n * fr[k])
                kp = np.sqrt(kx ** 2 + ky ** 2 + kz_ ** 2) * (n / Lh)          # h/Mpc
                f = b0 / (1 + kp / k0) ** alpha * th(kp * R)
                if deconv:
                    f /= (sinc(kx / 2) * sinc(ky / 2) * sinc(kz_ / 2)) ** 2
                out[i, j, k] = dk[i, j, k] * f
    dz = np.fft.ifftn(out).real
    return dz * (1 + zmean) + zmean


def test_apply_zreion_matches_package_kernel_on_small_grid():
    rng = np.random.default_rng(2)
    n, L = 8, 40.0
    d = rng.normal(0, 0.3, size=(n, n, n)).astype(np.float32); d -= d.mean()
    for R, dec in ((0.0, False), (1.0, False), (1.0, True), (0.0, True)):
        ref = _reference_loops(d, L, 8.0, 0.2, 0.9, LP22_COSMO["h"], R, dec)
        mine = apply_zreion(d, L, 8.0, 0.2, 0.9, rsmooth_hmpc=R, deconvolve=dec)
        assert np.max(np.abs(ref - mine)) < 2e-4, (R, dec, np.max(np.abs(ref - mine)))


def test_smoothing_reduces_and_deconvolution_raises_variance():
    pk = linear_pk(backend="eh")
    d, dk = gaussian_delta(N, L, pk, seed=13)
    dm = d * float(growth_factor_ratio(LP22_COSMO["Om"], 8.0))
    s0 = apply_zreion(dm, L).std()
    assert apply_zreion(dm, L, rsmooth_hmpc=1.0).std() < s0
    assert apply_zreion(dm, L, deconvolve=True).std() > s0
