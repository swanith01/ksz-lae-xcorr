"""field_units_check on a synthetic box whose velocity is built from first principles."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.field_units_check import KM_PER_MPC, band_median, compare_to_linear_theory
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio

N, L, OM = 64, 400.0, 0.3086
Z, H_KMS, F = 9.0, 1500.0, 1.0


def P0(k):                       # arbitrary smooth z=0 'linear' spectrum [Mpc^3]
    return 2.0e4 * (k / 0.05) / (1 + (k / 0.05) ** 3.2)


def Pk_lin(k, z):
    return P0(k) * float(growth_factor_ratio(OM, z)) ** 2


def _fields(kind, seed=0):
    rng = np.random.default_rng(seed)
    kf = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kf, kf, kf, indexing="ij")
    k = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2); k[0, 0, 0] = 1.0
    white = np.fft.fftn(rng.normal(size=(N, N, N)))
    d0 = white * np.sqrt(P0(k) * N ** 3 / L ** 3); d0[0, 0, 0] = 0.0
    D = float(growth_factor_ratio(OM, Z)); a = 1 / (1 + Z); H_s = H_KMS / KM_PER_MPC
    # comoving rate: dx/dt = f H D * i k_z/k^2 * delta_0 ; peculiar = a * that
    vk = 1j * KZ / k ** 2 * (F * H_s * D) * d0
    if kind == "peculiar":
        vk = vk * a
    return np.fft.ifftn(d0).real, np.fft.ifftn(vk).real


@pytest.mark.parametrize("kind,expect", [("peculiar", 1.0), ("comoving_rate", (1 + Z) ** 2)])
def test_recovers_units_hypothesis_and_flat_in_k(kind, expect):
    d0, v = _fields(kind)
    r = compare_to_linear_theory(d0, v, L, Z, Pk_lin, H_KMS, F, n_kbins=10)
    med, scat = band_median(r, "R_v", 0.05, 0.4)
    assert med == pytest.approx(expect, rel=0.12)
    assert scat < 0.25                                   # flat in k (mode-count noise only)
    med_d, _ = band_median(r, "R_d_z0", 0.05, 0.4)
    assert med_d == pytest.approx(1.0, rel=0.12)         # z=0-normalised IC: ratio to P_lin(k,0) is 1
    med_dz, _ = band_median(r, "R_d_z", 0.05, 0.4)
    assert med_dz == pytest.approx(float(growth_factor_ratio(OM, Z)) ** -2, rel=0.12)
