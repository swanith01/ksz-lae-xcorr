"""tracers/suite.py on synthetic data: gridding, selection, bias recovery, ST reference."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.tracers import suite as S  # noqa: E402
from ksz_lae_xcorr.zreion.box import gaussian_delta, linear_pk  # noqa: E402


def test_spec_names_roundtrip_and_parse():
    for spec in S.default_specs():
        back = S.spec_from_name(spec.name)
        assert back.kind == spec.kind
        assert np.isclose(np.log10(back.cut), np.log10(spec.cut), atol=0.06) if spec.kind != "lbg" else back.cut == spec.cut
    ps = S.parse_specs("halo:1e10, lae:1e42, lbg:-19")
    assert [p.kind for p in ps] == ["halo", "lae", "lbg"]
    assert len(S.parse_specs("default")) == len(S.default_specs())
    q = S.parse_specs("halo:10.5;lae:42.5;halo:1e11")
    assert np.isclose(q[0].cut, 10 ** 10.5) and np.isclose(q[1].cut, 10 ** 42.5) and q[2].cut == 1e11


def test_count_grid_matches_histogramdd_and_wraps():
    rng = np.random.default_rng(1)
    n, L = 16, 80.0
    c = rng.uniform(-5, L + 5, (5000, 3))                 # includes out-of-box -> periodic wrap
    g = S.count_grid(c, n, L / n, chunk=1234)
    ref, _ = np.histogramdd((c % L), bins=[np.linspace(0, L, n + 1)] * 3)
    assert g.shape == (n, n, n) and g.sum() == 5000
    assert np.allclose(g, ref)


def test_select_members_halo_lae_lbg():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 100, (1000, 3)).astype(np.float32)
    masses = 10 ** rng.uniform(8, 12, 1000).astype(np.float32)
    ids = np.sort(rng.choice(1000, 200, replace=False))
    lum = 10 ** rng.uniform(41, 44, 200)
    muv = rng.uniform(-22, -15, 200)
    c, m = S.select_members(S.TracerSpec("halo", 1e10), coords, masses)
    assert np.all(m > 1e10) and len(c) == len(m)
    c, m = S.select_members(S.TracerSpec("lae", 1e43), coords, masses, lae=(ids, lum))
    assert len(m) == int(np.sum(lum >= 1e43))
    c, m = S.select_members(S.TracerSpec("lbg", -19.0), coords, masses, lbg=(ids, muv))
    assert len(m) == int(np.sum(muv < -19.0))
    c, m = S.select_members(S.TracerSpec("lae", 1e43), coords, masses, lae=None)      # catalogue missing -> empty
    assert len(m) == 0


def test_bias_recovery_on_poisson_tracer():
    n, L, b_true = 48, 240.0, 3.0
    pk = linear_pk(backend="eh")
    delta, _ = gaussian_delta(n, L, pk, seed=3)
    delta = (delta * 0.12 / np.std(delta)).astype(np.float64)        # b*rms = 0.36: the 1+b*delta clip is negligible
    rate = 40.0 * np.clip(1.0 + b_true * delta, 0.0, None)
    grid = np.random.default_rng(4).poisson(rate).astype(np.float32)
    res = S.halo_bias_from_grids(grid, delta, L, kmax=0.3)
    assert abs(res["b_lin"] - b_true) < 0.35, res["b_lin"]
    assert res["n_obj"] == grid.sum()
    # unbiased tracer -> b = 1
    grid1 = np.random.default_rng(5).poisson(40.0 * np.clip(1.0 + delta, 0, None)).astype(np.float32)
    assert abs(S.halo_bias_from_grids(grid1, delta, L, kmax=0.3)["b_lin"] - 1.0) < 0.25
    # empty tracer
    assert np.isnan(S.halo_bias_from_grids(np.zeros((n, n, n), np.float32), delta, L)["b_lin"])


def test_sheth_tormen_bias_trends():
    sig = S.SigmaOfM(Om=0.3086, Ob=0.0489, h=0.6777, ns=0.9665, backend="eh")
    assert sig(1e9) > sig(1e12)                                   # sigma falls with mass
    b = S.sheth_tormen_bias(np.array([1e9, 1e10, 1e11, 1e12]), 8.0, sig)
    assert np.all(np.diff(b) > 0)                                 # more massive -> more biased
    assert S.sheth_tormen_bias(np.array([1e11]), 10.0, sig)[0] > S.sheth_tormen_bias(np.array([1e11]), 6.0, sig)[0]
    assert S.sheth_tormen_bias(np.array([1e12]), 8.0, sig)[0] > 5.0       # rare haloes at z=8 are strongly biased
    assert np.isnan(S.mean_bias_of_masses(np.array([]), 8.0, sig))
    assert np.isclose(S.mean_bias_of_masses(np.full(10, 1e11), 8.0, sig), S.sheth_tormen_bias(np.array([1e11]), 8.0, sig)[0])


class _PowerLawSigma:
    """stub sigma(M) = (M / 1e13)^-0.2 (D = 1 at z=0): wide dynamic range in nu without needing a power spectrum."""
    Om = 0.3
    rho_m = 4e10

    def __call__(self, m):
        return (np.asarray(m, dtype=float) / 1e13) ** -0.2

    def dlnsigma_dlnm(self, m):
        return np.full_like(np.asarray(m, dtype=float), -0.2)


def test_sheth_tormen_mass_function_is_normalised():
    """integral of M dn/dlnM / rho_m over ln M = integral of f(nu) dnu = 1 (the multiplicity function is normalised)."""
    from ksz_lae_xcorr.tracers.suite import sheth_tormen_dndlnm
    sig = _PowerLawSigma()
    lnm = np.linspace(np.log(1e-25), np.log(1e45), 20000)         # nu from ~1e-5 to >30
    m = np.exp(lnm)
    f = sheth_tormen_dndlnm(m, 0.0, sig) * m / sig.rho_m
    assert abs(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(lnm)) - 1.0) < 0.02


def test_sheth_tormen_abundance_trends_with_the_real_sigma_of_m():
    from ksz_lae_xcorr.tracers.suite import SigmaOfM, sheth_tormen_n_gt
    sig = SigmaOfM(Om=0.3086, Ob=0.0489, h=0.6777, ns=0.9665, sigma8=0.8102, backend="auto")
    n = {(z, m): sheth_tormen_n_gt(m, z, sig) for z in (6.0, 8.0, 10.0) for m in (1e9, 1e10, 1e11)}
    for z in (6.0, 8.0, 10.0):
        assert n[(z, 1e9)] > n[(z, 1e10)] > n[(z, 1e11)] > 0
    for m in (1e9, 1e10, 1e11):
        assert n[(6.0, m)] > n[(8.0, m)] > n[(10.0, m)]
    assert 0.02 < n[(6.0, 1e10)] < 0.12                  # ~0.06 Mpc^-3 (comoving) at z=6
    assert sig.dlnsigma_dlnm(1e10) < 0
