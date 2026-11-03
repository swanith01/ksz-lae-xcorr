import numpy as np

from ksz_lae_xcorr.snr import sim_snr as ss


def test_modes_match_integer_sum():
    e = np.array([100.0, 200.0, 400.0])
    m = ss.modes_in_range(e, 100, 400)
    exact = [sum(2 * l + 1 for l in range(100, 200)), sum(2 * l + 1 for l in range(200, 400))]
    assert np.allclose(m, exact, rtol=0.01)
    assert np.allclose(ss.modes_in_range(e, 150, 300), [200 ** 2 - 150 ** 2, 300 ** 2 - 200 ** 2])


def test_constant_spectra_closed_form():
    e = np.geomspace(100, 10000, 21)
    Cs, Cg, N, fs = 1e-3, 2e-4, 5e-6, 0.05
    t = ss.snr_terms(np.full(20, Cs ** 2), np.full(20, Cg), N, e, fs)
    expect = fs * (10000 ** 2 - 100 ** 2) * Cs ** 2 / (N * Cg + Cs ** 2)
    assert np.isclose(t.sum(), expect, rtol=1e-10)


def test_cumulative_ends_at_total_and_is_monotonic():
    rng = np.random.default_rng(0)
    e1, e2 = np.geomspace(150, 30000, 30), np.geomspace(190, 40000, 30)
    t1, t2 = rng.random(29), rng.random(29)
    L = np.geomspace(150, 1e5, 60)
    c = ss.cumulative_snr([t1, t2], [e1, e2], L, ell_min=100)
    assert np.all(np.diff(c) >= -1e-12)
    assert np.isclose(c[-1], np.sqrt(t1.sum() + t2.sum()))


def test_debias_never_exceeds_raw_and_kills_pure_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, size=(10, 50))
    assert np.all(ss.signal_sq(noise, True) <= ss.signal_sq(noise, False) + 1e-15)
    assert ss.signal_sq(noise, True).mean() < 0.3 * ss.signal_sq(noise, False).mean() + 0.05
    sig = 5 + 0.1 * noise
    assert np.allclose(ss.signal_sq(sig, True), 25, rtol=0.05)


def test_non_overlapping_windows():
    z0 = np.arange(6.5, 13.01, 0.5)
    i = ss.non_overlapping(z0, np.ones_like(z0))
    assert np.allclose(z0[i], np.arange(6.5, 13.01, 1.0))


def test_survey_cg_and_shot_noise():
    Cg = ss.survey_cg(np.array([3.0, 1.0]), 1.0, 2.0, 100.0, 10.0, 0.5)
    assert np.allclose(Cg, [4 * 2 * 10 + 2, 0 + 2])
    chi = lambda z: 1000.0 * np.asarray(z)
    n = lambda z: np.full_like(np.asarray(z, float), 1e-3)
    C = ss.window_shot_noise(n, 7.0, 8.0, chi, 7500.0)
    assert np.isclose(C, 1.0 / (1e-3 * 1000.0 * 7500.0 ** 2))   # 1/N per sr


def test_field_ell_min_and_overlap():
    from ksz_lae_xcorr.snr import feasibility as fe
    assert np.isclose(fe.field_ell_min(1.0), 2 * np.pi / np.deg2rad(1.0))       # ~360 for 1 deg2
    assert fe.field_ell_min(1e4) == 100.0
    assert fe.overlap("HSC SILVERRUSH (NB816+NB921)", "SPT-3G Main") == 0.0
    assert fe.overlap("HSC SILVERRUSH (NB816+NB921)", "ACT DR6") == 1.0
