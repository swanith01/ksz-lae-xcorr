"""tests/test_feasibility.py -- snr/feasibility.py: filter normalisation, noise scalings, calibration against La Plante Table 2."""
import numpy as np
import pytest

from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils.config import load_config

CFG = "configs/fiducial.yaml"


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(CFG)
    comp = fe.lp_components(cfg)
    sig = fe.SignalModel(cfg)
    scale, raw = fe.calibrate_to_lp_table2(sig, comp)
    return cfg, comp, sig, scale, raw


def test_filter_peak_is_one(setup):
    _, comp, *_ = setup
    f = fe.experiment_filter(fe.EXPERIMENTS["SO"], comp)
    assert np.isclose(f.f.max(), 1.0)
    assert f.C_T2T2 > 0 and f.sigma2_reion > 0


def test_white_noise_level_and_beam():
    n0 = fe.white_noise_Cl(np.array([1.0]), 10.0, 1.4)[0]
    assert np.isclose(n0, (10.0 * np.pi / 180 / 60) ** 2, rtol=1e-3)               # (10 uK arcmin)^2 in rad^2
    n = fe.white_noise_Cl(np.array([3000.0]), 10.0, 1.4)[0]
    assert n > n0                                                                  # beam deconvolution raises the noise at high ell


def test_calibration_reproduces_lp_table2(setup):
    _, comp, sig, scale, raw = setup
    got = fe.lp_benchmark(sig, comp)
    for k, v in fe.LP_TABLE2_ILC_NO_SHOT.items():
        assert got[k] == pytest.approx(v, rel=0.05)
    # one common rescaling for the three experiments (the filter scaling is then validated, not fitted)
    vals = np.array(list(raw.values()))
    assert vals.max() / vals.min() < 1.05
    # the white-noise column was not used for the calibration
    w = fe.lp_benchmark(sig, comp, white=True)
    for k, v in fe.LP_TABLE2_WHITE_NO_SHOT.items():
        assert w[k] == pytest.approx(v, rel=0.3)


def test_snr_scales_as_sqrt_fsky_and_shot_noise_hurts(setup):
    _, comp, sig, *_ = setup
    ref = fe.experiment_filter(fe.EXPERIMENTS["SO"], comp)
    so = fe.experiment_filter(fe.EXPERIMENTS["SO"], comp)
    b = lambda z: np.full_like(np.asarray(z, float), 5.0)       # noqa: E731
    s1 = fe.snr_window(sig, so, ref, 8.5, 1.0, b, 0.01, None, n_ell=40)
    s4 = fe.snr_window(sig, so, ref, 8.5, 1.0, b, 0.04, None, n_ell=40)
    assert s4 / s1 == pytest.approx(2.0, rel=0.05)
    s_shot = fe.snr_window(sig, so, ref, 8.5, 1.0, b, 0.01, 1e5, n_ell=40)       # 1e5 galaxies per steradian: shot-noise dominated
    assert s_shot < s1


def test_late_history_has_more_signal_at_low_z(setup):
    cfg, comp, sig, scale, _ = setup
    late = fe.SignalModel(cfg, z_shift=1.0, scale=scale)
    b = lambda z: np.full_like(np.asarray(z, float), 5.0)       # noqa: E731
    assert late.D(6.6, 0.2, 1000.0, b) > sig.D(6.6, 0.2, 1000.0, b)


def test_limber_matter_power_decreases_with_window_width(setup):
    _, _, sig, *_ = setup
    narrow = fe.limber_cmm(sig.toy, 8.0, 0.2, 1000.0)
    wide = fe.limber_cmm(sig.toy, 8.0, 2.0, 1000.0)
    assert narrow > wide > 0


def test_lae_density_and_abundance_matching(setup):
    cfg, *_ = setup
    n1 = fe.lae_number_density(1e42, ("Umeda25", 5.7))
    n2 = fe.lae_number_density(1e43, ("Umeda25", 5.7))
    assert n1 > n2 > 0
    hb = fe.HaloBias(cfg)
    b_hi, M_hi = hb.bias_for_density(1e-5, 6.6)
    b_lo, M_lo = hb.bias_for_density(1e-3, 6.6)
    assert M_hi > M_lo and b_hi > b_lo > 1
