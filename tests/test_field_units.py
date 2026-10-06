"""utils/field_units.py + the field_convention switch of the wrap-cycle builder."""
import logging
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import (
    build_wrapcycle_field_data,
    compute_wrapcycle_auto_power,
    implied_velocity_kms,
    normalisation_keys,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher
from ksz_lae_xcorr.utils.field_units import (
    delta_at_z_from_ic,
    growth_factor_ratio,
    peculiar_velocity_from_raw,
)

from test_ksz_auto_wrapcycle import _cfg, _write_tree

OM = 0.3086


def _ode_growth(Om, z_eval):
    """Independent D(z)/D(0): D'' + (2 + dlnE/dlna) D' - 1.5 Om a^-3/E^2 D = 0 in ln a."""
    def E2(a):
        return Om * a ** -3 + (1 - Om)

    def rhs(la, y):
        a = np.exp(la)
        dlnE = -1.5 * Om * a ** -3 / E2(a)
        return [y[1], -(2 + dlnE) * y[1] + 1.5 * Om * a ** -3 / E2(a) * y[0]]

    a0 = 1e-4
    sol = solve_ivp(rhs, [np.log(a0), 0.0], [a0, a0], t_eval=np.sort(np.log(1 / (1 + np.asarray(z_eval))))[::1],
                    rtol=1e-10, atol=1e-14)
    return sol.y[0]


def test_growth_matches_independent_ode_and_limits():
    zs = np.array([9.0, 6.0, 1.0, 0.0])
    ode = _ode_growth(OM, zs)
    ode = ode / ode[-1]
    np.testing.assert_allclose(growth_factor_ratio(OM, zs), ode, rtol=2e-3)
    assert growth_factor_ratio(OM, 0.0) == pytest.approx(1.0)
    # Einstein-de Sitter: D = a
    zs2 = np.array([0.0, 1.0, 5.0])
    np.testing.assert_allclose(growth_factor_ratio(1.0, zs2), 1 / (1 + zs2), rtol=1e-4)
    # the number that explains the old 'correction factor 0.78' at z=9.04: a/D = 0.784
    assert (1 / 10.04) / float(growth_factor_ratio(OM, 9.04)) == pytest.approx(0.784, abs=0.003)


def test_conversions():
    assert peculiar_velocity_from_raw(10.0, 9.0) == pytest.approx(1.0)
    d = np.array([-22.0, 0.0, 4.5])
    np.testing.assert_allclose(delta_at_z_from_ic(d, OM, 9.0), d * growth_factor_ratio(OM, 9.0))


def test_physical_vs_legacy_scaling_and_floor(tmp_path):
    _write_tree(tmp_path)
    cfg = _cfg(tmp_path)
    lg = logging.getLogger("t_fu")
    fds = {}
    for conv in ("legacy", "physical"):
        fds[conv] = build_wrapcycle_field_data(cfg, 1, 7, Stitcher(cfg), lg,
                                                norm=normalisation_keys(cfg), field_convention=conv)
    leg, phy = fds["legacy"], fds["physical"]
    # snapshots span z=5.6..8.0 -> v scaled by 1/(1+z) in [1/9, 1/6.6]
    rv = np.sqrt(np.mean(phy["velocity_lc"] ** 2) / np.mean(leg["velocity_lc"] ** 2))
    assert 1 / 9.0 * 0.9 < rv < 1 / 6.6 * 1.1
    # linear growth ratio at those snapshots: D(8)/D0 .. D(5.6)/D0
    lo, hi = growth_factor_ratio(OM, 8.0), growth_factor_ratio(OM, 5.6)
    rd = np.sqrt(np.mean((phy["density_lc"] - 1) ** 2) / np.mean((leg["density_lc"] - 1) ** 2))
    assert lo * 0.8 < rd < hi * 1.2
    assert phy["density_lc"].min() >= 0.0 and phy["field_convention"] == "physical"
    # kSZ power falls by ~ (1/(1+z))^2 * growth^2-ish: strictly lower
    a = compute_wrapcycle_auto_power(cfg, leg)
    b = compute_wrapcycle_auto_power(cfg, phy)
    assert np.nanmedian(b["full"]["D_diag"]) < 0.1 * np.nanmedian(a["full"]["D_diag"])
    assert b["field_convention"] == "physical"
    iv = implied_velocity_kms(cfg, b, z_targets=(6.5, 7.0))
    assert len(iv) == 2 and all(np.isfinite(v) for _, _, v in iv)
    with pytest.raises(ValueError):
        build_wrapcycle_field_data(cfg, 1, 7, Stitcher(cfg), lg, field_convention="nope")
