"""
snr/roman_hls_benchmark.py
============================
Stage 1 (Girish, 2026-09-07 Slack): reproduce La Plante, Sipple & Lidz
2022 (ApJ 928, 162; arXiv:2111.13717) under MATCHED assumptions, as a
pass/fail check on the estimator before trusting it for real LAE/LBG
catalogues.

Rather than use our own (still partial) LBG catalogue for the galaxy
field, this module builds the SAME linear-bias-weighted density field the
paper itself uses for its cross-correlation SIGNAL (their Eq. 6-7):

    delta_g(nhat) = INT dz Wg(z) bg(z) delta_m[chi(z) nhat, chi(z)]

directly from this simulation's own density field. This isolates "does
OUR simulation's kSZ^2 x matter-bispectrum physics produce a comparable
signal to theirs" from "does our specific galaxy catalogue match Roman
HLS" -- two different questions. See scripts/10_stage1_lbg_benchmark.py
for the catalogue-based (real LBG counts) complementary check -- running
both and comparing is more informative than either alone.

CAVEATS, stated plainly rather than glossed over:
- bluetides_bias_gz below is the exact fit from Waters et al. 2016
  (bg(z) = 2.1(1+z) - 5.3, their Sec 5/Fig 14) -- this IS the real curve
  La Plante+2022 cites, not an approximation of it.
- This computes the CLUSTERING-ONLY signal and, when used for a full S/N
  (not just D_ell), should be compared against the paper's own "without
  shot noise" Table 2 column -- sidesteps needing their exact N_g(z)
  (Waters+2016 Figure 3, also not digitized here). Shot noise is a
  secondary refinement.
- Our box is 300 Mpc; theirs is 2 h^-1 Gpc (2LPT, 1024^3 particles), and
  their peak signal (ell~1000) is a LARGE-SCALE (low-ell) mode -- the
  SAME regime correlation.coherence_decomposition was built to check for
  periodicity contamination in the kSZ auto-power. Worth revisiting once
  real numbers are in from both, rather than treating these as unrelated.
- Cosmology differs slightly from the paper's Planck18-like values
  (Om0=0.316, Ob0=0.049, h=0.673 vs this repo's 21cmFAST-default
  Om0=0.3086, Ob0=0.0489, h=0.6777) -- a few percent, shouldn't matter
  for an order-of-magnitude check.
- Instrument_noise() in cmb_filter.py implements Eq. 11 (naive,
  instrument-only noise) -- this repo does NOT yet implement the more
  realistic post-ILC residual-foreground noise the paper's headline 13-sigma
  number actually uses (their Table 2 'ILC Noise' column, orphics-based).
  Any S/N built from this repo's current instrument_noise() should be
  checked against Table 2's 'Instrument Noise' column, NOT the abstract's
  13-sigma figure -- those are different columns (35 vs 13 for CMB-HD/
  Fiducial/with shot noise).
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import KGrid, cross_power_2d, make_ell, to_Cell, to_Dell
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology

# Paper's Figure 4 peak, read off by eye: ell(ell+1)C_ell/(2pi) ~ 0.02 uK^2
# near ell~1000, roughly independent of z0/dz (SO filter; broadly similar
# for CMB-S4/CMB-HD per their Sec 3). Use as a rough visual reference only.
PAPER_FIG4_PEAK_DELL_UK2 = 0.02
PAPER_FIG4_PEAK_ELL = 1000.0

# Patchy-regime window convention, ported from ksz-pipeline's
# scripts/14_closure_test.py (XHI_MIN_PATCHY/XHI_MAX_PATCHY): exclude any
# z where the volume-averaged x_HI is outside this range -- i.e. either
# already fully ionized (x_HI < XHI_MIN_PATCHY, no patchiness left to
# source a kSZ signal) or reionization hasn't meaningfully started yet
# (x_HI > XHI_MAX_PATCHY, x_e negligible). Same values as ksz-pipeline
# uses, since this is a definitional choice, not something tied to their
# specific simulation.
XHI_MIN_PATCHY = 1.0e-4
XHI_MAX_PATCHY = 1.0 - 1.0e-4

# La Plante+2022 Fig 9 discussion: "we truncate the window to have a
# minimum value of z=6, both for the signal and noise components of the
# S/N" -- avoids double-counting/contaminating the reionization-era
# signal with post-reionization (z<6) contributions. Applied as a hard
# floor below, same as the paper does.
Z_FLOOR_REIONIZATION = 6.0


def clamp_window_to_patchy_regime(z0: float, dz: float, z_lc: np.ndarray,
                                   xHI_lc: np.ndarray) -> tuple[float, float]:
    """
    Adjust a requested (z0, dz) top-hat window to:
      (a) never dip below Z_FLOOR_REIONIZATION (La Plante+2022 Fig 9's
          explicit z>=6 truncation), and
      (b) exclude any z where this seed's own volume-averaged x_HI falls
          outside [XHI_MIN_PATCHY, XHI_MAX_PATCHY] (ksz-pipeline's patchy-
          regime convention, ported as-is).

    Returns (z_lo_clamped, z_hi_clamped) -- the actual bounds to use,
    generally narrower than the naive (z0-dz/2, z0+dz/2). Deterministic
    given the same inputs, so calling this independently from multiple
    places (e.g. once for the galaxy field, once for an x_HI report) gives
    consistent bounds without needing to thread extra state around.

    Raises if nothing in the requested window survives both constraints
    -- a real "this window doesn't correspond to any patchy-era data"
    condition, not something to silently paper over.
    """
    z_lo_req, z_hi_req = z0 - dz / 2, z0 + dz / 2
    z_lo_floor = max(z_lo_req, Z_FLOOR_REIONIZATION)
    if z_lo_floor >= z_hi_req:
        raise ValueError(
            f"Requested window [{z_lo_req:.3f}, {z_hi_req:.3f}] lies entirely "
            f"below the z>={Z_FLOOR_REIONIZATION} reionization floor -- "
            f"nothing left after clamping."
        )

    xHI_mean_z = xHI_lc.mean(axis=(0, 1))
    in_patchy = (xHI_mean_z >= XHI_MIN_PATCHY) & (xHI_mean_z <= XHI_MAX_PATCHY)
    # Exclusive upper bound, matching the (>=lo, <hi) convention used
    # everywhere else in this module (build_bias_weighted_galaxy_field,
    # compute_volume_averaged_xHI) -- an inconsistent inclusive '<=' here
    # caused a real off-by-one mismatch against those functions, caught
    # by test_uniform_bias_and_window_reduces_to_plain_mean.
    in_window = (z_lc >= z_lo_floor) & (z_lc < z_hi_req)
    combined = in_patchy & in_window
    if not np.any(combined):
        raise ValueError(
            f"No z in [{z_lo_floor:.3f}, {z_hi_req:.3f}] falls in the patchy "
            f"regime (x_HI in [{XHI_MIN_PATCHY}, {XHI_MAX_PATCHY}]) for this "
            f"seed -- widen the window, or this seed's ionization history "
            f"doesn't overlap the requested range at all."
        )
    z_survive = z_lc[combined]
    # +epsilon on the upper edge: z_survive.max() is an exact z_lc grid
    # value, and downstream callers use searchsorted+exclusive-slice
    # (z_lc[zi_lo:zi_hi]) -- passing z_hi as an exact grid point would
    # have searchsorted find that point's own index and the slice would
    # then EXCLUDE it. The tiny epsilon ensures the true max surviving
    # point is actually included, not silently dropped by an off-by-one.
    return float(z_survive.min()), float(z_survive.max()) + 1e-9


def chi_eff_power_weighted(cfg, field_data_seed: dict, z_lo: float, z_hi: float) -> float:
    """
    Power-weighted mean comoving distance over [z_lo, z_hi], matching
    ksz-pipeline's validated definition (their
    scripts/14_closure_test.py::chi_eff_power_weighted):

        chi_eff = INT w(z)^2 chi(z) dz / INT w(z)^2 dz

    where w(z) is the RMS (over transverse pixels) of the per-slice kSZ
    integrand. REUSES
    correlation.coherence_decomposition.compute_ksz_slices for that
    integrand -- this repo's own trusted, tested kSZ construction
    (verified bit-for-bit against build_projected_maps) -- rather than
    reimplementing a separate visibility/patchy-mask formula that could
    drift from it.

    Replaces the single-z reference chi=comoving_distance(z0) used
    elsewhere in roman_hls_benchmark.py -- a cruder approximation, see
    this module's earlier docstring note and
    coherence_decomposition.py's own caveat about the single-chi
    approximation.
    """
    from ksz_lae_xcorr.correlation.coherence_decomposition import compute_ksz_slices

    theta_slices, chi_mpc, z_arr = compute_ksz_slices(cfg, field_data_seed)
    window_mask = (z_arr >= z_lo) & (z_arr <= z_hi)
    if window_mask.sum() < 2:
        raise ValueError(
            f"Window [{z_lo:.3f}, {z_hi:.3f}] contains <2 LOS pixels in this "
            f"seed's kSZ integrand grid -- cannot compute chi_eff."
        )

    w_z = np.sqrt(np.mean(theta_slices[:, :, window_mask] ** 2, axis=(0, 1)))
    chi_z = chi_mpc[window_mask]
    z_z = z_arr[window_mask]

    # Manual trapezoidal rule -- np.trapz was removed in newer NumPy
    # (renamed np.trapezoid in 2.0+, see the same fix already applied in
    # snr/cmb_filter.py), needs to work regardless of NumPy version.
    def _trapz(y, x):
        return np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x))

    num = _trapz(w_z**2 * chi_z, z_z)
    den = _trapz(w_z**2, z_z)
    if den == 0:
        raise ValueError(
            "Zero total weight in the chi_eff window -- the kSZ integrand "
            "is uniformly zero there (e.g. x_e=0 everywhere), so a power-"
            "weighted mean chi is undefined."
        )
    return float(num / den)


def bluetides_bias_gz(z):
    """
    Roman HLS LBG linear bias, from Waters et al. 2016 (MNRAS 463, 3520;
    arXiv:1605.05670) Sec 5, their Figure 14 fit:

        bg(z) = 2.1*(1+z) - 5.3

    calibrated to bg = 13.4 +/- 1.8 at z=8. This is the REAL BlueTides
    fit (not an interpolation) -- it independently checks against the two
    anchor points La Plante+2022's text quotes (bg~9 at z=6: 2.1*7-5.3=9.4;
    bg~20 at z=12: 2.1*13-5.3=22.0), confirming this is the same curve
    they're citing.

    Clipped flat OUTSIDE [6, 12] on BOTH ends -- not just below z=6.
    Neither Waters+2016 nor La Plante+2022 discuss this bias beyond z~12
    (their own text stops quoting it there); left unclipped above, the
    linear formula grows without bound (bg~32 by z=17!) with nothing
    behind it -- caught via scripts/13's D_ell-vs-z sweep producing an
    unphysical "peak" at z~17-18, x_HI~1.0 (fully neutral, where genuine
    kSZ signal should be near zero, not maximal) that turned out to be
    this runaway extrapolation, not a real reionization feature. Fixed
    2026-09-08 -- if re-deriving this from a fuller Waters+2016 digitization
    later, re-check whether a real (not just flat-clipped) high-z falloff
    is more appropriate.
    """
    z = np.asarray(z, dtype=np.float64)
    z_clipped = np.clip(z, 6.0, 12.0)
    return 2.1 * (1.0 + z_clipped) - 5.3


def build_bias_weighted_galaxy_field(cfg, field_data_seed: dict, z0: float, dz: float,
                                      bias_fn=bluetides_bias_gz, clamp_to_patchy: bool = True) -> np.ndarray:
    """
    Eq. 6-7 of La Plante+2022: delta_g = INT dz Wg(z) bg(z) delta_m(chi(z) nhat, chi(z)),
    Wg a top-hat of width dz centered at z0, normalized so INT dz Wg(z) = 1.

    Built directly from field_data_seed['density_lc'] ((1+delta), this
    repo's own convention per projected_maps.py) -- no external galaxy
    catalogue needed. Each LOS pixel is weighted by its own thickness in
    z (via np.gradient of the pixel z values actually present), so an
    unevenly-spaced z_lc grid is handled correctly rather than assuming
    uniform pixel spacing.

    clamp_to_patchy (default True): adjusts the requested (z0, dz) window
    via clamp_window_to_patchy_regime before building the field -- floors
    at z=6 (La Plante+2022 Fig 9) and excludes x_HI outside
    [XHI_MIN_PATCHY, XHI_MAX_PATCHY] (ksz-pipeline's patchy-regime
    convention). Pass False to get the raw, unclamped (z0-dz/2, z0+dz/2)
    window (e.g. for reproducing earlier, pre-clamp results).

    Returns a 2D (Nx, Ny) field -- NOT mean-subtracted (caller's choice).
    """
    z_lc = field_data_seed["z_lc"]
    if clamp_to_patchy:
        z_lo, z_hi = clamp_window_to_patchy_regime(z0, dz, z_lc, field_data_seed["xHI_lc"])
    else:
        z_lo, z_hi = z0 - dz / 2, z0 + dz / 2

    zi_lo = int(np.searchsorted(z_lc, z_lo))
    zi_hi = int(np.searchsorted(z_lc, z_hi))
    if zi_hi <= zi_lo:
        raise ValueError(
            f"Window [{z_lo:.3f}, {z_hi:.3f}] contains no LOS pixels in this "
            f"seed's z_lc grid (range [{z_lc.min():.3f}, {z_lc.max():.3f}]) -- "
            f"widen dz or move z0."
        )

    z_win = z_lc[zi_lo:zi_hi]
    delta_m = field_data_seed["density_lc"][:, :, zi_lo:zi_hi] - 1.0  # (1+delta) -> delta

    if len(z_win) > 1:
        dz_pix = np.gradient(z_win)
    else:
        dz_pix = np.array([dz])
    Wg = dz_pix / np.sum(dz_pix)  # top-hat, normalized to sum to 1 over this window

    bg = bias_fn(z_win)
    weight = (Wg * bg)[None, None, :]
    return np.sum(delta_m * weight, axis=2)


def compute_bias_weighted_cross_power(cfg, kg: KGrid, filtered_kSZ2_seed: np.ndarray,
                                       field_data_seed: dict, z0: float, dz: float,
                                       bias_fn=bluetides_bias_gz) -> dict:
    """
    One seed's D_ell for filtered-kSZ^2 x (bias-weighted matter field) --
    the quantity to compare against the paper's own Figure 4/5.
    filtered_kSZ2_seed: one entry of
    snr.snr_forecast.build_filtered_kSZ2_maps[experiment][seed].

    Returns {'ell', 'D_ell', 'D_err', 'z0', 'dz', 'chi_c'}.
    """
    cosmo = get_cosmology(cfg)
    delta_g = build_bias_weighted_galaxy_field(cfg, field_data_seed, z0, dz, bias_fn=bias_fn)

    chi_c = cosmo.comoving_distance(z0).to_value("Mpc")
    ell_c = make_ell(kg.k_centers, chi_c)

    sig = filtered_kSZ2_seed.astype(np.float64)
    sig = sig - sig.mean()
    P, Pe, _ = cross_power_2d(sig, delta_g - delta_g.mean(), kg)
    C, Ce = to_Cell(P, Pe, chi_c)
    D, De = to_Dell(ell_c, C, Ce, T_CMB_uK=constants.T_CMB_UK)
    return {"ell": ell_c, "D_ell": D, "D_err": De, "z0": z0, "dz": dz, "chi_c": chi_c}


def compute_volume_averaged_xHI(field_data_seed: dict, z0: float, dz: float,
                                 clamp_to_patchy: bool = True) -> float:
    """
    Volume-averaged neutral fraction over the SAME top-hat window
    (z0, dz) used for the galaxy field above -- so a reported x_HI value
    is directly comparable to whichever (z0, dz) point produced a given
    D_ell, same pairing the paper's own Figure 5 upper x-axis provides
    (their zreion model's x_HII(z); this is OUR simulation's own x_HI(z),
    not assumed to match theirs).

    clamp_to_patchy (default True): same clamping as
    build_bias_weighted_galaxy_field -- pass the SAME value to both calls
    for a given (z0, dz) so the reported x_HI actually corresponds to the
    window the galaxy field was built on. clamp_window_to_patchy_regime
    is deterministic given the same (z0, dz, z_lc, xHI_lc), so this stays
    consistent automatically as long as both calls use the same field_data_seed.
    """
    z_lc = field_data_seed["z_lc"]
    if clamp_to_patchy:
        z_lo, z_hi = clamp_window_to_patchy_regime(z0, dz, z_lc, field_data_seed["xHI_lc"])
    else:
        z_lo, z_hi = z0 - dz / 2, z0 + dz / 2
    zi_lo = int(np.searchsorted(z_lc, z_lo))
    zi_hi = int(np.searchsorted(z_lc, z_hi))
    if zi_hi <= zi_lo:
        raise ValueError(
            f"Window [{z_lo:.3f}, {z_hi:.3f}] contains no LOS pixels in this "
            f"seed's z_lc grid (range [{z_lc.min():.3f}, {z_lc.max():.3f}])."
        )
    return float(np.mean(field_data_seed["xHI_lc"][:, :, zi_lo:zi_hi]))
