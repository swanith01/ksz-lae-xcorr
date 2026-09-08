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
                                      bias_fn=bluetides_bias_gz) -> np.ndarray:
    """
    Eq. 6-7 of La Plante+2022: delta_g = INT dz Wg(z) bg(z) delta_m(chi(z) nhat, chi(z)),
    Wg a top-hat of width dz centered at z0, normalized so INT dz Wg(z) = 1.

    Built directly from field_data_seed['density_lc'] ((1+delta), this
    repo's own convention per projected_maps.py) -- no external galaxy
    catalogue needed. Each LOS pixel is weighted by its own thickness in
    z (via np.gradient of the pixel z values actually present), so an
    unevenly-spaced z_lc grid is handled correctly rather than assuming
    uniform pixel spacing.

    Returns a 2D (Nx, Ny) field -- NOT mean-subtracted (caller's choice).
    """
    z_lc = field_data_seed["z_lc"]
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


def compute_volume_averaged_xHI(field_data_seed: dict, z0: float, dz: float) -> float:
    """
    Volume-averaged neutral fraction over the SAME top-hat window
    (z0, dz) used for the galaxy field above -- so a reported x_HI value
    is directly comparable to whichever (z0, dz) point produced a given
    D_ell, same pairing the paper's own Figure 5 upper x-axis provides
    (their zreion model's x_HII(z); this is OUR simulation's own x_HI(z),
    not assumed to match theirs).

    Same window-finding logic as build_bias_weighted_galaxy_field --
    raises under the same condition (empty window).
    """
    z_lc = field_data_seed["z_lc"]
    z_lo, z_hi = z0 - dz / 2, z0 + dz / 2
    zi_lo = int(np.searchsorted(z_lc, z_lo))
    zi_hi = int(np.searchsorted(z_lc, z_hi))
    if zi_hi <= zi_lo:
        raise ValueError(
            f"Window [{z_lo:.3f}, {z_hi:.3f}] contains no LOS pixels in this "
            f"seed's z_lc grid (range [{z_lc.min():.3f}, {z_lc.max():.3f}])."
        )
    return float(np.mean(field_data_seed["xHI_lc"][:, :, zi_lo:zi_hi]))
