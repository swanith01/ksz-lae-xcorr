"""
correlation/velocity_reconstruction.py
=========================================
Linear ("continuity equation") reconstruction of the line-of-sight
peculiar velocity from a real-space 3D overdensity field -- NOT machine
learning. Built 2026-09-30 per Girish's request, following Eqs. 11-12 of
Gong & Bean 2026 (arXiv:2609.36355, "Probing Baryons with the Kinematic
Sunyaev-Zel'dovich Effect and Machine Learning-Derived Peculiar
Velocities using DESI DR2 and ACT DR6"), whose own linear baseline is
their citations [56] Guachalla, Schaan, Hadzhiyska & Ferraro 2024 (Phys.
Rev. D 109, 103533, arXiv:2312.12435), [57] Hadzhiyska, Ferraro,
Guachalla & Schaan 2024 (Phys. Rev. D 109, 103534, arXiv:2312.12434),
and the classic Zel'dovich/MultiGrid reconstruction algorithm of [92]
White 2015 (MNRAS 450, 3822, arXiv:1504.03677).

THE ADAPTATION TO OUR SETTING (why this is simpler than the paper's own
pipeline): the paper reconstructs from an OBSERVED, REDSHIFT-SPACE,
survey-masked galaxy catalog, so their Eq. 11 divides by (b + f*mu^2) --
the Kaiser factor that relates redshift-space tracer overdensity to
real-space matter overdensity -- and uses pyrecon's MultiGrid solver to
handle the survey mask/geometry. We instead have a full, periodic,
REAL-SPACE coeval box directly from the simulation: no redshift-space
distortion to correct for (mu-dependence drops out entirely) and no
survey mask (so a simple FFT solves the same linear PDE exactly, with no
need for an iterative real-space solver). What remains is exactly the
plain continuity-equation relation:

    v_los(k) = i * a(z) * H(z) * f(z) * (k_los / k^2) * W_G(k) * delta(k) / b

which is EXACTLY the same formula already validated in this repo
(correlation.velocity_convention_check.theoretical_velocity_over_density_
ratio), used earlier this session to confirm py21cmfast's raw velocity_z
convention from first principles. That module checked this relation at
the POWER-SPECTRUM level (P_v(k)/P_delta(k) ratio); this module solves
it as an actual FIELD reconstruction (FFT then inverse FFT), using the
repo's own already-tested growth_rate/growth_factor helpers
(lightcone.stitch._growth_rate_linder, matching this repo's cosmology,
not ksz-pipeline's).

delta must be a REAL-SPACE, ZERO-MEAN overdensity field (matter delta =
(1+delta)-1, or a tracer overdensity (count-mean)/mean -- same
convention as correlation.power_spectra.make_overdensity elsewhere in
this repo) -- NOT the 1+delta convention used for density_lc.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def _k_grid_3d_components(n_side: int, box_len_mpc: float):
    """KX, KY, KZ, kmag on the natural 3D FFT grid, angular-frequency
    convention -- matches direct_bispectrum._k_grid_3d and
    velocity_convention_check.radial_power_spectrum_3d exactly, so a
    field built here is on the same k-grid as those modules' spectra."""
    kfreq = np.fft.fftfreq(n_side, d=box_len_mpc / n_side) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    return KX, KY, KZ, kmag


def reconstruct_velocity_los(cfg, delta: np.ndarray, box_len_mpc: float, z: float,
                              bias: float = 1.0, r_smooth_mpc: float | None = None,
                              los_axis: int = 2) -> np.ndarray:
    """
    Linear/continuity-equation reconstruction of the LOS peculiar
    velocity from a real-space overdensity field.

    delta: real-space, zero-mean overdensity (Nx, Ny, Nz) -- matter
        delta for the bias=1.0 default (the cleanest first test: use the
        box's own matter density field directly, no bias uncertainty at
        all), or a tracer overdensity (halo/LAE/LBG count-based) with an
        appropriate `bias` (e.g. snr.roman_hls_benchmark.bluetides_bias_gz
        for a density-proxy-style bias, or a measured tracer bias) to
        first map tracer overdensity back to an implied matter
        overdensity, same role as the paper's Eq. 11 bias term.
    box_len_mpc, z: this snapshot's box size and redshift.
    bias: linear bias b such that delta_tracer = b * delta_matter (real
        space, no RSD term needed here -- see module docstring). Default
        1.0 for a direct matter-field reconstruction.
    r_smooth_mpc: optional Gaussian smoothing scale (real-space sigma,
        Mpc) applied as exp(-k^2 R_s^2 / 2) before reconstructing, same
        role as the paper's W_G(k) (they use R_s=12.5 h^-1 Mpc) -- the
        linear approximation is only trustworthy on large/smoothed
        scales (this repo's own velocity_convention_check found the
        raw-field power-spectrum ratio starts drifting from linear
        theory above k~0.7 Mpc^-1 on real data), so smoothing suppresses
        the small-scale modes where this reconstruction is not expected
        to work. None (default) applies no smoothing.
    los_axis: which Cartesian axis is the line of sight (default 2 = z,
        matching direct_bispectrum.build_momentum_field_3d's convention
        of using velocity_z as the only LOS direction 21cmFAST v4
        provides at this grid resolution).

    Returns v_los_mpc_s: real (Nx, Ny, Nz) array, in Mpc/s -- the SAME
    units/convention as this repo's own velocity_z field
    (lightcone.stitch.Stitcher's velocity_lc), so it can be compared to
    or substituted for the native field directly, with no further
    conversion.
    """
    from ksz_lae_xcorr.lightcone.stitch import _growth_rate_linder

    n = delta.shape[0]
    if delta.shape != (n, n, n):
        raise ValueError(f"Expected a cubic box, got shape {delta.shape}.")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2 -- got {los_axis}.")
    if bias == 0:
        raise ValueError("bias=0 is unphysical (implies infinite matter overdensity for any "
                          "nonzero tracer overdensity) -- cannot reconstruct.")

    cosmo = get_cosmology(cfg)
    a = 1.0 / (1.0 + z)
    H_kms_mpc = cosmo.H(z).to_value("km/s/Mpc")
    f = _growth_rate_linder(cosmo, z)

    KX, KY, KZ, kmag = _k_grid_3d_components(n, box_len_mpc)
    K_LOS = (KX, KY, KZ)[los_axis]

    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = 1j * K_LOS / kmag**2
    kernel[kmag == 0] = 0.0  # no reconstructed bulk-flow/DC mode -- standard convention

    if r_smooth_mpc is not None:
        kernel = kernel * np.exp(-0.5 * (kmag * r_smooth_mpc) ** 2)

    delta_k = np.fft.fftn(delta)
    psi_los_k = kernel * delta_k / bias
    psi_los_x = np.real(np.fft.ifftn(psi_los_k))

    v_los_kms = a * H_kms_mpc * f * psi_los_x
    return v_los_kms * constants.MPC_PER_KM_S_TO_S


def radial_cross_power_spectrum_3d(field_a: np.ndarray, field_b: np.ndarray,
                                    box_len_mpc: float, n_kbins: int = 15):
    """
    Isotropic 3D cross-power spectrum of two real fields on the same
    grid, same binning/normalization convention as
    velocity_convention_check.radial_power_spectrum_3d (so an auto- and
    a cross-spectrum computed with these two functions sit on identical
    k bins and can be combined directly, e.g. into a correlation
    coefficient).

    Returns {'k_centers', 'P_cross'}.
    """
    if field_a.shape != field_b.shape:
        raise ValueError(f"Shape mismatch: field_a{field_a.shape} vs field_b{field_b.shape}.")
    n = field_a.shape[0]
    a0 = field_a - field_a.mean()
    b0 = field_b - field_b.mean()
    a_k = np.fft.fftn(a0)
    b_k = np.fft.fftn(b0)
    vol = box_len_mpc**3
    cross_3d = np.real(np.conj(a_k) * b_k) * vol / n**6

    kfreq = np.fft.fftfreq(n, d=box_len_mpc / n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_fund = 2 * np.pi / box_len_mpc
    k_nyq = np.pi * n / box_len_mpc
    k_edges = np.geomspace(k_fund, k_nyq, n_kbins + 1)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])

    P_cross = np.full(n_kbins, np.nan)
    for j in range(n_kbins):
        mask = (kmag >= k_edges[j]) & (kmag < k_edges[j + 1])
        if np.any(mask):
            P_cross[j] = np.mean(cross_3d[mask])
    return {"k_centers": k_centers, "P_cross": P_cross}


def compare_reconstructed_to_native(v_reconstructed: np.ndarray, v_native: np.ndarray,
                                     box_len_mpc: float, n_kbins: int = 15) -> dict:
    """
    Reconstruction-fidelity diagnostics between a reconstructed LOS
    velocity field and the simulation's own native LOS velocity field --
    both level A (Fourier/power-spectrum) and level B (real-space/pixel)
    checks, so either can be plotted:

      - P_rec(k), P_native(k), P_cross(k): auto- and cross-power spectra
        (reuses velocity_convention_check.radial_power_spectrum_3d for
        the two autos, and radial_cross_power_spectrum_3d above for the
        cross, all on identical k bins).
      - r(k) = P_cross(k) / sqrt(P_rec(k) * P_native(k)): the
        scale-dependent correlation coefficient -- the standard
        reconstruction-fidelity diagnostic in this literature (c.f. the
        paper's own single scalar r=0.91 velocity calibration factor,
        Sec. III C / IV B -- this is the k-resolved version, more
        informative since we have the full field, not just a catalog
        statistic). r=1 at a given k means the reconstruction is a
        perfect (up to amplitude) predictor of the true field at that
        scale; r=0 means no relation at all.
      - transfer(k) = sqrt(P_rec(k) / P_native(k)): amplitude ratio,
        separate from r(k) -- a reconstruction can be well-correlated
        (r~1) but systematically under/over-amplitude (transfer != 1),
        which r(k) alone would not reveal.
      - pixel_pearson_r: single real-space Pearson correlation
        coefficient between the two fields' raw pixel values (flattened)
        -- a cheap, non-Fourier sanity check requested alongside the
        power-spectrum comparison.

    Returns {'k_centers', 'P_rec', 'P_native', 'P_cross', 'r_k',
    'transfer_k', 'pixel_pearson_r'}.
    """
    from ksz_lae_xcorr.correlation.velocity_convention_check import radial_power_spectrum_3d

    if v_reconstructed.shape != v_native.shape:
        raise ValueError(
            f"Shape mismatch: v_reconstructed{v_reconstructed.shape} vs "
            f"v_native{v_native.shape}."
        )

    p_rec = radial_power_spectrum_3d(v_reconstructed, box_len_mpc, n_kbins)
    p_native = radial_power_spectrum_3d(v_native, box_len_mpc, n_kbins)
    p_cross = radial_cross_power_spectrum_3d(v_reconstructed, v_native, box_len_mpc, n_kbins)

    k = p_rec["k_centers"]
    denom = np.sqrt(p_rec["P"] * p_native["P"])
    with np.errstate(divide="ignore", invalid="ignore"):
        r_k = np.where(denom > 0, p_cross["P_cross"] / denom, np.nan)
        transfer_k = np.where(p_native["P"] > 0, np.sqrt(p_rec["P"] / p_native["P"]), np.nan)

    pixel_pearson_r = float(np.corrcoef(v_reconstructed.ravel(), v_native.ravel())[0, 1])

    return {
        "k_centers": k,
        "P_rec": p_rec["P"],
        "P_native": p_native["P"],
        "P_cross": p_cross["P_cross"],
        "r_k": r_k,
        "transfer_k": transfer_k,
        "pixel_pearson_r": pixel_pearson_r,
    }
