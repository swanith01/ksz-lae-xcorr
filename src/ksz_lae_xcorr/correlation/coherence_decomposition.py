"""
correlation/coherence_decomposition.py
=======================================
P_diag / P_off decomposition of the stitched kSZ map's power, ported from
ksz-pipeline (src/ksz_pipeline/ksz/coherence_decomposition.py), where it was
built per the advisor's 2026-07-23 suggestion and validated: P_diag (the
sum of each LOS-pixel's own auto-power, no cross terms between pixels) was
shown to match ksz-pipeline's independent coeval-direct/Limber calculation
to 8.5%, robust across grouping choices (script 20's n_groups sweep) --
strong evidence that P_off (= P_total - P_diag, the genuine cross-pixel
correlation power) is essentially the box-periodicity artifact itself,
not a fixed number that needs a second independent (and much more
expensive) direct/Limber recomputation to characterize.

WHY THIS MODULE, RATHER THAN PORTING compute_cell/qperp_power (Girish,
2026-09-07 Slack): those build an INDEPENDENT kSZ auto-power from raw
coeval boxes (no stitching at all) -- more work, and a second code path
that itself needs validating. This module instead decomposes the EXISTING
stitched map you already have, algebraically, with no new simulation
output required: P_total = P_diag + P_off always holds exactly (see
test_coherence_decomposition.py), so P_diag is available immediately from
data already on disk.

SCOPE, STATED PLAINLY: this decomposes the kSZ (linear Delta T/T) AUTO-power
only -- i.e. it curbs periodicity in the SAME quantity that feeds
snr.cmb_filter.kSZ_reion_from_sim. It does NOT, as written, decompose the
kSZ2 x tracer CROSS-power that is the actual target statistic for this
paper -- kSZ2 is quadratic in the LOS-integrated field, so its cross-power
with a (separately, and more naively-stitched, see stitch_discrete) tracer
is a different, bispectrum-type object that this linear-field decomposition
does not directly cover. Treat this as curbing periodicity in one
ingredient (the own-kSZ auto-power / CMB filter), not as a solved answer
for the cross-correlation itself -- that generalization is future work,
worth scoping once Girish's Stage 1 benchmark (reproduce La Plante+22 under
matched assumptions) is in hand.

Usage: see scripts/09_coherence_decomposition.py for the CLI wrapper.
"""

from __future__ import annotations

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import KGrid, make_ell, to_Cell, to_Dell
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def compute_ksz_slices(cfg, field_data_seed: dict):
    """
    Per-LOS-pixel kSZ integrand -- mirrors
    correlation.projected_maps.build_projected_maps's kSZ_map formula
    EXACTLY, just without the final np.sum(axis=2). This guarantees
    theta_slices.sum(axis=2) reproduces build_projected_maps's own
    kSZ_map bit-for-bit (checked in tests/test_coherence_decomposition.py,
    not just asserted here) -- if that check ever fails, the bug is in
    THIS function drifting from projected_maps.py, not a physics result.

    Parameters
    ----------
    cfg : Config
    field_data_seed : dict, one seed's entry as produced by
        io.loaders.load_lightcone_products -- needs 'z_lc', 'xHI_lc',
        'density_lc' (1+delta), 'velocity_lc' (Mpc/s).

    Returns
    -------
    theta_slices : ndarray (Nx, Ny, Nz) -- per-LOS-pixel Theta_i(x,y).
        theta_slices.sum(axis=2) == build_projected_maps(...)['kSZ'][seed].
    chi_mpc      : ndarray (Nz,) -- comoving distance [Mpc] to each pixel.
    z            : ndarray (Nz,) -- redshift of each pixel (same trim as
        build_projected_maps: cfg.box.z_min <= z < cfg.box.z_max).
    """
    cosmo = get_cosmology(cfg)
    tau_pref = constants.tau_prefactor(cfg)
    c_mpc_s = constants.c_mpc_per_s()

    z_lo, z_hi = cfg.box.z_min, cfg.box.z_max
    z_lc = field_data_seed["z_lc"]
    zi_lo = int(np.searchsorted(z_lc, z_lo))
    zi_hi = int(np.searchsorted(z_lc, z_hi))
    z = np.asarray(z_lc[zi_lo:zi_hi])

    xHI = field_data_seed["xHI_lc"][:, :, zi_lo:zi_hi]
    delta = field_data_seed["density_lc"][:, :, zi_lo:zi_hi]  # (1+delta)
    v_m = field_data_seed["velocity_lc"][:, :, zi_lo:zi_hi]   # Mpc/s
    x_e = 1.0 - xHI

    d_com = cosmo.comoving_distance(z).to_value("Mpc")
    ds = np.abs(np.gradient(d_com))
    a_arr = 1.0 / (1.0 + z)

    x_e_mean = x_e.mean(axis=(0, 1))
    dtau = tau_pref * x_e_mean * (1.0 + z) ** 2 * ds
    tau_arr = np.cumsum(dtau)
    e_tau = np.exp(-tau_arr)

    v_over_c = v_m / c_mpc_s
    integrand = (tau_pref * delta * x_e * v_over_c
                 * (1.0 / a_arr ** 2)[None, None, :]
                 * e_tau[None, None, :]
                 * ds[None, None, :])
    theta_slices = (-integrand).astype(np.float64)
    return theta_slices, d_com.astype(np.float64), z


def _bin_2d_power(power_2d: np.ndarray, kg: KGrid) -> np.ndarray:
    """Radially bin a 2D power array onto kg.k_bins (mean per bin) --
    same binning convention as power_spectra.cross_power_2d, reused here
    so P_diag/P_off/P_total sit on identical k-bins to every other
    spectrum in this repo (rather than a separately parameterized grid)."""
    P1d = np.full(kg.n_kbins, np.nan)
    for j in range(kg.n_kbins):
        mask = (kg.kgrid >= kg.k_bins[j]) & (kg.kgrid < kg.k_bins[j + 1])
        if mask.sum() == 0:
            continue
        P1d[j] = np.mean(power_2d[mask])
    return P1d


def decompose_p_total_diag_off(cfg, theta_slices: np.ndarray, chi_eff_mpc: float):
    """
    FFT each LOS-pixel slice independently (after subtracting ITS OWN
    mean -- mean is linear, so this still reproduces the standard
    mean-subtracted overdensity convention for the coherently-summed
    total map exactly, while making the per-slice decomposition
    well-defined), then build P_total/P_diag/P_off and convert to D_ell
    using this repo's own make_ell/to_Cell/to_Dell (ell = k*chi_eff_mpc,
    flat-sky Limber convention, same as everywhere else in this repo --
    NOT ksz-pipeline's astropy-Planck18 chi, see module note).

    chi_eff_mpc: single reference comoving distance for the ell=k*chi
    conversion of this seed's z-range (e.g. chi at the bin's z-center,
    matching correlation.power_spectra.ell_at_redshift's convention).
    For a wide z-range this single-chi flattening is an approximation --
    same caveat ksz-pipeline's own single-chi_eff version carries; see
    that repo's decompose_off_pairwise_chi if a per-pair chi becomes
    necessary later.

    Returns
    -------
    ell       : ndarray, valid multipoles (ell > 10)
    D_total   : ndarray -- should match build_projected_maps's kSZ map
                 put through the normal power_spectra pipeline
    D_diag    : ndarray -- compare this to any independent coeval-direct
                 calculation; this IS what Girish's P_diag refers to
    D_off     : ndarray -- CAN BE NEGATIVE (destructive interference
                 between periodic replicas is expected, not an error)
    """
    kg = KGrid(cfg)
    Nx, Ny, Nz = theta_slices.shape
    if (Nx, Ny) != (kg.n_side, kg.n_side):
        raise ValueError(
            f"theta_slices transverse shape {(Nx, Ny)} does not match "
            f"cfg.box.hii_dim={kg.n_side} -- KGrid's k-grid would silently "
            f"be built for the wrong pixel scale."
        )

    theta_zeroed = theta_slices - theta_slices.mean(axis=(0, 1), keepdims=True)
    theta_hat = np.fft.fftshift(np.fft.fft2(theta_zeroed, axes=(0, 1)), axes=(0, 1))
    norm = (kg.pix_size_mpc / kg.n_side) ** 2

    total_hat = theta_hat.sum(axis=-1)
    P_total_2d = norm * np.abs(total_hat) ** 2
    P_diag_2d = norm * np.sum(np.abs(theta_hat) ** 2, axis=-1)
    P_off_2d = P_total_2d - P_diag_2d

    P1d_total = _bin_2d_power(P_total_2d, kg)
    P1d_diag = _bin_2d_power(P_diag_2d, kg)
    P1d_off = _bin_2d_power(P_off_2d, kg)

    ell = make_ell(kg.k_centers, chi_eff_mpc)
    zeros = np.zeros_like(ell)
    Cl_total, _ = to_Cell(P1d_total, zeros, chi_eff_mpc)
    Cl_diag, _ = to_Cell(P1d_diag, zeros, chi_eff_mpc)
    Cl_off, _ = to_Cell(P1d_off, zeros, chi_eff_mpc)

    D_total, _ = to_Dell(ell, Cl_total, zeros, constants.T_CMB_UK)
    D_diag, _ = to_Dell(ell, Cl_diag, zeros, constants.T_CMB_UK)
    D_off, _ = to_Dell(ell, Cl_off, zeros, constants.T_CMB_UK)

    valid = (ell > 10) & np.isfinite(D_total)
    return ell[valid], D_total[valid], D_diag[valid], D_off[valid]


def group_slices_by_snapshot(cfg, theta_slices: np.ndarray, chi_mpc: np.ndarray,
                              z_snapshots) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum thin LOS-pixel slices into thicker groups, one per real coeval
    snapshot z (e.g. from lightcone.stitch.Stitcher.get_snapshot_redshifts),
    so 'diagonal' here matches what an independent per-snapshot direct
    calculation would treat as diagonal (each snapshot's own depth, not
    each thin n_lc_pix=512 LOS pixel). Bucket edges = comoving-distance
    midpoints between adjacent sorted snapshot redshifts.

    Raises rather than silently drops pixels that fall outside every
    bucket (can happen if z_snapshots doesn't fully bracket the
    lightcone's z-range) -- a bug to fix, not to ignore.
    """
    cosmo = get_cosmology(cfg)
    z_sorted = sorted(z_snapshots)
    chi_snap = np.array([cosmo.comoving_distance(z).to_value("Mpc") for z in z_sorted])

    if len(chi_snap) > 1:
        lo = chi_snap[0] - (chi_snap[1] - chi_snap[0]) / 2
        hi = chi_snap[-1] + (chi_snap[-1] - chi_snap[-2]) / 2
    else:
        lo, hi = chi_snap[0] - 1, chi_snap[0] + 1
    lo = min(lo, float(chi_mpc.min()) - 1e-6)
    hi = max(hi, float(chi_mpc.max()) + 1e-6)
    edges = np.sort(np.concatenate(([lo], 0.5 * (chi_snap[:-1] + chi_snap[1:]), [hi])))
    digit = np.digitize(chi_mpc, edges)

    grouped, chi_grouped = [], []
    n_used = 0
    for g in range(1, len(edges)):
        mask = digit == g
        if np.any(mask):
            grouped.append(theta_slices[:, :, mask].sum(axis=-1))
            chi_grouped.append(float(chi_mpc[mask].mean()))
            n_used += int(mask.sum())
    if n_used != theta_slices.shape[-1]:
        raise RuntimeError(
            f"group_slices_by_snapshot: {theta_slices.shape[-1] - n_used} of "
            f"{theta_slices.shape[-1]} LOS pixels were not assigned to any "
            f"group -- bucket edges do not cover the full chi_mpc range. "
            f"Fix the edges, don't ignore this."
        )
    return np.stack(grouped, axis=-1), np.array(chi_grouped)


def random_shift_slices(theta_slices: np.ndarray, seed: int | None = None) -> np.ndarray:
    """
    Control test (ported as-is -- purely geometric, no ksz-pipeline
    dependency): independent random cyclic (toroidal) shift per slice in
    (x, y). Preserves each slice's own power spectrum exactly (translation
    is a pure Fourier phase rotation) but destroys any FIXED cross-slice
    spatial alignment -- i.e. destroys periodicity-induced correlation
    from stitching identical/correlated box copies at zero relative
    offset, while leaving P_diag unchanged. Run decompose_p_total_diag_off
    on the shifted slices: P_off should drop toward zero if P_off really
    is a periodicity artifact and not some other real effect.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = theta_slices.shape
    shifted = np.empty_like(theta_slices)
    for i in range(Nz):
        dx, dy = int(rng.integers(0, Nx)), int(rng.integers(0, Ny))
        shifted[:, :, i] = np.roll(theta_slices[:, :, i], shift=(dx, dy), axis=(0, 1))
    return shifted


def cross_power_by_dchi(theta_slices: np.ndarray, chi_mpc: np.ndarray,
                         box_len_mpc: float, n_dchi_bins: int = 20):
    """
    Pairwise real-space cross-correlation between every pair of slices
    i != j, binned by radial separation |chi_i - chi_j| -- the Delta-chi
    overlay that most directly shows periodicity (real, correlated power
    peaking at separations that are multiples of the box's comoving depth
    along the LOS, box_len_mpc).

    COST WARNING (ported as-is): O(Nz^2) pairs, pure-Python double loop.
    Fine for snapshot-grouped input (Nz ~ 65 for the fiducial 300 Mpc run
    -- ~2000 pairs). NOT fine on raw n_lc_pix=512 LOS pixels (~130,000
    pairs) -- always call group_slices_by_snapshot first.

    Returns
    -------
    dchi_bin_centers : ndarray [Mpc]
    cross_power_mean : ndarray -- mean cross_ij in each Delta-chi bin;
        a peak (or oscillation) near integer multiples of box_len_mpc is
        the periodicity signature.
    cross_power_std  : ndarray
    n_pairs_per_bin  : ndarray (int)
    """
    Nx, Ny, Nz = theta_slices.shape
    pix_area = (box_len_mpc / Nx) * (box_len_mpc / Ny)
    theta_zeroed = theta_slices - theta_slices.mean(axis=(0, 1), keepdims=True)

    pairs_dchi, pairs_cross = [], []
    for i in range(Nz):
        for j in range(i + 1, Nz):
            dchi = abs(chi_mpc[i] - chi_mpc[j])
            cross = 2.0 * np.sum(theta_zeroed[:, :, i] * theta_zeroed[:, :, j]) * pix_area
            pairs_dchi.append(dchi)
            pairs_cross.append(cross)

    pairs_dchi = np.array(pairs_dchi)
    pairs_cross = np.array(pairs_cross)

    bins = np.linspace(0, pairs_dchi.max(), n_dchi_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_out = np.full(n_dchi_bins, np.nan)
    std_out = np.full(n_dchi_bins, np.nan)
    n_out = np.zeros(n_dchi_bins, dtype=int)

    digit = np.digitize(pairs_dchi, bins)
    for i in range(n_dchi_bins):
        mask = digit == i + 1
        n_out[i] = mask.sum()
        if n_out[i] > 0:
            mean_out[i] = pairs_cross[mask].mean()
            std_out[i] = pairs_cross[mask].std()

    return centers, mean_out, std_out, n_out
