"""
correlation/direct_bispectrum.py
==================================
Direct/coeval (non-stitched) estimator for the kSZ^2 x galaxy cross-power
-- the piece flagged repeatedly this session as missing: everything else
built so far (cross_correlation.py, snr_forecast.py, roman_hls_benchmark.py)
runs on the STITCHED lightcone, with the periodicity risk that implies
(coherence_decomposition.py). This module is the "direct" counterpart for
the CROSS term specifically (coherence_decomposition already covers the
kSZ AUTO term).

KEY ARCHITECTURAL INSIGHT (worth stating explicitly -- this is what makes
the scope much smaller than "implement a general bispectrum estimator"):

La Plante+2022's Eq. 12 full bispectrum-Limber integral looks
intimidating, but their own Sec 3 discussion (right after Eq 12) does the
hard work for us: in the ell << L regime where essentially ALL the S/N
comes from (their Fig 10 -- most of the signal is squeezed
configurations), the bispectrum collapses to its SQUEEZED-TRIANGLE limit.
A bispectrum correlating one soft (large-scale) density mode with two
hard (small-scale, back-to-back) momentum modes is EXACTLY the same
object, by the standard squeezed-bispectrum <-> mode-coupling identity
used throughout large-scale-structure non-Gaussianity work, as:

    locally high-pass-filter the momentum field, square it in real
    space (giving a map of LOCAL small-scale power), and cross-correlate
    THAT with the soft large-scale density field.

That is precisely the filter -> square -> cross-correlate recipe already
built for the STITCHED estimator (snr_forecast.py, roman_hls_benchmark.py)
-- just a 3D k-space filter instead of a 2D ell-space one. The direct/
coeval version does NOT need a fundamentally different estimator, only
the SAME recipe applied to each coeval snapshot's own 3D box (no
stitching, hence no periodicity), Limber-summed over snapshots instead of
computed once on a periodically-tiled lightcone.

PIPELINE, per snapshot z (no stitching):
  1. Build the 3D momentum field q(x) = (1+delta_m)(1+delta_x) v_los/c
     from THAT SNAPSHOT'S OWN raw coeval fields -- NOT the stitched
     lightcone. v_los uses one fixed Cartesian box axis (z-axis, matching
     the only velocity component 21cmFAST v4 actually provides at this
     grid resolution -- see halos/coeval_pipeline.py's field access
     notes) as the line-of-sight direction, the standard simplification
     for a coeval box with no preferred direction of its own.
  2. High-pass filter q(x) in 3D k-space at k_hard = L_peak/chi(z),
     mapping the CMB filter's peak ell (~3000-5000, experiment-dependent,
     see cmb_filter.py's f(ell)) into a physical k at this redshift --
     the 3D analog of applying f(ell) before squaring.
  3. Square in real space: q_filtered(x)^2 -- this snapshot's local
     "kSZ^2-like" field.
  4. Cross-correlate q_filtered^2 against the bias-weighted density field
     bg(z)*delta_m(x) (same Eq 6-7 model as roman_hls_benchmark.py), at
     LARGE scales k_soft ~ ell/chi(z) -- this 3D cross-power IS the
     squeezed-bispectrum contribution at this z.
  5. Weight by g(chi)^2 * bg(z) / chi^4 * dchi (Eq 12's radial kernel,
     Wg folded in by which snapshots are included) and Limber-sum over
     the snapshot list -- see limber_sum_snapshots.

WHAT THIS MODULE DOES NOT YET DO: load real per-snapshot coeval box files
(halos/coeval_pipeline.py's actual saved arrays) -- every function here
is tested against SYNTHETIC 3D boxes to validate the estimator math is
correct, the same approach used for coherence_decomposition.py before it
was pointed at real data. Wiring in real per-snapshot loading is the
next concrete step once this is trusted -- see scripts/09's
Stitcher.get_snapshot_redshifts for the pattern of finding real snapshot
directories, and coeval_pipeline.py's field-access notes for exact array
names/locations on disk.
"""

from __future__ import annotations

import numpy as np


def build_momentum_field_3d(density: np.ndarray, xHI: np.ndarray, velocity_los: np.ndarray,
                             c_mpc_s: float) -> np.ndarray:
    """
    q(x) = (1+delta_m) * (1+delta_x) * v_los/c -- same physical
    combination as the kSZ integrand in projected_maps.py and La
    Plante+2022 Eq. 4, but here on a single snapshot's raw 3D box (no
    line-of-sight integral, no stitching).

    density: (1+delta_m), same convention as projected_maps.py.
    xHI: neutral fraction on [0,1]. delta_x is defined via
    1+delta_x = x_e/mean(x_e), matching La Plante+2022 Eq. 4's convention.
    velocity_los: one Cartesian velocity component (Mpc/s) used as the
    line-of-sight direction -- pass velocity_z (the only component
    21cmFAST v4 provides at this grid, per coeval_pipeline.py).

    All three inputs must be the same shape (Nx, Ny, Nz).
    """
    if not (density.shape == xHI.shape == velocity_los.shape):
        raise ValueError(
            f"Shape mismatch: density{density.shape}, xHI{xHI.shape}, "
            f"velocity_los{velocity_los.shape} -- must all match."
        )
    x_e = 1.0 - xHI
    x_e_mean = x_e.mean()
    if x_e_mean <= 0:
        raise ValueError("Mean ionized fraction is zero -- fully neutral snapshot, "
                          "no kSZ signal possible here, don't silently divide by zero.")
    delta_x_plus1 = x_e / x_e_mean
    return density * delta_x_plus1 * (velocity_los / c_mpc_s)


def _k_grid_3d(n_side: int, box_len_mpc: float) -> np.ndarray:
    """|k| on the natural 3D FFT grid for an n_side^3 box of side box_len_mpc,
    in Mpc^-1 (angular frequency convention, matching power_spectra.py)."""
    kfreq = np.fft.fftfreq(n_side, d=box_len_mpc / n_side) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    return np.sqrt(KX**2 + KY**2 + KZ**2)


def highpass_filter_3d(field: np.ndarray, box_len_mpc: float, k_cut: float) -> np.ndarray:
    """
    Real-space high-pass filtered version of field: zero every 3D Fourier
    mode with |k| < k_cut, inverse-transform back. Sharp cutoff (not a
    smooth window) -- fine for this order-of-magnitude architecture pass;
    a smoother window is a refinement, not a correctness issue.
    """
    n = field.shape[0]
    if field.shape != (n, n, n):
        raise ValueError(f"Expected a cubic box, got shape {field.shape}.")
    kmag = _k_grid_3d(n, box_len_mpc)
    field_k = np.fft.fftn(field)
    field_k[kmag < k_cut] = 0.0
    return np.real(np.fft.ifftn(field_k))


def squeezed_bispectrum_proxy_3d(local_power_field: np.ndarray, delta_g_field: np.ndarray,
                                  box_len_mpc: float, k_soft_bins: np.ndarray) -> dict:
    """
    Step 4: 3D cross-power spectrum between a local-small-scale-power
    field (e.g. the output of squaring a high-pass-filtered momentum
    field) and a large-scale bias-weighted density/galaxy field, radially
    binned in k_soft. This IS the squeezed-bispectrum contribution at
    this snapshot's redshift, by the mode-coupling identity described in
    this module's docstring.

    k_soft_bins: bin EDGES (ascending), length n_bins+1.

    Returns {'k_centers', 'P_cross'} -- P_cross[j] is NaN for any bin with
    no populated modes (raise-free, matches the convention used
    elsewhere in this repo, e.g. coherence_decomposition._bin_2d_power).
    """
    n = local_power_field.shape[0]
    if local_power_field.shape != delta_g_field.shape:
        raise ValueError(
            f"Shape mismatch: local_power_field{local_power_field.shape} vs "
            f"delta_g_field{delta_g_field.shape}."
        )
    kmag = _k_grid_3d(n, box_len_mpc)

    a = local_power_field - local_power_field.mean()
    b = delta_g_field - delta_g_field.mean()
    a_k = np.fft.fftn(a)
    b_k = np.fft.fftn(b)
    vol = box_len_mpc**3
    cross_k = np.real(a_k * np.conj(b_k)) * vol / n**6

    n_bins = len(k_soft_bins) - 1
    k_centers = 0.5 * (k_soft_bins[:-1] + k_soft_bins[1:])
    P_cross = np.full(n_bins, np.nan)
    for j in range(n_bins):
        mask = (kmag >= k_soft_bins[j]) & (kmag < k_soft_bins[j + 1])
        if np.any(mask):
            P_cross[j] = np.mean(cross_k[mask])
    return {"k_centers": k_centers, "P_cross": P_cross}


def compute_snapshot_bispectrum_contribution(density: np.ndarray, xHI: np.ndarray,
                                              velocity_los: np.ndarray, delta_g_field: np.ndarray,
                                              box_len_mpc: float, c_mpc_s: float,
                                              k_hard: float, k_soft_bins: np.ndarray) -> dict:
    """
    Steps 1-4 chained for one snapshot: build q(x), high-pass at k_hard,
    square, cross-correlate against delta_g_field at k_soft_bins.

    Returns {'k_centers', 'P_cross'} -- one snapshot's raw (unweighted by
    the Eq 12 radial kernel -- see limber_sum_snapshots for that)
    squeezed-bispectrum-proxy contribution.
    """
    q = build_momentum_field_3d(density, xHI, velocity_los, c_mpc_s)
    q_filtered = highpass_filter_3d(q, box_len_mpc, k_hard)
    local_power = q_filtered**2
    return squeezed_bispectrum_proxy_3d(local_power, delta_g_field, box_len_mpc, k_soft_bins)


def limber_sum_snapshots(per_snapshot_results: list[dict], chi_list: np.ndarray,
                          g_chi_list: np.ndarray, bg_list: np.ndarray,
                          dchi_list: np.ndarray) -> dict:
    """
    Step 5: combine per-snapshot squeezed_bispectrum_proxy_3d outputs into
    the final radial (k_soft) sum, weighted by Eq. 12's
    g(chi)^2 * bg(chi) / chi^4 * dchi kernel (Wg is implicit in which
    snapshots/dchi are included in the lists).

    per_snapshot_results: list of dicts from compute_snapshot_bispectrum_
    contribution, one per snapshot, all sharing the same k_soft_bins
    (hence same k_centers) -- raises if they don't.
    chi_list, g_chi_list, bg_list, dchi_list: same length as
    per_snapshot_results, one value per snapshot.

    Returns {'k_centers', 'P_cross_summed'} -- still in k-space; converting
    k_centers to ell via a reference chi (make_ell) is the caller's job,
    same pattern as everywhere else in this repo.
    """
    n_snap = len(per_snapshot_results)
    if not (len(chi_list) == len(g_chi_list) == len(bg_list) == len(dchi_list) == n_snap):
        raise ValueError(
            f"Length mismatch: {n_snap} snapshot results but "
            f"chi_list={len(chi_list)}, g_chi_list={len(g_chi_list)}, "
            f"bg_list={len(bg_list)}, dchi_list={len(dchi_list)}."
        )
    if n_snap == 0:
        raise ValueError("No snapshots provided -- nothing to sum.")

    k_centers_ref = per_snapshot_results[0]["k_centers"]
    stacked = np.zeros_like(k_centers_ref)
    for i, res in enumerate(per_snapshot_results):
        if not np.allclose(res["k_centers"], k_centers_ref):
            raise ValueError(
                f"Snapshot {i}'s k_centers don't match snapshot 0's -- were "
                f"these computed with different k_soft_bins? Cannot sum."
            )
        weight = g_chi_list[i] ** 2 * bg_list[i] / chi_list[i] ** 4 * dchi_list[i]
        contribution = np.nan_to_num(res["P_cross"], nan=0.0)
        stacked = stacked + weight * contribution

    return {"k_centers": k_centers_ref, "P_cross_summed": stacked}
