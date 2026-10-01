#!/usr/bin/env python3
"""
scripts/15_direct_bispectrum_vs_stitched.py
==============================================
The comparison this whole session has been building toward: kSZ^2 x
galaxy D_ell computed the DIRECT way (correlation/direct_bispectrum.py,
per-snapshot, no stitching, no periodicity risk) against the STITCHED way
(snr/roman_hls_benchmark.py, scripts/11's output) -- same box, same bias
model, same z-window, so any difference is attributable to the
stitching/periodicity mechanism itself, not a different physical setup.

Reuses lightcone.stitch.Stitcher.load_field_box directly for raw
per-snapshot fields (density, xHI, velocity_z) -- NOT a new loader with
independently-guessed unit conversions. That function already handles the
DIM->HII_DIM density downsampling and the velocity_z unit conversion
(/(1+z)*3.086e19) exactly as the rest of this repo trusts them; reusing it
means this script can't silently drift into a different convention.

g(chi) (Eq. 5's visibility function) needs the FULL optical-depth history
from z_min up to each snapshot -- not just the snapshots inside the
analysis z-window -- so this script loads x_e_mean(z) for EVERY snapshot
first (cheap: just a mean, not the full field) to build a proper
cumulative tau(z), then only builds the full 3D momentum/bispectrum
machinery for snapshots actually inside the z0+-dz/2 window being
analyzed.

MULTI-SEED (2026-09-10): --seeds accepts multiple seeds (default: all of
cfg.box.seeds). Each seed is computed fully independently (its own
optical-depth history, its own per-snapshot bispectrum contributions),
then aggregated as mean +/- std across seeds at each ell. This is real,
multiplicative compute cost -- N seeds = N times the work of one.

HONEST STATUS: this is the first time correlation/direct_bispectrum.py
has touched real data. Every piece (loader reuse, g(chi) construction,
k_hard/k_soft mapping, bg(z) reuse from roman_hls_benchmark.py) is
individually justified above and in the modules it calls, but the
COMBINATION has only been synthetic-tested at the estimator level
(test_direct_bispectrum.py), not end-to-end against a real box. Expect to
need adjustment once real output is in hand -- report exactly what
happens, don't assume it's already right.

Usage:
    python scripts/15_direct_bispectrum_vs_stitched.py
    python scripts/15_direct_bispectrum_vs_stitched.py --seeds 1 2 3 --z0 9.5 --dz 1.0
"""

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation.direct_bispectrum import (
    build_delta_g_field,
    build_tau_history,
    compute_snapshot_bispectrum_contribution,
    limber_sum_snapshots,
)
from ksz_lae_xcorr.correlation.velocity_reconstruction import (
    measure_large_scale_bias,
    reconstruct_velocity_los,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.io.la_plante_reference import load_dell_vs_ell_band
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.figio import compute_symlog_linthresh, save_fig


def _get_velocity(source, z, stitcher, seed, logger, cfg, delta_matter, r_smooth_mpc):
    """One snapshot's v_los, for one velocity source.

    'native': the simulation's own velocity_z (unchanged behaviour).
    'halo_reconstructed': v_los reconstructed via the linear continuity
    equation (velocity_reconstruction.reconstruct_velocity_los) FROM THAT
    SNAPSHOT'S OWN halo catalogue (mass_cut=cfg.tracers.halo_mass_cut_msun),
    bias measured empirically from this box (measure_large_scale_bias,
    same reasoning as scripts/25: no bias fit in this repo is calibrated
    to this specific mass cut), smoothed at r_smooth_mpc (default 18.4 Mpc,
    the paper's R_s, and the value scripts/25 found necessary -- an
    unsmoothed halo reconstruction is shot-noise-dominated).

    Returns (v_los, bias_used) or (None, None) if this snapshot's halo
    catalogue is empty (nothing to reconstruct from -- caller should skip).
    """
    if source == "native":
        return np.asarray(stitcher.load_field_box(seed, z, "vz")), None

    if source == "halo_reconstructed":
        counts = stitcher.load_halo_grid(seed, z, logger).astype(np.float64)
        if counts.sum() <= 0:
            return None, None
        delta_h = (counts - counts.mean()) / counts.mean()
        bias_result = measure_large_scale_bias(delta_h, delta_matter, cfg.box.box_len_mpc)
        v_rec = reconstruct_velocity_los(cfg, delta_h, cfg.box.box_len_mpc, z,
                                          bias=bias_result["b_eff"], r_smooth_mpc=r_smooth_mpc)
        return v_rec, bias_result["b_eff"]

    raise ValueError(f"Unknown velocity source '{source}'.")


def compute_direct_dell_one_seed(cfg, stitcher, seed, z0, dz, ell_peak_filter, ell_edges, ell_centers,
                                  tracer="density_proxy", velocity_sources=("native",),
                                  recon_r_smooth=18.4):
    """One seed's D_ell(ell) at the given (z0, dz) window, for EACH velocity
    source requested (so native and halo_reconstructed can be compared
    from the exact same snapshots/delta_g, not two separate runs).

    Returns {source: (ell_direct, D_direct)} -- a source is simply absent
    from the dict if every snapshot in the window was unusable for it
    (e.g. halo_reconstructed with no halos above the mass cut anywhere in
    the window).

    tracer: 'density_proxy' (default, La Plante+2022 Eq. 6-7 bias-weighted
    matter field), 'lae', or 'lbg' (real tracer counts -- see
    correlation.direct_bispectrum.build_delta_g_field's docstring for the
    honest sparsity caveat on real LAE/LBG counts at the per-snapshot level).
    velocity_sources: subset of ('native', 'halo_reconstructed')."""
    logger = setup_logger(seed, cfg.paths.lightcone_root)
    all_snap_z = stitcher.get_snapshot_redshifts(seed, logger)
    z_sorted, chi_sorted, tau_cumulative, x_e_mean = build_tau_history(cfg, stitcher, seed, all_snap_z, logger)

    in_window = (z_sorted >= z0 - dz / 2) & (z_sorted < z0 + dz / 2)
    z_window = z_sorted[in_window]
    if len(z_window) == 0:
        print(f"  seed {seed}: NO snapshots in window z0={z0}, dz={dz} -- skipping this seed.")
        return {}
    print(f"  seed {seed}: {len(z_window)} snapshots inside window: z={z_window.min():.3f}-{z_window.max():.3f}")

    c_mpc_s = constants.c_mpc_per_s()
    tau_pref = constants.tau_prefactor(cfg)

    per_snapshot_results = {src: [] for src in velocity_sources}
    chi_g_dchi = {src: ([], [], [], []) for src in velocity_sources}  # chi, g_chi, bg, dchi
    n_skipped_empty_g = 0
    n_skipped_empty_v = {src: 0 for src in velocity_sources}
    bias_log = {src: [] for src in velocity_sources}

    for z in z_window:
        i = np.searchsorted(z_sorted, z)
        chi_z = chi_sorted[i]
        tau_z = tau_cumulative[i]
        g_chi = tau_pref * x_e_mean[i] * (1.0 + z) ** 2 * np.exp(-tau_z)

        density = np.asarray(stitcher.load_field_box(seed, z, "density"))
        xHI = np.asarray(stitcher.load_field_box(seed, z, "xH"))
        delta_matter = density.astype(np.float64) - 1.0
        delta_matter -= delta_matter.mean()

        delta_g = build_delta_g_field(stitcher, seed, z, tracer, logger,
                                       bias_fn=bluetides_bias_gz, density=density)
        if delta_g is None:
            n_skipped_empty_g += 1
            continue

        k_hard = ell_peak_filter / chi_z
        k_soft_edges = ell_edges / chi_z
        dchi = float(np.abs(np.gradient(chi_sorted))[i])

        for src in velocity_sources:
            v_los, bias_used = _get_velocity(src, z, stitcher, seed, logger, cfg, delta_matter, recon_r_smooth)
            if v_los is None:
                n_skipped_empty_v[src] += 1
                continue
            if bias_used is not None:
                bias_log[src].append(bias_used)

            result = compute_snapshot_bispectrum_contribution(
                density, xHI, v_los, delta_g, cfg.box.box_len_mpc, c_mpc_s, k_hard, k_soft_edges
            )
            result["k_centers"] = ell_centers
            per_snapshot_results[src].append(result)

            chi_list, g_chi_list, bg_list, dchi_list = chi_g_dchi[src]
            chi_list.append(chi_z)
            g_chi_list.append(g_chi)
            bg_list.append(1.0)
            dchi_list.append(dchi)

    if n_skipped_empty_g:
        print(f"  seed {seed}: {n_skipped_empty_g}/{len(z_window)} snapshots skipped "
              f"(all-zero {tracer} counts in this window -- affects every velocity source)")

    out = {}
    for src in velocity_sources:
        if n_skipped_empty_v[src]:
            print(f"  seed {seed}: [{src}] {n_skipped_empty_v[src]}/{len(z_window)} snapshots skipped "
                  f"(no usable velocity for this source)")
        if bias_log[src]:
            b_arr = np.array(bias_log[src])
            print(f"  seed {seed}: [{src}] halo bias over {len(b_arr)} snapshots: "
                  f"mean={b_arr.mean():.3f}, range={b_arr.min():.3f}-{b_arr.max():.3f}")
        if not per_snapshot_results[src]:
            print(f"  seed {seed}: [{src}] no usable snapshots at all -- skipping this source for this seed.")
            continue
        chi_list, g_chi_list, bg_list, dchi_list = chi_g_dchi[src]
        summed = limber_sum_snapshots(per_snapshot_results[src], np.array(chi_list), np.array(g_chi_list),
                                        np.array(bg_list), np.array(dchi_list))
        ell_direct = summed["k_centers"]
        D_direct = ell_direct * (ell_direct + 1) * summed["P_cross_summed"] * constants.T_CMB_UK**2 / (2 * np.pi)
        out[src] = (ell_direct, D_direct)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Seeds to average over (default: all of cfg.box.seeds)")
    parser.add_argument("--z0", type=float, default=9.5)
    parser.add_argument("--dz", type=float, default=1.0)
    parser.add_argument("--ell-peak-filter", type=float, default=3500.0,
                         help="Representative CMB-filter peak ell for k_hard=ell_peak/chi(z) "
                              "(paper's filters peak ~3000-5000 depending on experiment; "
                              "this is a single representative value, not experiment-specific yet)")
    parser.add_argument("--n-ell", type=int, default=12)
    parser.add_argument("--ell-min", type=float, default=200.0)
    parser.add_argument("--ell-max", type=float, default=6000.0)
    parser.add_argument("--tracer", type=str, default="density_proxy",
                         choices=["density_proxy", "lae", "lbg"],
                         help="Galaxy field: La Plante+2022 bias-weighted density proxy (default), "
                              "or real LAE/LBG tracer counts (see build_delta_g_field's docstring "
                              "for the sparsity caveat on real counts)")
    parser.add_argument("--velocity", type=str, default="native",
                         choices=["native", "halo_reconstructed", "both"],
                         help="Source of v_los in the momentum field. 'native' (default, unchanged "
                              "behaviour): the simulation's own velocity_z. 'halo_reconstructed': "
                              "v_los reconstructed per-snapshot from that snapshot's OWN halo "
                              "catalogue via the linear continuity equation (measured bias, "
                              "r_smooth=--recon-r-smooth) -- tests how much kSZ^2 x <tracer> signal "
                              "survives without access to the true velocity field. 'both' computes "
                              "and overlays both from the exact same snapshots.")
    parser.add_argument("--recon-r-smooth", type=float, default=18.4,
                         help="Gaussian smoothing scale (Mpc) for --velocity halo_reconstructed -- "
                              "default is this repo's h=0.6777 conversion of the reference paper's "
                              "R_s=12.5 h^-1 Mpc, already found necessary in scripts/25 (an "
                              "unsmoothed halo reconstruction is shot-noise-dominated).")
    parser.add_argument("--stitched-csv", type=str, default=None,
                         help="Path to scripts/11's stage1_dell_points CSV for the SO row, "
                              "to overlay (default: guess from --out-dir/z0)")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    ell_edges = np.geomspace(args.ell_min, args.ell_max, args.n_ell + 1)
    ell_centers = np.sqrt(ell_edges[:-1] * ell_edges[1:])

    velocity_sources = {"native": ("native",), "halo_reconstructed": ("halo_reconstructed",),
                         "both": ("native", "halo_reconstructed")}[args.velocity]

    print(f"Running {len(seeds)} seed(s): {seeds}, tracer={args.tracer}, velocity={velocity_sources}")
    per_seed_D = {src: [] for src in velocity_sources}
    for seed in seeds:
        res = compute_direct_dell_one_seed(cfg, stitcher, seed, args.z0, args.dz,
                                            args.ell_peak_filter, ell_edges, ell_centers, tracer=args.tracer,
                                            velocity_sources=velocity_sources, recon_r_smooth=args.recon_r_smooth)
        for src, (ell_s_, D_s_) in res.items():
            per_seed_D[src].append(D_s_)

    if not any(per_seed_D.values()):
        print("No seeds produced usable data for any velocity source -- nothing to plot.")
        return 1

    ell_direct = ell_centers
    stats = {}  # src -> (D_mean, D_std, n_used)
    for src in velocity_sources:
        if not per_seed_D[src]:
            print(f"[{src}]: no seed produced usable data -- omitted from the plot.")
            continue
        stacked = np.array(per_seed_D[src])
        D_mean = np.mean(stacked, axis=0)
        D_std = np.std(stacked, axis=0) if stacked.shape[0] > 1 else np.zeros_like(D_mean)
        n_used = stacked.shape[0]
        stats[src] = (D_mean, D_std, n_used)
        print(f"\nDirect/coeval [{src}] result (mean +/- std over {n_used} seed(s)):")
        for e, d, s in zip(ell_direct, D_mean, D_std):
            print(f"  ell={e:.0f}: D_ell={d:.4g} +/- {s:.4g} uK^2")

    if not stats:
        print("No velocity source produced usable data -- nothing to plot.")
        return 1

    n_used_ref = next(iter(stats.values()))[2]
    seed_tag = f"{n_used_ref}seeds" if n_used_ref > 1 else f"seed{seeds[0]}"
    tracer_tag = "" if args.tracer == "density_proxy" else f"_{args.tracer}"
    velocity_tag = "" if args.velocity == "native" else f"_v-{args.velocity}"

    for src, (D_mean, D_std, n_used) in stats.items():
        src_tag = "" if src == "native" and args.velocity == "native" else f"_{src}"
        csv_path = os.path.join(args.out_dir, f"direct_bispectrum_{seed_tag}{tracer_tag}{src_tag}_z{args.z0:.1f}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ell", "D_ell_mean_uK2", "D_ell_std_uK2", "n_seeds"])
            for e, d, s in zip(ell_direct, D_mean, D_std):
                w.writerow([f"{e:.6f}", f"{d:.6e}", f"{s:.6e}", n_used])
        print(f"Saved: {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    src_style = {
        "native": dict(fmt="o-", color="darkgreen",
                        label_fmt="Direct/coeval, native velocity (mean +/- std, {n} seed{p}, no stitching)"),
        "halo_reconstructed": dict(fmt="^--", color="mediumorchid",
                                     label_fmt="Direct/coeval, halo-RECONSTRUCTED velocity "
                                                "(mean +/- std, {n} seed{p}, r_smooth=" +
                                                f"{args.recon_r_smooth:.1f} Mpc)"),
    }
    for src, (D_mean, D_std, n_used) in stats.items():
        style = src_style[src]
        label = style["label_fmt"].format(n=n_used, p="s" if n_used > 1 else "")
        ax.errorbar(ell_direct, D_mean, yerr=D_std, fmt=style["fmt"], color=style["color"],
                    capsize=3, label=label)

    lp_band = load_dell_vs_ell_band()
    ax.fill_between(lp_band["ell_lo"], lp_band["lo"],
                     np.interp(lp_band["ell_lo"], lp_band["ell_hi"], lp_band["hi"]),
                     color="black", alpha=0.15,
                     label="La Plante+2022 band (digitized, x_HII~0.43)")

    stitched_csv = args.stitched_csv or os.path.join(args.out_dir, f"stage1_dell_points_z{args.z0:.1f}.csv")
    ell_s, D_s = [], []
    if os.path.exists(stitched_csv):
        with open(stitched_csv) as f:
            for row in csv.DictReader(f):
                if row["experiment"] == "SO":
                    ell_s.append(float(row["ell"]))
                    D_s.append(float(row["D_ell_uK2"]))
        if ell_s:
            ax.plot(ell_s, D_s, "s--", color="firebrick", alpha=0.7, label="Stitched (scripts/11, SO filter, all seeds)")
    else:
        print(f"  (no stitched comparison overlay -- {stitched_csv} not found; run scripts/11 first for that)")

    # symlog, not log: these values can be legitimately negative
    # (noise-dominated bins), and a plain log axis would silently drop
    # every negative point rather than showing it -- symlog keeps a
    # linear region near zero and goes log-scale for larger magnitudes
    # on both sides, so nothing gets hidden.
    all_D_means = np.concatenate([D_mean for D_mean, _, _ in stats.values()])
    linthresh = compute_symlog_linthresh(all_D_means, lp_band["hi"], np.array(D_s))

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$] (symlog)")
    velocity_desc = {"native": "native velocity", "halo_reconstructed": "halo-reconstructed velocity",
                      "both": "native vs halo-reconstructed velocity"}[args.velocity]
    ax.set_title(f"Direct/coeval ({args.tracer}, {velocity_desc}) vs stitched vs La Plante+2022 -- "
                 f"z0={args.z0}, dz={args.dz}\n({cfg.box.box_len_mpc:.0f} Mpc box vs paper's larger box -- "
                 f"trend/amplitude both shown, nothing normalized away", fontsize=11)
    ax.legend(fontsize=8)

    outpath = os.path.join(args.out_dir, f"direct_vs_stitched_{seed_tag}{tracer_tag}{velocity_tag}_z{args.z0:.1f}.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
