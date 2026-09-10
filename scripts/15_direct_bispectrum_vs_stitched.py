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
    build_tau_history,
    compute_snapshot_bispectrum_contribution,
    limber_sum_snapshots,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.io.la_plante_reference import load_dell_vs_ell_band
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.figio import save_fig


def compute_direct_dell_one_seed(cfg, stitcher, seed, z0, dz, ell_peak_filter, ell_edges, ell_centers):
    """One seed's D_ell(ell) at the given (z0, dz) window. Returns
    (ell_direct, D_direct) or None if the window is empty for this seed."""
    logger = setup_logger(seed, cfg.paths.lightcone_root)
    all_snap_z = stitcher.get_snapshot_redshifts(seed, logger)
    z_sorted, chi_sorted, tau_cumulative, x_e_mean = build_tau_history(cfg, stitcher, seed, all_snap_z, logger)

    in_window = (z_sorted >= z0 - dz / 2) & (z_sorted < z0 + dz / 2)
    z_window = z_sorted[in_window]
    if len(z_window) == 0:
        print(f"  seed {seed}: NO snapshots in window z0={z0}, dz={dz} -- skipping this seed.")
        return None
    print(f"  seed {seed}: {len(z_window)} snapshots inside window: z={z_window.min():.3f}-{z_window.max():.3f}")

    c_mpc_s = constants.c_mpc_per_s()
    tau_pref = constants.tau_prefactor(cfg)

    per_snapshot_results = []
    chi_list, g_chi_list, bg_list, dchi_list = [], [], [], []

    for z in z_window:
        i = np.searchsorted(z_sorted, z)
        chi_z = chi_sorted[i]
        tau_z = tau_cumulative[i]
        g_chi = tau_pref * x_e_mean[i] * (1.0 + z) ** 2 * np.exp(-tau_z)

        density = np.asarray(stitcher.load_field_box(seed, z, "density"))
        xHI = np.asarray(stitcher.load_field_box(seed, z, "xH"))
        v_los = np.asarray(stitcher.load_field_box(seed, z, "vz"))

        bg_z = float(bluetides_bias_gz(z))
        delta_g = bg_z * (density - 1.0)

        k_hard = ell_peak_filter / chi_z
        k_soft_edges = ell_edges / chi_z
        result = compute_snapshot_bispectrum_contribution(
            density, xHI, v_los, delta_g, cfg.box.box_len_mpc, c_mpc_s, k_hard, k_soft_edges
        )
        result["k_centers"] = ell_centers
        per_snapshot_results.append(result)

        dchi = float(np.abs(np.gradient(chi_sorted))[i])
        chi_list.append(chi_z)
        g_chi_list.append(g_chi)
        bg_list.append(1.0)
        dchi_list.append(dchi)

    summed = limber_sum_snapshots(per_snapshot_results, np.array(chi_list), np.array(g_chi_list),
                                    np.array(bg_list), np.array(dchi_list))
    ell_direct = summed["k_centers"]
    D_direct = ell_direct * (ell_direct + 1) * summed["P_cross_summed"] * constants.T_CMB_UK**2 / (2 * np.pi)
    return ell_direct, D_direct


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

    print(f"Running {len(seeds)} seed(s): {seeds}")
    per_seed_D = []
    for seed in seeds:
        res = compute_direct_dell_one_seed(cfg, stitcher, seed, args.z0, args.dz,
                                            args.ell_peak_filter, ell_edges, ell_centers)
        if res is not None:
            per_seed_D.append(res[1])

    if not per_seed_D:
        print("No seeds produced usable data -- nothing to plot.")
        return 1

    stacked = np.array(per_seed_D)
    ell_direct = ell_centers
    D_mean = np.mean(stacked, axis=0)
    D_std = np.std(stacked, axis=0) if len(per_seed_D) > 1 else np.zeros_like(D_mean)
    n_used = len(per_seed_D)

    print(f"\nDirect/coeval result (mean +/- std over {n_used} seed(s)):")
    for e, d, s in zip(ell_direct, D_mean, D_std):
        print(f"  ell={e:.0f}: D_ell={d:.4g} +/- {s:.4g} uK^2")

    seed_tag = f"{n_used}seeds" if n_used > 1 else f"seed{seeds[0]}"
    csv_path = os.path.join(args.out_dir, f"direct_bispectrum_{seed_tag}_z{args.z0:.1f}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell", "D_ell_mean_uK2", "D_ell_std_uK2", "n_seeds"])
        for e, d, s in zip(ell_direct, D_mean, D_std):
            w.writerow([f"{e:.6f}", f"{d:.6e}", f"{s:.6e}", n_used])
    print(f"Saved: {csv_path}")

    fig, (ax, ax_norm) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax.errorbar(ell_direct, D_mean, yerr=D_std, fmt="o-", color="darkgreen", capsize=3,
                label=f"Direct/coeval (mean +/- std, {n_used} seed{'s' if n_used > 1 else ''}, no stitching)")

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

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$]")
    ax.set_title(f"Absolute amplitude -- {n_used} seed{'s' if n_used > 1 else ''}, z0={args.z0}, dz={args.dz}")
    ax.legend(fontsize=8)

    # Shape-only comparison -- each curve normalized to its own peak, so a
    # shape match is visible even while the absolute amplitude is still off.
    # Only the POSITIVE part of D_mean is meaningful to peak-normalize
    # (matches scripts/11's convention for the same reason). Error bars
    # scaled by the same normalization factor as the central value.
    pos = D_mean > 0
    if np.any(pos):
        peak = D_mean[pos].max()
        ax_norm.errorbar(ell_direct, D_mean / peak, yerr=D_std / peak, fmt="o-", color="darkgreen",
                          capsize=3, label="Direct/coeval (shape only)")
    lp_hi_peak = lp_band["hi"].max()
    ax_norm.fill_between(lp_band["ell_lo"], lp_band["lo"] / lp_hi_peak,
                          np.interp(lp_band["ell_lo"], lp_band["ell_hi"], lp_band["hi"]) / lp_hi_peak,
                          color="black", alpha=0.15, label="La Plante+2022 band (shape only)")
    if ell_s:
        D_s_arr = np.array(D_s)
        pos_s = D_s_arr > 0
        if np.any(pos_s):
            ax_norm.plot(ell_s, D_s_arr / D_s_arr[pos_s].max(), "s--", color="firebrick", alpha=0.7,
                         label="Stitched (shape only)")
    ax_norm.axhline(0, color="gray", lw=0.5)
    ax_norm.set_xscale("log")
    ax_norm.set_xlabel(r"$\ell$")
    ax_norm.set_ylabel(r"$D_\ell / D_\ell^{\rm peak}$ (each curve normalized to its own peak)")
    ax_norm.set_title("Shape only -- peak-normalized")
    ax_norm.legend(fontsize=8)

    fig.suptitle(f"Direct/coeval vs stitched vs La Plante+2022 -- {cfg.box.box_len_mpc:.0f} Mpc box, "
                 f"same bias model, same window", fontsize=12)
    outpath = os.path.join(args.out_dir, f"direct_vs_stitched_{seed_tag}_z{args.z0:.1f}.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
