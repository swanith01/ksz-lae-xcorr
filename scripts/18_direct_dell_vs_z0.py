#!/usr/bin/env python3
"""
scripts/18_direct_dell_vs_z0.py
==================================
The missing piece (2b): D_ell vs z0, computed the DIRECT way (per-
snapshot, no stitching -- same machinery as scripts/15), at three fixed
ell values (500, 1000, 3000) chosen to match exactly what's available in
the digitized La Plante+2022 reference bands (data/reference/la_plante_2022,
io/la_plante_reference.py) -- so the comparison is a real overlay, not
just a shape gesture.

This is the direct-method analog of scripts/13 (which only exists for the
STITCHED pathway -- currently broken, see today's velocity/stitching
investigation). Building this separately, rather than patching scripts/13,
because the whole point is to have a version that does NOT depend on the
still-broken stitching step.

COMPUTE COST WARNING: each z0 point requires a full snapshot-level 3D FFT
bispectrum computation (same cost as one scripts/15 run) for every
snapshot in its window. Uses NON-OVERLAPPING windows (z0 step == dz) so
no snapshot's expensive computation is repeated across sweep points, but
this is still real, multi-snapshot-times-multi-z0 compute -- run via
qsub, not interactively.

Usage:
    python scripts/18_direct_dell_vs_z0.py --seed 1
    python scripts/18_direct_dell_vs_z0.py --seed 1 --z0-min 7 --z0-max 16 --dz 1.5
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
from ksz_lae_xcorr.io.la_plante_reference import load_dell_vs_z0_bands
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig

TARGET_ELLS = [500.0, 1000.0, 3000.0]  # matches the digitized La Plante z0-bands exactly


def compute_dell_at_target_ells(cfg, stitcher, seed, z0, dz, chi_sorted, tau_cumulative,
                                 x_e_mean, z_sorted, ell_peak_filter, n_ell_grid,
                                 ell_min_grid, ell_max_grid):
    """One z0 window's D_ell at TARGET_ELLS, via the same
    filter->square->cross-correlate->Limber-sum machinery as scripts/15."""
    in_window = (z_sorted >= z0 - dz / 2) & (z_sorted < z0 + dz / 2)
    z_window = z_sorted[in_window]
    if len(z_window) == 0:
        return None

    ell_edges = np.geomspace(ell_min_grid, ell_max_grid, n_ell_grid + 1)
    ell_centers = np.sqrt(ell_edges[:-1] * ell_edges[1:])
    c_mpc_s = constants.c_mpc_per_s()
    tau_pref = constants.tau_prefactor(cfg)

    per_snapshot_results = []
    chi_list, g_chi_list, bg_list, dchi_list = [], [], [], []
    x_hii_window = []

    for z in z_window:
        i = np.searchsorted(z_sorted, z)
        chi_z = chi_sorted[i]
        tau_z = tau_cumulative[i]
        g_chi = tau_pref * x_e_mean[i] * (1.0 + z) ** 2 * np.exp(-tau_z)
        x_hii_window.append(x_e_mean[i])  # x_e_mean IS the ionized fraction already
                                            # (see build_tau_history) -- do NOT flip it
                                            # again here. An earlier version of this line
                                            # did `1.0 - x_e_mean[i]`, silently reporting
                                            # the NEUTRAL fraction while labeling it
                                            # x_HII -- caught 2026-09-10 when the reported
                                            # "x_HII" rose toward 1.0 at HIGH z, backwards
                                            # from real physics (x_HII should fall toward 0
                                            # going to higher z, before reionization).

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

    valid = np.isfinite(D_direct)
    if valid.sum() < 3:
        return None
    D_at_targets = np.interp(np.log(TARGET_ELLS), np.log(ell_direct[valid]), D_direct[valid])
    return {"D_at_targets": D_at_targets, "x_hii": float(np.mean(x_hii_window)),
            "n_snapshots": len(z_window)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z0-min", type=float, default=5.0)
    parser.add_argument("--z0-max", type=float, default=16.0)
    parser.add_argument("--dz", type=float, default=1.5, help="Window width -- z0 step matches this "
                         "exactly, giving non-overlapping windows (no repeated snapshot compute)")
    parser.add_argument("--ell-peak-filter", type=float, default=3500.0)
    parser.add_argument("--n-ell-grid", type=int, default=8)
    parser.add_argument("--ell-min-grid", type=float, default=300.0)
    parser.add_argument("--ell-max-grid", type=float, default=5000.0)
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    print(f"Seed {args.seed}: finding real coeval snapshot redshifts...")
    all_snap_z = stitcher.get_snapshot_redshifts(args.seed, logger)
    print(f"  Found {len(all_snap_z)} snapshots: z={all_snap_z.min():.4f} -> {all_snap_z.max():.4f}")

    print("Building full optical-depth history (once per seed, reused across all z0)...")
    z_sorted, chi_sorted, tau_cumulative, x_e_mean = build_tau_history(
        cfg, stitcher, args.seed, all_snap_z, logger)

    z0_grid = np.arange(args.z0_min, args.z0_max + 1e-6, args.dz) + args.dz / 2
    z0_used, D_at_ell = [], {ell: [] for ell in TARGET_ELLS}
    x_hii_used = []

    for z0 in z0_grid:
        print(f"\n  z0={z0:.3f} (window +-{args.dz/2:.2f}):")
        res = compute_dell_at_target_ells(
            cfg, stitcher, args.seed, z0, args.dz, chi_sorted, tau_cumulative, x_e_mean, z_sorted,
            args.ell_peak_filter, args.n_ell_grid, args.ell_min_grid, args.ell_max_grid
        )
        if res is None:
            print("    no snapshots in this window -- skipping")
            continue
        z0_used.append(z0)
        x_hii_used.append(res["x_hii"])
        for ell, D in zip(TARGET_ELLS, res["D_at_targets"]):
            D_at_ell[ell].append(D)
        print(f"    {res['n_snapshots']} snapshots, x_HII~{res['x_hii']:.3f}, "
              f"D_ell={dict(zip(TARGET_ELLS, res['D_at_targets']))}")

    z0_used = np.array(z0_used)
    if len(z0_used) == 0:
        print("No usable z0 points -- nothing to plot.")
        return 1

    csv_path = os.path.join(args.out_dir, f"direct_dell_vs_z0_seed{args.seed}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell", "z0", "D_ell_uK2", "x_HII"])
        for ell in TARGET_ELLS:
            for z0, D, x in zip(z0_used, D_at_ell[ell], x_hii_used):
                w.writerow([ell, f"{z0:.4f}", f"{D:.6e}", f"{x:.4f}"])
    print(f"\nSaved: {csv_path}")

    lp_bands = load_dell_vs_z0_bands()
    colors = {500.0: "tab:blue", 1000.0: "tab:orange", 3000.0: "tab:green"}

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    for ell in TARGET_ELLS:
        ax.plot(z0_used, D_at_ell[ell], "o-", color=colors[ell],
                 label=f"Direct/coeval, ell={ell:.0f}")
        lp = lp_bands[int(ell)]
        ax.fill_between(lp["z0_lo"], lp["lo"],
                          np.interp(lp["z0_lo"], lp["z0_hi"], lp["hi"]),
                          color=colors[ell], alpha=0.15,
                          label=f"La Plante+2022 (digitized), ell={ell:.0f}")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$]")
    ax.set_title(f"D_ell vs z0, direct/coeval method -- seed {args.seed}, dz={args.dz}", pad=30)
    ax.text(0.02, 0.02, f"{cfg.box.box_len_mpc:.0f} Mpc box vs paper's larger box -- amplitude "
            f"not expected to match,\nshape/z0-dependence is the comparison",
            transform=ax.transAxes, fontsize=7, style="italic", va="bottom", alpha=0.7)
    ax.legend(fontsize=8, ncol=2)

    order = np.argsort(z0_used)
    z_sorted_ax, x_sorted_ax = z0_used[order], np.array(x_hii_used)[order]
    if np.all(np.diff(x_sorted_ax) <= 0) or np.all(np.diff(x_sorted_ax) >= 0):
        from scipy.interpolate import interp1d
        z_to_x = interp1d(z_sorted_ax, x_sorted_ax, bounds_error=False,
                           fill_value=(x_sorted_ax[0], x_sorted_ax[-1]))
        x_to_z = interp1d(x_sorted_ax, z_sorted_ax, bounds_error=False,
                           fill_value=(z_sorted_ax[0], z_sorted_ax[-1])) \
            if x_sorted_ax[0] < x_sorted_ax[-1] else \
            interp1d(x_sorted_ax[::-1], z_sorted_ax[::-1], bounds_error=False,
                     fill_value=(z_sorted_ax[-1], z_sorted_ax[0]))
        try:
            secax = ax.secondary_xaxis("top", functions=(z_to_x, x_to_z))
            secax.set_xlabel(r"$x_{\rm HI}$ (this simulation)")
        except Exception as e:
            print(f"  (secondary x_HI axis skipped: {e})")

    outpath = os.path.join(args.out_dir, f"direct_dell_vs_z0_seed{args.seed}.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
