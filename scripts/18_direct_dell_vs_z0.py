#!/usr/bin/env python3
"""
scripts/18_direct_dell_vs_z0.py
==================================
The missing piece (2b): D_ell vs z0, computed the DIRECT way (per-
snapshot, no stitching -- same machinery as scripts/15), at three fixed
ell values (500, 1000, 3000) chosen to match exactly what's available in
the digitized La Plante+2022 reference bands (data/reference/la_plante_2022,
io/la_plante_reference.py) -- so the comparison is a real overlay, not
just a shape gesture. Also overlays the STITCHED pathway's own z0-sweep
(scripts/13's output, if available) -- shown as-is, including if it's
near zero, since that's the current honest state of that pathway.

This is the direct-method analog of scripts/13 (which only exists for the
STITCHED pathway -- currently broken, see today's velocity/stitching
investigation). Building this separately, rather than patching scripts/13,
because the whole point is to have a version that does NOT depend on the
still-broken stitching step.

MULTI-SEED (2026-09-10): --seeds accepts multiple seeds (default: all of
cfg.box.seeds). For each seed, the FULL z0 sweep is computed
independently (its own optical-depth history, its own per-snapshot
bispectrum contributions at every z0), then aggregated as mean +/- std
across seeds at each (ell, z0) point. This multiplies the compute cost
by the number of seeds -- N seeds = N times the work of one, on top of
the z0-sweep's own cost.

COMPUTE COST WARNING: each z0 point requires a full snapshot-level 3D FFT
bispectrum computation (same cost as one scripts/15 run) for every
snapshot in its window, times every seed requested. Uses NON-OVERLAPPING
windows (z0 step == dz) so no snapshot's expensive computation is
repeated across sweep points WITHIN one seed. Run via qsub, not
interactively -- for all 10 seeds this could run many hours.

Usage:
    python scripts/18_direct_dell_vs_z0.py --seeds 1
    python scripts/18_direct_dell_vs_z0.py --seeds 1 2 3 --z0-min 5 --z0-max 16 --dz 1.5
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

TARGET_ELLS = [500.0, 1000.0, 3000.0]


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
        x_hii_window.append(x_e_mean[i])

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


def run_one_seed(cfg, stitcher, seed, z0_grid, dz, ell_peak_filter, n_ell_grid,
                  ell_min_grid, ell_max_grid):
    """Full z0 sweep for one seed. Returns {z0: {'D': {ell: val}, 'x_hii': val}}."""
    logger = setup_logger(seed, cfg.paths.lightcone_root)
    print(f"Seed {seed}: finding real coeval snapshot redshifts...")
    all_snap_z = stitcher.get_snapshot_redshifts(seed, logger)
    print(f"  Found {len(all_snap_z)} snapshots: z={all_snap_z.min():.4f} -> {all_snap_z.max():.4f}")

    print(f"  Building full optical-depth history for seed {seed}...")
    z_sorted, chi_sorted, tau_cumulative, x_e_mean = build_tau_history(cfg, stitcher, seed, all_snap_z, logger)

    out = {}
    for z0 in z0_grid:
        print(f"\n  seed {seed}, z0={z0:.3f} (window +-{dz/2:.2f}):")
        res = compute_dell_at_target_ells(
            cfg, stitcher, seed, z0, dz, chi_sorted, tau_cumulative, x_e_mean, z_sorted,
            ell_peak_filter, n_ell_grid, ell_min_grid, ell_max_grid
        )
        if res is None:
            print("    no snapshots in this window -- skipping")
            continue
        out[z0] = {"D": dict(zip(TARGET_ELLS, res["D_at_targets"])), "x_hii": res["x_hii"]}
        print(f"    {res['n_snapshots']} snapshots, x_HII~{res['x_hii']:.3f}, D_ell={out[z0]['D']}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Seeds to average over (default: all of cfg.box.seeds)")
    parser.add_argument("--z0-min", type=float, default=5.0)
    parser.add_argument("--z0-max", type=float, default=16.0)
    parser.add_argument("--dz", type=float, default=1.5, help="Window width -- z0 step matches this "
                         "exactly, giving non-overlapping windows (no repeated snapshot compute)")
    parser.add_argument("--ell-peak-filter", type=float, default=3500.0)
    parser.add_argument("--n-ell-grid", type=int, default=8)
    parser.add_argument("--ell-min-grid", type=float, default=300.0)
    parser.add_argument("--ell-max-grid", type=float, default=5000.0)
    parser.add_argument("--stitched-csv", type=str, default=None,
                         help="Path to scripts/13's dell_vs_z_multiell CSV, to overlay "
                              "(default: guess from --out-dir/SO)")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    z0_grid = np.arange(args.z0_min, args.z0_max + 1e-6, args.dz) + args.dz / 2

    print(f"Running {len(seeds)} seed(s): {seeds}")
    per_seed_out = {}
    for seed in seeds:
        per_seed_out[seed] = run_one_seed(cfg, stitcher, seed, z0_grid, args.dz, args.ell_peak_filter,
                                           args.n_ell_grid, args.ell_min_grid, args.ell_max_grid)

    z0_common = sorted(set.intersection(*(set(d.keys()) for d in per_seed_out.values())))
    if not z0_common:
        print("No z0 point has data from every requested seed -- nothing to plot.")
        return 1
    n_used = len(seeds)

    z0_used = np.array(z0_common)
    D_mean_at_ell, D_std_at_ell = {}, {}
    x_hii_used = np.array([np.mean([per_seed_out[s][z0]["x_hii"] for s in seeds]) for z0 in z0_common])
    for ell in TARGET_ELLS:
        vals = np.array([[per_seed_out[s][z0]["D"][ell] for s in seeds] for z0 in z0_common])
        D_mean_at_ell[ell] = vals.mean(axis=1)
        D_std_at_ell[ell] = vals.std(axis=1) if n_used > 1 else np.zeros(len(z0_common))

    seed_tag = f"{n_used}seeds" if n_used > 1 else f"seed{seeds[0]}"
    csv_path = os.path.join(args.out_dir, f"direct_dell_vs_z0_{seed_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell", "z0", "D_ell_mean_uK2", "D_ell_std_uK2", "x_HII", "n_seeds"])
        for ell in TARGET_ELLS:
            for z0, D, s, x in zip(z0_used, D_mean_at_ell[ell], D_std_at_ell[ell], x_hii_used):
                w.writerow([ell, f"{z0:.4f}", f"{D:.6e}", f"{s:.6e}", f"{x:.4f}", n_used])
    print(f"\nSaved: {csv_path}")

    lp_bands = load_dell_vs_z0_bands()
    colors = {500.0: "tab:blue", 1000.0: "tab:orange", 3000.0: "tab:green"}

    stitched_csv = args.stitched_csv or os.path.join(args.out_dir, "dell_vs_z_multiell_SO.csv")
    stitched = {ell: {"z0": [], "D": []} for ell in TARGET_ELLS}
    if os.path.exists(stitched_csv):
        with open(stitched_csv) as f:
            for row in csv.DictReader(f):
                ell = float(row["ell"])
                if ell in stitched:
                    stitched[ell]["z0"].append(float(row["z0"]))
                    stitched[ell]["D"].append(float(row["D_ell_uK2"]))
    else:
        print(f"  (no stitched overlay -- {stitched_csv} not found; run scripts/13 first for that)")

    fig, (ax, ax_norm) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for ell in TARGET_ELLS:
        ax.errorbar(z0_used, D_mean_at_ell[ell], yerr=D_std_at_ell[ell], fmt="o-", color=colors[ell],
                     capsize=3, label=f"Direct/coeval, ell={ell:.0f} ({n_used} seed{'s' if n_used > 1 else ''})")
        lp = lp_bands[int(ell)]
        lp_hi_interp = np.interp(lp["z0_lo"], lp["z0_hi"], lp["hi"])
        ax.fill_between(lp["z0_lo"], lp["lo"], lp_hi_interp,
                          color=colors[ell], alpha=0.15,
                          label=f"La Plante+2022 (digitized), ell={ell:.0f}")
        if stitched[ell]["z0"]:
            order_s = np.argsort(stitched[ell]["z0"])
            zs = np.array(stitched[ell]["z0"])[order_s]
            Ds = np.array(stitched[ell]["D"])[order_s]
            ax.plot(zs, Ds, "s--", color=colors[ell], alpha=0.5, ms=4,
                     label=f"Stitched, ell={ell:.0f}")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$]")
    ax.set_title(f"Absolute amplitude -- {n_used} seed{'s' if n_used > 1 else ''}, dz={args.dz}", pad=30)
    ax.text(0.02, 0.02, f"{cfg.box.box_len_mpc:.0f} Mpc box vs paper's larger box -- amplitude "
            f"not expected to match,\nshape/z0-dependence is the comparison",
            transform=ax.transAxes, fontsize=7, style="italic", va="bottom", alpha=0.7)
    ax.legend(fontsize=7, ncol=2)

    for ell in TARGET_ELLS:
        D_arr = D_mean_at_ell[ell]
        pos = D_arr > 0
        if np.any(pos):
            peak = D_arr[pos].max()
            ax_norm.errorbar(z0_used, D_arr / peak, yerr=D_std_at_ell[ell] / peak, fmt="o-",
                              color=colors[ell], capsize=3, label=f"Direct/coeval, ell={ell:.0f} (shape only)")
        lp = lp_bands[int(ell)]
        lp_hi_interp = np.interp(lp["z0_lo"], lp["z0_hi"], lp["hi"])
        lp_peak = lp_hi_interp.max()
        ax_norm.fill_between(lp["z0_lo"], lp["lo"] / lp_peak, lp_hi_interp / lp_peak,
                              color=colors[ell], alpha=0.15,
                              label=f"La Plante+2022, ell={ell:.0f} (shape only)")
        if stitched[ell]["z0"]:
            order_s = np.argsort(stitched[ell]["z0"])
            zs = np.array(stitched[ell]["z0"])[order_s]
            Ds = np.array(stitched[ell]["D"])[order_s]
            pos_s = Ds > 0
            if np.any(pos_s):
                ax_norm.plot(zs, Ds / Ds[pos_s].max(), "s--", color=colors[ell], alpha=0.5, ms=4,
                             label=f"Stitched, ell={ell:.0f} (shape only)")

    ax_norm.axhline(0, color="gray", lw=0.5)
    ax_norm.set_xlabel(r"$z_0$")
    ax_norm.set_ylabel(r"$D_\ell / D_\ell^{\rm peak}$ (each curve normalized to its own peak)")
    ax_norm.set_title("Shape only -- peak-normalized", pad=30)
    ax_norm.legend(fontsize=6, ncol=2)

    for this_ax in (ax, ax_norm):
        order = np.argsort(z0_used)
        z_sorted_ax, x_sorted_ax = z0_used[order], x_hii_used[order]
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
                secax = this_ax.secondary_xaxis("top", functions=(z_to_x, x_to_z))
                secax.set_xlabel(r"$x_{\rm HI}$ (this simulation)")
            except Exception as e:
                print(f"  (secondary x_HI axis skipped: {e})")

    fig.suptitle(f"D_ell vs z0, direct/coeval vs stitched vs La Plante+2022", fontsize=13)
    outpath = os.path.join(args.out_dir, f"direct_dell_vs_z0_{seed_tag}.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
