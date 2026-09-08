#!/usr/bin/env python3
"""
scripts/13_dell_vs_z_multi_ell.py
====================================
D_ell(z0) at a handful of fixed ell values, matching La Plante+2022
Figure 5 exactly: their Fig 5 shows the same statistic peaking near
x_HII~0.2 (early reionization), roughly independent of which ell mode
you look at. This script checks whether OUR simulation shows the same
"hump" near ITS OWN reionization midpoint.

Sweeps z0 across a grid (fixed dz), reusing
snr.roman_hls_benchmark.compute_bias_weighted_cross_power at each z0,
extracts D_ell at a few fixed target ell values by interpolating each
z0's own (ell, D_ell) curve, and plots D_ell vs z0 -- one line per ell,
with this simulation's own volume-averaged x_HI(z0) (NOT the paper's
zreion model -- ours) on a secondary top axis, same layout as their Fig 5.

Requires scripts/04_compute_xcorr.py to have been run first.

Usage:
    python scripts/13_dell_vs_z_multi_ell.py
    python scripts/13_dell_vs_z_multi_ell.py --experiment SO --target-ells 500,1000,3000
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.snr.cmb_filter import build_cmb_filter_ingredients
from ksz_lae_xcorr.snr.roman_hls_benchmark import (
    compute_bias_weighted_cross_power,
    compute_volume_averaged_xHI,
)
from ksz_lae_xcorr.snr.snr_forecast import build_filtered_kSZ2_maps
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--experiment", type=str, default="SO",
                         help="Which CMB experiment's filter to use (must be in cfg.snr.experiments)")
    parser.add_argument("--dz", type=float, default=1.0, help="Galaxy window width (paper's Fig 5: dz=1)")
    parser.add_argument("--z-min", type=float, default=6.5)
    parser.add_argument("--z-max", type=float, default=18.0)
    parser.add_argument("--n-z", type=int, default=15, help="Number of z0 points to sweep")
    parser.add_argument("--target-ells", type=str, default="500,1000,3000",
                         help="Comma-separated ell values to trace vs z0 (paper's Fig 5 uses 500,1000,3000,10000)")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    target_ells = [float(x) for x in args.target_ells.split(",")]
    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    xcorr_path = f"{cfg.paths.products_root}/cross_results.pkl"
    print(f"Loading cross-correlation products from {xcorr_path} ...")
    with open(xcorr_path, "rb") as f:
        maps = pickle.load(f)["maps"]

    auto_path = f"{cfg.paths.products_root}/auto_results.pkl"
    with open(auto_path, "rb") as f:
        auto_results_ksz = pickle.load(f)["kSZ"]

    print("Loading field data (density + xHI)...")
    field_data, _ = load_lightcone_products(cfg, seeds)

    kg = KGrid(cfg)
    filt = build_cmb_filter_ingredients(cfg, kg, auto_results_ksz)
    filtered_kSZ2 = build_filtered_kSZ2_maps(cfg, kg, maps["kSZ"], filt, seeds)
    if args.experiment not in filtered_kSZ2:
        raise ValueError(f"'{args.experiment}' not in cfg.snr.experiments: {list(filtered_kSZ2.keys())}")
    filtered_map_this_exp = filtered_kSZ2[args.experiment]

    z0_grid = np.linspace(args.z_min, args.z_max, args.n_z)
    D_at_ell = {ell: [] for ell in target_ells}
    x_hi_vals = []
    z0_used = []

    for z0 in z0_grid:
        D_seeds = {ell: [] for ell in target_ells}
        x_hi_seeds = []
        for seed in seeds:
            if seed not in field_data or seed not in filtered_map_this_exp:
                continue
            try:
                res = compute_bias_weighted_cross_power(
                    cfg, kg, filtered_map_this_exp[seed], field_data[seed], z0, args.dz
                )
                x_hi_seeds.append(compute_volume_averaged_xHI(field_data[seed], z0, args.dz))
            except ValueError:
                continue  # window empty for this seed's z_lc range -- skip, not fatal
            valid = np.isfinite(res["D_ell"]) & (res["ell"] > 10)
            if valid.sum() < 3:
                continue
            interp = interp1d(res["ell"][valid], res["D_ell"][valid], bounds_error=False, fill_value=np.nan)
            for ell in target_ells:
                D_seeds[ell].append(interp(ell))

        if not x_hi_seeds:
            continue
        z0_used.append(z0)
        x_hi_vals.append(np.nanmedian(x_hi_seeds))
        for ell in target_ells:
            vals = [v for v in D_seeds[ell] if np.isfinite(v)]
            D_at_ell[ell].append(np.nanmedian(vals) if vals else np.nan)

    z0_used = np.array(z0_used)
    x_hi_vals = np.array(x_hi_vals)
    print(f"\n{args.experiment} filter, dz={args.dz}, {len(z0_used)} z0 points with usable data:")
    for ell in target_ells:
        arr = np.array(D_at_ell[ell])
        if np.any(np.isfinite(arr)):
            i_peak = np.nanargmax(arr)
            print(f"  ell={ell:.0f}: peak D_ell={arr[i_peak]:.4g} uK^2 at z0={z0_used[i_peak]:.2f} "
                  f"(x_HI~{x_hi_vals[i_peak]:.2f})")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for ell in target_ells:
        ax.plot(z0_used, D_at_ell[ell], marker="o", ms=4, label=f"$\\ell$={ell:.0f}")
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$]")
    ax.set_title(f"D_ell vs z0 -- {args.experiment} filter, dz={args.dz}\n"
                 f"({cfg.box.box_len_mpc:.0f} Mpc box; secondary axis is OUR sim's own x_HI(z), not the paper's model)")
    ax.legend(fontsize=9)

    # Secondary top x-axis in volume-averaged x_HI, matching the paper's
    # Fig 5 layout -- interpolation-based, only valid if x_HI(z0) is
    # monotonic over the swept range (true for a normal reionization
    # history; falls back silently to no secondary axis if not).
    order = np.argsort(z0_used)
    z_sorted, x_sorted = z0_used[order], x_hi_vals[order]
    if np.all(np.diff(x_sorted) <= 0) or np.all(np.diff(x_sorted) >= 0):
        z_to_x = interp1d(z_sorted, x_sorted, bounds_error=False, fill_value=(x_sorted[0], x_sorted[-1]))
        x_to_z = interp1d(x_sorted, z_sorted, bounds_error=False, fill_value=(z_sorted[0], z_sorted[-1])) \
            if x_sorted[0] < x_sorted[-1] else \
            interp1d(x_sorted[::-1], z_sorted[::-1], bounds_error=False, fill_value=(z_sorted[-1], z_sorted[0]))
        try:
            secax = ax.secondary_xaxis("top", functions=(z_to_x, x_to_z))
            secax.set_xlabel(r"$x_{\rm HI}$ (this simulation)")
        except Exception as e:
            print(f"  (secondary x_HI axis skipped: {e})")
    else:
        print("  (secondary x_HI axis skipped: x_HI(z0) not monotonic over this range)")

    outpath = os.path.join(args.out_dir, f"dell_vs_z_multiell_{args.experiment}.pdf")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {outpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
