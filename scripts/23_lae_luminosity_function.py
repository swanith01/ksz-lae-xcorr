#!/usr/bin/env python3
"""
scripts/23_lae_luminosity_function.py
========================================
Our simulated LAE luminosity function, swept across many real redshifts
and colored by z (one smooth line per z, continuous colormap + colorbar),
overlaid with Kageura et al. 2025's real observational data points
(their own Table 2 measurements, with error bars, at their own five
redshift bins) -- matching the comparison style used in that paper's own
Figure 6 and in this group's prior plots.

Unlike everything else built for the kSZ comparison, this checks the LAE
catalogue's actual luminosity distribution directly, independent of any
kSZ, stitching, or cross-correlation machinery at all.

NOTE: Umeda et al. 2025's own measurement (a second real reference,
Subaru narrow-band survey) is not yet included -- see
io/lae_luminosity_function_reference.py's module docstring for why.

Usage:
    python scripts/23_lae_luminosity_function.py
    python scripts/23_lae_luminosity_function.py --seeds 1 2 3 --n-z-lines 8
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.lae_luminosity_function_reference import KAGEURA2025_LF_POINTS
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.tracers.luminosity_function import compute_luminosity_function
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-z-lines", type=int, default=12,
                         help="How many of our simulation's real snapshot redshifts to show as "
                              "lines, evenly spread across the range with real LAE data")
    parser.add_argument("--log-l-min", type=float, default=41.5)
    parser.add_argument("--log-l-max", type=float, default=44.0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    logger = setup_logger(seeds[0], cfg.paths.lightcone_root)
    all_snap_z = np.sort(stitcher.get_halo_redshifts(seeds[0]))
    print(f"Found {len(all_snap_z)} real snapshot redshifts: z={all_snap_z.min():.2f}-{all_snap_z.max():.2f}")

    idx = np.linspace(0, len(all_snap_z) - 1, args.n_z_lines).round().astype(int)
    z_lines = sorted(set(all_snap_z[idx]))

    all_z_for_color = list(z_lines) + list(KAGEURA2025_LF_POINTS.keys())
    z_min_color, z_max_color = min(all_z_for_color), max(all_z_for_color)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=z_min_color, vmax=z_max_color)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    print("Computing simulated LF at each line redshift...")
    for z in z_lines:
        result = compute_luminosity_function(cfg, seeds, z, args.log_l_min, args.log_l_max, args.n_bins)
        valid = np.isfinite(result["phi"]) & (result["phi"] > 0)
        if not np.any(valid):
            print(f"  z={z:.2f}: no data, skipping this line")
            continue
        log_phi = np.log10(result["phi"][valid])
        ax.plot(result["log_l_centers"][valid], log_phi, "-", color=cmap(norm(z)), lw=1.3, alpha=0.85)
        print(f"  z={z:.2f}: {result['n_seeds_with_data']}/{len(seeds)} seeds, "
              f"{result['n_objects_total']} total LAEs")

    print("Overlaying Kageura+2025 real data points...")
    for z, d in KAGEURA2025_LF_POINTS.items():
        log_phi = np.array(d["log_phi"])
        yerr = np.array([d["log_phi_err_lo"], d["log_phi_err_hi"]])
        ax.errorbar(d["log_l"], log_phi, yerr=yerr, fmt="o", color=cmap(norm(z)),
                     markeredgecolor="black", markeredgewidth=0.6, ms=7, capsize=3, zorder=5)

    ax.plot([], [], "o", color="gray", markeredgecolor="black", markeredgewidth=0.6,
             label="Kageura+2025 (real data, JWST/NIRSpec)")
    ax.plot([], [], "-", color="gray", label="This simulation (lines, colored by z)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$z$")

    ax.set_xlabel(r"$\log_{10}(L_{\rm Ly\alpha}$ [erg/s])")
    ax.set_ylabel(r"$\log_{10}\phi$ [Mpc$^{-3}$ dex$^{-1}$]")
    ax.set_title(f"LAE luminosity function ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    ax.legend(loc="lower left", fontsize=8)

    outpath = os.path.join(args.out_dir, "lae_luminosity_function.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"\nSaved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
