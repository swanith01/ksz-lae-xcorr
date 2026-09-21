#!/usr/bin/env python3
"""
scripts/23_lae_luminosity_function.py
========================================
Our simulated LAE luminosity function (from the raw lya_lum_obs catalogue
values, tracers/luminosity_function.py) against real observed LAE
luminosity functions (Konno et al. 2018, PASJ 70, S16 -- the SILVERRUSH
survey, at z=5.7 and z=6.6, both within our simulation's real LAE data
range). Unlike everything else built for the kSZ comparison, this checks
the LAE catalogue's actual luminosity distribution directly, independent
of any kSZ, stitching, or cross-correlation machinery at all.

Usage:
    python scripts/23_lae_luminosity_function.py
    python scripts/23_lae_luminosity_function.py --z 5.7 6.6 --seeds 1 2 3
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.lae_luminosity_function_reference import (
    KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED,
    schechter_phi_per_dex,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.tracers.luminosity_function import compute_luminosity_function
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--z", type=float, nargs="+", default=[5.7, 6.6],
                         help="Redshifts to compute at -- default matches the two Konno+2018 "
                              "reference redshifts exactly. If your snapshot grid doesn't land "
                              "exactly here, the nearest real snapshot z is used instead.")
    parser.add_argument("--log-l-min", type=float, default=41.0)
    parser.add_argument("--log-l-max", type=float, default=44.0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)

    fig, axes = plt.subplots(1, len(args.z), figsize=(7 * len(args.z), 6), constrained_layout=True)
    if len(args.z) == 1:
        axes = [axes]

    for ax, z_target in zip(axes, args.z):
        z_actual = z_target
        try:
            logger = setup_logger(seeds[0], cfg.paths.lightcone_root)
            all_snap_z = stitcher.get_halo_redshifts(seeds[0])
            z_actual = float(all_snap_z[np.argmin(np.abs(all_snap_z - z_target))])
        except Exception as e:
            print(f"  Could not find real snapshot redshifts, using requested z={z_target} as-is ({e})")

        print(f"z_target={z_target} -> using real snapshot z={z_actual:.4f}")
        result = compute_luminosity_function(cfg, seeds, z_actual, args.log_l_min, args.log_l_max, args.n_bins)
        print(f"  {result['n_seeds_with_data']}/{len(seeds)} seeds have data, "
              f"{result['n_objects_total']} total LAEs across those seeds")

        valid = np.isfinite(result["phi"]) & (result["phi"] > 0)
        if np.any(valid):
            ax.errorbar(result["log_l_centers"][valid], result["phi"][valid],
                         yerr=result["phi_err"][valid], fmt="o-", color="darkgreen",
                         capsize=3, label=f"This sim (z={z_actual:.2f}, {result['n_seeds_with_data']} seeds)")
        else:
            ax.text(0.5, 0.5, "No LAE luminosity data at this z\n(lya_lum_obs not yet delivered "
                    "for this snapshot, or genuinely zero)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="firebrick")

        if z_target in KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED:
            alpha, log_l_star, log_phi_star = KONNO2018_SCHECHTER_PARAMS_ALPHA_FIXED[z_target]
            log_l_smooth = np.linspace(args.log_l_min, args.log_l_max, 200)
            phi_ref = schechter_phi_per_dex(log_l_smooth, alpha, log_l_star, log_phi_star)
            ax.plot(log_l_smooth, phi_ref, "--", color="black",
                     label=f"Konno+2018, z={z_target} (alpha={alpha} fixed)")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(L_{\rm Ly\alpha}$ [erg/s])")
        ax.set_ylabel(r"$\Phi(L)$ [Mpc$^{-3}$ dex$^{-1}$]")
        ax.set_title(f"LAE luminosity function, z~{z_target}")
        ax.legend(fontsize=8)

    outpath = os.path.join(args.out_dir, "lae_luminosity_function.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"\nSaved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
