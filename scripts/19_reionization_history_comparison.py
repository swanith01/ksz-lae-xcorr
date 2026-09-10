#!/usr/bin/env python3
"""
scripts/19_reionization_history_comparison.py
=================================================
Our own simulation's reionization history (x_HII vs z, ionized fraction --
same x_e_mean already computed inside build_tau_history for every other
direct-method script) overplotted against the digitized La Plante+2022
scenarios (Fiducial/Early/Short -- data/reference/la_plante_2022,
io/la_plante_reference.py).

Cheap -- no 3D FFTs, just per-snapshot field means -- fine to run
interactively, all 10 seeds, no qsub needed.

Usage:
    python scripts/19_reionization_history_comparison.py
    python scripts/19_reionization_history_comparison.py --seeds 1 2 3
"""

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation.direct_bispectrum import build_tau_history
from ksz_lae_xcorr.io.la_plante_reference import load_reionization_histories
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Seeds to use (default: all seeds in cfg.box.seeds)")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    per_seed_xhii = []
    z_common = None

    for seed in seeds:
        logger = setup_logger(seed, cfg.paths.lightcone_root)
        print(f"Seed {seed}: computing x_HII(z) history...")
        all_snap_z = stitcher.get_snapshot_redshifts(seed, logger)
        z_sorted, chi_sorted, tau_cumulative, x_e_mean = build_tau_history(
            cfg, stitcher, seed, all_snap_z, logger)
        if z_common is None:
            z_common = z_sorted
        elif not np.allclose(z_sorted, z_common):
            print(f"  WARNING: seed {seed}'s snapshot z grid differs from the first "
                  f"seed's -- interpolating onto the common grid instead of stacking directly.")
            x_e_mean = np.interp(z_common, z_sorted, x_e_mean)
        per_seed_xhii.append(x_e_mean)
        print(f"  x_HII range: [{x_e_mean.min():.4f}, {x_e_mean.max():.4f}]")

    stacked = np.array(per_seed_xhii)
    x_hii_median = np.median(stacked, axis=0)
    x_hii_lo = np.percentile(stacked, 16, axis=0) if len(seeds) > 1 else x_hii_median
    x_hii_hi = np.percentile(stacked, 84, axis=0) if len(seeds) > 1 else x_hii_median

    csv_path = os.path.join(args.out_dir, "our_reionization_history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "x_HII_median", "x_HII_p16", "x_HII_p84", "n_seeds"])
        for z, med, lo, hi in zip(z_common, x_hii_median, x_hii_lo, x_hii_hi):
            w.writerow([f"{z:.4f}", f"{med:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(seeds)])
    print(f"\nSaved: {csv_path}")

    lp = load_reionization_histories()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.plot(z_common, x_hii_median, color="black", lw=2,
            label=f"This simulation (median, {len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    if len(seeds) > 1:
        ax.fill_between(z_common, x_hii_lo, x_hii_hi, color="black", alpha=0.15,
                         label="This simulation (16-84th percentile)")

    styles = {"Fiducial": "-", "Early": "--", "Short": ":"}
    colors = {"Fiducial": "tab:blue", "Early": "tab:orange", "Short": "tab:green"}
    for name, d in lp.items():
        ax.plot(d["z"], d["x_HII"], styles[name], color=colors[name],
                label=f"La Plante+2022 ({name}, digitized)")

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$x_{\rm HII}$")
    ax.set_title("Reionization history -- this simulation vs La Plante+2022 scenarios")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    outpath = os.path.join(args.out_dir, "reionization_history_comparison.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
