#!/usr/bin/env python3
"""
scripts/12_plot_coherence_summary.py
=======================================
One summary plot of the 10-seed periodicity result, for showing Girish --
reuses plotting.spectra_plots.plot_coherence_decomposition (no new
plotting logic) on the seed-aggregated pickle scripts/09 already saved.

Requires scripts/09_coherence_decomposition.py to have been run first.

Usage:
    python scripts/12_plot_coherence_summary.py
"""

import argparse
import os
import pickle
import sys

from ksz_lae_xcorr.plotting.spectra_plots import plot_coherence_decomposition
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agg_path = f"{cfg.paths.products_root}/coherence_decomposition_seed_agg.pkl"
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"{agg_path} not found -- run scripts/09 first.")

    with open(agg_path, "rb") as f:
        agg = pickle.load(f)

    ell = agg["D_diag"]["ell"]
    D_total = agg["D_total"]["median"]
    D_diag = agg["D_diag"]["median"]
    D_off = agg["D_off"]["median"]
    n_seeds = agg["D_diag"]["n_seeds"]

    os.makedirs(args.out_dir, exist_ok=True)
    plot_coherence_decomposition(cfg, ell, D_total, D_diag, D_off,
                                  seed=f"median_{n_seeds}seeds", out_dir=args.out_dir)
    outpath = os.path.join(args.out_dir, f"coherence_decomposition_seedmedian_{n_seeds}seeds.pdf")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    sys.exit(main())
