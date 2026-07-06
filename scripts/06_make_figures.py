#!/usr/bin/env python3
"""
scripts/06_make_figures.py
=============================
Generates all figures for the paper from saved products in data/products/
(written by scripts 02, 04, 05). Fully non-interactive -- no notebook
clicking required, per the reproducibility checklist.

Usage:
    python scripts/06_make_figures.py
"""

import argparse
import os
import pickle
import sys

from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.plotting.lightcone_panels import plot_four_tracer_panel, plot_xhi_halo_overlay
from ksz_lae_xcorr.plotting.spectra_plots import plot_dell_vs_ell, plot_dell_vs_z, plot_snr_vs_z
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading lightcone products for panel plots...")
    field_data, tracer_data = load_lightcone_products(cfg, seeds)
    for seed in seeds:
        if seed not in field_data or seed not in tracer_data:
            continue
        plot_xhi_halo_overlay(cfg, field_data[seed]["xHI_lc"][:, cfg.box.hii_dim // 2, :],
                               tracer_data[seed]["halo_count_lc"][:, cfg.box.hii_dim // 2, :],
                               field_data[seed]["z_lc"], args.out_dir, seed)
        plot_four_tracer_panel(cfg, field_data[seed]["xHI_lc"][:, cfg.box.hii_dim // 2, :],
                                tracer_data[seed]["halo_count_lc"][:, cfg.box.hii_dim // 2, :],
                                tracer_data[seed]["lae_count_lc"][:, cfg.box.hii_dim // 2, :],
                                tracer_data[seed]["lbg_count_lc"][:, cfg.box.hii_dim // 2, :],
                                field_data[seed]["z_lc"], args.out_dir, seed)

    xcorr_path = f"{cfg.paths.products_root}/cross_results.pkl"
    if os.path.exists(xcorr_path):
        print("Loading cross-correlation products...")
        with open(xcorr_path, "rb") as f:
            xcorr = pickle.load(f)
        cross_results = xcorr["cross_results"]
        for tracer in cfg.correlation.tracers:
            for seed in seeds:
                if seed not in cross_results[tracer]:
                    continue
                plot_dell_vs_ell(cross_results, tracer, "kSZ2", seed, args.out_dir)
            plot_dell_vs_z(cross_results, tracer, "kSZ2", ell_target=3000, seeds=seeds, out_dir=args.out_dir)
    else:
        print(f"  Skipping cross-power figures -- {xcorr_path} not found "
              "(run scripts/04_compute_xcorr.py first)")

    snr_path = f"{cfg.paths.products_root}/snr_results.pkl"
    if os.path.exists(snr_path):
        print("Loading SNR products...")
        with open(snr_path, "rb") as f:
            snr = pickle.load(f)
        plot_snr_vs_z(cfg, snr["SN_results"], snr["z_cents"], args.out_dir)
    else:
        print(f"  Skipping SNR figure -- {snr_path} not found "
              "(run scripts/05_compute_snr.py first)")

    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
