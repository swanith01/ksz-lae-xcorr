#!/usr/bin/env python3
"""
scripts/10_stage1_lbg_benchmark.py
=====================================
Girish's Stage 1 (2026-09-07 Slack): "reproduce the published kSZ^2-galaxy
result under matched assumptions ... define a clear pass/fail comparison
with La Plante et al." before touching real LAE selection in Stage 2.

La Plante, Sipple & Lidz 2022 (ApJ 928, 162; arXiv:2111.13717) use Roman
HLS Lyman-break galaxies, NOT LAEs, cross-correlated with filtered-and-
squared kSZ maps for SO/CMB-S4/CMB-HD. This script runs the SAME
estimator already built for the LAE forecast (snr/snr_forecast.py,
snr/cmb_filter.py -- filter f(ell)=F(ell)b(ell) Eq.8/11, S/N Eq.13) but
pointed at this repo's LBG tracer instead, as the closest available match
to that paper's target population.

WHAT THIS IS NOT YET: a verified match to La Plante+2022's own numbers.
Their bias assumption (b_g ~ 9 at z=6 for Roman HLS) and exact redshift
binning are not yet cross-checked against what our LBG catalogue (cut at
configs/fiducial.yaml tracers.lbg_muv_cut) actually produces -- that
comparison is the next step once this runs against real data. This script
gets the plumbing in place and produces a first real number to look at.

Requires scripts/04_compute_xcorr.py to have been run first.

Usage:
    python scripts/10_stage1_lbg_benchmark.py
"""

import argparse
import os
import pickle
import sys

from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.snr.snr_forecast import run_snr_pipeline
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--xcorr-products", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    xcorr_path = args.xcorr_products or f"{cfg.paths.products_root}/cross_results.pkl"

    print(f"Loading cross-correlation products from {xcorr_path} ...")
    with open(xcorr_path, "rb") as f:
        products = pickle.load(f)
    maps = products["maps"]

    print("Loading tracer data for LBG auto-power...")
    _, tracer_data = load_lightcone_products(cfg, seeds)

    n_with_lbg = sum(1 for s in seeds if s in tracer_data and "lbg_count_lc" in tracer_data[s])
    print(f"  {n_with_lbg}/{len(seeds)} seeds have LBG lightcones")
    if n_with_lbg == 0:
        print("  No LBG data available -- nothing to benchmark. Exiting.")
        return 1

    auto_path = f"{cfg.paths.products_root}/auto_results.pkl"
    if not os.path.exists(auto_path):
        raise FileNotFoundError(f"{auto_path} not found -- run scripts/04_compute_xcorr.py first.")
    with open(auto_path, "rb") as f:
        auto_results_ksz = pickle.load(f)["kSZ"]

    kg = KGrid(cfg)
    result = run_snr_pipeline(cfg, kg, maps["kSZ"], tracer_data, seeds, auto_results_ksz,
                               tracer_key="lbg_count_lc")

    os.makedirs(cfg.paths.products_root, exist_ok=True)
    out_path = f"{cfg.paths.products_root}/stage1_lbg_snr_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved: {out_path}")

    print("\nStage 1 (LBG, matched-tracer) S/N -- NOT YET a verified match to")
    print("La Plante+2022's own published numbers, see module docstring:")
    for name in cfg.snr.experiments:
        sn = result["SN_results"][name]
        if sn:
            total = sum(v**2 for v in sn.values()) ** 0.5
            print(f"  {name:8s}: total S/N = {total:.3f}  ({len(sn)} z-bins)")
        else:
            print(f"  {name:8s}: no z-bins with usable signal")

    return 0


if __name__ == "__main__":
    sys.exit(main())
