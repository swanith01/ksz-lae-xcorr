#!/usr/bin/env python3
"""
scripts/05_compute_snr.py
============================
LAE-only SNR forecast: builds the CMB filter, filters the kSZ maps per
experiment, computes filtered-kSZ^2 x LAE cross-power, and forecasts S/N
vs redshift via the discrete-ell-sum estimator (La Plante+2022 Eq. 13).

Requires scripts/04_compute_xcorr.py to have been run first (reads
data/products/cross_results.pkl for the kSZ maps and auto-power).

Note: this deliberately only ever touches the LAE tracer -- see
configs/fiducial.yaml `snr.tracer` and src/ksz_lae_xcorr/snr/*.py docstrings.

Usage:
    python scripts/05_compute_snr.py
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
    parser.add_argument("--xcorr-products", type=str, default=None,
                         help="Path to cross_results.pkl from script 04")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    xcorr_path = args.xcorr_products or f"{cfg.paths.products_root}/cross_results.pkl"

    print(f"Loading cross-correlation products from {xcorr_path} ...")
    with open(xcorr_path, "rb") as f:
        products = pickle.load(f)
    maps = products["maps"]

    print("Loading tracer data for LAE auto-power...")
    _, tracer_data = load_lightcone_products(cfg, seeds)

    # auto_results_ksz[seed] = (D_ell, D_err) -- linear kSZ auto-power,
    # produced by scripts/04_compute_xcorr.py via correlation.auto_power.
    auto_path = f"{cfg.paths.products_root}/auto_results.pkl"
    if not os.path.exists(auto_path):
        raise FileNotFoundError(f"{auto_path} not found -- run scripts/04_compute_xcorr.py first.")
    with open(auto_path, "rb") as f:
        auto_results_ksz = pickle.load(f)["kSZ"]

    kg = KGrid(cfg)
    result = run_snr_pipeline(cfg, kg, maps["kSZ"], tracer_data, seeds, auto_results_ksz)

    os.makedirs(cfg.paths.products_root, exist_ok=True)
    out_path = f"{cfg.paths.products_root}/snr_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved: {out_path}")

    for name in cfg.snr.experiments:
        sn = result["SN_results"][name]
        if sn:
            total = sum(v**2 for v in sn.values()) ** 0.5
            print(f"  {name:8s}: total S/N = {total:.3f}")


if __name__ == "__main__":
    sys.exit(main())
