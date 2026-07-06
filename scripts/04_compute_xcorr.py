#!/usr/bin/env python3
"""
scripts/04_compute_xcorr.py
==============================
Loads stitched lightcones, builds projected 2D maps (kSZ, kSZ^2, xe^2, v^2,
v_proj, halo/LAE/LBG overdensity), computes cross-power spectra for all
signal x tracer combinations, and saves the results to data/products/.

Usage:
    python scripts/04_compute_xcorr.py
"""

import argparse
import pickle
import sys

from ksz_lae_xcorr.correlation.auto_power import compute_auto_spectra
from ksz_lae_xcorr.correlation.cross_correlation import compute_cross_spectra
from ksz_lae_xcorr.correlation.projected_maps import build_projected_maps
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds

    print("Loading stitched lightcone products...")
    field_data, tracer_data = load_lightcone_products(cfg, seeds)
    print(f"  {len(field_data)}/{len(seeds)} seeds have field data; "
          f"{len(tracer_data)}/{len(seeds)} have tracer data")

    print("Building projected 2D maps...")
    maps = build_projected_maps(cfg, field_data, tracer_data, seeds)

    print("Computing cross-power spectra (halo, LAE, LBG)...")
    cross_results = compute_cross_spectra(cfg, maps, tracer_data, seeds)

    print("Computing auto-power spectra (needed by the SNR kSZ-reion filter)...")
    auto_results = compute_auto_spectra(cfg, maps, seeds)

    import os
    os.makedirs(cfg.paths.products_root, exist_ok=True)

    out_path = f"{cfg.paths.products_root}/cross_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"maps": maps, "cross_results": cross_results}, f)
    print(f"Saved: {out_path}")

    auto_path = f"{cfg.paths.products_root}/auto_results.pkl"
    with open(auto_path, "wb") as f:
        pickle.dump(auto_results, f)
    print(f"Saved: {auto_path}")


if __name__ == "__main__":
    sys.exit(main())
