#!/usr/bin/env python3
"""
scripts/03_make_type_b_grids.py
==================================
CLI wrapper around ksz_lae_xcorr.tracers.type_b_grids. Builds the
physical-value (halo mass, Lya luminosity, MUV) lightcone grids used only
by the proportional-marker diagnostic plots -- not part of the SNR pipeline.

Usage:
    python scripts/03_make_type_b_grids.py
"""

import argparse
import sys

from ksz_lae_xcorr.tracers.type_b_grids import build_type_b_grids
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seed if args.seed is not None else cfg.box.seeds
    build_type_b_grids(cfg, seeds)


if __name__ == "__main__":
    sys.exit(main())
