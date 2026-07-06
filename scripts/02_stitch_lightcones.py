#!/usr/bin/env python3
"""
scripts/02_stitch_lightcones.py
==================================
CLI wrapper around ksz_lae_xcorr.lightcone.stitch. Stitches per-redshift
coeval boxes + external LAE/LBG catalogues into full 3D lightcones.

Usage:
    python scripts/02_stitch_lightcones.py                # all configured seeds
    python scripts/02_stitch_lightcones.py --seed 1 3 5    # specific seeds
"""

import argparse
import sys

from ksz_lae_xcorr.lightcone.stitch import stitch_seeds
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seed if args.seed is not None else cfg.box.seeds
    stitch_seeds(cfg, seeds)


if __name__ == "__main__":
    sys.exit(main())
