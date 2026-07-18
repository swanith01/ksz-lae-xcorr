#!/usr/bin/env python3
"""
scripts/07_stitch_lae_value_fields.py
========================================
Stitches full 3D LAE luminosity + REW value fields (lc_lae_lum.npz,
lc_lae_rew.npz), needed by the realistic survey-selection S/N extension
(scripts/08_compute_realistic_snr.py). Not needed for the core pipeline
(scripts 01-06) -- only run this if you want the SILVERRUSH/Roman-Grism
realistic forecasts, not just the "optimistic" S/N from script 05.

Requires Jahaan's pipeline to provide lya_rew_obs alongside lya_lum_obs
(see data/README.md) -- a NEW external dependency beyond the core pipeline.

Usage:
    python scripts/07_stitch_lae_value_fields.py --seed 1
"""

import argparse
import sys

from ksz_lae_xcorr.lightcone.value_fields import stitch_value_fields
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seed if args.seed is not None else cfg.box.seeds
    stitch_value_fields(cfg, seeds)


if __name__ == "__main__":
    sys.exit(main())
