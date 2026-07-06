#!/usr/bin/env python3
"""
scripts/01_run_coeval_seed.py
===============================
CLI wrapper around ksz_lae_xcorr.halos.coeval_pipeline. Runs py21cmfast
coeval boxes + two-pass halo catalogs for one seed.

Usage:
    python scripts/01_run_coeval_seed.py --seed 1
    python scripts/01_run_coeval_seed.py --seed 1 --config configs/variants/my_variant.yaml
"""

import argparse
import sys

from ksz_lae_xcorr.halos.coeval_pipeline import run_seed
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_seed(cfg, seed=args.seed, log=lambda msg: print(msg, flush=True))


if __name__ == "__main__":
    sys.exit(main())
