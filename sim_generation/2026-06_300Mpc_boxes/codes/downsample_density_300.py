"""
downsample_density_300.py
==========================
Block-averages hires_density.npy (600,600,600) -> density_300.npy (300,300,300)
for every redshift snapshot in every seed. Purely additive: writes a new file,
never touches or deletes hires_density.npy.

Usage:
    python downsample_density_300.py              # all seeds
    python downsample_density_300.py --seed 3     # single seed
"""
import numpy as np
import os
import sys
import glob
import argparse
import logging
from datetime import datetime

ROOT = "/user1/swanith/kSZ2_LAE_project_22Jun2026"
ALL_SEEDS = list(range(1, 11))
DIM_HI  = 600
DIM_LO  = 300
FACTOR  = DIM_HI // DIM_LO   # = 2


def setup_logger(seed):
    log_dir = os.path.join(ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir,
                f'downsample_seed{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger(f'downsample_seed{seed}')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def block_average(hi):
    """(600,600,600) -> (300,300,300) via exact 2x2x2 mean-pooling."""
    return hi.reshape(DIM_LO, FACTOR, DIM_LO, FACTOR, DIM_LO, FACTOR).mean(axis=(1, 3, 5))


def process_seed(seed):
    logger = setup_logger(seed)
    seed_dir = os.path.join(ROOT, f'seed_{seed}')
    if not os.path.isdir(seed_dir):
        logger.error(f"seed dir not found: {seed_dir}")
        return
    z_dirs = sorted(glob.glob(os.path.join(seed_dir, 'coeval_z*')))
    logger.info(f"seed {seed}: {len(z_dirs)} redshift snapshots found")
    n_done, n_skip, n_fail = 0, 0, 0
    for zdir in z_dirs:
        hi_path = os.path.join(zdir, 'hires_density.npy')
        lo_path = os.path.join(zdir, 'density_300.npy')
        if not os.path.exists(hi_path):
            logger.warning(f"  missing hires_density.npy in {zdir}, skipping")
            n_fail += 1
            continue
        if os.path.exists(lo_path):
            n_skip += 1
            continue
        try:
            hi = np.load(hi_path).astype(np.float32)
            if hi.shape != (DIM_HI, DIM_HI, DIM_HI):
                logger.error(f"  unexpected shape {hi.shape} in {hi_path}, skipping")
                n_fail += 1
                continue
            lo = block_average(hi).astype(np.float32)
            np.save(lo_path, lo)
            n_done += 1
            del hi, lo
        except Exception as e:
            logger.error(f"  FAILED on {zdir}: {e}")
            n_fail += 1
    logger.info(f"seed {seed} complete: {n_done} written, {n_skip} already done, {n_fail} failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='+', default=ALL_SEEDS)
    args = parser.parse_args()
    print(f"Downsampling {DIM_HI}^3 -> {DIM_LO}^3 (factor {FACTOR}), seeds: {args.seed}")
    for seed in args.seed:
        process_seed(seed)
    print("\nAll done.")
