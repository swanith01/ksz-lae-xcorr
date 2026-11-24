#!/usr/bin/env python3
# =============================================================================
# verify_outputs.py
# kSZ2 x LAE Project  —  22 Jun 2026
# =============================================================================
# Run this after all PBS jobs finish to verify:
#   - All expected redshift nodes are present for all seeds
#   - Every .npy file exists and has the correct shape
#   - No files are suspiciously small (truncated write)
#
# USAGE:
#   python verify_outputs.py
#   python verify_outputs.py --seeds 1 3    # check specific seeds only
# =============================================================================

import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument(
    "--root",
    default=os.path.expanduser("~/kSZ2_LAE_project_22Jun2026"),
    help="Project root directory",
)
args = parser.parse_args()

PROJECT_ROOT = args.root
SEEDS        = args.seeds

# Expected shapes
HII_DIM = 512
DIM     = 1024

EXPECTED_FIELDS = {
    "hires_density.npy"       : (DIM,     DIM,     DIM),
    "hires_vz.npy"            : (DIM,     DIM,     DIM),
    "neutral_fraction.npy"    : (HII_DIM, HII_DIM, HII_DIM),
    "kinetic_temperature.npy" : (HII_DIM, HII_DIM, HII_DIM),
    # halo files are particle lists — shapes vary, just check they exist and are non-empty
    "halo_coords.npy"         : None,
    "halo_masses.npy"         : None,
}

# Minimum file sizes (bytes) — a truncated write will be much smaller
MIN_BYTES = {
    "hires_density.npy"       : DIM**3 * 4 - 1024,        # ~4 GB
    "hires_vz.npy"            : DIM**3 * 4 - 1024,
    "neutral_fraction.npy"    : HII_DIM**3 * 4 - 1024,    # ~512 MB
    "kinetic_temperature.npy" : HII_DIM**3 * 4 - 1024,
    "halo_coords.npy"         : 100,                       # at least something
    "halo_masses.npy"         : 100,
}

print(f"\n{'='*70}")
print(f"  kSZ2 x LAE  —  Output Verification")
print(f"  Root : {PROJECT_ROOT}")
print(f"  Seeds: {SEEDS}")
print(f"{'='*70}\n")

all_ok   = True
n_total  = 0
n_ok     = 0
n_errors = 0

for seed in SEEDS:
    seed_dir = os.path.join(PROJECT_ROOT, f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        print(f"  ✗  seed {seed}: directory missing — {seed_dir}")
        all_ok = False
        n_errors += 1
        continue

    coeval_dirs = sorted(glob.glob(os.path.join(seed_dir, "coeval_z*")))
    if len(coeval_dirs) == 0:
        print(f"  ✗  seed {seed}: no coeval_z* directories found")
        all_ok = False
        n_errors += 1
        continue

    seed_errors = []
    for cdir in coeval_dirs:
        z_str = os.path.basename(cdir).replace("coeval_z", "")
        for fname, expected_shape in EXPECTED_FIELDS.items():
            n_total += 1
            fpath = os.path.join(cdir, fname)

            # --- existence ---
            if not os.path.isfile(fpath):
                seed_errors.append(f"    MISSING  z={z_str}  {fname}")
                n_errors += 1
                continue

            # --- size sanity ---
            fsize = os.path.getsize(fpath)
            if fsize < MIN_BYTES[fname]:
                seed_errors.append(
                    f"    TOO SMALL  z={z_str}  {fname}  "
                    f"({fsize/1e9:.2f} GB < {MIN_BYTES[fname]/1e9:.2f} GB expected)"
                )
                n_errors += 1
                continue

            # --- shape check (only for fixed-shape fields) ---
            if expected_shape is not None:
                try:
                    arr = np.load(fpath, mmap_mode="r")
                    if arr.shape != expected_shape:
                        seed_errors.append(
                            f"    WRONG SHAPE  z={z_str}  {fname}  "
                            f"got {arr.shape}, expected {expected_shape}"
                        )
                        n_errors += 1
                        continue
                except Exception as e:
                    seed_errors.append(f"    LOAD ERROR  z={z_str}  {fname}  {e}")
                    n_errors += 1
                    continue

            n_ok += 1

    if seed_errors:
        print(f"  ✗  seed {seed}: {len(coeval_dirs)} nodes,  "
              f"{len(seed_errors)} errors:")
        for e in seed_errors:
            print(e)
        all_ok = False
    else:
        # Print halo count at a sample redshift as a sanity check
        sample_dir  = coeval_dirs[len(coeval_dirs)//2]   # middle redshift
        sample_z    = os.path.basename(sample_dir).replace("coeval_z", "")
        halo_masses = np.load(os.path.join(sample_dir, "halo_masses.npy"), mmap_mode="r")
        print(f"  ✓  seed {seed}: {len(coeval_dirs)} nodes OK  |  "
              f"sample z={sample_z}  n_halos={len(halo_masses):,}")

print(f"\n{'='*70}")
print(f"  Files checked : {n_total}")
print(f"  OK            : {n_ok}")
print(f"  Errors        : {n_errors}")
if all_ok:
    print(f"  STATUS        : ✓ ALL GOOD — ready to hand to Jahaan")
else:
    print(f"  STATUS        : ✗ ERRORS FOUND — check above before proceeding")
print(f"{'='*70}\n")
