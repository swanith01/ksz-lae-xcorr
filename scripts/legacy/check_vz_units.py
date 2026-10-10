#!/usr/bin/env python3
"""
Diagnostic: is the stitched lightcone velocity product (lc_vz.npz, read by
io/loaders.py) actually in km/s as assumed, or is it already in Mpc/s (same
bug class as the old lightcone/stitch.py velocity_z issue)?

No heavy compute -- just loads a couple of small arrays and prints stats.
Run via qsub -I like everything else, from inside the repo:

    python diag_vz_units.py --seed 1

Compares three things for the SAME seed:
  1. lc_vz.npz's raw 'lc' array (what loaders.py calls vz_kms, pre-conversion)
  2. that same array after loaders.py's km/s -> Mpc/s conversion (vz_mpc_s)
  3. the raw per-snapshot velocity_z.npy coeval box (read via Stitcher.
     load_field_box, same quantity confirmed to be already-Mpc/s by last
     session's continuity-equation check), at the box's own z_min as the
     nearest comparison point

If (1)'s magnitude looks like physical peculiar velocities in km/s
(order ~1 to a few 1000), the loader's conversion is doing the right thing
and the amplitude bug is elsewhere. If (1) is already ~1e-14 to 1e-18
(i.e. Mpc/s-scale, same ballpark as (3)), then loaders.py is silently
re-applying a km/s->Mpc/s conversion to a quantity that's already in
Mpc/s -- exactly the old stitch.py bug, just in a second code path that
fix never reached.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_dir = os.path.join(cfg.paths.lightcone_root, f"seed_{args.seed}")
    vz_path = os.path.join(seed_dir, "lc_vz.npz")

    print(f"Loading raw stitched product: {vz_path}")
    vz_data = np.load(vz_path)
    vz_raw = vz_data["lc"].astype(np.float64)
    z_arr = vz_data["z_arr"]

    c_kms = constants.C_KMS
    c_mpc_s = constants.c_mpc_per_s()
    vz_mpc_s = vz_raw / c_kms * c_mpc_s

    print("\n--- (1) raw lc_vz.npz['lc'] array, BEFORE any conversion ---")
    print(f"  shape: {vz_raw.shape}")
    print(f"  min/max/mean/std: {vz_raw.min():.6e} / {vz_raw.max():.6e} / "
          f"{vz_raw.mean():.6e} / {vz_raw.std():.6e}")

    print("\n--- (2) after loaders.py's assumed km/s -> Mpc/s conversion ---")
    print(f"  min/max/mean/std: {vz_mpc_s.min():.6e} / {vz_mpc_s.max():.6e} / "
          f"{vz_mpc_s.mean():.6e} / {vz_mpc_s.std():.6e}")

    print("\n--- (3) raw per-snapshot velocity_z.npy (confirmed Mpc/s), "
          "via Stitcher.load_field_box, for comparison ---")
    logger = setup_logger("diag_vz_units")
    stitcher = Stitcher(cfg)
    z_compare = float(z_arr[len(z_arr) // 2])  # a z actually in this lightcone
    try:
        vz_coeval = stitcher.load_field_box(args.seed, z_compare, "vz")
        vz_coeval = np.asarray(vz_coeval, dtype=np.float64)
        print(f"  (at z={z_compare:.3f}) min/max/mean/std: "
              f"{vz_coeval.min():.6e} / {vz_coeval.max():.6e} / "
              f"{vz_coeval.mean():.6e} / {vz_coeval.std():.6e}")
    except Exception as e:
        print(f"  (could not load a directly comparable coeval snapshot: {e})")
        vz_coeval = None

    print("\n--- verdict ---")
    scale_raw = np.abs(vz_raw).std()
    scale_mpc_s_converted = np.abs(vz_mpc_s).std()
    print(f"  raw lc_vz std: {scale_raw:.3e}")
    print(f"  converted (assumed-km/s->Mpc/s) std: {scale_mpc_s_converted:.3e}")
    if vz_coeval is not None:
        scale_coeval = np.abs(vz_coeval).std()
        print(f"  raw coeval velocity_z.npy std (known Mpc/s): {scale_coeval:.3e}")
        ratio_raw_vs_coeval = scale_raw / scale_coeval if scale_coeval else float("nan")
        ratio_converted_vs_coeval = scale_mpc_s_converted / scale_coeval if scale_coeval else float("nan")
        print(f"  raw lc_vz / coeval velocity_z ratio: {ratio_raw_vs_coeval:.3e}")
        print(f"  converted lc_vz / coeval velocity_z ratio: {ratio_converted_vs_coeval:.3e}")
        print()
        if 0.1 < ratio_raw_vs_coeval < 10:
            print("  ==> raw lc_vz is ALREADY the same order of magnitude as the known-"
                  "Mpc/s coeval field. loaders.py's km/s->Mpc/s conversion is almost "
                  "certainly being WRONGLY applied to an already-Mpc/s quantity -- "
                  "this is the bug (same class as the old stitch.py issue, different file).")
        elif 0.1 < ratio_converted_vs_coeval < 10:
            print("  ==> only AFTER conversion does lc_vz land in the same ballpark as "
                  "the known-Mpc/s coeval field. loaders.py's conversion looks correct; "
                  "the amplitude bug must be elsewhere (patchy-window masking, chi_eff, "
                  "or decompose_p_total_diag_off's normalization).")
        else:
            print("  ==> neither raw nor converted lc_vz lands near the coeval field's "
                  "scale -- something else is going on, needs a closer look (possibly "
                  "box-size/pixel-count mismatch between the stitched lightcone grid "
                  "and the coeval box, or z_compare isn't actually comparable).")


if __name__ == "__main__":
    main()
