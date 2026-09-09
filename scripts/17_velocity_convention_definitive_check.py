#!/usr/bin/env python3
"""
scripts/17_velocity_convention_definitive_check.py
=====================================================
Runs the definitive, physics-based velocity-convention check
(correlation/velocity_convention_check.py) on a real coeval snapshot:
measures P_v_raw(k) and P_delta(k) from the SAME real data, compares to
the linear-theory (continuity equation) prediction, and reports the
empirical correction factor needed at each k -- settling what
py21cmfast v4's raw velocity_z actually is, from physics and real data,
not from documentation (none was found) or an assumed formula (which
may be for the wrong py21cmfast version).

HOW TO READ THE OUTPUT:
- correction_factor_kms ~ 1.0 at every k -> raw velocity_z is ALREADY a
  genuine physical velocity in km/s. No conversion needed at all.
- correction_factor_mpc_s ~ 1.0 at every k -> raw velocity_z is ALREADY
  in Mpc/s directly (consistent with the earlier empirical hint from
  scripts/16: raw v/c ~ 3.9e-3 with zero conversion).
- Neither ~1, but CONSTANT across k -> there's a real missing scalar
  factor (could be D(z), or a(z), or something else) -- the constant's
  value will hint at what.
- Not constant across k (e.g. varies smoothly, like 1/k or k) -> something
  more structural is off (e.g. a derivative/gradient convention
  mismatch), worth a closer look before trusting any of this.

Usage:
    python scripts/17_velocity_convention_definitive_check.py --seed 1 --z 9.446145
"""

import argparse
import os
import sys

import numpy as np

from ksz_lae_xcorr.correlation.velocity_convention_check import check_velocity_convention
from ksz_lae_xcorr.lightcone.stitch import Stitcher
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z", type=float, required=True,
                         help="Exact snapshot z (6 decimal places, matching a real coeval_z{z:.6f} dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stitcher = Stitcher(cfg)

    print(f"Loading seed {args.seed}, z={args.z:.6f} ...")
    density = np.asarray(stitcher.load_field_box(args.seed, args.z, "density"))  # (1+delta), on HII_DIM
    path = os.path.join(cfg.paths.coeval_root, f"seed_{args.seed}",
                         f"coeval_z{args.z:.6f}", "velocity_z.npy")
    raw_velocity_z = np.load(path)
    print(f"  density shape={density.shape}, raw velocity_z shape={raw_velocity_z.shape}")
    if density.shape != raw_velocity_z.shape:
        print("  WARNING: shapes differ -- density.py's block-averaging inside "
              "load_field_box should already match velocity_z's HII_DIM grid. "
              "If this fires, something upstream changed.")

    result = check_velocity_convention(cfg, density, raw_velocity_z,
                                        cfg.box.box_len_mpc, args.z, n_kbins=15)

    print(f"\n{'k [1/Mpc]':>12s} {'P_v_raw':>14s} {'P_delta':>14s} {'ratio_obs':>12s} "
          f"{'ratio_theory':>13s} {'corr_kms':>12s} {'corr_mpc_s':>12s}")
    for i in range(len(result["k_centers"])):
        k = result["k_centers"][i]
        print(f"{k:12.4f} {result['P_v_raw'][i]:14.4e} {result['P_delta'][i]:14.4e} "
              f"{result['ratio_observed'][i]:12.4e} {result['ratio_theory_kms'][i]:13.4e} "
              f"{result['correction_factor_kms'][i]:12.4e} {result['correction_factor_mpc_s'][i]:12.4e}")

    valid = np.isfinite(result["correction_factor_kms"])
    if np.any(valid):
        cf_kms = result["correction_factor_kms"][valid]
        cf_mpcs = result["correction_factor_mpc_s"][valid]
        print(f"\nSummary across valid k bins:")
        print(f"  correction_factor_kms:    median={np.median(cf_kms):.4e}  "
              f"spread(std/median)={np.std(cf_kms)/abs(np.median(cf_kms)):.2%}")
        print(f"  correction_factor_mpc_s:  median={np.median(cf_mpcs):.4e}  "
              f"spread(std/median)={np.std(cf_mpcs)/abs(np.median(cf_mpcs)):.2%}")
        print("\nSee this script's module docstring for how to read these numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
