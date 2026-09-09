#!/usr/bin/env python3
"""
scripts/16_velocity_conversion_check.py
==========================================
Cheap, fast diagnostic (loads ONE raw velocity_z.npy file, no heavy
compute) comparing the CURRENT LIVE conversion in
lightcone.stitch.Stitcher.load_field_box (box/(1+z)*3.086e19 -- no
growth factor, growth rate, or Hubble parameter at all) against the NEW,
physically-complete velocity_z_to_mpc_per_s (D(z)*f(z)*H(z)/(1+z),
matching ksz-pipeline's own validated conversion).

Run this BEFORE touching the live pipeline -- it tells us whether the
suspected bug is real and how large it is, on actual data, cheaply.

Usage:
    python scripts/16_velocity_conversion_check.py --seed 1 --z 9.446100
"""

import argparse
import os
import sys

import numpy as np

from ksz_lae_xcorr.lightcone.stitch import velocity_z_to_mpc_per_s
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z", type=float, required=True,
                         help="Exact snapshot z (must match a real coeval_z{z:.6f} directory)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = os.path.join(cfg.paths.coeval_root, f"seed_{args.seed}",
                         f"coeval_z{args.z:.6f}", "velocity_z.npy")
    if not os.path.exists(path):
        print(f"Not found: {path}")
        print("Check --z matches a real snapshot exactly (6 decimal places).")
        return 1

    raw = np.load(path)
    print(f"Loaded {path}, shape={raw.shape}")
    print(f"Raw velocity_z: rms={np.std(raw):.6g}, mean={np.mean(raw):.6g}, "
          f"min={raw.min():.6g}, max={raw.max():.6g}")

    old_mpc_s = np.array(raw) / (1 + args.z) * 3.086e19
    old_kms = old_mpc_s / constants.MPC_PER_KM_S_TO_S

    new_mpc_s = velocity_z_to_mpc_per_s(cfg, np.array(raw), args.z)
    new_kms = new_mpc_s / constants.MPC_PER_KM_S_TO_S

    c_mpc_s = constants.c_mpc_per_s()

    print(f"\n{'':20s} {'rms [Mpc/s]':>16s} {'rms [km/s]':>14s} {'rms v/c':>12s}")
    print(f"{'OLD (live)':20s} {np.std(old_mpc_s):16.6e} {np.std(old_kms):14.4f} {np.std(old_mpc_s)/c_mpc_s:12.4e}")
    print(f"{'NEW (D*f*H)':20s} {np.std(new_mpc_s):16.6e} {np.std(new_kms):14.4f} {np.std(new_mpc_s)/c_mpc_s:12.4e}")

    ratio = np.std(old_mpc_s) / np.std(new_mpc_s) if np.std(new_mpc_s) != 0 else float("inf")
    print(f"\nOLD/NEW ratio: {ratio:.4g}x")
    print("\nFor reference: a physically realistic peculiar velocity is v/c ~ 1e-4 to 1e-3")
    print("(hundreds of km/s or less). Whichever conversion lands in that range is")
    print("the physically plausible one -- this alone is a strong diagnostic, before")
    print("even considering which formula 'should' be right on theoretical grounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
