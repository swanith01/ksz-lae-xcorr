#!/usr/bin/env python3
"""
scripts/26_ksz_auto_power_wrapcycle.py
========================================
ONE seed's kSZ auto-power from a WRAP-CYCLE stitched lightcone (the
periodicity fix from ksz-pipeline; see lightcone/wrap_cycle.py).  Submit one
PBS job per seed (pbs/pbs_ksz_wrapcycle.sh, pbs/submit_ksz_wrapcycle.sh),
then combine with scripts/27_ksz_auto_power_aggregate.py.

Reads the raw coeval snapshots directly (no lc_*.npz needed) -- so it is
independent of the old stitched lightcone products and of io/loaders.py.
Velocity is used AS-IS: our py21cmfast-v4 velocity_z is already Mpc/s (do not
apply ksz-pipeline's v3 Zel'dovich D*f*H/(1+z) conversion).

Normalisation conventions (recorded in the output; see utils/optical_depth.py):
  --ne-convention helium  (default) helium-inclusive n_e = 0.82 * rho_b/m_p,
                          matching ksz-pipeline's validated ne0_cgs().
                          'legacy' = repo's historical tau_prefactor (n_e is
                          1.22x too high -> D_ell 1.49x too high).
  --tau0 analytic         (default) adds the optical depth below z_min
                          (~0.03) to e^{-tau}.  'none' = historical.

Output: <products_root>/ksz_auto_wrapcycle/seed{S}_wo{K}.pkl  (small)
        wrap_cycle_seed = S + K   (K = --wrap-cycle-seed-offset, default 0),
        so changing K repeats the run with fresh per-cycle rotation angles.

Usage:
    python scripts/26_ksz_auto_power_wrapcycle.py --seed 1
    python scripts/26_ksz_auto_power_wrapcycle.py --seed 1 --wrap-cycle-seed-offset 100
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np

from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import (
    build_wrapcycle_field_data,
    compute_wrapcycle_auto_power,
    normalisation_keys,
    summarise_at_ell,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.utils.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--wrap-cycle-seed-offset", type=int, default=0)
    ap.add_argument("--mode", choices=("grid-wrap", "wrap"), default="grid-wrap",
                    help="scipy periodic mode; 'wrap' reproduces ksz-pipeline bit-for-bit")
    ap.add_argument("--ne-convention", choices=("helium", "legacy"), default="helium")
    ap.add_argument("--tau0", choices=("analytic", "none"), default="analytic")
    ap.add_argument("--out-dir", default=None, help="default: <products_root>/ksz_auto_wrapcycle")
    args = ap.parse_args()

    cfg = load_config(args.config)
    wrap_seed = args.seed + args.wrap_cycle_seed_offset
    out_dir = args.out_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seed{args.seed}_wo{args.wrap_cycle_seed_offset}.pkl")

    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    t0 = time.time()
    logger.info(f"scripts/26 seed={args.seed} wrap_cycle_seed={wrap_seed} mode={args.mode} "
                f"ne={args.ne_convention} tau0={args.tau0}")

    norm = normalisation_keys(cfg, ne_convention=args.ne_convention, tau0_mode=args.tau0)
    logger.info(f"normalisation: ne_scale={norm['ne_scale']:.4f}  tau0={norm['tau0']:.5f}")

    stitcher = Stitcher(cfg)
    fd = build_wrapcycle_field_data(cfg, args.seed, wrap_seed, stitcher, logger,
                                    mode=args.mode, norm=norm)
    logger.info(f"stitched in {time.time() - t0:.0f}s; decomposing...")
    res = compute_wrapcycle_auto_power(cfg, fd)
    res["meta"] = {
        "seed": args.seed, "wrap_cycle_seed": wrap_seed,
        "wrap_cycle_seed_offset": args.wrap_cycle_seed_offset, "mode": args.mode,
        "ne_convention": args.ne_convention, "tau0_mode": args.tau0,
        "config": args.config, "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": time.time() - t0,
    }
    with open(out_path, "wb") as f:
        pickle.dump(res, f)

    for key in ("patchy", "full"):
        r = res[key]
        s = summarise_at_ell(r, 3000.0)
        logger.info(f"[{key}] z=[{r['z_lo']:.2f},{r['z_hi']:.2f}] chi_eff={r['chi_eff']:.0f} Mpc | "
                    f"ell~{s['ell']:.0f}: D_total={s['D_total']:.4g} D_diag={s['D_diag']:.4g} "
                    f"D_off={s['D_off']:.4g} uK^2  D_off/D_total={s['D_off_over_total']:.1%}")
    logger.info(f"saved {out_path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    sys.exit(main())
