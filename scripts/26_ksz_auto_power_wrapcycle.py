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
    implied_velocity_kms,
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
    ap.add_argument("--field-convention", choices=("physical", "legacy"), default="physical",
                    help="physical (default): velocity_z/(1+z) and delta(z)=D(z)/D(0)*hires_density "
                         "(both missing before 2026-10-06, see utils/field_units.py); "
                         "legacy: raw fields as in the first run (~700x too high)")
    ap.add_argument("--out-dir", default=None, help="default: <products_root>/ksz_auto_wrapcycle")
    ap.add_argument("--save-maps", action="store_true",
                    help="ALSO save the La Plante map-estimator inputs (kSZ map + bias-weighted "
                         "galaxy window maps, a few MB) to seed{S}_wo{K}_lpmaps.pkl for scripts/28. "
                         "Default OFF: the standard run is unchanged.")
    ap.add_argument("--z0-grid", type=float, nargs="+", default=None,
                    help="window centres for --save-maps (default 6.5..13 step 0.5)")
    ap.add_argument("--dz-list", type=float, nargs="+", default=None,
                    help="window widths for --save-maps (default 1.0, as LP Fig 5)")
    ap.add_argument("--ksz-z-min", type=float, default=None,
                    help="lowest z in the kSZ MAP for --save-maps (default 6.0; an assumption: "
                         "LP's map is reionization-era only)")
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
                                    mode=args.mode, norm=norm, field_convention=args.field_convention)
    logger.info(f"stitched in {time.time() - t0:.0f}s; decomposing...")
    res = compute_wrapcycle_auto_power(cfg, fd)
    res["meta"] = {
        "seed": args.seed, "wrap_cycle_seed": wrap_seed,
        "wrap_cycle_seed_offset": args.wrap_cycle_seed_offset, "mode": args.mode,
        "ne_convention": args.ne_convention, "tau0_mode": args.tau0,
        "field_convention": args.field_convention, "config": args.config, "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": time.time() - t0,
    }
    with open(out_path, "wb") as f:
        pickle.dump(res, f)

    for key in ("patchy", "full"):
        r = res[key]
        s = summarise_at_ell(r, 3000.0)
        logger.info(f"[{key}] z=[{r['z_lo']:.2f},{r['z_hi']:.2f}] chi_eff={r['chi_eff']:.0f} Mpc | "
                    f"ell~{s['ell']:.0f}: D_total={s['D_total']:.4g} D_diag={s['D_diag']:.4g} "
                    f"D_off={s['D_off']:.4g} uK^2  D_off/D_total={s['D_off_over_total']:.1%}  [per-1Mpc-pixel baseline]")
        if "D_diag_grouped" in s:
            logger.info(f"[{key}] snapshot-GROUPED ({r['n_groups']} groups): D_diag_grouped={s['D_diag_grouped']:.4g} "
                        f"D_off_grouped={s['D_off_grouped']:.4g} uK^2  D_off/D_total={s['D_off_grouped_over_total']:.1%}  "
                        f"(the validated baseline; expect small after wrap-cycle)")
    for zt, xe, vk in implied_velocity_kms(cfg, res):
        logger.info(f"[implied v] z={zt:5.2f} x_e={xe:.3f} rms[(1+d)v]={vk:8.1f} km/s (expect ~100-300 at z=7-10)")
    logger.info(f"saved {out_path}  ({time.time() - t0:.0f}s total)")

    if args.save_maps:
        from ksz_lae_xcorr.correlation import lp_maps
        del res
        prod = lp_maps.build_lp_products(
            cfg, fd,
            z0_grid=args.z0_grid or lp_maps.DEFAULT_Z0_GRID,
            dz_list=args.dz_list or lp_maps.DEFAULT_DZ_LIST,
            kSZ_z_min=args.ksz_z_min if args.ksz_z_min is not None else lp_maps.DEFAULT_KSZ_Z_MIN)
        prod["meta"] = {"seed": args.seed, "wrap_cycle_seed": wrap_seed,
                        "wrap_cycle_seed_offset": args.wrap_cycle_seed_offset, "mode": args.mode,
                        "ne_convention": args.ne_convention, "tau0_mode": args.tau0,
                        "field_convention": args.field_convention,
                        "created": datetime.now().isoformat(timespec="seconds")}
        lp_path = os.path.join(out_dir, f"seed{args.seed}_wo{args.wrap_cycle_seed_offset}_lpmaps.pkl")
        with open(lp_path, "wb") as f:
            pickle.dump(prod, f)
        logger.info(f"[lp-maps] kSZ map z>={prod['kSZ_z_min']} chi_eff={prod['chi_eff_kSZ']:.0f} Mpc "
                    f"rms(dT/T)={prod['kSZ_map'].std():.3g}; {len(prod['windows'])} windows; "
                    f"saved {lp_path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    sys.exit(main())
