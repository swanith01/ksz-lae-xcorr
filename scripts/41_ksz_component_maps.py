#!/usr/bin/env python3
"""
scripts/41_ksz_component_maps.py
==================================
Component decomposition of the kSZ^2 field, step 1 (per seed; heavy: it rebuilds the seed's wrap-cycle lightcone):
the filtered-and-squared maps built from the ionisation field alone (x), the velocity field alone (v), the density
alone (d) and the real kSZ integrand (full), each in the two orderings

    A = [F sum_i X_i]^2   (project, then square)      B = sum_i [F X_i]^2   (square every slice, then project)

see correlation/ksz_components.py for the exact definitions.  The geometry, normalisation conventions and wrap-cycle
seed are those of scripts/26 / 37 (same lightcone, same filter), so A_full equals filter_and_square(kSZ map) of the
stored products -- this is checked and printed.

Needs the seed's kSZ products from scripts/26 --save-maps (seed<S>_wo<K>_lpmaps.pkl) for >= 2 seeds, because the filter's
C_kSZreion is the seed median of the map auto-powers (as in scripts/28).

Output: <out-dir>/seed<S>_wo<K>_components.pkl  (8 maps of 300x300 float32, a few MB)

Usage:
    python scripts/41_ksz_component_maps.py --seed 1
    qsub -v SEED=1 pbs/pbs_ksz_components.sh
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import build_wrapcycle_field_data, normalisation_keys
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.utils.config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--in-dir", default=None, help="kSZ products (default <products_root>/ksz_auto_wrapcycle)")
    ap.add_argument("--out-dir", default="data/products_tracers/components")
    ap.add_argument("--experiment", default="SO")
    ap.add_argument("--cltt-file", default=None)
    ap.add_argument("--mode", default="grid-wrap")
    ap.add_argument("--ne-convention", default="helium")
    ap.add_argument("--tau0", default="analytic")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    in_dir = args.in_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    prods = kc.load_lpmaps_products(in_dir, args.wrap_offset)
    if len(prods) < 2 or args.seed not in prods:
        sys.exit(f"need >= 2 seeds incl. seed {args.seed} with _lpmaps.pkl in {in_dir}; found {sorted(prods)}")
    kg, filt, chi_ref = kc.build_filter_from_products(cfg, prods, kc.cl_tt_provider(cfg, args.cltt_file), args.experiment)
    f2d = kc.filter_grid(kg, filt["ell_grid"], filt["fl"], chi_ref)

    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    t0 = time.time()
    wrap_seed = args.seed + args.wrap_offset
    logger.info(f"scripts/41 seed={args.seed} wrap_seed={wrap_seed}  chi_ref={chi_ref:.0f} Mpc  ({len(prods)} seeds for the filter)")
    norm = normalisation_keys(cfg, ne_convention=args.ne_convention, tau0_mode=args.tau0)
    fd = build_wrapcycle_field_data(cfg, args.seed, wrap_seed, Stitcher(cfg), logger, mode=args.mode, norm=norm,
                                    field_convention="physical")
    logger.info(f"stitched in {time.time() - t0:.0f}s; building component maps")
    prod = prods[args.seed]
    maps = kc.component_maps(cfg, fd, f2d, kSZ_z_min=prod["kSZ_z_min"])
    del fd

    # ---- checks -------------------------------------------------------------------------------------------------
    k_new, k_old = maps.pop("_ksz_map"), prod["kSZ_map"]
    rel = float(np.abs(k_new - k_old).max() / np.abs(k_old).max())
    ref = lp_maps.filter_and_square(k_old, filt["ell_grid"], filt["fl"], kg, chi_ref)
    relA = float(np.abs(maps["A_full"] - ref).max() / ref.max())
    logger.info(f"check: kSZ map rebuilt vs stored: max|diff|/max = {rel:.2e};  A_full vs filter_and_square: {relA:.2e}")
    if rel > 1e-3 or relA > 1e-3:
        logger.warning("the rebuilt kSZ map does NOT match the stored one -- check the config / conventions")
    rho = {v: kc.map_correlation(maps[v], maps["A_full"]) for v in kc.VARIANTS}
    for v in kc.VARIANTS:
        logger.info(f"  {v:7s} mean={maps[v].mean():.3e}  rms/mean={maps[v].std() / maps[v].mean():.2f}  "
                    f"r(map, A_full)={rho[v]:+.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"seed{args.seed}_wo{args.wrap_offset}_components.pkl")
    with open(path, "wb") as f:
        pickle.dump({"maps": maps, "seed": args.seed, "chi_ref": chi_ref, "rho_with_A_full": rho,
                     "check": {"ksz_map_rel": rel, "A_full_rel": relA}, "kSZ_z_min": prod["kSZ_z_min"]}, f)
    logger.info(f"saved {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
