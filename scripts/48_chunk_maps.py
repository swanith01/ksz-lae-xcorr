#!/usr/bin/env python3
"""
scripts/48_chunk_maps.py
==========================
Maps to LOOK at (step 1, one seed, swarm; heavy because it rebuilds the seed's wrap-cycle lightcone like scripts/26 / 41):
for a few redshift windows (dz = 1) save

  * the kSZ^2 map built from that window's line-of-sight pixels ALONE, for the full kSZ integrand, the x_e-only and the v-only
    ingredient, both project-then-square (A) and square-then-project (B)        (correlation/ksz_components.window_chunk_maps)
  * the tracer map of the same window for matter x Waters+16 and for the tracer cuts you ask for (scripts/37 products)
  * the whole-lightcone filtered kSZ^2 map that the estimator really cross-correlates (A_full of the stored kSZ map)

scripts/49 then puts them side by side.  Output: <out-dir>/seed<S>_wo<K>_chunks.pkl (a few tens of MB).

Usage:
    python scripts/48_chunk_maps.py --seed 1
    qsub -v SEED=1 pbs/pbs_chunk_maps.sh
Needs the kSZ products of >= 2 seeds (scripts/26 --save-maps) for the filter, as scripts/41.
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


def pick_windows(prod: dict, z0_list, dz: float = 1.0) -> list:
    """Indices of the stored dz=`dz` windows whose z0 is nearest to each requested z0 (unique, ascending z0)."""
    cand = [i for i, w in enumerate(prod["windows"]) if abs(w["dz"] - dz) < 1e-9]
    if not cand:
        raise ValueError(f"no dz={dz} windows in the products")
    got = {min(cand, key=lambda i: abs(prod["windows"][i]["z0"] - z)) for z in z0_list}
    return sorted(got, key=lambda i: prod["windows"][i]["z0"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--in-dir", default=None, help="kSZ products (default <products_root>/ksz_auto_wrapcycle)")
    ap.add_argument("--maps-root", default="data/products_tracers/maps", help="scripts/37 tracer maps")
    ap.add_argument("--tracers", default="halo_M10.0,lae_L42.5,lbg_MUV-19.0", help="comma-separated tracer names (besides matter)")
    ap.add_argument("--z0", default="12;11;10;9;8;7", help="window centres to save, ';'-separated (nearest stored dz=1 windows)")
    ap.add_argument("--out-dir", default="data/products_tracers/chunks")
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
    prod = prods[args.seed]
    wi = pick_windows(prod, [float(z) for z in args.z0.split(";")])
    wins = [prod["windows"][i] for i in wi]

    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    log = logger.info
    t0 = time.time()
    log(f"scripts/48 seed={args.seed}: windows z0 = {[w['z0'] for w in wins]}")
    norm = normalisation_keys(cfg, ne_convention=args.ne_convention, tau0_mode=args.tau0)
    fd = build_wrapcycle_field_data(cfg, args.seed, args.seed + args.wrap_offset, Stitcher(cfg), logger, mode=args.mode, norm=norm,
                                    field_convention="physical")
    log(f"stitched in {time.time() - t0:.0f}s; building the chunk maps")
    chunks = kc.window_chunk_maps(cfg, fd, f2d, wins, kSZ_z_min=prod["kSZ_z_min"])
    del fd

    gal = {"matter_Waters": [prod["windows"][i]["gal_map"].astype(np.float32) for i in wi]}
    for name in [t for t in args.tracers.split(",") if t]:
        p = os.path.join(args.maps_root, name, f"seed{args.seed}_wo{args.wrap_offset}_lpmaps.pkl")
        if not os.path.exists(p):
            log(f"  (no tracer maps for {name}: {p})")
            continue
        with open(p, "rb") as f:
            tp = pickle.load(f)
        gal[name] = [tp["windows"][i]["gal_map"].astype(np.float32) for i in wi]
    S_global = lp_maps.filter_and_square(prod["kSZ_map"], filt["ell_grid"], filt["fl"], kg, chi_ref).astype(np.float32)

    out = {"seed": args.seed, "chi_ref": chi_ref, "S_global": S_global,
           "windows": [{k: w[k] for k in ("z0", "dz", "z_lo", "z_hi", "chi_eff", "x_hii")} | {"maps": c, "n_pix": c["n_pix"]}
                       for w, c in zip(wins, chunks)],
           "gal": gal}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"seed{args.seed}_wo{args.wrap_offset}_chunks.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log(f"saved {path}  ({time.time() - t0:.0f}s); tracers {sorted(gal)}")


if __name__ == "__main__":
    sys.exit(main())
