#!/usr/bin/env python3
"""
scripts/44_coeval_cross.py
============================
D_ell^cross from COEVAL snapshot boxes (correlation/coeval_cross.py) next to the stitched-lightcone estimate of the SAME seed,
window by window, for the matter x Waters+16-bias reference and for chosen tracer cuts.  One seed per job (pbs/pbs_coeval_cross.sh).

Per window (z0, dz = 1; the windows, chi_eff and filter are those of scripts/26 + 28):
  coeval    : every snapshot overlapping the window contributes the cross-power of  [F Theta_slab]^2  with the tracer
              map of the SAME slab of its own box; summed over snapshots (no stitching, no interpolation, no rotations)
  lightcone : [F Theta_lightcone]^2 x the window's tracer map from scripts/37  (what scripts/28 computes)
Both use the same filter, the same weights W_i and the same window top hat, so any difference is the stitching / cross-snapshot
pairs.  Tracers: `matter` (b_g(z) delta_m, as scripts/26) and the suite specs (scripts/37; tracer overdensity n/nbar - 1).

Output: <out-dir>/seed<S>_wo<K>_coeval.pkl  with, for every tracer name, D_coeval[window, ell target] and D_lightcone[window, ell target].

Usage:
    python scripts/44_coeval_cross.py --seed 1
    python scripts/44_coeval_cross.py --seed 1 --specs "halo:10;lae:42.5;lbg:-19"      # the default
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

from ksz_lae_xcorr.correlation import coeval_cross as cc
from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import normalisation_keys
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.lightcone.wrap_cycle import build_los_z_grid
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.tracers import suite
from ksz_lae_xcorr.utils import constants
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology

MATTER = "matter_Waters"


def snapshot_boxes(stitcher, seed, z, specs, need_tracers):
    """theta cube (-(1+delta) x_e v/c), matter overdensity cube, tracer overdensity cubes {name: cube} for one snapshot."""
    xH = np.asarray(stitcher.load_field_box(seed, z, "xH", convention="physical"), np.float32)
    dens = np.maximum(1.0 + np.asarray(stitcher.load_field_box(seed, z, "density", convention="physical"), np.float32), 0.0)
    vz = np.asarray(stitcher.load_field_box(seed, z, "vz", convention="physical"), np.float32)
    theta = (-(dens * (1.0 - xH) * vz / np.float32(constants.c_mpc_per_s()))).astype(np.float32)
    cubes = {MATTER: dens - 1.0}
    if specs and need_tracers:
        cat = suite.load_catalogues(stitcher, seed, z)
        tr = suite.snapshot_tracers(cat, specs, stitcher.ngrid, stitcher.cell, max_mass_sample=1)
        for sp in specs:
            g = tr[sp.name]["grid"].astype(np.float32)
            m = float(g.mean())
            cubes[sp.name] = (g / m - 1.0).astype(np.float32) if m > 0 else np.zeros_like(g)
    return theta, cubes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--specs", default="halo:10;lae:42.5;lbg:-19", help="tracer cuts (suite spec string) besides matter; 'none' for matter only")
    ap.add_argument("--in-dir", default=None, help="scripts/26 products (default <products_root>/ksz_auto_wrapcycle)")
    ap.add_argument("--maps-root", default="data/products_tracers/maps", help="scripts/37 tracer window maps (lightcone side)")
    ap.add_argument("--out-dir", default="data/products_tracers/coeval")
    ap.add_argument("--experiment", default="SO")
    ap.add_argument("--cltt-file", default=None)
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--zmax-snap-pad", type=float, default=0.3)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    in_dir = args.in_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    prods = kc.load_lpmaps_products(in_dir, args.wrap_offset)
    if len(prods) < 2 or args.seed not in prods:
        sys.exit(f"need >= 2 seeds incl. seed {args.seed} with _lpmaps.pkl in {in_dir}; found {sorted(prods)}")
    kg, filt, chi_ref = kc.build_filter_from_products(cfg, prods, kc.cl_tt_provider(cfg, args.cltt_file), args.experiment)
    f2d = kc.filter_grid(kg, filt["ell_grid"], filt["fl"], chi_ref)
    prod = prods[args.seed]
    wins = [w for w in prod["windows"] if abs(w["dz"] - args.dz) < 1e-9]
    specs = [] if args.specs == "none" else suite.parse_specs(args.specs)
    names = [MATTER] + [sp.name for sp in specs]

    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    log = logger.info
    st = Stitcher(cfg, field_convention="physical")
    cosmo = get_cosmology(cfg)
    ngrid, cell = int(cfg.box.hii_dim), float(cfg.box.box_len_mpc) / int(cfg.box.hii_dim)
    z_arr, _ = build_los_z_grid(cosmo, cfg.box.z_min, cfg.box.z_max, cell)
    snap_all = np.asarray(sorted(float(z) for z in st.get_snapshot_redshifts(args.seed, logger)))
    z_top = max(w["z_hi"] for w in wins) + args.zmax_snap_pad
    snap = snap_all[:int(np.searchsorted(snap_all, z_top)) + 1]
    log(f"scripts/44 seed={args.seed}: {len(wins)} windows (z0 {wins[0]['z0']}..{wins[-1]['z0']}), {len(snap)} snapshots up to z={snap[-1]:.2f}, "
        f"tracers {names}")

    # ---- weights W_i on the LOS grid (x_e,mean(z) interpolated between the snapshot means, like the lightcone slices) ----
    xe_snap = np.array([1.0 - float(np.mean(st.load_field_box(args.seed, z, "xH", convention="physical"))) for z in snap])
    zi_lo0 = int(np.searchsorted(z_arr, cfg.box.z_min))
    z_trim = z_arr[zi_lo0:int(np.searchsorted(z_arr, cfg.box.z_max))]
    xe_pix = np.interp(z_trim, snap, xe_snap)
    norm = {"ne_scale": prod.get("ne_scale", 1.0), "tau0": prod.get("tau0", 0.0)}
    wt = kc.slice_weights_from_means(cosmo, constants.tau_prefactor(cfg) * float(norm["ne_scale"]), float(norm["tau0"]), z_trim, xe_pix)
    W = wt["W"]
    assign = cc.pixel_assignment(z_trim, snap)

    # ---- windows: which (window, group) pairs each snapshot serves ----
    plan = {}                       # snapshot index -> [(window index, pixel indices)]
    wmeta = []
    for wi, w in enumerate(wins):
        zi_lo, zi_hi = int(np.searchsorted(z_trim, w["z_lo"])), int(np.searchsorted(z_trim, w["z_hi"]))
        wmeta.append({"zi": (zi_lo, zi_hi), "wz": cc.window_pixel_weights(z_trim, zi_lo, zi_hi), "x_hii": float(np.mean(xe_pix[zi_lo:zi_hi]))})
        for si, idx in cc.window_groups(assign, zi_lo, zi_hi):
            plan.setdefault(si, []).append((wi, idx))
    D_acc = {n: [None] * len(wins) for n in names}
    ell_ref = None
    t0 = time.time()
    for si in sorted(plan):
        z = float(snap[si])
        theta, cubes = snapshot_boxes(st, args.seed, z, specs, True)
        for wi, idx in plan[si]:
            wz = wmeta[wi]["wz"][idx - wmeta[wi]["zi"][0]]
            w_gal = {n: wz * (bluetides_bias_gz(z_trim[idx]) if n == MATTER else 1.0) for n in names}
            res = cc.snapshot_group_cross(theta, {n: cubes[n] for n in names}, W[idx], w_gal, f2d, kg, wins[wi]["chi_eff"])
            ell_ref = res["ell"]
            for n in names:
                D_acc[n][wi] = res["D"][n] if D_acc[n][wi] is None else D_acc[n][wi] + res["D"][n]
        log(f"  snapshot z={z:.3f}: {len(plan[si])} window(s), {sum(len(i) for _, i in plan[si])} pixel-slabs ({time.time() - t0:.0f}s)")

    # ---- results at the ell targets + the lightcone estimate of the same seed ----
    targets = lp_maps.LP_ELL_TARGETS
    f2_lc = lp_maps.filter_and_square(prod["kSZ_map"], filt["ell_grid"], filt["fl"], kg, chi_ref)
    out = {"seed": args.seed, "windows": [{k: w[k] for k in ("z0", "dz", "z_lo", "z_hi", "chi_eff")} | {"x_hii_coeval": wmeta[i]["x_hii"],
                                                                                                  "x_hii_lightcone": w["x_hii"]}
                                          for i, w in enumerate(wins)],
           "ell_targets": np.array(targets), "names": names, "D_coeval": {}, "D_lightcone": {}, "chi_ref": chi_ref}
    for n in names:
        Dc = np.full((len(wins), len(targets)), np.nan)
        Dl = np.full((len(wins), len(targets)), np.nan)
        if n == MATTER:
            gmaps = {(w["dz"], w["z0"]): w["gal_map"] for w in prod["windows"]}
        else:
            p = os.path.join(args.maps_root, n, f"seed{args.seed}_wo{args.wrap_offset}_lpmaps.pkl")
            gmaps = {}
            if os.path.exists(p):
                with open(p, "rb") as f:
                    gmaps = {(w["dz"], w["z0"]): w["gal_map"] for w in pickle.load(f)["windows"]}
            else:
                log(f"  (no lightcone tracer maps for {n}: {p})")
        for wi, w in enumerate(wins):
            if D_acc[n][wi] is not None:
                Dc[wi] = lp_maps.interp_at_ell(ell_ref, D_acc[n][wi], targets)
            g = gmaps.get((w["dz"], w["z0"]))
            if g is not None:
                c = lp_maps.lp_cross_dell(f2_lc, g, kg, w["chi_eff"])
                Dl[wi] = lp_maps.interp_at_ell(c["ell"], c["D_ell"], targets)
        out["D_coeval"][n], out["D_lightcone"][n] = Dc, Dl
        i = int(np.nanargmax(np.nanmax(np.abs(Dl), axis=1))) if np.isfinite(Dl).any() else 0
        log(f"  {n:>16}: window z0={wins[i]['z0']}: coeval {np.round(Dc[i], 4)}  lightcone {np.round(Dl[i], 4)} uK^2 at ell {list(map(int, targets))}")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"seed{args.seed}_wo{args.wrap_offset}_coeval.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log(f"saved {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
