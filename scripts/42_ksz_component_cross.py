#!/usr/bin/env python3
"""
scripts/42_ksz_component_cross.py
===================================
Component decomposition of the kSZ^2 x tracer cross-power, step 2 (light: 2D FFTs only, run on swarm).

For every tracer folder <maps-root>/<name>/seed<S>_wo<K>_lpmaps.pkl (scripts/37; its windows carry the tracer maps) and
every seed with <comp-dir>/seed<S>_wo<K>_components.pkl (scripts/41) it computes the fractional cross-power

    D^frac_ell(variant, window) = ell(ell+1)/2pi  C_ell( S_variant/<S_variant> - 1  x  delta_tracer )

for the 8 variants A/B x {x, v, d, full} at ell = 500, 1000, 3000 (correlation/ksz_components.py), and the seed mean +- s.e.m.
The base kSZ products (--base-dir, scripts/26) give an extra "matter_Waters" tracer (b_g(z) delta_m, the reference run).

Output per tracer in --out-dir:
    components_<name>.npz   D (n_seed, n_variant, n_window, n_ell), x_hii (n_seed, n_window), z0, dz, seeds, variants, ell_targets
    components_<name>.csv   z0, dz, x_hii, mean_<variant>_l<ell>, sem_<variant>_l<ell>
    components_rho.csv      Pearson r between every variant map and A_full, per seed

Usage:
    python scripts/42_ksz_component_cross.py --maps-root data/products_tracers/maps --comp-dir data/products_tracers/components
"""
import argparse
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np

from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.utils.config import load_config

TARGETS = lp_maps.LP_ELL_TARGETS


def load_components(comp_dir, offset):
    out = {}
    for p in sorted(glob.glob(os.path.join(comp_dir, f"seed*_wo{offset}_components.pkl"))):
        s = int(re.search(r"seed(\d+)_wo", os.path.basename(p)).group(1))
        with open(p, "rb") as f:
            out[s] = pickle.load(f)
    return out


def tracer_products(maps_root, offset):
    """{name: {seed: path}} for every <maps_root>/<name>/seed*_wo<offset>_lpmaps.pkl."""
    out = {}
    for d in sorted(glob.glob(os.path.join(maps_root, "*", ""))):
        name = os.path.basename(os.path.dirname(d))
        ps = {int(re.search(r"seed(\d+)_wo", os.path.basename(p)).group(1)): p
              for p in glob.glob(os.path.join(d, f"seed*_wo{offset}_lpmaps.pkl"))}
        if ps:
            out[name] = ps
    return out


def cross_one_tracer(name, paths, comps, kg, targets=TARGETS, dz=1.0):
    """Arrays for one tracer: D (n_seed, n_variant, n_window, n_target), x_hii (n_seed, n_window), z0, dz, seeds."""
    seeds = sorted(set(paths) & set(comps))
    if not seeds:
        return None
    wins = {}
    for s in seeds:
        with open(paths[s], "rb") as f:
            wins[s] = {(w["dz"], w["z0"]): w for w in pickle.load(f)["windows"]}
    keys = sorted({k for s in seeds for k in wins[s] if abs(k[0] - dz) < 1e-9}, key=lambda k: k[1])
    D = np.full((len(seeds), len(kc.VARIANTS), len(keys), len(targets)), np.nan)
    xh = np.full((len(seeds), len(keys)), np.nan)
    for si, s in enumerate(seeds):
        maps = comps[s]["maps"]
        for ki, key in enumerate(keys):
            w = wins[s].get(key)
            if w is None:
                continue
            xh[si, ki] = w["x_hii"]
            for vi, v in enumerate(kc.VARIANTS):
                c = kc.frac_cross_dell(maps[v], w["gal_map"], kg, w["chi_eff"])
                D[si, vi, ki] = lp_maps.interp_at_ell(c["ell"], c["D_ell"], targets)
    return {"D": D, "x_hii": xh, "z0": np.array([k[1] for k in keys]), "dz": np.array([k[0] for k in keys]),
            "seeds": np.array(seeds), "variants": np.array(kc.VARIANTS), "ell_targets": np.array(targets, float)}


def write_outputs(name, res, out_dir):
    np.savez(os.path.join(out_dir, f"components_{name}.npz"), **res)
    D = res["D"]
    n_ok = np.sum(np.isfinite(D), axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(D, axis=0)
        sem = np.nanstd(D, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))
    cols = {"z0": res["z0"], "dz": res["dz"], "x_hii": np.nanmean(res["x_hii"], axis=0)}
    for vi, v in enumerate(kc.VARIANTS):
        for ti, t in enumerate(res["ell_targets"]):
            cols[f"mean_{v}_l{int(t)}"] = mean[vi, :, ti]
            cols[f"sem_{v}_l{int(t)}"] = sem[vi, :, ti]
    with open(os.path.join(out_dir, f"components_{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        w.writerows(zip(*[np.asarray(c) for c in cols.values()]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--maps-root", default="data/products_tracers/maps")
    ap.add_argument("--comp-dir", default="data/products_tracers/components")
    ap.add_argument("--base-dir", default=None, help="scripts/26 products (default <products_root>/ksz_auto_wrapcycle); "
                                                     "gives the matter_Waters reference; 'none' to skip")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output_components")
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--dz", type=float, default=1.0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    kg = KGrid(cfg)
    comps = load_components(args.comp_dir, args.wrap_offset)
    if not comps:
        sys.exit(f"no seed*_wo{args.wrap_offset}_components.pkl in {args.comp_dir} (run scripts/41 per seed)")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"component seeds: {sorted(comps)}")
    tr = tracer_products(args.maps_root, args.wrap_offset)
    if args.base_dir != "none":
        base = args.base_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
        bp = {int(re.search(r"seed(\d+)_wo", os.path.basename(p)).group(1)): p
              for p in glob.glob(os.path.join(base, f"seed*_wo{args.wrap_offset}_lpmaps.pkl"))}
        if bp:
            tr = {"matter_Waters": bp, **tr}
    rho_rows = []
    for name, paths in tr.items():
        res = cross_one_tracer(name, paths, comps, kg, dz=args.dz)
        if res is None:
            print(f"  {name}: no common seeds, skipped")
            continue
        write_outputs(name, res, args.out_dir)
        print(f"  {name:>16}: seeds {list(res['seeds'])}, {len(res['z0'])} windows")
    for s in sorted(comps):
        for v, r in comps[s]["rho_with_A_full"].items():
            rho_rows.append({"seed": s, "variant": v, "r_with_A_full": r})
    with open(os.path.join(args.out_dir, "components_rho.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "variant", "r_with_A_full"])
        w.writeheader(); w.writerows(rho_rows)
    print("written to", args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
