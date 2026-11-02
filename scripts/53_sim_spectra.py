#!/usr/bin/env python3
"""
scripts/53_sim_spectra.py
===========================
Step 1 of the SIMULATION-based S/N (snr/sim_snr.py): binned spectra for every CMB experiment, from maps that already exist.

Input = the per-seed La Plante map products, physical field units (wrap-cycle path):
    matter x Waters+16 bias : <products_root>/ksz_auto_wrapcycle/seed*_wo0_lpmaps.pkl      (scripts/26 --save-maps)
    each tracer             : data/products_tracers/maps/<tracer>/seed*_wo0_lpmaps.pkl     (scripts/37)
NOT the stitched lc_*.npz of scripts 02/04/05/08: those still load the raw (legacy) velocity and density by default.

For each input directory: the CMB filter of every experiment (snr.feasibility, f = F b with peak 1; C_kSZ,reion = seed median
of the simulated map's own auto-power, or La Plante's with --ksz-reion lp), the filtered kSZ map squared, and per seed and
window the binned Cs = C_l(kSZ_f^2 x delta_g) [uK^2] and Cg = C_l(delta_g x delta_g).  Writes <out-dir>/<name>.npz (a few MB).
Light: about a minute per tracer, 2 CPUs.  Run on swarm (the pickles live there), then copy <out-dir> to the desktop for 54.

    python scripts/53_sim_spectra.py --in-dir data/products/ksz_auto_wrapcycle --name matter_waters
    python scripts/53_sim_spectra.py --maps-root data/products_tracers/maps            # every tracer folder
    python scripts/53_sim_spectra.py --maps-root data/products_tracers/maps --in-dir <products_root>/ksz_auto_wrapcycle
"""
import argparse
import dataclasses
import glob
import os
import pickle
import re
import sys
import time

import numpy as np

from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.snr import sim_snr as ss
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def experiment_variants():
    """(label, Experiment): every experiment of snr.feasibility; the ILC ones also with white noise only (as in scripts/52)."""
    out = []
    for name, e in fe.EXPERIMENTS.items():
        out.append((name + (" (post-ILC)" if e.ilc_key else " (white noise)"), e))
        if e.ilc_key:
            out.append((name + " (white noise)", dataclasses.replace(e, ilc_key=None)))
    return out


def load_seeds(in_dir, wo):
    out = {}
    for p in sorted(glob.glob(os.path.join(in_dir, f"seed*_wo{wo}_lpmaps.pkl"))):
        s = int(re.search(r"seed(\d+)_", os.path.basename(p)).group(1))
        with open(p, "rb") as f:
            out[s] = pickle.load(f)
    return out


def tracer_n_of_z(prod, box_len):
    """callable z -> n_3D [Mpc^-3] from the tracer's snapshot object counts (None for the matter products)."""
    tr = prod.get("tracer")
    if not tr or "n_obj_by_z" not in tr:
        return None
    zs = np.array(sorted(tr["n_obj_by_z"]), float)
    n = np.array([tr["n_obj_by_z"][z] for z in sorted(tr["n_obj_by_z"])], float) / box_len ** 3
    return lambda z: np.interp(z, zs, n)


def run_dir(cfg, in_dir, name, out_dir, wo, ksz_reion_mode, variants):
    t0 = time.time()
    prods = load_seeds(in_dir, wo)
    if len(prods) < 2:
        print(f"  {name}: {len(prods)} seed(s) in {in_dir}, need >= 2 -- skipped")
        return None
    seeds = sorted(prods)
    kg = KGrid(cfg)
    cosmo = get_cosmology(cfg)
    chi_of_z = lambda z: cosmo.comoving_distance(np.asarray(z, float)).to_value("Mpc")
    chi_ref = float(np.median([prods[s]["chi_eff_kSZ"] for s in seeds]))
    comp = fe.lp_components(cfg)
    ell = comp["ell"]
    kr_sim = lp_maps.kSZ_reion_Cl_on_grid([prods[s]["kSZ_map"] for s in seeds], kg, chi_ref, ell)
    kr = kr_sim if ksz_reion_mode == "sim" else None
    filts = [fe.experiment_filter(e, comp, ksz_reion=kr) for _, e in variants]

    # windows matched across seeds by (dz, z0)
    keys = sorted({(w["dz"], w["z0"]) for s in seeds for w in prods[s]["windows"]})
    nb = len(kg.k_centers)
    nS, nE, nW = len(seeds), len(variants), len(keys)
    Cs = np.full((nS, nE, nW, nb), np.nan)
    Cg = np.full((nS, nW, nb), np.nan)
    meta = {k: np.full(nW, np.nan) for k in ("z0", "dz", "z_lo", "z_hi", "chi_eff", "x_hii")}
    for si, s in enumerate(seeds):
        p = prods[s]
        sq = [lp_maps.filter_and_square(p["kSZ_map"], ell, fs.f, kg, chi_ref) * ss.T_CMB_UK2 for fs in filts]   # uK^2 maps
        wmap = {(w["dz"], w["z0"]): w for w in p["windows"]}
        for wi, key in enumerate(keys):
            w = wmap.get(key)
            if w is None:
                continue
            for k in meta:
                if not np.isfinite(meta[k][wi]):
                    meta[k][wi] = w[k]
            g = w["gal_map"]
            Cg[si, wi] = ss.binned_cl(g, g, kg, w["chi_eff"])
            for ei, m in enumerate(sq):
                Cs[si, ei, wi] = ss.binned_cl(m, g, kg, w["chi_eff"])
    x_mean = np.array([np.nanmean([w["x_hii"] for s in seeds for w in prods[s]["windows"] if (w["dz"], w["z0"]) == k])
                       for k in keys])
    edges = np.array([ss.ell_edges(kg, c) for c in meta["chi_eff"]])
    nz = tracer_n_of_z(prods[seeds[0]], float(cfg.box.box_len_mpc))
    shot = np.array([ss.window_shot_noise(nz, meta["z_lo"][i], meta["z_hi"][i], chi_of_z, meta["chi_eff"][i]) if nz else 0.0
                     for i in range(nW)])
    tr = prods[seeds[0]].get("tracer") or {}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        path, name=name, kind=tr.get("kind", "matter"), cut=float(tr.get("cut", np.nan)), seeds=np.array(seeds),
        experiments=np.array([lab for lab, _ in variants]), Cs=Cs, Cg=Cg, edges=edges, shot_sim=shot,
        z0=meta["z0"], dz=meta["dz"], z_lo=meta["z_lo"], z_hi=meta["z_hi"], chi_eff=meta["chi_eff"], x_hii=x_mean,
        CT2T2=np.array([fs.C_T2T2 for fs in filts]), sigma2_reion=np.array([fs.sigma2_reion for fs in filts]),
        filter_ell=ell, filter_f=np.array([fs.f for fs in filts]), kSZ_reion_sim=kr_sim, kSZ_reion_lp=comp["kSZ_reion_30sim_mean"],
        chi_ref=chi_ref, ksz_reion_mode=ksz_reion_mode, n_obj_z=np.array(sorted((tr.get("n_obj_by_z") or {}).items()), float).reshape(-1, 2),
    )
    j = int(np.argmin(np.abs(ell - 3000)))
    print(f"  {name}: {nS} seeds, {nW} windows, {nb} l-bins -> {path}  ({time.time() - t0:.0f}s);  "
          f"sim kSZ_reion D(3000) = {kr_sim[j] * 3000 * 3001 / (2 * np.pi):.3g} uK^2 (La Plante {comp['kSZ_reion_30sim_mean'][j] * 3000 * 3001 / (2 * np.pi):.3g})")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--in-dir", default=None, help="one folder of seed*_wo*_lpmaps.pkl (e.g. the matter x Waters bias run)")
    ap.add_argument("--name", default="matter_waters", help="output name for --in-dir")
    ap.add_argument("--maps-root", default=None, help="folder of tracer folders written by scripts/37 (each becomes <tracer>.npz)")
    ap.add_argument("--out-dir", default="data/products_tracers/sim_spectra")
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--ksz-reion", default="sim", choices=("sim", "lp"),
                    help="C_kSZ,reion in the filter: the simulated map's own (default, as scripts/28) or La Plante's Fig. 1 curve")
    ap.add_argument("--n-kbins", type=int, default=40, help="log k-bins (cfg.correlation.n_kbins is 25)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg.correlation.n_kbins = args.n_kbins
    if args.in_dir is None and args.maps_root is None:
        args.in_dir = os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    variants = experiment_variants()
    print(f"experiments: {', '.join(lab for lab, _ in variants)}")
    if args.in_dir:
        run_dir(cfg, args.in_dir, args.name, args.out_dir, args.wrap_offset, args.ksz_reion, variants)
    if args.maps_root:
        for d in sorted(glob.glob(os.path.join(args.maps_root, "*", ""))):
            run_dir(cfg, d, os.path.basename(os.path.normpath(d)), args.out_dir, args.wrap_offset, args.ksz_reion, variants)


if __name__ == "__main__":
    sys.exit(main())
