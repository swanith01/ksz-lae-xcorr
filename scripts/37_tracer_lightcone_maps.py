#!/usr/bin/env python3
"""
scripts/37_tracer_lightcone_maps.py
=====================================
Cross-correlation suite, step 1 (per seed): galaxy-like tracers (halo mass cuts, LAE luminosity cuts, LBG MUV cuts, see
tracers/suite.py) on the SAME wrap-cycle lightcone as the kSZ map, so that the existing La Plante-style estimator
(scripts/28) can be run with each tracer in place of  b_g(z) * delta_m.

For every tracer `name` this writes  <out-root>/<name>/seed<S>_wo<K>_lpmaps.pkl : a copy of the seed's kSZ products
(scripts/26 --save-maps) whose windows' `gal_map` is replaced by the projected tracer overdensity

    gal_map = sum_i W_i ( n_i / nbar(z_i) - 1 ),     W = the same top-hat-in-z weights as lp_maps.build_lp_galaxy_map,

with n_i the tracer count per 1 Mpc^3 cell on the lightcone (nearest-grid-point counts, interpolated between snapshots
exactly like the density) and nbar(z) the tracer mean density (exact, from the snapshot object counts).  The tracer carries
its own bias and shot noise; nothing is multiplied by a bias.  Because the kSZ map, filter and windows are untouched, the
result drops into scripts/28 unchanged:

    python scripts/28_lp_cross_vs_z0.py --in-dir <out-root>/<name> --out-dir <...>/<name>

Geometry is rebuilt exactly as in correlation/ksz_auto_wrapcycle.build_wrapcycle_field_data (same cosmology, LOS grid,
wrap_cycle_seed = seed + wrap offset); only the redshift range is truncated at the highest window (the LOS origin and
spacing are untouched, so the pixels coincide).  Heavy part: reading the halo catalogues (IO), then ~0.5 GB of
lightcone per tracer in memory -> use pbs/pbs_tracer_maps.sh.

Usage:
    python scripts/37_tracer_lightcone_maps.py --seed 1
    python scripts/37_tracer_lightcone_maps.py --seed 1 --config configs/variants/zreion_xh.yaml --specs "halo:10;lae:42.5;lbg:-19"
"""
import argparse
import copy
import os
import pickle
import sys
import time
from collections import OrderedDict

import numpy as np

from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.lightcone.wrap_cycle import build_los_z_grid, stitch_wrapcycle
from ksz_lae_xcorr.tracers import suite
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology


class SnapshotGrids:
    """Tracer count grids of the snapshots the stitcher asks for (all specs computed in ONE pass over a snapshot's
    catalogue, the last few snapshots kept) + the exact mean density of every spec at every loaded snapshot."""

    def __init__(self, stitcher, seed, specs, keep=3, log=print):
        self.st, self.seed, self.specs, self.keep, self.log = stitcher, seed, specs, keep, log
        self.cache: OrderedDict = OrderedDict()
        self.nbar: dict = {}          # (name, z) -> mean count per cell
        self.nobj: dict = {}

    def load_box(self, z, name):
        z = round(float(z), 6)
        if z not in self.cache:
            t0 = time.time()
            cat = suite.load_catalogues(self.st, self.seed, z)
            tr = suite.snapshot_tracers(cat, self.specs, self.st.ngrid, self.st.cell, max_mass_sample=1)
            for sp in self.specs:
                self.nbar[(sp.name, z)] = float(tr[sp.name]["grid"].mean())
                self.nobj[(sp.name, z)] = tr[sp.name]["n"]
            self.cache[z] = {k: v["grid"] for k, v in tr.items()}
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
            self.log(f"  snapshot z={z:.4f}: tracer grids built in {time.time() - t0:.0f}s "
                     + " ".join(f"{sp.name}:{self.nobj[(sp.name, z)]:,}" for sp in self.specs[:3]) + " ...")
        return self.cache[z][name]

    def nbar_curve(self, name):
        pts = sorted((z, v) for (n, z), v in self.nbar.items() if n == name)
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def window_maps(prod: dict, z_arr: np.ndarray, dens_lc: np.ndarray) -> list:
    """gal_map for every stored window, with the arithmetic of lp_maps.build_lp_galaxy_map (bias = 1)."""
    fd = {"z_lc": z_arr, "density_lc": dens_lc}
    out = []
    for w in prod["windows"]:
        zi_lo, zi_hi = int(np.searchsorted(z_arr, w["z_lo"])), int(np.searchsorted(z_arr, w["z_hi"]))
        out.append(lp_maps.build_lp_galaxy_map(fd, zi_lo, zi_hi, bias_fn=lambda z: np.ones_like(z)).astype(np.float32))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml", help="decides the kSZ products to start from (in-dir)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--specs", default="default")
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--in-dir", default=None, help="default: <products_root>/ksz_auto_wrapcycle (seed*_wo*_lpmaps.pkl)")
    ap.add_argument("--out-root", default="data/products_tracers/maps")
    ap.add_argument("--mode", default="grid-wrap")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    specs = suite.parse_specs(args.specs)
    in_dir = args.in_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    in_path = os.path.join(in_dir, f"seed{args.seed}_wo{args.wrap_offset}_lpmaps.pkl")
    with open(in_path, "rb") as f:
        prod = pickle.load(f)
    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    log = logger.info
    st = Stitcher(cfg, field_convention="physical")
    ngrid, cell = int(cfg.box.hii_dim), float(cfg.box.box_len_mpc) / int(cfg.box.hii_dim)

    z_arr, chi = build_los_z_grid(get_cosmology(cfg), cfg.box.z_min, cfg.box.z_max, cell)
    z_need = max(w["z_hi"] for w in prod["windows"]) + 0.2
    n_keep = int(np.searchsorted(z_arr, z_need)) + 2
    z_arr, chi = z_arr[:n_keep], chi[:n_keep]
    snap_z = [round(float(z), 6) for z in st.get_snapshot_redshifts(args.seed, logger)]
    wrap_seed = args.seed + args.wrap_offset
    log(f"scripts/37 seed={args.seed} wrap_seed={wrap_seed}: {len(z_arr)} LOS pixels up to z={z_arr[-1]:.2f}, "
        f"{len(specs)} tracers, {len(prod['windows'])} windows (kSZ products: {in_path})")

    grids = SnapshotGrids(st, args.seed, specs, log=log)
    t0 = time.time()
    lc = stitch_wrapcycle(grids.load_box, snap_z, z_arr, chi, ngrid=ngrid, cell_size=cell, wrap_cycle_seed=wrap_seed,
                          fields=[sp.name for sp in specs], mode=args.mode,
                          progress=lambda k, n: log(f"  stitched {k + 1}/{n} LOS pixels ({time.time() - t0:.0f}s)")
                          if (k + 1) % 500 == 0 or k + 1 == n else None)
    log(f"stitched all tracers in {time.time() - t0:.0f}s")

    for sp in specs:
        zs, nb = grids.nbar_curve(sp.name)
        nbar_pix = np.interp(z_arr, zs, nb)
        cube = lc.pop(sp.name)
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = np.where(nbar_pix[None, None, :] > 0, cube / nbar_pix[None, None, :], 1.0).astype(np.float32)
        del cube
        maps = window_maps(prod, z_arr, dens)
        del dens
        out = copy.deepcopy(prod)
        for w, g in zip(out["windows"], maps):
            w["gal_map"] = g
        out["tracer"] = {"name": sp.name, "kind": sp.kind, "cut": sp.cut, "label": sp.label,
                         "n_obj_by_z": {z: grids.nobj[(sp.name, z)] for z in zs}}
        d = os.path.join(args.out_root, sp.name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"seed{args.seed}_wo{args.wrap_offset}_lpmaps.pkl"), "wb") as f:
            pickle.dump(out, f)
        rms = [float(np.std(g)) for g in maps]
        log(f"  {sp.name:>16}: saved; window-map rms {min(rms):.3g}..{max(rms):.3g}")
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
