#!/usr/bin/env python3
"""
scripts/61_lae_igm.py  (24 Nov 2026)
======================================
Recompute the IGM attenuation (tau + observed stage) of Jahaan's intrinsic LAE catalogue on the REPAIRED 300 Mpc boxes:
evolved density (density_evolved.npy) and displaced halo positions (halo_coords_eulerian.npy), both written by
sim_generation/2026-06_300Mpc_boxes/repair/repair_seed.py.  The intrinsic catalogue (which haloes host LAEs, their
UV magnitudes, REW, L_Lya and voff) is reused unchanged: halo_ids are rows of the saved halo arrays, and the repair
keeps the row order.  Physics: src/ksz_lae_xcorr/lae/igm.py (port of Jahaan's current tau.py / observed_prop.py).

Outputs use the same file names as Jahaan's obs_prop, so the tracer pipeline only needs lae_catalogue_root changed:
    <out-root>/obs_prop/{lya_lum_obs,rew_obs,halo_ids_obs,MUV_obs}/<name>_z<z>_300_300_s<seed>.npy
    <out-root>/igm/T_igm/T_igm_z<z>_300_300_s<seed>.npy      (transmission of every intrinsic LAE)
    <out-root>/summary_s<seed>.csv                               (counts, old vs new, per snapshot)

    python scripts/61_lae_igm.py --seed 1 --workers 8
"""
import argparse
import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from ksz_lae_xcorr.lae import igm

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--boxes-root", default="/user1/swanith/kSZ2_LAE_project_22Jun2026")
ap.add_argument("--int-root", default="/user1/jahaan/swanith/int_prop", help="Jahaan's intrinsic catalogue (read only)")
ap.add_argument("--old-obs-root", default="/user1/jahaan/swanith/obs_prop", help="old observed catalogue, for counts only")
ap.add_argument("--out-root", default="/user1/swanith/lae_catalogue_v2")
ap.add_argument("--box-len", type=float, default=300.0, help="cMpc")
ap.add_argument("--n", type=int, default=300, help="HII_DIM")
ap.add_argument("--n-fwd", type=int, default=None, help="sightline length in cells (default n // 2, as Jahaan)")
ap.add_argument("--spec", choices=("cell", "fine"), default="cell", help="spectral grid (cell = Jahaan's)")
ap.add_argument("--zmin", type=float, default=0.0)
ap.add_argument("--zmax", type=float, default=99.0)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
n_fwd = args.n_fwd or args.n // 2
dx = args.box_len / args.n
tag = lambda z: f"z{z:.6f}_{args.n}_{int(args.box_len)}_s{args.seed}.npy"
OUT_FIELDS = ("lya_lum_obs", "rew_obs", "halo_ids_obs", "MUV_obs")


def snapshot(z):
    t0 = time.time()
    zd = os.path.join(args.boxes_root, f"seed_{args.seed}", f"coeval_z{z:.6f}")
    outs = {f: os.path.join(args.out_root, "obs_prop", f, f"{f}_{tag(z)}") for f in OUT_FIELDS}
    tpath = os.path.join(args.out_root, "igm", "T_igm", f"T_igm_{tag(z)}")
    row = dict(z=f"{z:.6f}")
    try:
        cat = {k: np.load(os.path.join(args.int_root, k, f"{k}_{tag(z)}")) for k in ("halo_ids", "voff", "lya_lum", "rew", "MUV")}
    except FileNotFoundError:
        return dict(row, status="no intrinsic catalogue")
    old = os.path.join(args.old_obs_root, "lya_lum_obs", f"lya_lum_obs_{tag(z)}")
    row.update(N_intrinsic=cat["halo_ids"].size, N_obs_old=np.load(old).size if os.path.exists(old) else "")
    if all(os.path.exists(p) for p in outs.values()) and not args.overwrite:
        return dict(row, N_obs_new=np.load(outs["lya_lum_obs"]).size, status="exists")
    for needed in ("density_evolved.npy", "halo_coords_eulerian.npy"):
        if not os.path.exists(os.path.join(zd, needed)):
            return dict(row, status=f"missing {needed} (run the repair first)")
    if cat["halo_ids"].size:
        fields = dict(delta=np.load(os.path.join(zd, "density_evolved.npy")),
                      x_hi=np.load(os.path.join(zd, "neutral_fraction.npy")),
                      temp=np.load(os.path.join(zd, "kinetic_temperature.npy")),
                      vz_raw=np.load(os.path.join(zd, "velocity_z.npy")))
        coords = np.load(os.path.join(zd, "halo_coords_eulerian.npy"), mmap_mode="r")[np.sort(cat["halo_ids"])]
        order = np.argsort(np.argsort(cat["halo_ids"]))          # back to catalogue order
        coords = np.asarray(coords)[order]
        T = igm.attenuate(z, fields, coords, cat["voff"], dx, n_fwd, spec=args.spec)
    else:
        T = np.empty(0)
    lum, rew = T * cat["lya_lum"], T * cat["rew"]
    sel = (lum > 1e42) & (rew > 0)
    for f, arr in (("lya_lum_obs", lum[sel]), ("rew_obs", rew[sel]), ("halo_ids_obs", cat["halo_ids"][sel]),
                   ("MUV_obs", cat["MUV"][sel])):
        os.makedirs(os.path.dirname(outs[f]), exist_ok=True)
        np.save(outs[f], arr)
    os.makedirs(os.path.dirname(tpath), exist_ok=True)
    np.save(tpath, T.astype(np.float32))
    return dict(row, N_obs_new=int(sel.sum()), T_median=round(float(np.median(T)), 4) if T.size else "",
                T_mean=round(float(T.mean()), 4) if T.size else "", status=f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    zs = sorted(float(p.split("coeval_z")[1]) for p in glob.glob(os.path.join(args.boxes_root, f"seed_{args.seed}", "coeval_z*")))
    zs = [z for z in zs if args.zmin <= z <= args.zmax]
    if not zs:
        sys.exit("no snapshots found")
    print(f"seed {args.seed}: {len(zs)} snapshots, n_fwd {n_fwd}, spec {args.spec}, workers {args.workers}", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap(snapshot, zs):
            print("  " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)
            rows.append(r)
    keys = ["z", "N_intrinsic", "N_obs_old", "N_obs_new", "T_median", "T_mean", "status"]
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, f"summary_s{args.seed}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["status"].startswith("missing")]
    print(f"done; {len(bad)} snapshots missing repaired inputs", flush=True)
    sys.exit(1 if bad else 0)
