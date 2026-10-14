#!/usr/bin/env python3
"""
scripts/35_tracer_bias_suite.py
=================================
Linear bias b(z), number density and object counts of a SUITE of tracers cut from the 21cmFAST halo catalogue
(halo mass cuts, LAE Lya-luminosity cuts, LBG MUV cuts), for ONE seed, on every coeval snapshot.

    b_lin = sum_{k<kmax} P_tm / sum_{k<kmax} P_mm      (tracer-matter cross-spectrum: shot noise does not bias it)

matter = the pipeline's own delta(z) (field_convention='physical': D(z)/D(0) x the IC density, 1 Mpc cells), so b is
the bias the La Plante-style delta_g = b_g(z) delta_m would have to MATCH.  Also writes, for the same sample, the
analytic Sheth-Mo-Tormen mean bias from the sample's own halo masses (reference only).

Heavy part = reading the full halo catalogue of each snapshot (tens of millions of haloes at low z): submit with
pbs/pbs_tracer_bias.sh (one job per seed).  No CAMB needed (the analytic reference falls back to Eisenstein-Hu).

Usage:
    python scripts/35_tracer_bias_suite.py --seed 1
    python scripts/35_tracer_bias_suite.py --seed 1 --specs "halo:1e10,halo:1e11,lae:42.5,lbg:-19" --every 4
Outputs (in --out-dir, default data/products_tracers/bias):
    tracer_bias_seed<S>.csv   one row per (snapshot, tracer): z, name, kind, cut, n_obj, n_per_mpc3, b_lin, b_lin_err, b_ST
    tracer_bias_seed<S>.npz   the full b(k) curves
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.tracers import suite
from ksz_lae_xcorr.utils.config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--specs", default="default", help="'default' or e.g. 'halo:1e10,lae:42.5,lbg:-19'")
    ap.add_argument("--kmax", type=float, default=0.15, help="Mpc^-1: b_lin uses modes below this")
    ap.add_argument("--every", type=int, default=1, help="use every N-th snapshot (smoke tests)")
    ap.add_argument("--z-min", type=float, default=5.0)
    ap.add_argument("--z-max", type=float, default=20.0)
    ap.add_argument("--out-dir", default="data/products_tracers/bias")
    ap.add_argument("--sigma8", type=float, default=suite.SIGMA8_21CMFAST)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    specs = suite.parse_specs(args.specs)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    st = Stitcher(cfg, field_convention="physical")
    c = cfg.cosmology
    sig = suite.SigmaOfM(Om=c.Om0, Ob=c.Ob0, h=c.H0 / 100.0, ns=c.ns, sigma8=args.sigma8, backend="auto")
    vol = float(cfg.box.box_len_mpc) ** 3
    zs = [z for z in st.get_snapshot_redshifts(args.seed, logger) if args.z_min <= z <= args.z_max][::args.every]
    print(f"seed {args.seed}: {len(zs)} snapshots, {len(specs)} tracers: {[s.name for s in specs]}", flush=True)

    rows, bk, t0 = [], {}, time.time()
    for iz, z in enumerate(sorted(zs)):
        cat = suite.load_catalogues(st, args.seed, z)
        tr = suite.snapshot_tracers(cat, specs, st.ngrid, st.cell)
        dm = np.asarray(st.load_field_box(args.seed, z, "density", convention="physical"), dtype=np.float32)
        for spec in specs:
            t = tr[spec.name]
            r = suite.halo_bias_from_grids(t["grid"], dm, float(cfg.box.box_len_mpc), kmax=args.kmax)
            b_st = suite.mean_bias_of_masses(t["mass_sample"], z, sig)
            rows.append(dict(z=z, name=spec.name, kind=spec.kind, cut=spec.cut, n_obj=r["n_obj"],
                             n_per_mpc3=r["n_obj"] / vol, b_lin=r["b_lin"], b_lin_err=r["b_lin_err"], b_ST=b_st))
            bk[f"{spec.name}__z{z:.4f}"] = r["b_k"]
            bk["k"] = r["k"]
        print(f"  z={z:7.3f} ({iz + 1}/{len(zs)})  {time.time() - t0:6.0f}s   "
              + "  ".join(f"{s.name}: N={tr[s.name]['n']:,}" for s in specs[:3]), flush=True)
        del cat, tr, dm

    out_csv = os.path.join(args.out_dir, f"tracer_bias_seed{args.seed}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(args.out_dir, f"tracer_bias_seed{args.seed}.npz"), **bk)
    print("wrote", out_csv)

    from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
    print("\nb_lin at selected redshifts (Waters+16 / La Plante bias in the header):")
    for zt in (6.0, 7.0, 8.0, 10.0):
        zn = min(zs, key=lambda x: abs(x - zt))
        print(f"  z={zn:.2f}  b_Waters={float(bluetides_bias_gz(zn)):.1f}")
        for r in rows:
            if r["z"] == zn:
                print(f"     {r['name']:>16}  N={r['n_obj']:>12,.0f}  b_lin={r['b_lin']:6.2f}  b_ST={r['b_ST']:6.2f}")


if __name__ == "__main__":
    sys.exit(main())
