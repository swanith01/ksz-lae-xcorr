#!/usr/bin/env python3
"""
scripts/39_tracer_luminosity_functions.py
===========================================
Halo mass function, UV luminosity function (LBG catalogue) and Lya luminosity function (LAE catalogue) of ONE seed, on
every coeval snapshot -- the numbers that scripts/40 compares with Sheth-Tormen and with the observed luminosity functions
(data/reference/luminosity_functions).

    halo_mf   dn/dlog10(M)   [Mpc^-3 dex^-1]  for ALL haloes of the snapshot (bins of 0.25 dex, 8 <= log M <= 13),
              plus the Sheth-Tormen value at the bin centre (same cosmology / sigma8 as scripts/35)
    uvlf      phi(M_UV)      [Mpc^-3 mag^-1]  of the LBG catalogue (MUV_lbg), bins of 0.5 mag; also the median halo
              mass in each bin (the LBG sample only contains haloes above its own mass cut, so the faint end is cut off)
    lae_lf    phi(log L_Lya) [Mpc^-3 dex^-1]  of the LAE catalogue (lya_lum_obs = observed, i.e. after IGM attenuation),
              bins of 0.1 dex

Only masses and the two catalogue value files are read (no positions) -> light compared with scripts/35, but it still
touches the 73M-halo mass file of every snapshot: use pbs/pbs_tracer_lf.sh.

    python scripts/39_tracer_luminosity_functions.py --seed 1
Output (--out-dir, default data/products_tracers/lf):  lf_seed<S>.csv  with columns
    kind, z, x, dx, n, phi, phi_err, median_log10M, phi_ST          (x = log10 M | M_UV | log10 L; phi_ST only for halo_mf)
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

HALO_EDGES = np.arange(8.0, 13.0001, 0.25)
UV_EDGES = np.arange(-24.0, -12.0001, 0.5)
LAE_EDGES = np.arange(41.0, 45.0001, 0.1)


def _hist_rows(kind, z, values, edges, vol, per_unit, extra=None, st=None):
    """one row per bin: counts, phi = counts / (vol * bin width), Poisson error."""
    n, _ = np.histogram(values, bins=edges)
    dx = np.diff(edges)
    x = 0.5 * (edges[1:] + edges[:-1])
    phi = n / (vol * dx)
    err = np.where(n > 0, phi / np.sqrt(np.maximum(n, 1)), np.nan)
    rows = []
    for i in range(len(n)):
        rows.append(dict(kind=kind, z=z, x=x[i], dx=dx[i], n=int(n[i]), phi=phi[i], phi_err=err[i],
                         median_log10M=extra[i] if extra is not None else np.nan, phi_ST=st[i] if st is not None else np.nan))
    return rows


def snapshot_lfs(cat: dict, z: float, vol: float, sig=None) -> list:
    rows = []
    masses = np.asarray(cat["masses"], dtype=np.float64)
    logm = np.log10(masses[masses > 0])
    st = None
    if sig is not None:
        centre = 0.5 * (HALO_EDGES[1:] + HALO_EDGES[:-1])
        st = suite.sheth_tormen_dndlnm(10 ** centre, z, sig) * np.log(10.0)          # per dex
    rows += _hist_rows("halo_mf", z, logm, HALO_EDGES, vol, "dex", st=st)
    if cat["lbg"] is not None:
        ids, muv = cat["lbg"]
        muv = np.asarray(muv, dtype=np.float64)
        lm = np.log10(masses[np.asarray(ids, dtype=np.int64)])
        which = np.digitize(muv, UV_EDGES) - 1
        med = np.array([np.median(lm[which == i]) if np.any(which == i) else np.nan for i in range(len(UV_EDGES) - 1)])
        rows += _hist_rows("uvlf", z, muv, UV_EDGES, vol, "mag", extra=med)
    if cat["lae"] is not None:
        _, lum = cat["lae"]
        lum = np.asarray(lum, dtype=np.float64)
        rows += _hist_rows("lae_lf", z, np.log10(lum[lum > 0]), LAE_EDGES, vol, "dex")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--every", type=int, default=1, help="use every N-th snapshot (smoke tests)")
    ap.add_argument("--z-min", type=float, default=5.0)
    ap.add_argument("--z-max", type=float, default=20.0)
    ap.add_argument("--out-dir", default="data/products_tracers/lf")
    ap.add_argument("--sigma8", type=float, default=suite.SIGMA8_21CMFAST)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    st = Stitcher(cfg, field_convention="physical")
    c = cfg.cosmology
    sig = suite.SigmaOfM(Om=c.Om0, Ob=c.Ob0, h=c.H0 / 100.0, ns=c.ns, sigma8=args.sigma8, backend="auto")
    vol = float(cfg.box.box_len_mpc) ** 3
    zs = sorted(z for z in st.get_snapshot_redshifts(args.seed, logger) if args.z_min <= z <= args.z_max)[::args.every]
    print(f"seed {args.seed}: {len(zs)} snapshots", flush=True)
    rows, t0 = [], time.time()
    for iz, z in enumerate(zs):
        cat = suite.load_catalogues(st, args.seed, z)
        got = snapshot_lfs(cat, z, vol, sig)
        rows += got
        nh = sum(r["n"] for r in got if r["kind"] == "halo_mf")
        nl = sum(r["n"] for r in got if r["kind"] == "lae_lf")
        nu = sum(r["n"] for r in got if r["kind"] == "uvlf")
        print(f"  z={z:7.3f} ({iz + 1}/{len(zs)})  {time.time() - t0:6.0f}s  haloes(8<logM<13)={nh:,}  LBG={nu:,}  LAE={nl:,}", flush=True)
        del cat
    out = os.path.join(args.out_dir, f"lf_seed{args.seed}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
