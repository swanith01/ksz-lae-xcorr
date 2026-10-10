#!/usr/bin/env python3
"""
scripts/34_zreion_on_21cmfast.py
===================================
Apply the zreion model (src/ksz_lae_xcorr/zreion, verified against the public `zreion` package) to the density
field of an EXISTING py21cmfast seed, and compare the resulting ionisation field with 21cmFAST's own:

  * x_HII(z): 21cmFAST vs zreion (same z) vs La Plante+22 Fig. 6 (digitised),
  * MORPHOLOGY at equal mean x_HII (zreion thresholded to 21cmFAST's x_HII at each z): cross-correlation coefficient
    r(k) and power ratio P_zre/P_21cmFAST of the neutral-fraction fields, plus a slice figure.

Optionally (--write-root) writes a parallel coeval root in which ONLY neutral_fraction.npy is replaced by the zreion
field (density and velocity files are symlinks to the originals), so the existing pipeline can be rerun unchanged with
    configs/variants/zreion_xh.yaml   (coeval_root -> that root, products_root -> data/products_zreion).

    python scripts/34_zreion_on_21cmfast.py --seed 1                        # compare only (light: ~2-3 GB, minutes)
    python scripts/34_zreion_on_21cmfast.py --seed 1 --write-root /user1/swanith/zreion_coeval --match z

Run on the cluster (compute node) -- the boxes live there.  Density: z=0-normalised IC (hires_density) scaled to
z_mean and Zel'dovich-displaced (--lpt za) or left linear (--lpt linear).  --coarsen K block-averages the IC by K first
(the paper's cells are 2 h^-1 Mpc = 2.9 Mpc, ours 1 Mpc): zreion's history depends on the cell size.
"""
import argparse
import logging
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.la_plante_reference import load_reionization_histories
from ksz_lae_xcorr.lightcone.stitch import Stitcher
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig
from ksz_lae_xcorr.utils.grid import block_average_downsample
from ksz_lae_xcorr.zreion.convert import (binned_spectra, write_zreion_coeval_root, xhi_from_zre,
                                          xhi_quantile_matched, zre_from_ic)

K_REPORT = (0.05, 0.15, 0.5)          # Mpc^-1
Z_REPORT = (6.0, 7.0, 8.0, 9.0, 10.0, 12.0)


def nearest(zs, z):
    return float(zs[int(np.argmin(np.abs(zs - z)))])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zmean", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k0", type=float, default=0.9, help="h/Mpc")
    ap.add_argument("--rsmooth", type=float, default=1.0, help="h^-1 Mpc (package default 1.0)")
    ap.add_argument("--no-deconvolve", action="store_true", help="package default is deconvolve=True")
    ap.add_argument("--lpt", choices=("za", "linear"), default="za")
    ap.add_argument("--coarsen", type=int, default=1, help="block-average the IC by this factor before zreion")
    ap.add_argument("--write-root", default=None, help="write a parallel coeval root with zreion neutral fractions")
    ap.add_argument("--match", choices=("z", "quantile"), default="z",
                    help="for --write-root: 'z' = pure zreion x(z); 'quantile' = zreion morphology at 21cmFAST's x_HII(z)")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("zre"); t0 = time.time()
    cfg = load_config(args.config)
    st = Stitcher(cfg, field_convention="legacy")
    h = cfg.cosmology.H0 / 100.0; Om0 = float(cfg.cosmology.Om0)
    L, n = float(cfg.box.box_len_mpc), int(cfg.box.hii_dim)
    snap = st.get_snapshot_redshifts(args.seed, log)
    zic = nearest(snap, args.zmean)
    print(f"seed {args.seed}: {len(snap)} snapshots z={snap.min():.2f}-{snap.max():.2f}; IC density from z={zic:.3f}; "
          f"box {L:.0f} Mpc, {n}^3 cells; zreion zmean={args.zmean} alpha={args.alpha} k0={args.k0} rsmooth={args.rsmooth} "
          f"deconvolve={not args.no_deconvolve} lpt={args.lpt} coarsen={args.coarsen}", flush=True)

    ic = np.asarray(st.load_field_box(args.seed, zic, "density"), dtype=np.float32)       # (n,n,n) block-averaged IC
    if len(snap) > 1:                                                                      # IC should not depend on z
        z2 = nearest(snap, zic + 3.0 if zic + 3.0 <= snap.max() else zic - 3.0)
        ic2 = np.asarray(st.load_field_box(args.seed, z2, "density"), dtype=np.float32)
        print(f"  check: hires_density identical at z={zic:.2f} and z={z2:.2f}? rms diff / rms = "
              f"{np.sqrt(np.mean((ic - ic2) ** 2)) / ic.std():.2e}", flush=True)
        del ic2
    nc = n // args.coarsen
    ic_c = block_average_downsample(ic, nc) if args.coarsen > 1 else ic
    zre_c = zre_from_ic(ic_c, L, Om0, h, args.zmean, args.alpha, args.k0, args.rsmooth, not args.no_deconvolve, args.lpt)
    zre = np.repeat(np.repeat(np.repeat(zre_c, args.coarsen, 0), args.coarsen, 1), args.coarsen, 2) if args.coarsen > 1 else zre_c
    print(f"  z_re: mean={zre.mean():.3f} std={zre.std():.3f}  ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- history and morphology at the reporting redshifts
    ref = load_reionization_histories()["Fiducial"]
    zs_all = snap[(snap >= 5.0) & (snap <= 16.0)]
    hist = []
    for z in zs_all:
        x21 = 1.0 - float(np.mean(st.load_field_box(args.seed, float(z), "xH")))
        xzr = 1.0 - float(np.mean(xhi_from_zre(zre, float(z))))
        hist.append((float(z), x21, xzr))
    hist = np.array(hist)
    print("\n   z     x_HII(21cmFAST)  x_HII(zreion)  x_HII(paper Fig.6)")
    for zr in Z_REPORT:
        i = int(np.argmin(np.abs(hist[:, 0] - zr)))
        print(f"  {hist[i, 0]:5.2f}    {hist[i, 1]:6.3f}          {hist[i, 2]:6.3f}        {np.interp(hist[i, 0], ref['z'], ref['x_HII']):6.3f}")

    print("\nMorphology at EQUAL mean x_HII (zreion thresholded to 21cmFAST's x_HII at each z):")
    print("   z    x_HII   " + "  ".join(f"r(k={k})  Pzre/P21(k={k})" for k in K_REPORT))
    morph = {}
    for zr in Z_REPORT:
        zz = nearest(snap, zr)
        xh21 = np.asarray(st.load_field_box(args.seed, zz, "xH"), dtype=np.float32)
        x_t = 1.0 - float(xh21.mean())
        if x_t < 0.02 or x_t > 0.98:
            print(f"  {zz:5.2f}  {x_t:6.3f}   (too close to 0 or 1 to compare)")
            continue
        xz = xhi_quantile_matched(zre, x_t)
        sp = binned_spectra(1.0 - xh21, 1.0 - xz, L)
        morph[zz] = (sp, xh21, xz)
        cols = []
        for k in K_REPORT:
            j = int(np.nanargmin(np.abs(sp["k"] - k)))
            cols.append(f"{sp['r'][j]:6.3f}   {sp['Pbb'][j] / sp['Paa'][j]:8.3f}      ")
        print(f"  {zz:5.2f}  {x_t:6.3f}   " + "  ".join(cols))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"seed{args.seed}_zm{args.zmean:g}_a{args.alpha:g}_rs{args.rsmooth:g}_{args.lpt}" + (f"_c{args.coarsen}" if args.coarsen > 1 else "")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(ref["z"], ref["x_HII"], color="0.55", lw=2, label="La Plante+22 Fiducial (digitised)")
    ax.plot(hist[:, 0], hist[:, 1], "o-", ms=3, color="C0", label="21cmFAST (this seed)")
    ax.plot(hist[:, 0], hist[:, 2], "s-", ms=3, color="C1", label="zreion on same density")
    ax.set_xlim(5, 14); ax.set_xlabel("z"); ax.set_ylabel(r"$x_{\rm HII}$"); ax.legend(fontsize=7)
    ax = axes[1]
    zpick = nearest(np.array(sorted(morph)), 8.0) if morph else None
    if zpick is not None:
        sp, xh21, xz = morph[zpick]
        iy = xh21.shape[1] // 2
        ax.imshow(np.concatenate([xh21[:, iy, :], np.full((xh21.shape[0], 4), 0.5), xz[:, iy, :]], axis=1),
                  cmap="gray_r", origin="lower", interpolation="nearest")
        ax.set_title(f"x_HI slice, z={zpick:.2f}: 21cmFAST (left) | zreion, same mean x (right)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]
    for zz, (sp, _, _) in sorted(morph.items()):
        ax.semilogx(sp["k"], sp["r"], marker="o", ms=3, label=f"z={zz:.1f}")
    ax.axhline(1, color="k", lw=0.4); ax.set_ylim(-0.1, 1.05); ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$r(k)$ between $x_{\rm HII}$ fields"); ax.legend(fontsize=7)
    fig.tight_layout()
    print("\nfigure:", *save_fig(fig, os.path.join(args.out_dir, f"zreion_vs_21cmfast_{tag}")))
    np.savez(os.path.join(args.out_dir, f"zreion_vs_21cmfast_{tag}.npz"), hist=hist,
             **{f"r_z{zz:.2f}": sp["r"] for zz, (sp, _, _) in morph.items()}, k=next(iter(morph.values()))[0]["k"] if morph else [])

    if args.write_root:
        if args.match == "z":
            fn = lambda z: xhi_from_zre(zre, z)                                                    # noqa: E731
        else:
            fn = lambda z: xhi_quantile_matched(zre, 1.0 - float(np.mean(st.load_field_box(args.seed, z, "xH"))))   # noqa: E731
        zs = write_zreion_coeval_root(cfg.paths.coeval_root, args.write_root, args.seed, fn,
                                      z_range=(st.z_min - 0.5, st.z_max + 0.5), log=print)
        print(f"  -> use configs/variants/zreion_xh.yaml (edit coeval_root if you chose another --write-root); match={args.match}")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
