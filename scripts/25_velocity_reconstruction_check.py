#!/usr/bin/env python3
"""
scripts/25_velocity_reconstruction_check.py
==============================================
"Baby step 1" of the velocity-reconstruction request (2026-09-30, per
arXiv:2609.36355's Eqs. 11-12): reconstruct the LOS peculiar velocity
from ONE real coeval snapshot's own density field, using the linear
continuity-equation relation (NOT machine learning -- see
correlation/velocity_reconstruction.py's module docstring for the exact
formula and its provenance), and compare it against that SAME snapshot's
own native velocity_z field -- the one already confirmed physical this
session (correlation/velocity_convention_check.py).

This is a validation step, not yet a kSZ computation: does the
reconstruction actually recover the true velocity field, and on what
scales? Only once this looks sensible does it make sense to build a
"reconstructed velocity kSZ auto power" on top of it (baby step 2).

Reports:
  - power spectra: P_reconstructed(k), P_native(k), P_cross(k)
  - the scale-dependent correlation coefficient r(k) (the main fidelity
    diagnostic -- r=1 is perfect recovery at that k, r=0 is none)
  - the amplitude transfer function sqrt(P_rec/P_native)(k)
  - a pixel-level 2D histogram of reconstructed vs. native velocity
    values, plus the single-number Pearson correlation coefficient

Usage:
    python scripts/25_velocity_reconstruction_check.py --seed 1
    python scripts/25_velocity_reconstruction_check.py --seed 1 --z-index 30 --r-smooth 8.0
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation.velocity_reconstruction import (
    compare_reconstructed_to_native,
    reconstruct_velocity_los,
)
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z-index", type=int, default=None,
                         help="Index into this seed's sorted real snapshot redshifts "
                              "(default: the middle snapshot).")
    parser.add_argument("--bias", type=float, default=1.0,
                         help="Linear bias applied to the input field before "
                              "reconstructing (default 1.0 -- this script uses the "
                              "matter density field directly, so bias=1 is the "
                              "physically correct choice unless --tracer is used).")
    parser.add_argument("--r-smooth", type=float, default=None,
                         help="Optional Gaussian smoothing scale in Mpc (paper uses "
                              "12.5 h^-1 Mpc =~ 18.4 Mpc for this repo's h=0.6777). "
                              "Default: no smoothing.")
    parser.add_argument("--n-kbins", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    logger = setup_logger(args.seed, cfg.paths.lightcone_root)
    snap_z = stitcher.get_snapshot_redshifts(args.seed, logger)
    z_index = args.z_index if args.z_index is not None else len(snap_z) // 2
    z = float(snap_z[z_index])
    print(f"Seed {args.seed}: using snapshot z={z:.4f} "
          f"(index {z_index}/{len(snap_z)-1} of the real snapshot list)")

    print("Loading native density and velocity_z fields for this snapshot...")
    density_1pdelta = stitcher.load_field_box(args.seed, z, "density")
    v_native = stitcher.load_field_box(args.seed, z, "vz").astype(np.float64)
    delta = density_1pdelta.astype(np.float64) - 1.0
    delta -= delta.mean()  # exact zero-mean, matching reconstruct_velocity_los's assumption

    print(f"  density grid shape={delta.shape}, std(delta)={delta.std():.4f}")
    print(f"  native v_z: min={v_native.min():.3e}  max={v_native.max():.3e}  "
          f"std={v_native.std():.3e} Mpc/s")

    print(f"Reconstructing v_los (bias={args.bias}, r_smooth={args.r_smooth})...")
    v_rec = reconstruct_velocity_los(cfg, delta, cfg.box.box_len_mpc, z,
                                      bias=args.bias, r_smooth_mpc=args.r_smooth)
    print(f"  reconstructed v_z: min={v_rec.min():.3e}  max={v_rec.max():.3e}  "
          f"std={v_rec.std():.3e} Mpc/s")

    result = compare_reconstructed_to_native(v_rec, v_native, cfg.box.box_len_mpc,
                                              n_kbins=args.n_kbins)
    print(f"\nPixel-level Pearson r (real space, all cells): {result['pixel_pearson_r']:.4f}")
    finite = np.isfinite(result["r_k"])
    if finite.any():
        i_lo, i_hi = np.where(finite)[0][[0, -1]]
        print(f"r(k): k={result['k_centers'][i_lo]:.3f} Mpc^-1 -> r={result['r_k'][i_lo]:.3f}   "
              f"k={result['k_centers'][i_hi]:.3f} Mpc^-1 -> r={result['r_k'][i_hi]:.3f}")

    # --- Plot: power spectra, r(k)/transfer(k), pixel-level comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    ax = axes[0]
    k = result["k_centers"]
    ax.loglog(k, np.abs(result["P_native"]), "o-", label=r"$P_{\rm native}$", color="black")
    ax.loglog(k, np.abs(result["P_rec"]), "s-", label=r"$P_{\rm reconstructed}$", color="tab:blue")
    ax.loglog(k, np.abs(result["P_cross"]), "^-", label=r"$|P_{\rm cross}|$", color="tab:orange")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P(k)$ [(Mpc/s)$^2$ Mpc$^3$]")
    ax.set_title(f"Power spectra (seed {args.seed}, z={z:.2f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(k, result["r_k"], "o-", color="tab:green", label=r"$r(k)$")
    ax.plot(k, result["transfer_k"], "s-", color="tab:purple", label=r"transfer$(k)=\sqrt{P_{\rm rec}/P_{\rm native}}$")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel("value")
    ax.set_title("Reconstruction fidelity")
    ax.legend(fontsize=8)

    ax = axes[2]
    vr = v_rec.ravel()
    vn = v_native.ravel()
    lim = max(np.percentile(np.abs(vr), 99), np.percentile(np.abs(vn), 99))
    ax.hist2d(vn, vr, bins=80, range=[[-lim, lim], [-lim, lim]], cmap="viridis")
    ax.plot([-lim, lim], [-lim, lim], color="red", lw=1, ls="--", label="1:1")
    ax.set_xlabel(r"$v_{z,\rm native}$ [Mpc/s]")
    ax.set_ylabel(r"$v_{z,\rm reconstructed}$ [Mpc/s]")
    ax.set_title(f"Pixel-level (Pearson r={result['pixel_pearson_r']:.3f})")
    ax.legend(fontsize=8)

    fig.suptitle(f"Linear velocity reconstruction check -- seed {args.seed}, z={z:.3f}, "
                 f"bias={args.bias}, r_smooth={args.r_smooth}", fontsize=11)
    outpath = os.path.join(args.out_dir, f"velocity_reconstruction_check_seed{args.seed}_z{z:.2f}.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"\nSaved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
