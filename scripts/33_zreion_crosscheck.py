#!/usr/bin/env python3
"""
scripts/33_zreion_crosscheck.py
=================================
Cross-check src/ksz_lae_xcorr/zreion/box.py::apply_zreion against the PUBLIC `zreion` package (P. La Plante, MIT,
https://pypi.org/project/zreion) on the SAME density field, for the four (rsmooth, deconvolve) settings, and print
the x_HII(z) history of each against La Plante+22 Fig. 6 (digitised).

The public package needs Python <= 3.10, so run this in the throwaway env (NOT the main env):

    conda run -n zr310 pip install scipy matplotlib        # once
    PYTHONPATH=src /home/$USER/miniconda3/envs/zr310/bin/python scripts/33_zreion_crosscheck.py --n 256

The density is built with OUR code (Gaussian + Zel'dovich + TSC, same as scripts/32).
"""
import argparse
import os
import sys
import time

import numpy as np

import zreion as zr_pkg                                   # the public package (fails loudly if missing)
from ksz_lae_xcorr.io.la_plante_reference import load_reionization_histories
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio
from ksz_lae_xcorr.zreion import (LP22_COSMO, apply_zreion, gaussian_delta, linear_pk, xhii_history,
                                  zeldovich_density)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--cell-hmpc", type=float, default=2000.0 / 1024.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zmean", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k0", type=float, default=0.9)
    ap.add_argument("--lpt", choices=("zeldovich", "linear"), default="zeldovich")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    args = ap.parse_args()
    c = LP22_COSMO; n = args.n; h = c["h"]
    L_hmpc = n * args.cell_hmpc; L_mpc = L_hmpc / h
    print(f"n={n} box={L_hmpc:.1f} h^-1 Mpc ({L_mpc:.1f} Mpc); zreion package {getattr(zr_pkg, '__version__', '?')}")
    pk = linear_pk(backend="eh")
    d0, dk0 = gaussian_delta(n, L_mpc, pk, seed=args.seed)
    D = np.float32(growth_factor_ratio(c["Om"], args.zmean))
    dm = zeldovich_density(dk0 * D, n, L_mpc) if args.lpt == "zeldovich" else (d0 * D).astype(np.float32)
    del dk0, d0
    print(f"delta_m: std={dm.std():.3f} mean={dm.mean():.1e} min={dm.min():.3f}")
    zg = np.arange(4.0, 16.01, 0.25)
    ref = load_reionization_histories()["Fiducial"]
    zs = (6, 7, 8, 9, 10, 11, 12)
    print("\n paper Fiducial  :", "  ".join(f"z{z}={np.interp(z, ref['z'], ref['x_HII']):.3f}" for z in zs))
    for rs, dec in ((0.0, False), (1.0, False), (0.0, True), (1.0, True)):
        t0 = time.time()
        real = zr_pkg.apply_zreion_fast(dm.copy(), args.zmean, args.alpha, args.k0, L_hmpc, rsmooth=rs, deconvolve=dec)
        t_real = time.time() - t0
        mine = apply_zreion(dm, L_mpc, args.zmean, args.alpha, args.k0, rsmooth_hmpc=rs, deconvolve=dec)
        diff = np.abs(real.astype(np.float64) - mine)
        xr, xm = xhii_history(real, zg), xhii_history(mine, zg)
        print(f"\n rsmooth={rs:g} deconvolve={dec}:  max|z_re(real)-z_re(mine)| = {diff.max():.2e}  rms = {np.sqrt((diff ** 2).mean()):.2e}"
              f"   (z_re std real {real.std():.3f}, mine {mine.std():.3f};  real pkg {t_real:.1f} s)")
        print("   real pkg :", "  ".join(f"z{z}={np.interp(z, zg, xr):.3f}" for z in zs))
        print("   ours     :", "  ".join(f"z{z}={np.interp(z, zg, xm):.3f}" for z in zs))
        zm = float(np.interp(0.5, xr[::-1], zg[::-1])); z25 = float(np.interp(0.75, xr[::-1], zg[::-1])); z75 = float(np.interp(0.25, xr[::-1], zg[::-1]))
        print(f"   real pkg: x_HII=0.5 at z={zm:.2f}, 25-75% duration {z75 - z25:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
