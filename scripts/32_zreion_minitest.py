#!/usr/bin/env python3
"""
scripts/32_zreion_minitest.py
================================
Mini test of the zreion box (src/ksz_lae_xcorr/zreion/box.py): can we reproduce the La Plante+22 fiducial
reionisation history x_HII(z) (their Fig. 6, digitised in reionization_histories_digitized.csv) from a
Gaussian field + Zel'dovich density + zreion?  Also MEASURES time and peak memory so the full 2 h^-1 Gpc /
1024^3 box can be sized before it is attempted.

Default box: n = 256 cells of the paper's cell size (2000/1024 h^-1 Mpc = 1.953 h^-1 Mpc), i.e. a 0.5 h^-1 Gpc
sub-volume -- same resolution as the paper, 64x smaller volume.  The mean history depends on the cell size
(through the variance and skewness of delta_m), so keep --cell-hmpc at the paper value.

    python scripts/32_zreion_minitest.py                       # n=256, Zel'dovich, ~minutes, < 8 GB
    python scripts/32_zreion_minitest.py --n 128               # quicker
    python scripts/32_zreion_minitest.py --lpt linear          # no displacement: x_HII(z_mean) = 0.5 exactly
"""
import argparse
import os
import resource
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.la_plante_reference import load_reionization_histories
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio
from ksz_lae_xcorr.utils.figio import save_fig
from ksz_lae_xcorr.zreion import (LP22_COSMO, apply_zreion, gaussian_delta, linear_pk, xhii_history,
                                  zeldovich_density)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0     # KB -> GB (Linux)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--cell-hmpc", type=float, default=2000.0 / 1024.0, help="cell size [h^-1 Mpc] (paper: 1.953)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lpt", choices=("zeldovich", "linear"), default="zeldovich")
    ap.add_argument("--zmean", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k0", type=float, default=0.9, help="h Mpc^-1")
    ap.add_argument("--rsmooth", type=float, default=0.0, help="top-hat smoothing of z_re [h^-1 Mpc] (zreion package default: 1.0)")
    ap.add_argument("--deconvolve", action="store_true", help="divide by the CIC window (zreion package default: on)")
    ap.add_argument("--backend", choices=("auto", "camb", "eh"), default="auto")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    args = ap.parse_args()

    c = LP22_COSMO
    n = args.n
    boxsize = n * args.cell_hmpc / c["h"]                    # Mpc
    t0 = time.time(); T = {}
    def lap(name):
        T[name] = time.time() - t0; print(f"  [{T[name]:7.1f} s | peak {peak_gb():5.2f} GB] {name}", flush=True)

    print(f"n={n}  cell={args.cell_hmpc:.4f} h^-1 Mpc  box={boxsize:.1f} Mpc ({boxsize * c['h'] / 1000:.3f} h^-1 Gpc)  "
          f"lpt={args.lpt}  zmean={args.zmean} alpha={args.alpha} k0={args.k0} rsmooth={args.rsmooth} deconvolve={args.deconvolve}")
    pk = linear_pk(backend=args.backend); lap(f"linear P(k) [{args.backend}]")
    d0, dk0 = gaussian_delta(n, boxsize, pk, seed=args.seed); lap("Gaussian field")
    D = float(growth_factor_ratio(c["Om"], args.zmean))
    if args.lpt == "zeldovich":
        dm = zeldovich_density(dk0 * np.float32(D), n, boxsize); lap("Zel'dovich displacement + TSC")
    else:
        dm = (d0 * np.float32(D)).astype(np.float32); lap("linear density")
    del dk0
    zre = apply_zreion(dm, boxsize, zmean=args.zmean, alpha=args.alpha, k0_hmpc=args.k0,
                       rsmooth_hmpc=args.rsmooth, deconvolve=args.deconvolve); lap("zreion field")
    zg = np.arange(4.0, 16.01, 0.25)
    x = xhii_history(zre, zg); lap("history")

    sk = float(((dm - dm.mean()) ** 3).mean() / dm.std() ** 3)
    print(f"\ndelta_m(z={args.zmean}): std={dm.std():.3f} skew={sk:.2f} P(delta_m>0)={(dm > 0).mean():.3f};  "
          f"z_re: mean={zre.mean():.3f} std={zre.std():.3f} min={zre.min():.2f} max={zre.max():.2f}")
    ref = load_reionization_histories()
    print("\n  z    ours   paper-Fiducial   (digitised Fig. 6)")
    fz = ref["Fiducial"]
    for z in (6, 7, 8, 9, 10, 11, 12):
        xi = float(np.interp(z, zg, x)); xp = float(np.interp(z, fz["z"], fz["x_HII"]))
        print(f"  {z:2d}   {xi:5.3f}   {xp:5.3f}   diff {xi - xp:+.3f}")
    # reionisation midpoint / duration
    zmid = float(np.interp(0.5, x[::-1], zg[::-1])); z25 = float(np.interp(0.75, x[::-1], zg[::-1])); z75 = float(np.interp(0.25, x[::-1], zg[::-1]))
    zmid_p = float(np.interp(0.5, fz["x_HII"][::-1] if fz["x_HII"][0] > fz["x_HII"][-1] else fz["x_HII"],
                             fz["z"][::-1] if fz["x_HII"][0] > fz["x_HII"][-1] else fz["z"]))
    print(f"\nours: x_HII=0.5 at z={zmid:.2f}, 25-75% duration dz={z75 - z25:.2f};  paper Fiducial: x_HII=0.5 at z~{zmid_p:.2f}")

    scale = (1024 / n) ** 3
    print(f"\nRESOURCES  time {T['history']:.0f} s, peak memory {peak_gb():.2f} GB at n={n}.")
    print(f"  -> naive extrapolation to n=1024 (x{scale:.0f}): ~{T['history'] * scale / 3600:.1f} h, ~{peak_gb() * scale:.0f} GB "
          f"(the FFTs scale as n^3 log n and memory is dominated by float32 n^3 fields: plan on this as a lower bound)")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, col, ls in (("Fiducial", "C1", "-"), ("Early", "0.5", "--"), ("Short", "0.5", ":")):
        r = ref[name]; ax.plot(r["z"], r["x_HII"], color=col, ls=ls, lw=1.5, label=f"La Plante+22 {name} (digitised)")
    ax.plot(zg, x, color="C0", lw=2, label=f"ours: zreion, {args.lpt}, n={n}")
    ax.set_xlim(5, 14); ax.set_ylim(0, 1.02); ax.set_xlabel("z"); ax.set_ylabel(r"$x_{\rm HII}$"); ax.legend(fontsize=7)
    fig.tight_layout()
    tag = f"n{n}_{args.lpt}_seed{args.seed}" + (f"_rs{args.rsmooth:g}" if args.rsmooth else "") + ("_deconv" if args.deconvolve else "")
    print("figure:", *save_fig(fig, os.path.join(args.out_dir, f"zreion_minitest_{tag}")))
    np.savez(os.path.join(args.out_dir, f"zreion_minitest_{tag}.npz"), z=zg, x_HII=x)


if __name__ == "__main__":
    sys.exit(main())
