#!/usr/bin/env python3
"""
scripts/31_lp_cross_vs_xhii.py
================================
D_ell^cross against the volume-averaged ionised fraction x_HII (La Plante+22 Fig. 7, right panel) instead of z0.
Reads the CSV written by scripts/28_lp_cross_vs_z0.py (columns z0, dz, x_hii, mean_l*, sigma_mean_l*, ...) -- no
cluster access needed, run it anywhere, e.g. on the desktop after copying the CSV:

    python scripts/31_lp_cross_vs_xhii.py \\
        --csv data/plots/2026-10-10_lp_cross_10seeds/lp_cross_vs_z0_SO_wo0.csv \\
        --out-dir data/plots/2026-10-10_lp_cross_10seeds

Ours: seed-mean +- s.e.m. per galaxy window (x_HII = seed-mean x_HII of the window's z0).  Paper: the FIDUCIAL
solid curves of their Fig. 7 (data/reference/la_plante_2022/fig7_dell_vs_xhii_fiducial_digitized.csv).  The paper's
curves come from ONE history (theirs); ours from our own x_HII(z), so a shift in x_HII at fixed z0 is not an error.
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.la_plante_reference import load_fig7_vs_xhii
from ksz_lae_xcorr.utils.figio import save_fig

BLUE, ORANGE = "#2a78d6", "#eb6834"


def load_cross_csv(path: str) -> dict:
    a = np.genfromtxt(path, delimiter=",", names=True)
    return {n: a[n] for n in a.dtype.names}


def make_figure(cols: dict, paper: dict, dz: float = 1.0, title: str = ""):
    ells = sorted(int(n.split("_l")[1]) for n in cols if n.startswith("mean_l"))
    sel = np.isclose(cols["dz"], dz)
    fig, axes = plt.subplots(1, len(ells), figsize=(5 * len(ells), 4), sharex=True)
    for ax, ell in zip(np.atleast_1d(axes), ells):
        if ell in paper:
            ax.plot(paper[ell][0], paper[ell][1], color=ORANGE, lw=2, label="La Plante+22 Fig. 7 (Fiducial, digitised)")
        x, m, e = cols["x_hii"][sel], cols[f"mean_l{ell}"][sel], cols[f"sigma_mean_l{ell}"][sel]
        ax.errorbar(x, m, yerr=e, color=BLUE, marker="o", ms=3, ls="-", lw=1, label="ours: seed mean $\\pm$ s.e.m.")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(1.0, 0.0)                      # paper convention: fully ionised on the left
        ax.set_title(rf"$\ell={ell}$  {title}")
        ax.set_xlabel(r"$x_{\rm HII}$")
        ax.set_ylabel(r"$D_\ell^{\rm cross}\ [\mu{\rm K}^2]$")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="lp_cross_vs_z0_*.csv from scripts/28")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--title", default="(SO)")
    args = ap.parse_args()
    cols = load_cross_csv(args.csv)
    paper = load_fig7_vs_xhii()
    fig = make_figure(cols, paper, args.dz, args.title)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, os.path.basename(args.csv).replace("lp_cross_vs_z0", "lp_cross_vs_xhii").rsplit(".", 1)[0])
    print("figure:", *save_fig(fig, base))
    # numbers: ours vs paper at the paper's peak and at x_HII ~ 0.43
    sel = np.isclose(cols["dz"], args.dz)
    for ell in sorted(paper):
        xp, dp = paper[ell]
        ours_x = cols["x_hii"][sel]; ours_m = cols[f"mean_l{ell}"][sel]; ours_e = cols[f"sigma_mean_l{ell}"][sel]
        print(f"ell={ell}: paper peak {dp.max():.4f} at x_HII={xp[dp.argmax()]:.2f}")
        for x0 in (0.6, 0.43, 0.25, 0.15):
            i = np.argmin(np.abs(ours_x - x0))
            print(f"   x_HII~{x0:.2f}: ours (x={ours_x[i]:.2f}) {ours_m[i]:+.4f} +- {ours_e[i]:.4f}   paper {np.interp(ours_x[i], xp, dp):+.4f}")


if __name__ == "__main__":
    sys.exit(main())
