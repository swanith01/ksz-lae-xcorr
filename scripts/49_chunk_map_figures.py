#!/usr/bin/env python3
"""
scripts/49_chunk_map_figures.py
=================================
Look at the maps behind D_ell^cross(x_HII).  Reads scripts/48's pickle (one seed) and writes, in --out-dir,

  maps_estimator.png/.pdf   rows = redshift windows (z0, x_HII).  Column 1: the filtered kSZ^2 map of the WHOLE lightcone (the
                            same map in every row: it is what the estimator cross-correlates).  Other columns: the tracer map
                            of that window (matter x Waters+16, halo, LAE, LBG).  Every panel is smoothed on the scale of the
                            soft mode (l = --ell, i.e. sigma = chi_eff / l) and shown as a fractional fluctuation; the number in
                            the corner is the Pearson correlation of the panel with the kSZ^2 map of column 1.
  maps_chunk_<tracer>.png   the same window, but the kSZ^2 map is built from the window's own line-of-sight pixels:
                            full | x_e only | v only | tracer.  The numbers are the correlations of each kSZ^2 map with the
                            tracer map: which ingredient's pattern the tracer follows, and in which redshift window.
  maps_correlations.csv     all the numbers.

    python scripts/49_chunk_map_figures.py --chunks data/plots/2026-10-23_maps/seed1_wo0_chunks.pkl --out-dir data/plots/2026-10-23_maps
Options: --ell 1000, --order A|B (project-then-square or square-then-project for the chunk maps), --tracers a,b,c, --cell 1.0.
Reading the pictures: D_ell^cross at this ell is a (scale-dependent) cross-power, the corner Pearson number is its map-level,
dimensionless cousin: positive where the bright patches of the kSZ^2 map sit on over-dense tracer patches.
"""
import argparse
import csv
import os
import pickle
import sys

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.utils.figio import save_fig

TRACER_TITLES = {"matter_Waters": "matter $\\times$ Waters+16 bias"}


def smooth_frac(m, sigma_pix):
    """Fractional fluctuation (m / mean - 1 for the positive kSZ^2 maps, m - mean for the tracer overdensity maps) smoothed with a
    periodic Gaussian of width sigma_pix."""
    s = gaussian_filter(np.asarray(m, float), sigma_pix, mode="wrap")
    return s - s.mean()


def kSZ2_frac(S, sigma_pix):
    S = np.asarray(S, float)
    return smooth_frac(S / S.mean() - 1.0, sigma_pix)


def pearson(a, b):
    a = np.ravel(a) - np.mean(a)
    b = np.ravel(b) - np.mean(b)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _show(ax, img, vmax, title=None, corner=None):
    ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if corner:
        ax.text(0.04, 0.95, corner, transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def _vmax(img, q=98):
    return float(np.percentile(np.abs(img), q)) or 1.0


def sigma_pix(chi_eff, ell, cell):
    return (chi_eff / ell) / cell


def fig_estimator(d, tracers, ell, cell):
    n = len(d["windows"])
    fig, axes = plt.subplots(n, 1 + len(tracers), figsize=(2.5 * (1 + len(tracers)), 2.5 * n), squeeze=False, constrained_layout=True)
    rows = []
    for r, w in enumerate(d["windows"]):
        sg = sigma_pix(w["chi_eff"], ell, cell)
        S = kSZ2_frac(d["S_global"], sg)
        _show(axes[r, 0], S, _vmax(S), "whole-lightcone kSZ$^2$\n(estimator input)" if r == 0 else None)
        axes[r, 0].set_ylabel(f"$z_0$={w['z0']:g}\n$x_{{\\rm HII}}$={w['x_hii']:.2f}", fontsize=9)
        for c, t in enumerate(tracers):
            g = smooth_frac(d["gal"][t][r], sg)
            rr = pearson(S, g)
            rows.append({"window_z0": w["z0"], "x_hii": w["x_hii"], "tracer": t, "kind": "S_global", "pearson": rr})
            _show(axes[r, 1 + c], g, _vmax(g), (TRACER_TITLES.get(t, t) if r == 0 else None), f"r = {rr:+.2f}")
    fig.suptitle(f"smoothed at $\\ell$ = {ell:g}; colour = fractional fluctuation (red = above mean); r = Pearson correlation with the kSZ$^2$ map",
                 fontsize=9)
    return fig, rows


def fig_chunk(d, tracer, ell, cell, order):
    n = len(d["windows"])
    cols = [("full", "full kSZ$^2$ of this window"), ("x", "$x_e$ only"), ("v", "$v_{\\rm los}$ only")]
    fig, axes = plt.subplots(n, 4, figsize=(2.5 * 4, 2.5 * n), squeeze=False, constrained_layout=True)
    rows = []
    for r, w in enumerate(d["windows"]):
        sg = sigma_pix(w["chi_eff"], ell, cell)
        g = smooth_frac(d["gal"][tracer][r], sg)
        for c, (comp, title) in enumerate(cols):
            S = kSZ2_frac(w["maps"][f"{order}_{comp}"], sg)
            rr = pearson(S, g)
            rows.append({"window_z0": w["z0"], "x_hii": w["x_hii"], "tracer": tracer, "kind": f"{order}_{comp}", "pearson": rr})
            _show(axes[r, c], S, _vmax(S), title if r == 0 else None, f"r = {rr:+.2f}")
        _show(axes[r, 3], g, _vmax(g), (TRACER_TITLES.get(tracer, tracer) if r == 0 else None))
        axes[r, 0].set_ylabel(f"$z_0$={w['z0']:g}\n$x_{{\\rm HII}}$={w['x_hii']:.2f}", fontsize=9)
    fig.suptitle(f"kSZ$^2$ from this window's own pixels, {'project, then square' if order == 'A' else 'square, then project'}; "
                 f"smoothed at $\\ell$ = {ell:g}; r = correlation with the tracer map (right)", fontsize=9)
    return fig, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True, help="seed*_wo*_chunks.pkl from scripts/48")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ell", type=float, default=1000.0)
    ap.add_argument("--order", choices=("A", "B"), default="A")
    ap.add_argument("--tracers", default=None, help="comma-separated; default: all in the file")
    ap.add_argument("--chunk-tracers", default="lae_L42.5,matter_Waters", help="tracers that get their own chunk figure")
    ap.add_argument("--cell", type=float, default=1.0, help="Mpc per map pixel (300 Mpc / 300)")
    args = ap.parse_args(argv)
    with open(args.chunks, "rb") as f:
        d = pickle.load(f)
    out = args.out_dir or os.path.dirname(args.chunks) or "."
    os.makedirs(out, exist_ok=True)
    tracers = [t for t in (args.tracers.split(",") if args.tracers else list(d["gal"]))]
    tracers = [t for t in tracers if t in d["gal"]]
    print(f"seed {d['seed']}; windows z0 = {[w['z0'] for w in d['windows']]}; tracers {tracers}")
    fig, rows = fig_estimator(d, tracers, args.ell, args.cell)
    print("figure:", *save_fig(fig, os.path.join(out, "maps_estimator")))
    plt.close(fig)
    for t in [c for c in args.chunk_tracers.split(",") if c in d["gal"]]:
        fig, r2 = fig_chunk(d, t, args.ell, args.cell, args.order)
        print("figure:", *save_fig(fig, os.path.join(out, f"maps_chunk_{t}")))
        plt.close(fig)
        rows += r2
    with open(os.path.join(out, "maps_correlations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n  {'tracer':>16} {'kSZ^2 map':>10} " + " ".join(f"z0={w['z0']:<5g}" for w in d["windows"]))
    keys = sorted({(r["tracer"], r["kind"]) for r in rows})
    for t, k in keys:
        v = {r["window_z0"]: r["pearson"] for r in rows if r["tracer"] == t and r["kind"] == k}
        print(f"  {t:>16} {k:>10} " + " ".join(f"{v.get(w['z0'], np.nan):+8.2f}" for w in d["windows"]))


if __name__ == "__main__":
    sys.exit(main())
