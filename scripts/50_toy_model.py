#!/usr/bin/env python3
"""
scripts/50_toy_model.py
=========================
An analytic TOY for the shape of D_ell^cross(x_HII) (correlation/toy_model.py), laid over
  (1) La Plante+22 Fig. 7 (fiducial, ell = 500/1000/3000), and, if you give --ours-root,
  (2) our own measured curves (scripts/28 CSVs, one folder per tracer).

The toy has one shape parameter eps = (patchy-ionisation power) / (homogeneous velocity power) of the small-scale kSZ^2 of a slice,
and one amplitude per ell.  eps is chosen once, by the best match to the La Plante curve over x_HII = 0.05-0.9 (--eps to fix it).
Two pieces of the response of a slice to the long-wavelength density are drawn separately:
    dashed  : the density / velocity response of the kSZ^2 power of the slice (with the same ionised fraction);
    dotted  : the ionisation response (the ionised fraction of the slice is shifted by the large-scale density, zreion).

    python scripts/50_toy_model.py --out-dir data/plots/2026-10-23_toy
    python scripts/50_toy_model.py --ours-root data/plots/2026-10-16_tracer_cross --out-dir data/plots/2026-10-23_toy

Prints the eps scan, the amplitudes, and the x_HII of the peak of toy and data.
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation import toy_model as tm
from ksz_lae_xcorr.io.la_plante_reference import load_fig7_vs_xhii, load_reionization_histories
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig

LP_COLOR, TOY_COLOR, OURS_COLOR = "#D55E00", "#222222", "#0072B2"
TAG = "lp_cross_vs_z0_SO_wo0.csv"
X_RANGE = (0.1, 0.9)


def log_amp(model, target):
    """Best log-amplitude (and rms log misfit) of target ~ A model, over the points where both are positive."""
    model = np.maximum(model, 1e-3 * np.max(model))              # the toy can dip below zero where the ionisation response flips sign
    ok = (model > 0) & (target > 0)
    if ok.sum() < 3:
        return float("nan"), float("inf")
    r = np.log(target[ok]) - np.log(model[ok])
    a = r.mean()
    return float(np.exp(a)), float(np.sqrt(np.mean((r - a) ** 2)))


def toy_vs_x(toy, z0, ell, dz=1.0):
    c = toy.curve(z0, dz, ell)
    o = np.argsort(c["x_hii"])
    return {k: v[o] for k, v in c.items()}


def lp_targets(paper, ells, grid=None):
    grid = np.linspace(*X_RANGE, 40) if grid is None else grid
    return grid, {ell: np.interp(grid, *paper[ell]) for ell in ells}


def scan_eps(cfg, xb, paper, ells, eps_grid, z0, zmean=8.0, r_v=0.0):
    """rms log misfit (sum over ell) of the toy vs La Plante for each eps."""
    grid, tgt = lp_targets(paper, ells)
    out = []
    for eps in eps_grid:
        toy = tm.Toy(cfg, xb, eps=eps, r_v=r_v, zmean=zmean)
        tot = 0.0
        for ell in ells:
            c = toy_vs_x(toy, z0, ell)
            m = np.interp(grid, c["x_hii"], c["D"])
            tot += log_amp(m, tgt[ell])[1] ** 2
        out.append(np.sqrt(tot / len(ells)))
    return np.array(out)


def peak_x(x, y):
    y = np.asarray(y, float)
    return float(np.asarray(x)[int(np.nanargmax(y))]) if np.isfinite(y).any() else float("nan")


def read_ours(path, ells):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {"z0": np.array([float(r["z0"]) for r in rows]), "x_hii": np.array([float(r["x_hii"]) for r in rows])}
    for ell in ells:
        if f"mean_l{ell}" in rows[0]:
            d[ell] = (np.array([float(r[f"mean_l{ell}"]) for r in rows]), np.array([float(r.get(f"sigma_mean_l{ell}", "nan")) for r in rows]))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--ells", default="500,1000,3000")
    ap.add_argument("--eps", type=float, default=None, help="fix eps (default: best match to La Plante)")
    ap.add_argument("--r-v", type=float, default=0.0)
    ap.add_argument("--zmean", type=float, default=8.0)
    ap.add_argument("--ours-root", default=None, help="folder with <tracer>/lp_cross_vs_z0_SO_wo0.csv (scripts/28)")
    ap.add_argument("--out-dir", default="data/plots/toy")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ells = [int(e) for e in args.ells.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    h = load_reionization_histories()["Fiducial"]
    xb = tm.history_fn(h["z"], h["x_HII"])
    paper = load_fig7_vs_xhii()
    z0 = np.arange(5.6, 14.01, 0.2)
    eps_grid = np.logspace(-1, 2.5, 22)
    mis = scan_eps(cfg, xb, paper, ells, eps_grid, z0, args.zmean, args.r_v)
    eps = args.eps if args.eps is not None else float(eps_grid[int(np.argmin(mis))])
    print("eps scan (rms log misfit vs La Plante, x_HII 0.1-0.9):")
    for e, m in zip(eps_grid, mis):
        print(f"   eps = {e:8.2f}   {m:.2f}" + ("   <-- used" if abs(e - eps) < 1e-9 else ""))
    toy = tm.Toy(cfg, xb, eps=eps, r_v=args.r_v, zmean=args.zmean)

    # ---- (1) La Plante ------------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(ells) + 1, figsize=(4.2 * (len(ells) + 1), 3.8), constrained_layout=True)
    grid, tgt = lp_targets(paper, ells)
    for ax, ell in zip(axes, ells):
        c = toy_vs_x(toy, z0, ell)
        A, rms = log_amp(np.interp(grid, c["x_hii"], c["D"]), tgt[ell])
        ax.plot(*paper[ell], color=LP_COLOR, lw=2.2, label="La Plante+22 (fiducial)")
        ax.plot(c["x_hii"], A * c["D"], color=TOY_COLOR, lw=1.8, label=f"toy, $\\epsilon$={eps:.3g}")
        ax.plot(c["x_hii"], A * c["D_dens"], color=TOY_COLOR, ls="--", lw=1.2, label="  density/velocity response")
        ax.plot(c["x_hii"], A * c["D_ion"], color=TOY_COLOR, ls=":", lw=1.4, label="  ionisation response")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.4 * np.max(paper[ell][1]))
        ax.set_xlabel("$x_{\\rm HII}$"); ax.set_title(f"$\\ell$ = {ell}   (A = {A:.3g}, misfit {rms:.2f})", fontsize=9)
        ax.set_ylabel("$D_\\ell^{\\rm cross}$ [$\\mu$K$^2$]") if ax is axes[0] else None
        print(f"  ell={ell}: amplitude A = {A:.4g}; peak x_HII  LP {peak_x(*paper[ell]):.2f}  toy {peak_x(c['x_hii'], c['D']):.2f}"
              f"  (density/velocity part alone {peak_x(c['x_hii'], c['D_dens']):.2f}, ionisation part alone {peak_x(c['x_hii'], c['D_ion']):.2f})")
    axes[0].legend(fontsize=7, frameon=False)
    axes[-1].semilogx(eps_grid, mis, "o-", color=TOY_COLOR, ms=3)
    axes[-1].axvline(eps, color=LP_COLOR, lw=1); axes[-1].set_xlabel("$\\epsilon$ (patchy / homogeneous kSZ$^2$ power)")
    axes[-1].set_ylabel("rms log misfit"); axes[-1].set_title("which $\\epsilon$ matches La Plante", fontsize=9)
    print("figure:", *save_fig(fig, os.path.join(args.out_dir, "toy_vs_laplante")))
    plt.close(fig)

    # ---- (2) ours -----------------------------------------------------------------------------------------------------------
    if args.ours_root:
        paths = sorted(glob.glob(os.path.join(args.ours_root, "*", TAG)))
        if not paths:
            print(f"(no {TAG} under {args.ours_root})")
        else:
            fig, axes = plt.subplots(1, len(paths), figsize=(4.0 * len(paths), 3.6), constrained_layout=True, squeeze=False)
            print("\nours vs toy (ell = 1000; amplitude fitted per tracer):")
            for ax, p in zip(axes[0], paths):
                name = os.path.basename(os.path.dirname(p))
                d = read_ours(p, ells)
                o = np.argsort(d["x_hii"])
                tz = tm.Toy(cfg, tm.history_fn(d["z0"], d["x_hii"]), eps=eps, r_v=args.r_v, zmean=args.zmean)
                ell = 1000 if 1000 in d else [e for e in ells if e in d][0]
                y, s = d[ell]
                c = toy_vs_x(tz, d["z0"], ell)
                m = np.interp(d["x_hii"], c["x_hii"], c["D"])
                A, rms = log_amp(m, y)
                ax.errorbar(d["x_hii"], y, s, fmt="o", color=OURS_COLOR, ms=3.5, lw=1, label="measured")
                ax.plot(c["x_hii"], A * c["D"], color=TOY_COLOR, lw=1.6, label="toy (A fitted)")
                ax.plot(c["x_hii"], A * c["D_dens"], color=TOY_COLOR, ls="--", lw=1.0)
                ax.plot(c["x_hii"], A * c["D_ion"], color=TOY_COLOR, ls=":", lw=1.2)
                ax.set_title(f"{name}, $\\ell$={ell}", fontsize=9); ax.set_xlabel("$x_{\\rm HII}$")
                print(f"  {name:>16}: A = {A:.3g}  misfit {rms:.2f}   peak x_HII  data {peak_x(d['x_hii'], y):.2f}  toy {peak_x(c['x_hii'], c['D']):.2f}")
            axes[0, 0].legend(fontsize=7, frameon=False); axes[0, 0].set_ylabel("$D_\\ell^{\\rm cross}$ [$\\mu$K$^2$]")
            print("figure:", *save_fig(fig, os.path.join(args.out_dir, "toy_vs_ours")))
            plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
