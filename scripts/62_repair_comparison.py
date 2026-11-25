#!/usr/bin/env python3
"""
scripts/62_repair_comparison.py  (25 Nov 2026)
================================================
Before / after the 2026-11-24 box repair (sim_generation/2026-06_300Mpc_boxes/README.md):
  repair_diagnostics : (a) density std vs z -- frozen z = 0 initial field (what was saved) vs the evolved density;
                       (b) how far the haloes moved (median and 99th percentile) vs z.     [repair_summary.csv per seed]
  lae_counts         : (a) LAEs per snapshot vs z -- intrinsic, observed with the old catalogue, observed with the new
                       one (scripts/61); (b) mean Lya IGM transmission vs z.              [lae_catalogue_v2 summary_s*.csv]
  spectra_old_new    : seed-mean kSZ^2 x tracer cross-spectrum and tracer auto-spectrum, old vs new, for a few tracers
                       at two windows (SO post-ILC filter), with the new/old ratio.        [sim_spectra npz, old and new]

    python scripts/62_repair_comparison.py \
        --old-spectra data/products_tracers/sim_spectra --new-spectra data/products_tracers_repaired/sim_spectra \
        --repair-dir data/products_tracers_repaired/repair_summaries \
        --lae-dir data/products_tracers_repaired/lae_catalogue_v2_summaries \
        --out-dir data/plots/2026-11-25_repaired/repair_comparison
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
from matplotlib.lines import Line2D

from ksz_lae_xcorr.utils.figio import save_fig

FROZEN_STD = 3.4095        # density_300.npy std, identical at every z (codes/check_density.py, seed 6)
C_OLD, C_NEW, C_INT = "0.55", "#0072B2", "#E69F00"


def read_csvs(pattern):
    rows = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            rows[p] = list(csv.DictReader(f))
    return rows


def per_z(rows_by_file, col):
    """{z: [values over files]} for a numeric column."""
    out = {}
    for rows in rows_by_file.values():
        for r in rows:
            v = r.get(col, "")
            if v not in ("", None):
                try:
                    out.setdefault(float(r["z"]), []).append(float(v))
                except ValueError:
                    pass
    zs = np.array(sorted(out))
    return zs, [np.array(out[z]) for z in zs]


def band(ax, zs, vals, color, label, ls="-", marker="o"):
    m = np.array([v.mean() for v in vals])
    lo = np.array([v.min() for v in vals])
    hi = np.array([v.max() for v in vals])
    ax.plot(zs, m, ls, color=color, marker=marker, ms=3, lw=1.8, label=label)
    ax.fill_between(zs, lo, hi, color=color, alpha=0.2, lw=0)


def legend_below(ax, **kw):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8, **kw)


def fig_repair(repair_dir, out_dir):
    rows = read_csvs(os.path.join(repair_dir, "*repair_summary*.csv"))
    if not rows:
        print("no repair summaries in", repair_dir)
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    zs, v = per_z(rows, "density_std")
    a1.axhline(FROZEN_STD, color=C_OLD, lw=2, label="saved before the repair (density_300.npy): same at every z")
    band(a1, zs, v, C_NEW, f"evolved density (repaired), mean and range of {len(rows)} seeds")
    a1.set_yscale("log"); a1.set_xlabel("z"); a1.set_ylabel(r"std of $\delta$ (1 cMpc cells)")
    a1.set_title("density field", fontsize=10)
    legend_below(a1)
    zs, v = per_z(rows, "disp_median_cMpc")
    band(a2, zs, v, C_NEW, "median")
    zs, v = per_z(rows, "disp_p99_cMpc")
    band(a2, zs, v, C_INT, "99th percentile", ls="--", marker=None)
    a2.axhline(1.0, color=C_OLD, lw=0.8, ls=":", label="one cell (1 cMpc)")
    a2.set_xlabel("z"); a2.set_ylabel("halo displacement [cMpc]")
    a2.set_title("how far the haloes moved (saved: start positions; repaired: positions at z)", fontsize=10)
    legend_below(a2, ncol=3)
    print("figure:", *save_fig(fig, os.path.join(out_dir, "repair_diagnostics")))
    plt.close(fig)


def fig_lae(lae_dir, out_dir):
    rows = read_csvs(os.path.join(lae_dir, "summary_s*.csv"))
    if not rows:
        print("no LAE summaries in", lae_dir)
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for col, c, lab, ls in (("N_intrinsic", C_INT, "intrinsic (Jahaan's catalogue, unchanged)", "--"),
                            ("N_obs_old", C_OLD, "observed, old catalogue (Aug 2026)", "-"),
                            ("N_obs_new", C_NEW, "observed, new catalogue (scripts/61, repaired boxes)", "-")):
        zs, v = per_z(rows, col)
        v = [np.clip(x, 0.5, None) for x in v]
        band(a1, zs, v, c, lab, ls=ls)
    a1.set_yscale("log"); a1.set_xlabel("z"); a1.set_ylabel(r"LAEs per snapshot ($L_{\rm Ly\alpha}>10^{42}$, 300 cMpc box)")
    a1.set_title(f"LAE counts ({len(rows)} seeds; band = range)", fontsize=10)
    legend_below(a1)
    zs, v = per_z(rows, "T_mean")
    band(a2, zs, v, C_NEW, "mean")
    zs, v = per_z(rows, "T_median")
    band(a2, zs, v, C_INT, "median", ls="--", marker=None)
    a2.set_ylim(0, 1.05); a2.set_xlabel("z"); a2.set_ylabel("Ly$\\alpha$ IGM transmission of intrinsic LAEs")
    a2.set_title("new catalogue: transmission", fontsize=10)
    legend_below(a2, ncol=2)
    print("figure:", *save_fig(fig, os.path.join(out_dir, "lae_counts")))
    plt.close(fig)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fig_spectra(old_dir, new_dir, tracers, windows, experiment, out_dir):
    have = [t for t in tracers if os.path.exists(os.path.join(old_dir, f"{t}.npz")) and os.path.exists(os.path.join(new_dir, f"{t}.npz"))]
    if not have:
        print("no matching spectra in", old_dir, new_dir)
        return
    fig, axes = plt.subplots(3, len(have), figsize=(3.6 * len(have), 8.6), sharex=True, constrained_layout=True,
                             squeeze=False)
    rows = []
    wcol = ["#0072B2", "#D55E00", "#009E73"]
    for j, t in enumerate(have):
        o, n = load(os.path.join(old_dir, f"{t}.npz")), load(os.path.join(new_dir, f"{t}.npz"))
        ie = list(o["experiments"]).index(experiment)
        for k, z0 in enumerate(windows):
            iw = [i for i in range(len(o["z0"])) if abs(o["z0"][i] - z0) < 1e-6 and abs(o["dz"][i] - 1) < 1e-6]
            if not iw:
                continue
            iw = iw[0]
            e = o["edges"][iw]
            ell = np.sqrt(e[:-1] * e[1:])
            for d, ls, lab in ((o, "--", "old"), (n, "-", "repaired")):
                cs = d["Cs"][:, ie, iw, :]
                m = np.nanmean(cs, 0); s = np.nanstd(cs, 0) / np.sqrt(np.sum(np.isfinite(cs), 0).clip(1))
                axes[0, j].plot(ell, ell**2 * m, ls, color=wcol[k], lw=1.6)
                if ls == "-":
                    axes[0, j].fill_between(ell, ell**2 * (m - s), ell**2 * (m + s), color=wcol[k], alpha=0.18, lw=0)
                cg = np.nanmean(d["Cg"][:, iw, :], 0)
                axes[1, j].plot(ell, ell**2 * cg, ls, color=wcol[k], lw=1.6)
            mo, mn = np.nanmean(o["Cs"][:, ie, iw, :], 0), np.nanmean(n["Cs"][:, ie, iw, :], 0)
            go, gn = np.nanmean(o["Cg"][:, iw, :], 0), np.nanmean(n["Cg"][:, iw, :], 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                axes[2, j].plot(ell, mn / mo, "-", color=wcol[k], lw=1.6)
                axes[2, j].plot(ell, gn / go, ":", color=wcol[k], lw=1.6)
            sel = (ell > 300) & (ell < 3000) & np.isfinite(mo) & np.isfinite(mn)
            rows.append(dict(tracer=t, z0=z0, cross_ratio_l300_3000=float(np.sum(mn[sel]) / np.sum(mo[sel])) if sel.any() else np.nan,
                             auto_ratio_l300_3000=float(np.nansum(gn[sel]) / np.nansum(go[sel])) if sel.any() else np.nan))
        axes[0, j].set_title(t, fontsize=10)
        axes[0, j].axhline(0, color="0.7", lw=0.6)
        axes[1, j].set_yscale("log")
        axes[2, j].axhline(1, color="0.7", lw=0.6)
        axes[2, j].set_ylim(0, 3)
        axes[2, j].set_xscale("log"); axes[2, j].set_xlabel(r"$\ell$")
    axes[0, 0].set_ylabel(r"$\ell^2 C_\ell^{{\rm kSZ}^2_f\times g}$ (seed mean)")
    axes[1, 0].set_ylabel(r"$\ell^2 C_\ell^{gg}$ (seed mean)")
    axes[2, 0].set_ylabel("repaired / old")
    handles = [Line2D([], [], color=wcol[k], lw=2, label=f"window $z_0$ = {z0:g} ($\\Delta z$ = 1)") for k, z0 in enumerate(windows)]
    handles += [Line2D([], [], color="0.3", ls="--", label="old (frozen density, start positions, old LAE catalogue)"),
                Line2D([], [], color="0.3", ls="-", label="repaired (band: error of the seed mean)"),
                Line2D([], [], color="0.3", ls=":", label="ratio of the auto-spectra (bottom row)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"kSZ$^2$ x tracer cross-spectrum and tracer auto-spectrum, before vs after the repair ({experiment} filter)",
                 fontsize=10.5)
    print("figure:", *save_fig(fig, os.path.join(out_dir, "spectra_old_new")))
    plt.close(fig)
    with open(os.path.join(out_dir, "spectra_ratios.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nrepaired / old, summed over l = 300-3000:")
    for r in rows:
        print(f"  {r['tracer']:>15}  z0 {r['z0']:4.1f}:  cross {r['cross_ratio_l300_3000']:6.2f}   auto {r['auto_ratio_l300_3000']:6.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-spectra", default="data/products_tracers/sim_spectra")
    ap.add_argument("--new-spectra", default="data/products_tracers_repaired/sim_spectra")
    ap.add_argument("--repair-dir", default="data/products_tracers_repaired/repair_summaries")
    ap.add_argument("--lae-dir", default="data/products_tracers_repaired/lae_catalogue_v2_summaries")
    ap.add_argument("--tracers", default="matter_waters,halo_M10.0,lbg_MUV-19.0,lae_L42.5")
    ap.add_argument("--windows", default="7,8")
    ap.add_argument("--experiment", default="SO (post-ILC)")
    ap.add_argument("--out-dir", default="data/plots/2026-11-25_repaired/repair_comparison")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    fig_repair(args.repair_dir, args.out_dir)
    fig_lae(args.lae_dir, args.out_dir)
    fig_spectra(args.old_spectra, args.new_spectra, args.tracers.split(","), [float(w) for w in args.windows.split(",")],
                args.experiment, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
