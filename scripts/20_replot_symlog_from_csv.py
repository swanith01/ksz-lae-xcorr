#!/usr/bin/env python3
"""
scripts/20_replot_symlog_from_csv.py
=======================================
Re-plots an ALREADY-COMPUTED direct-method CSV (from scripts/15 or
scripts/18, either the old single-value format or the new mean+/-std
format) as a symlog plot, matching the current plotting style exactly --
WITHOUT redoing any of the expensive per-snapshot compute. Pure
plotting, runs in seconds, no cluster/qsub needed -- works on usha (or
any machine with this repo cloned and matplotlib installed) directly
from a CSV that's already sitting on disk or in git history.

Auto-detects which kind of CSV it's looking at:
  - Has a 'z0' column -> scripts/18-style (D_ell vs z0, several fixed ell)
  - No 'z0' column     -> scripts/15-style (D_ell vs ell, one fixed z0)
Handles both the OLD (single 'D_ell_uK2' column, no error bars) and NEW
('D_ell_mean_uK2' + 'D_ell_std_uK2') column formats -- whichever CSV you
already have on hand works, no need to regenerate it first.

Usage:
    # scripts/15-style CSV (D_ell vs ell):
    python scripts/20_replot_symlog_from_csv.py --csv path/to/direct_bispectrum_....csv

    # scripts/18-style CSV (D_ell vs z0):
    python scripts/20_replot_symlog_from_csv.py --csv path/to/direct_dell_vs_z0_....csv
"""

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.la_plante_reference import load_dell_vs_ell_band, load_dell_vs_z0_bands
from ksz_lae_xcorr.utils.figio import compute_symlog_linthresh, save_fig


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_dell_vs_ell(rows, out_stem, stitched_csv=None):
    """scripts/15-style: one z0, D_ell vs ell."""
    ell = np.array([float(r["ell"]) for r in rows])
    if "D_ell_mean_uK2" in rows[0]:
        D = np.array([float(r["D_ell_mean_uK2"]) for r in rows])
        D_err = np.array([float(r["D_ell_std_uK2"]) for r in rows])
        n_seeds = rows[0].get("n_seeds", "?")
        label = f"Direct/coeval (mean +/- std, {n_seeds} seed(s))"
    else:
        D = np.array([float(r["D_ell_uK2"]) for r in rows])
        D_err = np.zeros_like(D)
        label = "Direct/coeval"

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.errorbar(ell, D, yerr=D_err, fmt="o-", color="darkgreen", capsize=3, label=label)

    lp_band = load_dell_vs_ell_band()
    ax.fill_between(lp_band["ell_lo"], lp_band["lo"],
                     np.interp(lp_band["ell_lo"], lp_band["ell_hi"], lp_band["hi"]),
                     color="black", alpha=0.15, label="La Plante+2022 band (digitized, x_HII~0.43)")

    ell_s, D_s = [], []
    if stitched_csv and os.path.exists(stitched_csv):
        for row in _read_csv(stitched_csv):
            if row.get("experiment") == "SO":
                ell_s.append(float(row["ell"]))
                D_s.append(float(row["D_ell_uK2"]))
        if ell_s:
            ax.plot(ell_s, D_s, "s--", color="firebrick", alpha=0.7, label="Stitched (SO filter)")

    linthresh = compute_symlog_linthresh(D, lp_band["hi"], D_s)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$] (symlog)")
    ax.set_title("Direct/coeval vs La Plante+2022", fontsize=11)
    ax.legend(fontsize=8)

    save_fig(fig, out_stem)
    plt.close(fig)


def plot_dell_vs_z0(rows, out_stem):
    """scripts/18-style: several fixed ell, D_ell vs z0. Direct-vs-La-Plante
    only, no stitched overlay -- matches the original convention for this
    plot type (confirmed 2026-09-17); the ell-sweep plot above is the one
    that includes stitched."""
    ells = sorted(set(float(r["ell"]) for r in rows))
    colors_cycle = {ells[i]: c for i, c in enumerate(["tab:blue", "tab:orange", "tab:green", "tab:red"][:len(ells)])}
    has_err = "D_ell_mean_uK2" in rows[0]

    lp_bands = load_dell_vs_z0_bands()
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    all_D_for_thresh = []
    for ell in ells:
        sub = [r for r in rows if float(r["ell"]) == ell]
        z0 = np.array([float(r["z0"]) for r in sub])
        if has_err:
            D = np.array([float(r["D_ell_mean_uK2"]) for r in sub])
            D_err = np.array([float(r["D_ell_std_uK2"]) for r in sub])
        else:
            D = np.array([float(r["D_ell_uK2"]) for r in sub])
            D_err = np.zeros_like(D)
        order = np.argsort(z0)
        z0, D, D_err = z0[order], D[order], D_err[order]
        all_D_for_thresh.append(D)

        color = colors_cycle[ell]
        ax.errorbar(z0, D, yerr=D_err, fmt="o-", color=color, capsize=3,
                     label=f"Direct/coeval, ell={ell:.0f}")

        if int(ell) in lp_bands:
            lp = lp_bands[int(ell)]
            lp_hi_interp = np.interp(lp["z0_lo"], lp["z0_hi"], lp["hi"])
            ax.fill_between(lp["z0_lo"], lp["lo"], lp_hi_interp, color=color, alpha=0.15,
                              label=f"La Plante+2022, ell={ell:.0f}")
            all_D_for_thresh.append(lp["hi"])

    linthresh = compute_symlog_linthresh(*all_D_for_thresh)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$] (symlog)")
    ax.set_title("D_ell vs z0, direct/coeval vs La Plante+2022", fontsize=11)
    ax.legend(fontsize=8, ncol=2)

    save_fig(fig, out_stem)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, required=True, help="Path to an existing scripts/15 or scripts/18 CSV")
    parser.add_argument("--stitched-csv", type=str, default=None,
                         help="Optional companion stitched CSV (scripts/11 output) to overlay -- "
                              "only used for D_ell vs ell CSVs; the D_ell vs z0 plot is "
                              "direct-vs-La-Plante only, matching the original convention")
    parser.add_argument("--out", type=str, default=None,
                         help="Output file stem (no extension -- .pdf and .png both written). "
                              "Default: same name as --csv with '_symlog' appended")
    args = parser.parse_args()

    rows = _read_csv(args.csv)
    if not rows:
        print(f"{args.csv} is empty -- nothing to plot.")
        return 1

    out_stem = args.out or (os.path.splitext(args.csv)[0] + "_symlog")

    if "z0" in rows[0]:
        print(f"Detected scripts/18-style CSV (D_ell vs z0), {len(rows)} rows, "
              f"{len(set(r['ell'] for r in rows))} ell value(s)")
        plot_dell_vs_z0(rows, out_stem)
    else:
        print(f"Detected scripts/15-style CSV (D_ell vs ell), {len(rows)} rows")
        plot_dell_vs_ell(rows, out_stem, args.stitched_csv)

    print(f"Saved: {out_stem}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
