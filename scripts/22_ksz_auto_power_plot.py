#!/usr/bin/env python3
"""
scripts/22_ksz_auto_power_plot.py
====================================
Plots the kSZ-related auto-power spectra (kSZ, xe^2, v^2, v_proj,
v_proj^2), a basic "is this smooth" sanity check on the kSZ side, per
Girish's suggestion (2026-09-18) to validate simpler ingredients before
chasing the full cross-correlation.

REWRITTEN 2026-09-30, per Girish's follow-up on the first version of this
plot ("This is bad ... The ksz power is a few muK^2 ... why is it till
z=12.5? ... we want it for the full patchy window"):

  kSZ panel: previously plotted the RAW auto-power of the stitched kSZ
  map (correlation.auto_power.compute_auto_spectra), which showed ~1e-35
  values (and exact zero for kSZ^2) instead of the expected few-muK^2
  scale. Root cause was NOT a units bug -- to_Dell's T_CMB_uK convention
  here matches the validated coherence_decomposition.py convention
  exactly (single factor, squared internally). It was that the raw
  stitched kSZ map is dominated by the well-documented periodicity
  artifact: this box is repeated MANY times to fill the full z=5-20
  line of sight, and the resulting coherent cancellation between
  periodic replicas can drive the naive auto-power to near machine-zero.
  This is exactly what correlation.coherence_decomposition.py's P_diag/
  P_off split was built to isolate and remove -- P_diag was already
  validated against ksz-pipeline's independent direct/Limber calculation
  to 8.5%. This script now plots THAT (D_diag), computed over each
  seed's own actual patchy-reionization window (see
  compute_patchy_window_diag_power) rather than the raw P_total.

  "z=12.5" was never a truncation of the line-of-sight integral (the
  kSZ integrand always summed the full box.z_min-z_max range) -- it was
  only an arbitrary box-midpoint reference redshift used for the
  ell=k*chi Limber conversion. This script now uses each seed's real
  patchy window (z>=6 floor, x_HI strictly between 0 and 1) and a
  power-weighted chi_eff over that window instead, and reports the
  window explicitly in the plot title.

  kSZ^2 panel: DROPPED from this auto-power grid. kSZ^2 = kSZ_map^2 is a
  quadratic (bispectrum-type) quantity; the linear P_diag/P_off
  machinery used to fix the kSZ panel doesn't apply to it (see
  coherence_decomposition.py's own module docstring), and no
  periodicity-corrected version of the kSZ^2 SELF-auto-power exists yet.
  Rather than show a number that inherits the same cancellation problem
  with no fix applied, this panel is replaced with a note pointing to
  the kSZ^2 x tracer CROSS-power (correlation.cross_correlation.
  compute_cross_spectra), which IS the physically relevant statistic for
  this paper and IS already computed correctly.

  xe2 / v2 / v_proj / v_proj2: kept from compute_auto_spectra as before
  (unchanged) -- these aren't CMB-temperature quantities (no T_CMB_uK
  scaling applies), and only v_proj shares the "signed LOS sum" pattern
  that makes kSZ susceptible to catastrophic periodicity cancellation.
  Still evaluated at the box-midpoint z_ref for now; the title says so
  explicitly rather than implying they share the kSZ panel's patchy
  window.

Requires data/products/coherence_decomposition.pkl (scripts/09, RERUN
required to pick up the new 'patchy' field -- a copy from before
2026-09-30 will not have it) for the kSZ panel, and
data/products/auto_results.pkl (scripts/04) for the other panels.

Usage:
    python scripts/22_ksz_auto_power_plot.py
"""

import argparse
import datetime
import os
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation.power_spectra import KGrid, ell_at_redshift
from ksz_lae_xcorr.correlation.seed_stats import aggregate_coherence_over_seeds
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import compute_symlog_linthresh, save_fig

AUTO_MAP_NAMES = ["xe2", "v2", "v_proj", "v_proj2"]


def _check_mtime(path, label):
    mtime = os.path.getmtime(path)
    print(f"Loading: {path}")
    print(f"  Last modified: {datetime.datetime.fromtimestamp(mtime)} -- "
          f"confirm this is AFTER your most recent {label} rerun before trusting these numbers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--auto-results", type=str, default=None,
                         help="Path to auto_results.pkl (default: cfg.paths.products_root/auto_results.pkl)")
    parser.add_argument("--coherence-results", type=str, default=None,
                         help="Path to coherence_decomposition.pkl (default: cfg.paths.products_root/coherence_decomposition.pkl)")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    coherence_path = args.coherence_results or os.path.join(cfg.paths.products_root, "coherence_decomposition.pkl")
    _check_mtime(coherence_path, "scripts/09_coherence_decomposition.py")
    with open(coherence_path, "rb") as f:
        coherence_results = pickle.load(f)

    missing_patchy = [s for s, r in coherence_results.items() if "patchy" not in r]
    if missing_patchy:
        print(f"WARNING: seeds {missing_patchy} in {coherence_path} have no 'patchy' field -- "
              f"this is a pre-2026-09-30 coherence_decomposition.pkl. Rerun scripts/09 to fix.")

    auto_path = args.auto_results or os.path.join(cfg.paths.products_root, "auto_results.pkl")
    _check_mtime(auto_path, "scripts/04_compute_xcorr.py")
    with open(auto_path, "rb") as f:
        auto_results = pickle.load(f)

    kg = KGrid(cfg)
    z_box_mid = 0.5 * (cfg.box.z_min + cfg.box.z_max)
    ell_box_mid = ell_at_redshift(cfg, kg, z_box_mid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # --- Panel 1: kSZ, periodicity-corrected (D_diag), full patchy window ---
    ax = axes.flat[0]
    patchy_by_seed = {s: r["patchy"] for s, r in coherence_results.items() if "patchy" in r}
    if len(patchy_by_seed) >= 2:
        agg = aggregate_coherence_over_seeds(patchy_by_seed, list(patchy_by_seed.keys()), field="D_diag")
        z_los, z_his = zip(*[(r["z_lo"], r["z_hi"]) for r in patchy_by_seed.values()])
        ax.errorbar(agg["ell"], agg["median"], yerr=agg["sigma"], fmt="o-", capsize=3, color="darkblue")
        linthresh = compute_symlog_linthresh(agg["median"])
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xscale("log")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$D_\ell^{\rm kSZ}$ [$\mu K^2$] (symlog)")
        ax.set_title(
            f"kSZ auto-power, periodicity-corrected ($D_{{\\rm diag}}$)\n"
            f"({agg['n_seeds']} seeds, patchy window z=[{min(z_los):.1f}-{max(z_los):.1f}, "
            f"{min(z_his):.1f}-{max(z_his):.1f}])"
        )
        print(f"  kSZ (patchy, D_diag) at ell~3000: "
              f"{agg['median'][np.argmin(np.abs(agg['ell'] - 3000))]:.4g} +/- "
              f"{agg['sigma'][np.argmin(np.abs(agg['ell'] - 3000))]:.4g} uK^2")
    else:
        ax.set_title("kSZ -- need >=2 seeds with a 'patchy' field in coherence_decomposition.pkl")

    # --- Panel 2: kSZ^2 self-auto-power -- intentionally not shown ---
    ax = axes.flat[1]
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "kSZ$^2$ self-auto-power not shown here.\n\n"
        "kSZ$^2$ is quadratic in the LOS-integrated field; the\n"
        "periodicity fix used for the kSZ panel (P_diag/P_off)\n"
        "does not apply to it, and no corrected version exists\n"
        "yet for the self-auto-power.\n\n"
        "The physically relevant statistic -- kSZ$^2$ x tracer\n"
        "cross-power -- is already computed correctly by\n"
        "correlation.cross_correlation.compute_cross_spectra.",
        ha="center", va="center", fontsize=10, wrap=True,
        transform=ax.transAxes,
    )

    for ax, map_name in zip(axes.flat[2:], AUTO_MAP_NAMES):
        if map_name not in auto_results:
            ax.set_title(f"{map_name} -- not in auto_results.pkl")
            continue

        seeds = sorted(auto_results[map_name].keys())
        D_stack = np.array([auto_results[map_name][s][0] for s in seeds])
        D_mean = np.nanmean(D_stack, axis=0)
        D_std = np.nanstd(D_stack, axis=0) if len(seeds) > 1 else np.zeros_like(D_mean)

        n_nan = np.sum(np.isnan(D_stack))
        if n_nan:
            print(f"  {map_name}: {n_nan}/{D_stack.size} values are NaN across all seeds/bins")

        ax.errorbar(ell_box_mid, D_mean, yerr=D_std, fmt="o-", capsize=3, color="darkblue")
        linthresh = compute_symlog_linthresh(D_mean)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xscale("log")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$D_\ell$ (symlog)")
        ax.set_title(f"{map_name} auto-power ({len(seeds)} seeds, box-midpoint z={z_box_mid:.1f})")

    fig.suptitle(
        "kSZ-related auto-power spectra\n"
        "(kSZ panel: periodicity-corrected, seed's own patchy window -- "
        "other panels: raw stitched, box-midpoint z, unchanged)",
        fontsize=12,
    )
    outpath = os.path.join(args.out_dir, "ksz_auto_power_overview.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"\nSaved: {outpath} (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
