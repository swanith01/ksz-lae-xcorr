#!/usr/bin/env python3
"""
scripts/11_roman_hls_dell_comparison.py
==========================================
Girish's Stage 1 (2026-09-07 Slack): "define a clear pass/fail comparison
with La Plante et al." This script produces the actual comparison --
D_ell for filtered-kSZ^2 x (bias-weighted matter, Eq 6-7 of
arXiv:2111.13717) from THIS simulation, overlaid against a hand-read
reference point from their Figure 4 (ell(ell+1)C_ell/(2pi) ~ 0.02 uK^2
near ell~1000, their SO filter, roughly redshift-independent per their
Sec 3).

See src/ksz_lae_xcorr/snr/roman_hls_benchmark.py's module docstring for
every caveat on what this is and isn't (clustering-only, 2-point bias
interpolation, box-size mismatch, instrument-noise-only filter). This is
an order-of-magnitude / shape check, not a precision reproduction.

Requires scripts/04_compute_xcorr.py to have been run first (for the kSZ
maps and auto-power feeding the CMB filter).

Usage:
    python scripts/11_roman_hls_dell_comparison.py
    python scripts/11_roman_hls_dell_comparison.py --z0 9.5 --dz 1.0
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.snr.cmb_filter import build_cmb_filter_ingredients
from ksz_lae_xcorr.snr.roman_hls_benchmark import (
    PAPER_FIG4_PEAK_DELL_UK2,
    PAPER_FIG4_PEAK_ELL,
    compute_bias_weighted_cross_power,
)
from ksz_lae_xcorr.snr.snr_forecast import build_filtered_kSZ2_maps
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--z0", type=float, default=9.5,
                         help="Galaxy window central z (paper's Fig 3 example: z0=9.5)")
    parser.add_argument("--dz", type=float, default=1.0,
                         help="Galaxy window width (paper's Fig 3 example: dz=1.0)")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    xcorr_path = f"{cfg.paths.products_root}/cross_results.pkl"
    print(f"Loading cross-correlation products from {xcorr_path} ...")
    with open(xcorr_path, "rb") as f:
        products = pickle.load(f)
    maps = products["maps"]

    auto_path = f"{cfg.paths.products_root}/auto_results.pkl"
    if not os.path.exists(auto_path):
        raise FileNotFoundError(f"{auto_path} not found -- run scripts/04_compute_xcorr.py first.")
    with open(auto_path, "rb") as f:
        auto_results_ksz = pickle.load(f)["kSZ"]

    print("Loading field data (density) for the bias-weighted galaxy field...")
    field_data, _ = load_lightcone_products(cfg, seeds)

    kg = KGrid(cfg)
    filt = build_cmb_filter_ingredients(cfg, kg, auto_results_ksz)
    filtered_kSZ2 = build_filtered_kSZ2_maps(cfg, kg, maps["kSZ"], filt, seeds)

    results_per_exp = {}
    for name in cfg.snr.experiments:
        D_seeds = []
        ell_ref = None
        for seed in seeds:
            if seed not in field_data or seed not in filtered_kSZ2[name]:
                continue
            res = compute_bias_weighted_cross_power(
                cfg, kg, filtered_kSZ2[name][seed], field_data[seed], args.z0, args.dz
            )
            if ell_ref is None:
                ell_ref = res["ell"]
            D_seeds.append(res["D_ell"])
        if not D_seeds:
            print(f"  {name}: no seeds available, skipping")
            continue
        D_med = np.nanmedian(np.array(D_seeds), axis=0)
        D_std = np.nanstd(np.array(D_seeds), axis=0, ddof=1) if len(D_seeds) > 1 else np.zeros_like(D_med)
        results_per_exp[name] = {"ell": ell_ref, "D_med": D_med, "D_std": D_std, "n_seeds": len(D_seeds)}
        i_ref = int(np.argmin(np.abs(ell_ref - PAPER_FIG4_PEAK_ELL)))
        print(f"  {name}: D_ell(ell~{ell_ref[i_ref]:.0f}) = {D_med[i_ref]:.4g} uK^2 "
              f"({len(D_seeds)} seeds) -- paper's Fig 4 reads ~{PAPER_FIG4_PEAK_DELL_UK2} uK^2 at ell~1000")

    if not results_per_exp:
        print("No results for any experiment -- nothing to plot.")
        return 1

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for name, r in results_per_exp.items():
        ax.errorbar(r["ell"], r["D_med"], yerr=r["D_std"],
                     marker="o", ms=4, capsize=2, label=f"Ours -- {name} filter ({r['n_seeds']} seeds)")
    ax.axhline(PAPER_FIG4_PEAK_DELL_UK2, color="black", ls="--", lw=1,
               label=r"La Plante+2022 Fig.4 peak ($\approx$" + f"{PAPER_FIG4_PEAK_DELL_UK2}" + r" $\mu K^2$, hand-read)")
    ax.axvline(PAPER_FIG4_PEAK_ELL, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times\delta_g}/2\pi$ [$\mu K^2$]")
    ax.set_title(f"Stage 1 comparison -- z0={args.z0}, dz={args.dz}\n"
                 f"({cfg.box.box_len_mpc:.0f} Mpc box vs paper's 2 Gpc/h -- shape/order-of-magnitude only)")
    ax.legend(fontsize=8)
    outpath = os.path.join(args.out_dir, f"stage1_dell_comparison_z{args.z0:.1f}.pdf")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {outpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
