#!/usr/bin/env python3
"""
scripts/21_tracer_auto_power.py
==================================
D_ell^auto vs ell, at several z, for halo/LAE/LBG tracer counts alone --
NOT kSZ-related in any way, so this is independent of the still-broken
stitched kSZ pathway. Purpose: a basic sanity check ("is this smooth,
does it look like a real clustering signal") on the tracer fields
themselves, one level below the kSZ x tracer cross-correlation this
whole session has been chasing -- pulling back to something simpler and
separately checkable, per Girish's suggestion (2026-09-18).

Uses the real stitched tracer counts (Jahaan's catalogues, confirmed
correctly populated across all 10 seeds after the 2026-09-09 checkpoint
fix) -- NOT the direct/coeval per-snapshot method. A tracer auto-power
doesn't need kSZ or velocity at all, so stitching's periodicity risk is
the only caveat here, not the unresolved cross-power bug.

Usage:
    python scripts/21_tracer_auto_power.py
    python scripts/21_tracer_auto_power.py --seeds 1 2 3 --tracers halo lae lbg
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation.auto_power import compute_tracer_auto_spectra
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import compute_symlog_linthresh, save_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--tracers", type=str, nargs="+", default=["halo", "lae", "lbg"])
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading stitched tracer data for {len(seeds)} seed(s)...")
    _, tracer_data = load_lightcone_products(cfg, seeds)
    print(f"  {len(tracer_data)}/{len(seeds)} seeds have tracer data")

    print("Computing tracer auto-power (z-sweep)...")
    auto_results = compute_tracer_auto_spectra(cfg, tracer_data, seeds)

    for tracer in args.tracers:
        z_values = sorted(set().union(*(auto_results[tracer][s].keys() for s in seeds if s in auto_results[tracer])))
        if not z_values:
            print(f"  {tracer}: no data at all (tracer not stitched for any requested seed) -- skipping plot")
            continue

        n_show = min(6, len(z_values))
        idx = np.linspace(0, len(z_values) - 1, n_show).round().astype(int)
        z_to_show = [z_values[i] for i in idx]

        fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
        all_D_for_thresh = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(z_to_show)))

        for z_c, color in zip(z_to_show, colors):
            per_seed_D = [auto_results[tracer][s][z_c]["D_ell"] for s in seeds
                          if s in auto_results[tracer] and z_c in auto_results[tracer][s]]
            if not per_seed_D:
                continue
            ell = next(auto_results[tracer][s][z_c]["ell"] for s in seeds
                       if s in auto_results[tracer] and z_c in auto_results[tracer][s])
            D_mean = np.mean(per_seed_D, axis=0)
            ax.plot(ell, D_mean, "o-", color=color, label=f"z={z_c:.2f} ({len(per_seed_D)} seeds)")
            all_D_for_thresh.append(D_mean)

        linthresh = compute_symlog_linthresh(*all_D_for_thresh)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xscale("log")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{\rm auto}/2\pi$ (symlog, arbitrary units -- no T_CMB factor)")
        ax.set_title(f"{tracer} auto-power vs ell, at several z ({len(seeds)} seed(s))")
        ax.legend(fontsize=8)

        outpath = os.path.join(args.out_dir, f"tracer_auto_power_{tracer}.pdf")
        save_fig(fig, outpath)
        plt.close(fig)
        print(f"  {tracer}: {len(z_values)} z-slices computed, {len(z_to_show)} shown -- saved {outpath} (+ .png)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
