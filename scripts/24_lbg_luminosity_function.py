#!/usr/bin/env python3
"""
scripts/24_lbg_luminosity_function.py
========================================
Our simulated LBG UV luminosity function, swept across many real
redshifts and colored by z (one smooth line per z, continuous colormap +
colorbar) -- the LBG counterpart to scripts/23's LAE Lya LF, per Girish's
"can we have tracer LFs" (2026-09-30). Uses the FULL per-object MUV_lbg
catalogue (see tracers.luminosity_function.load_raw_lbg_muv), not the
cfg.tracers.lbg_muv_cut-selected subset the count-grid tracer uses for
cross-correlation -- a luminosity function should show the real
population; the muv_cut is shown as a reference line, not applied as a
cut on the data.

NOT YET INCLUDED: a published UV-LF comparison (the LBG counterpart to
scripts/23's Kageura+2025 overlay). I did not have a verified, directly-
quoted table of real published Schechter/points to use this pass (a
couple of arXiv/IOP fetches for Bouwens et al. 2021 / Harikane et al.
2023 were blocked in this sandbox) -- rather than guess numbers from
memory, which this repo's whole reference-data convention (see
io/lae_luminosity_function_reference.py's explicit provenance notes)
exists specifically to avoid, this is left for a follow-up once a real
table is in hand (either a working fetch, or Girish/Jahaan pointing at
the paper/table to digitize, the same way Kageura+2025's numbers were
sourced for the LAE version).

Usage:
    python scripts/24_lbg_luminosity_function.py
    python scripts/24_lbg_luminosity_function.py --seeds 1 2 3 --n-z-lines 8
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.tracers.luminosity_function import compute_uv_luminosity_function
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-z-lines", type=int, default=12,
                         help="How many of our simulation's real snapshot redshifts to show as "
                              "lines, evenly spread across the range with real LBG data")
    parser.add_argument("--m-uv-min", type=float, default=-24.0)
    parser.add_argument("--m-uv-max", type=float, default=-14.0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)

    stitcher = Stitcher(cfg)
    logger = setup_logger(seeds[0], cfg.paths.lightcone_root)
    all_snap_z = np.sort(stitcher.get_halo_redshifts(seeds[0]))
    print(f"Found {len(all_snap_z)} real snapshot redshifts: z={all_snap_z.min():.2f}-{all_snap_z.max():.2f}")

    idx = np.linspace(0, len(all_snap_z) - 1, args.n_z_lines).round().astype(int)
    z_lines = sorted(set(all_snap_z[idx]))

    z_min_color, z_max_color = min(z_lines), max(z_lines)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=z_min_color, vmax=z_max_color)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    print("Computing simulated UV LF at each line redshift...")
    any_data = False
    for z in z_lines:
        result = compute_uv_luminosity_function(cfg, seeds, z, args.m_uv_min, args.m_uv_max, args.n_bins)
        valid = np.isfinite(result["phi"]) & (result["phi"] > 0)
        if not np.any(valid):
            print(f"  z={z:.2f}: no data, skipping this line")
            continue
        any_data = True
        log_phi = np.log10(result["phi"][valid])
        ax.plot(result["m_uv_centers"][valid], log_phi, "-", color=cmap(norm(z)), lw=1.3, alpha=0.85)
        print(f"  z={z:.2f}: {result['n_seeds_with_data']}/{len(seeds)} seeds, "
              f"{result['n_objects_total']} total LBGs (uncut)")

    ax.axvline(cfg.tracers.lbg_muv_cut, color="gray", ls="--", lw=1.0,
               label=f"cfg.tracers.lbg_muv_cut = {cfg.tracers.lbg_muv_cut:.1f} "
                     f"(cross-corr. tracer cut -- LF above uses the FULL catalogue)")
    ax.plot([], [], "-", color="gray", label="This simulation (lines, colored by z)")

    if any_data:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$z$")

    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.invert_xaxis()  # brighter (more negative) on the right, standard UV-LF convention
    ax.set_ylabel(r"$\log_{10}\phi$ [Mpc$^{-3}$ mag$^{-1}$]")
    ax.set_title(
        f"LBG UV luminosity function ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})\n"
        f"no published comparison overlaid yet -- see script docstring"
    )
    ax.legend(loc="lower right", fontsize=8)

    outpath = os.path.join(args.out_dir, "lbg_luminosity_function.pdf")
    save_fig(fig, outpath)
    plt.close(fig)
    print(f"\nSaved: {outpath} (+ .png)")
    if not any_data:
        print("WARNING: no z had any LBG data at all -- check cfg.paths.lbg_catalogue_root "
              "and that MUV_lbg files exist for these seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
