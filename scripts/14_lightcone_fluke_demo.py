#!/usr/bin/env python3
"""
scripts/14_lightcone_fluke_demo.py
=====================================
Two deliverables for Girish, both from real data:

1. lightcone_aggregation_levels_seed{N}.pdf -- the SAME tracer shown at
   four aggregation levels, all with the faithful (imshow, log-alpha)
   rendering: individual seed/single slice, individual seed/transverse-
   summed, single-slice-averaged-over-seeds, and fully-averaged (summed
   transverse + averaged over seeds). Demonstrates the faithful rendering
   is internally consistent -- sparsity scales sensibly with how much
   data is aggregated, nothing anomalous at any level.

2. lightcone_fluke_reveal_seed{N}.pdf -- the actual point: the OLD
   scatter-based rendering (ax.scatter(xx, yy, s=clip(counts,1,20),
   alpha=0.6), removed from the real pipeline in an earlier commit but
   recreated HERE, isolated, for this one-time comparison only) next to
   the faithful rendering, on the EXACT SAME single-seed/single-slice
   data. Same numbers, two renderings -- makes the "the density was a
   rendering artifact, not real structure" case directly, not just by
   assertion.

DO NOT reuse _old_scatter_overlay_FOR_COMPARISON_ONLY elsewhere -- it is
deliberately NOT in plotting/lightcone_panels.py, so it can't accidentally
get reintroduced into the real pipeline.

Requires scripts/02 (stitching) to have produced real lightcone products.

Usage:
    python scripts/14_lightcone_fluke_demo.py
    python scripts/14_lightcone_fluke_demo.py --seed 3 --tracer halo
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.plotting.lightcone_panels import XH_CMAP, TRACER_COLORS, _rgba_overlay_log, _setup_axes
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.grid import aggregate_transverse


def _old_scatter_overlay_FOR_COMPARISON_ONLY(ax, lc_xHI, lc_count, z_arr, cfg):
    """
    Exact recreation of the REMOVED scatter-based overlay (see git history
    around the lightcone plotting fix), for this one side-by-side
    comparison only. This is deliberately isolated here, not restored to
    plotting/lightcone_panels.py -- do not import or reuse this function.

    Matches the ORIGINAL lc_panels_combined.png's "halo positions only"
    panel exactly (alpha=0.6, size up to 20, solid black) -- that's the
    specific rendering that made the lightcone look deceptively dense,
    not the fainter alpha=0.15 overlay variant that existed alongside it.
    """
    yy, xx = np.nonzero(lc_count)
    counts = lc_count[yy, xx]
    ax.scatter(xx, yy, s=np.clip(counts, 1, 20), c="black", linewidths=0, alpha=0.6, zorder=3)
    ax.set_xlim(0, len(z_arr))
    ax.set_ylim(0, cfg.box.hii_dim)
    _setup_axes(ax, cfg, z_arr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Individual seed to feature (default: first available)")
    parser.add_argument("--tracer", type=str, default="halo", choices=["halo", "lae", "lbg"])
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)
    count_key = {"halo": "halo_count_lc", "lae": "lae_count_lc", "lbg": "lbg_count_lc"}[args.tracer]
    color = TRACER_COLORS.get(args.tracer, (0.8, 0.2, 0.8))

    print("Loading lightcone products...")
    field_data, tracer_data = load_lightcone_products(cfg, seeds)
    available = [s for s in seeds if s in field_data and s in tracer_data and count_key in tracer_data[s]]
    if not available:
        print(f"No seeds have '{count_key}' -- nothing to show.")
        return 1
    seed0 = args.seed if args.seed is not None else available[0]
    if seed0 not in available:
        print(f"Seed {seed0} not available (have: {available}) -- using {available[0]} instead.")
        seed0 = available[0]

    hii_dim = cfg.box.hii_dim
    z_arr = field_data[seed0]["z_lc"]
    slice_idx = hii_dim // 2

    xHI_slice_s0 = field_data[seed0]["xHI_lc"][:, slice_idx, :]
    count_slice_s0 = tracer_data[seed0][count_key][:, slice_idx, :]

    xHI_sum_s0 = aggregate_transverse(field_data[seed0]["xHI_lc"], mode="mean")
    count_sum_s0 = aggregate_transverse(tracer_data[seed0][count_key], mode="sum")

    xHI_slice_all = np.mean([field_data[s]["xHI_lc"][:, slice_idx, :] for s in available], axis=0)
    count_slice_all = np.mean([tracer_data[s][count_key][:, slice_idx, :] for s in available], axis=0)

    xHI_sum_all = np.mean([aggregate_transverse(field_data[s]["xHI_lc"], mode="mean") for s in available], axis=0)
    count_sum_all = np.mean([aggregate_transverse(tracer_data[s][count_key], mode="sum") for s in available], axis=0)

    panels = [
        (f"Seed {seed0}, single slice (y={slice_idx})", xHI_slice_s0, count_slice_s0),
        (f"Seed {seed0}, transverse-summed", xHI_sum_s0, count_sum_s0),
        (f"Single slice, averaged over {len(available)} seeds", xHI_slice_all, count_slice_all),
        (f"Transverse-summed, averaged over {len(available)} seeds", xHI_sum_all, count_sum_all),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), dpi=110, constrained_layout=True, sharex=True)
    for ax, (title, xhi, count) in zip(axes, panels):
        im = ax.imshow(xhi, cmap=XH_CMAP, aspect="auto", origin="lower", vmin=0, vmax=1)
        ax.imshow(_rgba_overlay_log(count, color), aspect="auto", origin="lower", zorder=3)
        _setup_axes(ax, cfg, z_arr)
        nonzero_frac = np.mean(count > 0)
        ax.set_title(f"{title}  --  nonzero fraction: {nonzero_frac:.3%}")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02).set_label(r"$x_\mathrm{HI}$")
    fig.suptitle(f"{args.tracer} tracer -- faithful rendering at four aggregation levels ({cfg.box.box_len_mpc:.0f} Mpc box)",
                 fontsize=13)
    outpath1 = os.path.join(args.out_dir, f"lightcone_aggregation_levels_{args.tracer}_seed{seed0}.pdf")
    fig.savefig(outpath1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath1}")

    nonzero_frac_slice = np.mean(count_slice_s0 > 0)
    fig2, (ax_old, ax_new) = plt.subplots(2, 1, figsize=(16, 9), dpi=110, constrained_layout=True, sharex=True)

    _old_scatter_overlay_FOR_COMPARISON_ONLY(ax_old, xHI_slice_s0, count_slice_s0, z_arr, cfg)
    ax_old.set_title(f"OLD (removed) scatter rendering -- SAME data, nonzero fraction {nonzero_frac_slice:.3%}")

    im_new = ax_new.imshow(xHI_slice_s0, cmap=XH_CMAP, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax_new.imshow(_rgba_overlay_log(count_slice_s0, color), aspect="auto", origin="lower", zorder=3)
    _setup_axes(ax_new, cfg, z_arr)
    ax_new.set_title(f"Current faithful rendering -- SAME data, nonzero fraction {nonzero_frac_slice:.3%}")
    fig2.colorbar(im_new, ax=ax_new, pad=0.01, fraction=0.02).set_label(r"$x_\mathrm{HI}$")

    fig2.suptitle(f"Same {args.tracer} data, seed {seed0}, single slice -- two renderings\n"
                  f"(the top panel is what made the old lightcone look dense; it is the SAME "
                  f"{nonzero_frac_slice:.3%}-nonzero array as the bottom panel)", fontsize=12)
    outpath2 = os.path.join(args.out_dir, f"lightcone_fluke_reveal_{args.tracer}_seed{seed0}.pdf")
    fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {outpath2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
