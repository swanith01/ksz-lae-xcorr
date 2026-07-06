"""
plotting/lightcone_panels.py
==============================
2D lightcone slice plots (xHI + tracer overlays). Generalized from
replot_lightcone.py off the box/grid parameters in Config, instead of a
single hardcoded seed-1 400 Mpc example.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

XH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "green_white", ["#00441b", "#1a7c3a", "#52b365", "#b7e0b1", "white"]
)


def _setup_axes(ax, cfg, z_arr):
    n_lc_pix = len(z_arr)
    cell = cfg.box.box_len_mpc / cfg.box.hii_dim
    zlabels = [z for z in [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20] if cfg.box.z_min <= z <= z_arr.max()]
    zlocs = [(z - cfg.box.z_min) / (z_arr.max() - cfg.box.z_min) * n_lc_pix for z in zlabels]
    ylabels = np.linspace(0, cfg.box.box_len_mpc, 4)
    ylocs = ylabels / cell
    ax.set_xticks(zlocs)
    ax.set_xticklabels(zlabels)
    ax.set_xlabel("z")
    ax.set_yticks(ylocs)
    ax.set_yticklabels([f"{y:.0f}" for y in ylabels])
    ax.set_ylabel("cMpc")


def plot_xhi_halo_overlay(cfg, lc_xHI: np.ndarray, lc_halo_count: np.ndarray, z_arr: np.ndarray,
                           out_dir: str, seed: int) -> None:
    """3-panel figure: xHI alone, halo positions alone, overlay -- one seed."""
    os.makedirs(out_dir, exist_ok=True)
    yy, xx = np.nonzero(lc_halo_count)
    counts = lc_halo_count[yy, xx]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), dpi=120, sharex=True, constrained_layout=True)

    im0 = axes[0].imshow(lc_xHI, cmap=XH_CMAP, aspect="auto", origin="lower", vmin=0, vmax=1)
    _setup_axes(axes[0], cfg, z_arr)
    fig.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.025).set_label(r"$x_\mathrm{HI}$")
    axes[0].set_title(f"Seed {seed} -- xHI only")

    axes[1].scatter(xx, yy, s=np.clip(counts, 1, 20), c="black", linewidths=0, alpha=0.6)
    axes[1].set_xlim(0, len(z_arr))
    axes[1].set_ylim(0, cfg.box.hii_dim)
    _setup_axes(axes[1], cfg, z_arr)
    axes[1].set_title("Halo positions only")

    im2 = axes[2].imshow(lc_xHI, cmap=XH_CMAP, aspect="auto", origin="lower", vmin=0, vmax=1)
    axes[2].scatter(xx, yy, s=np.clip(counts, 1, 15), c="red", linewidths=0, alpha=0.15, zorder=3)
    _setup_axes(axes[2], cfg, z_arr)
    fig.colorbar(im2, ax=axes[2], pad=0.01, fraction=0.025).set_label(r"$x_\mathrm{HI}$")
    axes[2].set_title("Overlay")

    fig.suptitle(f"Seed {seed} -- {cfg.box.box_len_mpc:.0f} Mpc lightcone", fontsize=14)
    outpath = os.path.join(out_dir, f"lc_panels_combined_seed{seed}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_four_tracer_panel(cfg, lc_xHI, lc_halos, lc_lae, lc_lbg, z_arr, out_dir, seed) -> None:
    """4-row figure: xHI+all tracers, xHI+halos, xHI+LAEs, xHI+LBGs -- one seed."""
    os.makedirs(out_dir, exist_ok=True)
    yy_h, xx_h = np.nonzero(lc_halos)
    yy_l, xx_l = np.nonzero(lc_lae)
    yy_b, xx_b = np.nonzero(lc_lbg)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), dpi=100, constrained_layout=True, sharex=True)
    for ax in axes:
        _setup_axes(ax, cfg, z_arr)

    s0 = axes[0].imshow(lc_xHI, cmap=XH_CMAP, aspect="auto", vmin=0, vmax=1, origin="lower")
    axes[0].scatter(xx_h, yy_h, s=10, marker=".", c="red", linewidths=0, label="Halos", zorder=3)
    axes[0].scatter(xx_l, yy_l, s=40, marker="*", c="yellow", linewidths=0, label="LAEs", zorder=4)
    axes[0].scatter(xx_b, yy_b, s=15, marker="^", c="navy", linewidths=0, label="LBGs", zorder=4)
    axes[0].legend(loc="upper right", markerscale=4, framealpha=0.6)
    axes[0].set_title(f"Seed {seed}")
    fig.colorbar(s0, ax=axes[0], pad=0.01, fraction=0.025).set_label(r"$x_\mathrm{HI}$")

    for ax, yy, xx, color, marker, label in [
        (axes[1], yy_h, xx_h, "red", ".", "Halos"),
        (axes[2], yy_l, xx_l, "yellow", "*", "LAEs"),
        (axes[3], yy_b, xx_b, "navy", "^", "LBGs"),
    ]:
        s = ax.imshow(lc_xHI, cmap=XH_CMAP, aspect="auto", vmin=0, vmax=1, origin="lower")
        ax.scatter(xx, yy, s=6, marker=marker, c=color, linewidths=0, zorder=3)
        ax.text(0.99, 0.97, label, transform=ax.transAxes, ha="right", va="top",
                bbox=dict(fc="white", alpha=0.6, ec="none"))
        fig.colorbar(s, ax=ax, pad=0.01, fraction=0.025).set_label(r"$x_\mathrm{HI}$")

    axes[3].set_xlabel(r"$z$")
    outpath = os.path.join(out_dir, f"lightcone_seed{seed}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
