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

from ksz_lae_xcorr.utils.figio import save_fig

XH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "green_white", ["#00441b", "#1a7c3a", "#52b365", "#b7e0b1", "white"]
)

# Fixed per-tracer colors for the overlay plots below -- kept in one place
# so a given tracer always reads the same color across every figure.
TRACER_COLORS = {
    "halo": (0.82, 0.10, 0.10),   # red
    "lae": (1.00, 0.82, 0.00),    # yellow/gold
    "lbg": (0.10, 0.30, 0.70),    # navy
}


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


def _rgba_overlay_log(count: np.ndarray, rgb: tuple[float, float, float],
                       max_alpha: float = 0.9) -> np.ndarray:
    """
    Build an (H, W, 4) RGBA layer for a discrete count field: fully
    transparent where count==0, opaque `rgb` where count is at its max,
    log-scaled alpha in between -- an EXACT, per-pixel-faithful rendering
    of the array's own values.

    Deliberately replaces the old scatter-based overlay (removed --
    ax.scatter(xx, yy, s=clip(counts,1,20), alpha=0.6) drew a fixed-size,
    semi-transparent dot per nonzero cell; overlapping dots visually merge
    regardless of the array's real sparsity, so even ~genuinely-sparse
    real data (see ksz-lae-xcorr_HANDOFF.md: halos ~0.6%, LAE ~0.02%
    nonzero in a single slice) looked like a dense continuous field. An
    imshow of this RGBA array can only show exactly what's in the array --
    there is no marker-size or blending knob left to accidentally invent
    density that isn't there.
    """
    count = np.asarray(count, dtype=np.float64)
    vmax = float(count.max()) if count.size else 0.0
    if vmax <= 0:
        alpha = np.zeros_like(count)
    else:
        alpha = np.log1p(np.clip(count, 0, None)) / np.log1p(vmax)
    rgba = np.zeros(count.shape + (4,), dtype=np.float64)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0) * max_alpha
    return rgba


def plot_xhi_tracer_overlay(cfg, lc_xHI: np.ndarray, tracers: dict[str, np.ndarray],
                             z_arr: np.ndarray, out_dir: str, seed: int) -> None:
    """
    Faithful xHI + tracer overlay(s) -- one row per tracer in `tracers`
    (e.g. {'halo': lc_halo_count} or {'halo': ..., 'lae': ..., 'lbg': ...}),
    plus a combined row if more than one tracer is given. Each tracer is
    drawn as an exact per-pixel log-alpha RGBA layer (_rgba_overlay_log)
    on top of the xHI base image -- no scatter markers.

    Replaces plot_xhi_halo_overlay and plot_four_tracer_panel (both
    scatter-based, removed in the same change -- see git history and
    ksz-lae-xcorr_HANDOFF.md's lightcone-image-mismatch note for why).

    Aggregation is the CALLER's responsibility (pass already-2D arrays) --
    use utils.grid.aggregate_transverse with a consistent mode ('sum' for
    every discrete tracer here, since that's what these are) so this
    figure and plot_four_field_panels agree on what "the data" looks like.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = list(tracers.keys())
    n_rows = len(names) + (1 if len(names) > 1 else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.6 * n_rows), dpi=120,
                              sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    def _draw_base(ax):
        im = ax.imshow(lc_xHI, cmap=XH_CMAP, aspect="auto", origin="lower", vmin=0, vmax=1)
        _setup_axes(ax, cfg, z_arr)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025).set_label(r"$x_\mathrm{HI}$")

    for row, name in enumerate(names):
        ax = axes[row]
        _draw_base(ax)
        color = TRACER_COLORS.get(name, (0.8, 0.2, 0.8))
        ax.imshow(_rgba_overlay_log(tracers[name], color), aspect="auto", origin="lower", zorder=3)
        ax.set_title(f"Seed {seed} -- xHI + {name} (exact log-alpha overlay)")

    if len(names) > 1:
        ax = axes[-1]
        _draw_base(ax)
        for name in names:
            color = TRACER_COLORS.get(name, (0.8, 0.2, 0.8))
            ax.imshow(_rgba_overlay_log(tracers[name], color), aspect="auto", origin="lower", zorder=3)
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="none", markersize=10,
                       markerfacecolor=TRACER_COLORS.get(n, (0.8, 0.2, 0.8)),
                       markeredgewidth=0, label=n)
            for n in names
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.6)
        ax.set_title(f"Seed {seed} -- xHI + all tracers (exact log-alpha overlay)")

    fig.suptitle(f"Seed {seed} -- {cfg.box.box_len_mpc:.0f} Mpc lightcone", fontsize=14)
    outpath = os.path.join(out_dir, f"lc_tracer_overlay_seed{seed}.pdf")
    save_fig(fig, outpath, dpi=150)
    plt.close(fig)


def plot_four_field_panels(cfg, lc_xHI, lc_halos, lc_lae, lc_lbg, z_arr, out_dir, seed) -> None:
    """
    4-row figure: xHI, halos, LAE, LBG -- each field shown STANDALONE (own
    colormap, own colorbar), no xHI-background overlay. Use this alongside
    plot_xhi_tracer_overlay: this one is best for seeing each field's own
    structure/dynamic range in isolation, the overlay is best for seeing
    where tracers sit relative to the ionization state.

    Discrete tracer panels use a LOG color scale, not linear -- these fields
    are extremely sparse count data (the overwhelming majority of nonzero
    cells have count 1, with rare outlier cells reaching much higher). A
    linear scale stretched to the single brightest pixel makes every count=1
    cell look almost indistinguishable from true zero, hiding real structure
    -- not because the data is actually empty.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(16, 15), constrained_layout=True, sharex=True)

    im0 = axes[0].imshow(lc_xHI, cmap=XH_CMAP, aspect="auto", vmin=0, vmax=1, origin="lower")
    _setup_axes(axes[0], cfg, z_arr)
    fig.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.025).set_label(r"$x_\mathrm{HI}$")
    axes[0].set_title(f"Seed {seed} -- xHI")

    for ax, field, cmap, label in [
        (axes[1], lc_halos, "inferno", "Halo count"),
        (axes[2], lc_lae, "viridis", "LAE count"),
        (axes[3], lc_lbg, "cividis", "LBG count"),
    ]:
        vmax = max(field.max(), 1)
        norm = mcolors.LogNorm(vmin=1, vmax=vmax, clip=True)
        im = ax.imshow(field, cmap=cmap, aspect="auto", norm=norm, origin="lower")
        _setup_axes(ax, cfg, z_arr)
        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label(f"{label} (log scale; 0 shown as darkest)")
        ax.set_title(f"{label}  (nonzero fraction: {np.count_nonzero(field)/field.size:.2%})")

    axes[3].set_xlabel(r"Redshift $z$")
    outpath = os.path.join(out_dir, f"lightcone_fields_seed{seed}.pdf")
    save_fig(fig, outpath)
    plt.close(fig)



# plot_four_tracer_panel (scatter-based xHI+halos/LAEs/LBGs overlay) was
# removed here -- superseded by plot_xhi_tracer_overlay above, which does
# the same job (xHI + each tracer, plus a combined panel) with an exact
# per-pixel RGBA overlay instead of scatter markers. Call
# plot_xhi_tracer_overlay(cfg, lc_xHI, {"halo": ..., "lae": ..., "lbg": ...},
# z_arr, out_dir, seed) for the equivalent figure.
