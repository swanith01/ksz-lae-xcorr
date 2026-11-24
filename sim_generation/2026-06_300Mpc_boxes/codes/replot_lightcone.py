#!/usr/bin/env python3
"""
replot_lightcone.py
====================
Re-plot the saved quick_lightcone_seed1.npz as 3 separate panels:
halos alone, xHI alone, and the overlay (with better transparency so
the xHI colormap is actually visible under the halo points).

Usage:
    python replot_lightcone.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

DATA_PATH = "/user1/swanith/kSZ2_LAE_project_22Jun2026/quick_check/quick_lightcone_seed1.npz"
OUT_DIR   = "/user1/swanith/kSZ2_LAE_project_22Jun2026/quick_check"

BOX_LEN  = 300.0
HII_DIM  = 300
CELL     = BOX_LEN / HII_DIM
ZMIN     = 5.0

d = np.load(DATA_PATH)
lc_xHI = d["lc_xHI"]
lc_halo_count = d["lc_halo_count"]
z_arr = d["z_arr"]
N_LC_PIX = len(z_arr)

xH_cmap = mcolors.LinearSegmentedColormap.from_list(
    'green_white', ['#00441b', '#1a7c3a', '#52b365', '#b7e0b1', 'white']
)

zlabels = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
zlabels = [z for z in zlabels if ZMIN <= z <= z_arr.max()]
zlocs = [(z - ZMIN) / (z_arr.max() - ZMIN) * N_LC_PIX for z in zlabels]
ylabels = [0, 100, 200, 300]
ylocs = [y / CELL for y in ylabels]

def setup_axes(ax):
    ax.set_xticks(zlocs)
    ax.set_xticklabels(zlabels)
    ax.set_xlabel("z")
    ax.set_yticks(ylocs)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("cMpc")

yy, xx = np.nonzero(lc_halo_count)
counts = lc_halo_count[yy, xx]

# ---------------------------------------------------------------------------
# Panel 1 — xHI alone
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 5), dpi=120)
im = ax.imshow(lc_xHI, cmap=xH_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
setup_axes(ax)
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
cb.set_label(r"$x_\mathrm{HI}$")
ax.set_title("Seed 1 — xHI only")
plt.savefig(os.path.join(OUT_DIR, "lc_panel_xHI.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved lc_panel_xHI.png")

# ---------------------------------------------------------------------------
# Panel 2 — halos alone (on plain white background, density-shaded)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 5), dpi=120)
sc = ax.scatter(xx, yy, s=np.clip(counts, 1, 20), c='black',
                linewidths=0, alpha=0.6)
ax.set_xlim(0, N_LC_PIX)
ax.set_ylim(0, HII_DIM)
ax.set_facecolor('white')
setup_axes(ax)
ax.set_title("Seed 1 — halo positions only")
plt.savefig(os.path.join(OUT_DIR, "lc_panel_halos.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved lc_panel_halos.png")

# ---------------------------------------------------------------------------
# Panel 3 — overlay, with better-tuned transparency so xHI shows through
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 5), dpi=120)
im = ax.imshow(lc_xHI, cmap=xH_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
sc = ax.scatter(xx, yy, s=np.clip(counts, 1, 15), c='red',
                linewidths=0, alpha=0.15, zorder=3)
setup_axes(ax)
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
cb.set_label(r"$x_\mathrm{HI}$")
ax.set_title("Seed 1 — xHI + halo positions (overlay, low-alpha)")
plt.savefig(os.path.join(OUT_DIR, "lc_panel_overlay.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved lc_panel_overlay.png")

# ---------------------------------------------------------------------------
# Combined 3-row figure for convenience
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 14), dpi=120, sharex=True,
                          constrained_layout=True)

im0 = axes[0].imshow(lc_xHI, cmap=xH_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
setup_axes(axes[0])
cb0 = fig.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.025)
cb0.set_label(r"$x_\mathrm{HI}$")
axes[0].set_title("xHI only")

axes[1].scatter(xx, yy, s=np.clip(counts, 1, 20), c='black', linewidths=0, alpha=0.6)
axes[1].set_xlim(0, N_LC_PIX)
axes[1].set_ylim(0, HII_DIM)
setup_axes(axes[1])
axes[1].set_title("Halo positions only")

im2 = axes[2].imshow(lc_xHI, cmap=xH_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
axes[2].scatter(xx, yy, s=np.clip(counts, 1, 15), c='red', linewidths=0, alpha=0.15, zorder=3)
setup_axes(axes[2])
cb2 = fig.colorbar(im2, ax=axes[2], pad=0.01, fraction=0.025)
cb2.set_label(r"$x_\mathrm{HI}$")
axes[2].set_title("Overlay")

fig.suptitle("Seed 1 — 300 Mpc lightcone (quick check)", fontsize=14)
plt.savefig(os.path.join(OUT_DIR, "lc_panels_combined.png"), dpi=150, bbox_inches='tight')
print("Saved lc_panels_combined.png")
