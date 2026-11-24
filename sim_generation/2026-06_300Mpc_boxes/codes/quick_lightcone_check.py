#!/usr/bin/env python3
"""
quick_lightcone_check.py
=========================
Quick 2D lightcone (xHI + halo positions) for seed 1, 300 Mpc box.
Adapted from stitch_lightcones.py logic, simplified: single seed,
xHI + halos only (no LAE/LBG — not computed for this run), reads
directly from the run_coeval_seed.py output layout
(coeval_z{z}/neutral_fraction.npy, coeval_z{z}/halo_coords.npy).

Usage:
    python quick_lightcone_check.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
import re
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ── Cosmology (Planck-like, matches run_coeval_seed.py defaults) ────────────
COSMO = FlatLambdaCDM(H0=67.66, Om0=0.30964144154550644, Ob0=0.04897468161869667)

# ── Box parameters (300 Mpc production run) ──────────────────────────────────
BOX_LEN  = 300.0
HII_DIM  = 300
CELL     = BOX_LEN / HII_DIM   # 1.0 cMpc

# ── Lightcone parameters ──────────────────────────────────────────────────────
ZMIN      = 5.0
ZMAX      = 20.0
N_LC_PIX  = 512
ANGLE_DEG = 10.0   # small rotation to reduce periodic box repetition
X_SLICE   = HII_DIM // 2   # fixed transverse index for the 2D slice

SEED      = 1
SEED_DIR  = f"/user1/swanith/kSZ2_LAE_project_22Jun2026/seed_{SEED}"
OUT_DIR   = "/user1/swanith/kSZ2_LAE_project_22Jun2026/quick_check"
os.makedirs(OUT_DIR, exist_ok=True)


def comoving_distance_cMpc(z):
    return COSMO.comoving_distance(z).to(u.Mpc).value


def comoving_pixel(z, z0=ZMIN):
    d = comoving_distance_cMpc(z) - comoving_distance_cMpc(z0)
    return int(d / CELL) % HII_DIM


def periodic(n, ngrid=HII_DIM):
    return int(n) % ngrid


def rotate_index(i, j, k, angle_deg=ANGLE_DEG, ngrid=HII_DIM):
    a = np.deg2rad(angle_deg)
    ir = np.cos(a) * i - np.sin(a) * j
    jr = np.sin(a) * i + np.cos(a) * j
    return periodic(ir, ngrid), periodic(jr, ngrid), periodic(k, ngrid)


def get_slab(box, y_cell):
    """Extract a rotated 2D slab (transverse plane) at LoS index y_cell."""
    slab = np.zeros((HII_DIM, HII_DIM), dtype=np.float32)
    for ix in range(HII_DIM):
        for iy in range(HII_DIM):
            rx, ry, rz = rotate_index(ix, iy, y_cell)
            slab[ix, iy] = box[rx, ry, rz]
    return slab


def find_nearest(z, arr):
    return arr[np.argmin(np.abs(arr - z))]


# ---------------------------------------------------------------------------
# 1. Discover available redshifts
# ---------------------------------------------------------------------------

dirs = sorted(glob.glob(os.path.join(SEED_DIR, "coeval_z*")))
snap_z = []
for d in dirs:
    m = re.search(r"coeval_z([\d.]+)$", d)
    if m and os.path.exists(os.path.join(d, "DONE")):
        snap_z.append(float(m.group(1)))
snap_z = np.array(sorted(snap_z))
print(f"Found {len(snap_z)} completed redshifts: z={snap_z[0]:.4f} -> z={snap_z[-1]:.4f}")

# ---------------------------------------------------------------------------
# 2. Build a coarse halo grid + load xHI for every snapshot
#    (halo catalogs are per-halo lists; bin onto the HII_DIM grid as a
#    simple occupation count for visualization)
# ---------------------------------------------------------------------------

print("Loading xHI boxes and binning halo positions...")
xHI_boxes  = {}
halo_grids = {}

for z in snap_z:
    out_dir = os.path.join(SEED_DIR, f"coeval_z{z:.6f}")
    xHI_boxes[z] = np.load(os.path.join(out_dir, "neutral_fraction.npy"))

    coords = np.load(os.path.join(out_dir, "halo_coords.npy"))
    grid = np.zeros((HII_DIM, HII_DIM, HII_DIM), dtype=np.float32)
    ix = np.clip((coords[:, 0] / CELL).astype(int), 0, HII_DIM - 1)
    iy = np.clip((coords[:, 1] / CELL).astype(int), 0, HII_DIM - 1)
    iz = np.clip((coords[:, 2] / CELL).astype(int), 0, HII_DIM - 1)
    np.add.at(grid, (ix, iy, iz), 1.0)
    halo_grids[z] = grid
    print(f"  z={z:.4f}  xHI mean={xHI_boxes[z].mean():.4f}  N_halos={len(coords):,}")

# ---------------------------------------------------------------------------
# 3. Stitch into a 2D lightcone
# ---------------------------------------------------------------------------

print("\nStitching lightcone...")
z_arr = np.linspace(ZMIN, min(ZMAX, snap_z.max()), N_LC_PIX)

lc_xHI = np.zeros((HII_DIM, N_LC_PIX), dtype=np.float32)
lc_halo_count = np.zeros((HII_DIM, N_LC_PIX), dtype=np.float32)

for n, z in enumerate(z_arr):
    y_cell = comoving_pixel(z)
    nearest_z = find_nearest(z, snap_z)

    xHI_slab = get_slab(xHI_boxes[nearest_z], y_cell)
    lc_xHI[:, n] = xHI_slab[:, X_SLICE]

    halo_slab = get_slab(halo_grids[nearest_z], y_cell)
    lc_halo_count[:, n] = halo_slab[:, X_SLICE]

    if n % 64 == 0:
        print(f"  pixel {n}/{N_LC_PIX}  z={z:.3f}  nearest snapshot z={nearest_z:.4f}")

np.savez_compressed(
    os.path.join(OUT_DIR, f"quick_lightcone_seed{SEED}.npz"),
    lc_xHI=lc_xHI, lc_halo_count=lc_halo_count, z_arr=z_arr,
)
print(f"\nSaved lightcone arrays to {OUT_DIR}/quick_lightcone_seed{SEED}.npz")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

xH_cmap = mcolors.LinearSegmentedColormap.from_list(
    'green_white', ['#00441b', '#1a7c3a', '#52b365', '#b7e0b1', 'white']
)

yy, xx = np.nonzero(lc_halo_count)
counts = lc_halo_count[yy, xx]

fig, ax = plt.subplots(figsize=(16, 5), dpi=120)
im = ax.imshow(lc_xHI, cmap=xH_cmap, aspect='auto', origin='lower',
               vmin=0, vmax=1)
sc = ax.scatter(xx, yy, s=np.clip(counts, 1, 20), c='red',
                linewidths=0, alpha=0.5, zorder=3)

zlabels = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
zlabels = [z for z in zlabels if ZMIN <= z <= z_arr.max()]
zlocs = [(z - ZMIN) / (z_arr.max() - ZMIN) * N_LC_PIX for z in zlabels]
ax.set_xticks(zlocs)
ax.set_xticklabels(zlabels)
ax.set_xlabel("z")

ylabels = [0, 100, 200, 300]
ylocs = [y / CELL for y in ylabels]
ax.set_yticks(ylocs)
ax.set_yticklabels(ylabels)
ax.set_ylabel("cMpc")

cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
cb.set_label(r"$x_\mathrm{HI}$")
ax.set_title(f"Seed {SEED} — 300 Mpc lightcone (quick check): xHI + halo positions")

plot_path = os.path.join(OUT_DIR, f"quick_lightcone_seed{SEED}.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
