# %% [markdown]
# # 400 Mpc Lightcone — Visualisation & Analysis
# **Project:** 26Jun2026_400Mpc_halo_LAE_LBG  
# **Box:** 400 cMpc, 64³, 5 seeds, z=5.12→19.89  
# **Fields:** xHI, density, vz, halos, LAEs, LBGs  
# 
# Run `stitch_lightcones.py` on the cluster first, then rsync to desktop.

# %% [markdown]
# ## 0. Rsync from cluster
# Run this in a terminal before opening the notebook:

# %%
# Run in terminal — not a Python cell
# rsync -avz --progress \
#   swanith@swarm:/user1/swanith/kSZ2_halo_project/26Jun2026_400Mpc_halo_LAE_LBG/seed_* \
#   ~/Desktop/Research/'Semester 6'/Plots/LAE_cross/26Jun2026_400Mpc_halo_LAE_LBG/data/
print("See comment above for rsync command")

# %% [markdown]
# ## 1. Imports & parameters

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rc('axes', titlesize=18, labelsize=16)
plt.rc('legend', fontsize=12)

# %%
# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = os.path.expanduser(
    "~/Desktop/Research/Semester 6/Plots/LAE_cross/26Jun2026_HALO_LBG_LAE"
)
DATA_DIR  = PROJECT
PLOT_DIR  = os.path.join(PROJECT, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Box parameters ────────────────────────────────────────────────────────────
BOX_LEN  = 400.0
NGRID    = 64
CELL     = BOX_LEN / NGRID   # 6.25 cMpc
ZMIN     = 5.12
ZMAX     = 19.89
N_LC_PIX = 512
ALL_SEEDS = [1, 2, 3, 4, 5]

z_arr = np.linspace(ZMIN, ZMAX, N_LC_PIX)
dz    = (ZMAX - ZMIN) / N_LC_PIX

print(f"Project: {PROJECT}")
print(f"z_arr: {z_arr[0]:.3f} → {z_arr[-1]:.3f}  ({N_LC_PIX} pixels)")

# %%
for s in [1,2,3,4,5]:
    lae = np.load(f'seed_{s}/lc_lae.npz')['lc']
    yy, xx = np.nonzero(lae)
    if len(yy) > 0:
        print(f"seed {s}: N={len(xx)}  y_min={yy.min()}  y_max={yy.max()}")
    else:
        print(f"seed {s}: EMPTY")

# %% [markdown]
# ## 2. Check which seeds are ready

# %%
FIELDS = ['xH', 'density', 'vz', 'halos', 'lae', 'lbg']

print(f"{'Field':<12}" + "".join([f"  seed{s}" for s in ALL_SEEDS]))
print("-" * 50)
for field in FIELDS:
    row = f"{field:<12}"
    for s in ALL_SEEDS:
        fpath = os.path.join(DATA_DIR, f'seed_{s}', f'lc_{field}.npz')
        row += "  ✅    " if os.path.exists(fpath) else "  ❌    "
    print(row)

# %% [markdown]
# ## 3. Load lightcones

# %%
def load_lc(seed, field):
    """Load a lightcone array for a given seed and field."""
    path = os.path.join(DATA_DIR, f'seed_{seed}', f'lc_{field}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not yet computed: {path}")
    return np.load(path)['lc']   # shape (NGRID, N_LC_PIX)

def load_all_seeds(field):
    """Load a field for all available seeds. Returns dict seed→array."""
    out = {}
    for s in ALL_SEEDS:
        try:
            out[s] = load_lc(s, field)
            print(f"  seed {s}  {field}  shape={out[s].shape}  "
                  f"min={out[s].min():.4f}  max={out[s].max():.4f}")
        except FileNotFoundError as e:
            print(f"  seed {s}  {field}  NOT FOUND")
    return out

# ── Load (only runs for seeds that are ready) ─────────────────────────────────
print("Loading xHI...")
lc_xH = load_all_seeds('xH')

print("\nLoading density...")
lc_den = load_all_seeds('density')

print("\nLoading vz...")
lc_vz = load_all_seeds('vz')

print("\nLoading halos...")
lc_halos = load_all_seeds('halos')

print("\nLoading LAEs...")
lc_lae = load_all_seeds('lae')

print("\nLoading LBGs...")
lc_lbg = load_all_seeds('lbg')

# %% [markdown]
# ## 4. Plot helpers

# %%
# ── Colormaps ─────────────────────────────────────────────────────────────────
xH_cmap = mcolors.LinearSegmentedColormap.from_list(
    'green_white', ['#00441b', '#1a7c3a', '#52b365', '#b7e0b1', 'white']
)

# ── Z-axis ticks ─────────────────────────────────────────────────────────────
zlabels = [6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
zlocs   = [(x - ZMIN) / dz for x in zlabels]
ylabels = [0, 100, 200, 300, 400]
ylocs   = [x / CELL for x in ylabels]

def setup_axes(ax_list):
    for a in ax_list:
        a.set_xlim(0, N_LC_PIX)
        a.set_ylim(0, NGRID)
        a.set_xticks(zlocs)
        a.set_xticklabels(zlabels)
        a.set_yticks(ylocs)
        a.set_yticklabels(ylabels)

def add_colorbar(fig, mappable, ax, label):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="2.5%", pad=0.08)
    cb  = fig.colorbar(mappable, cax=cax)
    vmin, vmax = mappable.get_clim()
    locator = MaxNLocator(nbins=4, steps=[1,2,2.5,5,10])
    ticks   = locator.tick_values(vmin, vmax)
    ticks   = ticks[(ticks > vmin) & (ticks < vmax)]
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb.set_label(label, labelpad=10)
    cb.solids.set_edgecolor("face")
    return cb

print("Plot helpers defined.")

# %% [markdown]
# ## 5. Plot — one seed at a time

# %%
from matplotlib.pyplot import axes


def plot_seed(seed, save=True):
    """4-panel lightcone plot for one seed."""
    # Check all fields available
    for d, name in [(lc_xH, 'xH'), (lc_halos, 'halos'),
                    (lc_lae, 'lae'), (lc_lbg, 'lbg')]:
        if seed not in d:
            print(f"Seed {seed}: {name} not loaded, skipping plot")
            return

    xH     = lc_xH[seed]
    halos  = lc_halos[seed]
    lae    = lc_lae[seed]
    lbg    = lc_lbg[seed]

    yy_h, xx_h = np.nonzero(halos)
    yy_l, xx_l = np.nonzero(lae)
    yy_b, xx_b = np.nonzero(lbg)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), dpi=100,
                             constrained_layout=True, sharex=True)
    setup_axes(axes)

    # Row 0: xHI + all tracers
    s0 = axes[0].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    axes[0].scatter(xx_h, yy_h, s=10, marker='.',  c='red',    linewidths=0, label='Halos', zorder=3)
    axes[0].scatter(xx_l, yy_l, s=40, marker='*',  c='yellow', linewidths=0, label='LAEs',  zorder=4)
    axes[0].scatter(xx_b, yy_b, s=15, marker='^',  c='navy',   linewidths=0, label='LBGs',  zorder=4)
    axes[0].legend(loc='upper right', markerscale=4, framealpha=0.6)
    axes[0].set_title(f'Seed {seed}', fontsize=14)
    cb0 = fig.colorbar(s0, ax=axes[0], pad=0.01, fraction=0.025)
    cb0.set_label(r"$x_\mathrm{HI}$", labelpad=8)

    # Row 1: xHI + halos
    s1 = axes[1].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    axes[1].scatter(xx_h, yy_h, s=3, marker='.', c='red', linewidths=0, zorder=3)
    axes[1].text(0.99, 0.97, 'Halos', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=12,
                 bbox=dict(fc='white', alpha=0.6, ec='none'))
    cb1 = fig.colorbar(s1, ax=axes[1], pad=0.01, fraction=0.025)
    cb1.set_label(r"$x_\mathrm{HI}$", labelpad=8)

    # Row 2: xHI + LAEs
    s2 = axes[2].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    axes[2].scatter(xx_l, yy_l, s=30, marker='*', c='yellow', linewidths=0, zorder=3)
    axes[2].text(0.99, 0.97, 'LAEs', transform=axes[2].transAxes,
                 ha='right', va='top', fontsize=12,
                 bbox=dict(fc='white', alpha=0.6, ec='none'))
    cb2 = fig.colorbar(s2, ax=axes[2], pad=0.01, fraction=0.025)
    cb2.set_label(r"$x_\mathrm{HI}$", labelpad=8)

    # Row 3: xHI + LBGs
    s3 = axes[3].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    axes[3].scatter(xx_b, yy_b, s=4, marker='^', c='navy', linewidths=0, zorder=3)
    axes[3].text(0.99, 0.97, 'LBGs', transform=axes[3].transAxes,
                 ha='right', va='top', fontsize=12,
                 bbox=dict(fc='white', alpha=0.6, ec='none'))
    cb3 = fig.colorbar(s3, ax=axes[3], pad=0.01, fraction=0.025)
    cb3.set_label(r"$x_\mathrm{HI}$", labelpad=8)

    axes[3].set_xlabel(r'$z$', fontsize=16)
    fig.text(-0.01, 0.5, r'cMpc', va='center', rotation='vertical', fontsize=16)

    if save:
        seed_plot_dir = os.path.join(PLOT_DIR, f'seed_{seed}')
        os.makedirs(seed_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(seed_plot_dir, f'lightcone_seed{seed}.pdf'),
                    dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(seed_plot_dir, f'lightcone_seed{seed}.pdf'),
                    bbox_inches='tight')
        print(f"Saved seed {seed} plot")
    plt.show()

# ── Plot all available seeds ──────────────────────────────────────────────────
for s in ALL_SEEDS:
    plot_seed(s)

# %%
#Plotting Proportional marker size lightcones####

# ── Load Type B data ──────────────────────────────────────────────────────────
def load_typeB(seed, field):
    path = os.path.join(DATA_DIR, f'seed_{seed}', f'lc_{field}.npz')
    d = np.load(path)
    return d['lc_occ'], d['lc_val']

# Load all seeds
print("Loading Type B lightcones...")
lc_halos_occ, lc_halos_mass = {}, {}
lc_lae_occ,   lc_lae_lum   = {}, {}
lc_lbg_occ,   lc_lbg_muv   = {}, {}

for s in ALL_SEEDS:
    try:
        lc_halos_occ[s], lc_halos_mass[s] = load_typeB(s, 'halos_mass')
        lc_lae_occ[s],   lc_lae_lum[s]   = load_typeB(s, 'lae_lum')
        lc_lbg_occ[s],   lc_lbg_muv[s]   = load_typeB(s, 'lbg_muv')
        print(f"  seed {s} ✅  "
              f"mass: {lc_halos_mass[s][lc_halos_occ[s]>0].mean():.2e} Msun  "
              f"log10(L): {np.log10(lc_lae_lum[s][lc_lae_occ[s]>0]).mean():.2f}  "
              f"MUV: {lc_lbg_muv[s][lc_lbg_occ[s]>0].mean():.2f}")
    except FileNotFoundError:
        print(f"  seed {s} ❌ not found")

# ── Proportional marker plot ──────────────────────────────────────────────────
def size_from_values(vals, vmin, vmax, smin=4, smax=80):
    v = np.clip(vals, vmin, vmax)
    return smin + (smax - smin) * (v - vmin) / (vmax - vmin)

def plot_seed_typeB(seed, save=True):
    if seed not in lc_halos_mass:
        print(f"Seed {seed} not loaded")
        return

    xH     = lc_xH[seed]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), dpi=100,
                             constrained_layout=True, sharex=True)
    setup_axes(axes)

    # ── Row 0: xHI + halos, size/color ∝ log10(M) ────────────────────────
    s0 = axes[0].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    yy_h, xx_h = np.nonzero(lc_halos_occ[seed])
    log_mass    = np.log10(lc_halos_mass[seed][yy_h, xx_h])
    # Keep only top 20% most massive
    mass_cut = np.percentile(log_mass, 80)
    mask_h = log_mass >= mass_cut
    yy_h, xx_h, log_mass = yy_h[mask_h], xx_h[mask_h], log_mass[mask_h]
    lm_min, lm_max = np.percentile(log_mass, 5), np.percentile(log_mass, 95)
    sz_h = size_from_values(log_mass, lm_min, lm_max, smin=2, smax=30)
    sc0  = axes[0].scatter(xx_h, yy_h, s=sz_h, c=log_mass,
                           cmap='YlOrRd', vmin=lm_min, vmax=lm_max,
                           linewidths=0, zorder=3)
    axes[0].text(0.99, 0.97, 'Halos', transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=12,
                 bbox=dict(fc='white', alpha=0.6, ec='none'))
    cb0_xh = fig.colorbar(s0,  ax=axes[0], pad=0.01,  fraction=0.025)
    cb0_xh.set_label(r"$x_\mathrm{HI}$", labelpad=8)
    cb0_m  = fig.colorbar(sc0, ax=axes[0], pad=0.055, fraction=0.025)
    cb0_m.set_label(r"$\log_{10}(M_h\,[M_\odot])$", labelpad=8)

    # ── Row 1: xHI + LAEs, size/color ∝ log10(L_Lya) ────────────────────
    s1 = axes[1].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    yy_l, xx_l = np.nonzero(lc_lae_occ[seed])
    log_lum     = np.log10(lc_lae_lum[seed][yy_l, xx_l])
    # Keep only top 30% most luminous
    lum_cut = np.percentile(log_lum, 70)
    mask_l = log_lum >= lum_cut
    yy_l, xx_l, log_lum = yy_l[mask_l], xx_l[mask_l], log_lum[mask_l]
    ll_min, ll_max = np.percentile(log_lum, 5), np.percentile(log_lum, 95)
    sz_l = size_from_values(log_lum,  ll_min, ll_max, smin=2, smax=30)
    sc1  = axes[1].scatter(xx_l, yy_l, s=sz_l, c=log_lum,
                           cmap='plasma', vmin=ll_min, vmax=ll_max,
                           linewidths=0, zorder=3)
    axes[1].text(0.99, 0.97, 'LAEs', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=12,
                 bbox=dict(fc='white', alpha=0.6, ec='none'))
    cb1_xh = fig.colorbar(s1,  ax=axes[1], pad=0.01,  fraction=0.025)
    cb1_xh.set_label(r"$x_\mathrm{HI}$", labelpad=8)
    cb1_l  = fig.colorbar(sc1, ax=axes[1], pad=0.055, fraction=0.025)
    cb1_l.set_label(r"$\log_{10}(L_{\mathrm{Ly}\alpha}\,[\mathrm{erg\,s}^{-1}])$",
                    labelpad=8)

    # ── Row 2: xHI + LBGs, size/color ∝ -MUV ────────────────────────────
    s2 = axes[2].imshow(xH, cmap=xH_cmap, aspect='auto',
                        rasterized=True, vmin=0, vmax=1, origin='lower')
    yy_b, xx_b = np.nonzero(lc_lbg_occ[seed])
    neg_muv     = -lc_lbg_muv[seed][yy_b, xx_b]
    # Keep only MUV < -18 (bright LBGs)
    mask_b = neg_muv > 18.0
    yy_b, xx_b, neg_muv = yy_b[mask_b], xx_b[mask_b], neg_muv[mask_b]
    nm_min, nm_max = np.percentile(neg_muv, 5), np.percentile(neg_muv, 95)
    sz_b = size_from_values(neg_muv,  nm_min, nm_max, smin=2, smax=30)
    sc2  = axes[2].scatter(xx_b, yy_b, s=sz_b, c=neg_muv,
                           cmap='cool', vmin=nm_min, vmax=nm_max,
                           linewidths=0, zorder=3)
    axes[2].text(0.99, 0.97, 'LBGs', transform=axes[2].transAxes,
                 ha='right', va='top', fontsize=12,
                 bbox=dict(fc='white', alpha=0.6, ec='none'))
    cb2_xh = fig.colorbar(s2,  ax=axes[2], pad=0.01,  fraction=0.025)
    cb2_xh.set_label(r"$x_\mathrm{HI}$", labelpad=8)
    cb2_m  = fig.colorbar(sc2, ax=axes[2], pad=0.055, fraction=0.025)
    cb2_m.set_label(r"$-M_\mathrm{UV}$", labelpad=8)

    axes[2].set_xlabel(r'$z$', fontsize=16)
    fig.text(-0.01, 0.5, r'cMpc', va='center', rotation='vertical', fontsize=16)
    fig.suptitle(f'Seed {seed} — proportional markers', fontsize=14)

    if save:
        seed_plot_dir = os.path.join(PLOT_DIR, f'seed_{seed}')
        os.makedirs(seed_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(seed_plot_dir,
                    f'lightcone_typeB_seed{seed}.pdf'), dpi=200, bbox_inches='tight')
        print(f"Saved seed {seed} Type B plot")
    plt.show()

# ── Plot all seeds ────────────────────────────────────────────────────────────
for s in ALL_SEEDS:
    plot_seed_typeB(s)

# %% [markdown]
# ## 7. kSZ cross-correlations
# *Placeholder — details to be filled in.*

# %%
# =============================================================================
# CELL 2 — Load 3D lightcones into field_data and tracer_data
# Replaces the old FIELD_DIR / HALO_DIR / CATALOGUE_DIR loading
# =============================================================================

print('='*60)
print('CELL 2 — Loading 3D lightcones')
print('='*60)
SEEDS = [1, 2, 3, 4, 5]
LC_DIR_3D = os.path.expanduser(
    '~/Desktop/Research/Semester 6/Plots/LAE_cross/27Jun2026_400Mpc_halo_LAE_LBG_3D'
)

# LoS redshift array — must match stitch_lightcones_3D.py
z_lc = np.linspace(5.12, 19.89, 512)

# vz conversion: our lc_vz is already in km/s (applied in stitch_lightcones_3D.py)
# but Cell 4 needs both km/s AND Mpc/s
c_kms   = 299792.458
c_Mpc_s = c_kms / 3.08567758e19

field_data  = {}
tracer_data = {}

for s in SEEDS:
    seed_dir = os.path.join(LC_DIR_3D, f'seed_{s}')
    if not os.path.exists(seed_dir):
        print(f'  seed {s} ❌ not found')
        continue

    # ── continuous fields ─────────────────────────────────────────────────
    xHI     = np.load(os.path.join(seed_dir, 'lc_xH.npz'))    ['lc']  # (64,64,512)
    density = np.load(os.path.join(seed_dir, 'lc_density.npz'))['lc']  # (64,64,512) — δ
    vz_kms  = np.load(os.path.join(seed_dir, 'lc_vz.npz'))    ['lc']  # (64,64,512) km/s

    # density in our file is δ (overdensity), Cell 4 needs (1+δ)
    density_1pdelta = 1.0 + density.astype(np.float64)

    # velocity in Mpc/s for the kSZ integrand
    vz_Mpc_s = vz_kms.astype(np.float64) / c_kms * c_Mpc_s   # Mpc/s

    field_data[s] = {
        'z_lc'        : z_lc,
        'xHI_lc'      : xHI.astype(np.float64),
        'density_lc'  : density_1pdelta,        # (1+δ)
        'velocity_kms': vz_kms.astype(np.float64),
        'velocity_lc' : vz_Mpc_s,
    }

    # ── tracer fields ─────────────────────────────────────────────────────
    halos = np.load(os.path.join(seed_dir, 'lc_halos.npz'))['lc']  # (64,64,512)
    lae   = np.load(os.path.join(seed_dir, 'lc_lae.npz'))  ['lc']  # (64,64,512)
    lbg   = np.load(os.path.join(seed_dir, 'lc_lbg.npz'))  ['lc']  # (64,64,512)

    tracer_data[s] = {
        'z_nodes'      : z_lc,
        'halo_count_lc': halos.astype(np.float64),
        'lae_count_lc' : lae.astype(np.float64),
        'lbg_count_lc' : lbg.astype(np.float64),
    }

    print(f'  seed {s} ✅  '
          f'xHI [{xHI.min():.3f},{xHI.max():.3f}]  '
          f'vz [{vz_kms.min():.1f},{vz_kms.max():.1f}] km/s  '
          f'LAE nonzero={np.count_nonzero(lae)}')

print(f'\n  ✓ {len(field_data)}/{len(SEEDS)} seeds loaded')
print(f'  z_lc: {z_lc[0]:.3f} → {z_lc[-1]:.3f}  ({len(z_lc)} pixels)')
print(f'  Field shape: {xHI.shape}')

# %%
# =============================================================================
# CELL 2b — Sanity checks on loaded fields
# =============================================================================
print('='*60)
print('CELL 2b — Field sanity checks')
print('='*60)

for s in SEEDS:
    xHI     = field_data[s]['xHI_lc']
    delta   = field_data[s]['density_lc']      # (1+δ)
    v_kms   = field_data[s]['velocity_kms']
    v_Mpc   = field_data[s]['velocity_lc']
    halos   = tracer_data[s]['halo_count_lc']
    lae     = tracer_data[s]['lae_count_lc']
    lbg     = tracer_data[s]['lbg_count_lc']

    print(f'\n  seed {s}:')
    print(f'    xHI       rms={xHI.std():.4f}   mean={xHI.mean():.4f}   '
          f'[{xHI.min():.4f}, {xHI.max():.4f}]')
    print(f'    (1+δ)     rms={delta.std():.4f}  mean={delta.mean():.4f}  '
          f'[{delta.min():.4f}, {delta.max():.4f}]')
    print(f'    vz km/s   rms={v_kms.std():.2f}  mean={v_kms.mean():.4f}  '
          f'[{v_kms.min():.2f}, {v_kms.max():.2f}]')
    print(f'    vz Mpc/s  rms={v_Mpc.std():.4e}  mean={v_Mpc.mean():.4e}')
    print(f'    halos     rms={halos.std():.4f}  mean={halos.mean():.4f}  '
          f'max={halos.max():.0f}')
    print(f'    LAE       rms={lae.std():.4f}  mean={lae.mean():.6f}  '
          f'nonzero={np.count_nonzero(lae)}')
    print(f'    LBG       rms={lbg.std():.4f}  mean={lbg.mean():.6f}  '
          f'nonzero={np.count_nonzero(lbg)}')

print('\n✓ CELL 2b complete')

# %%
# =============================================================================
# CELL 3 — Constants, cosmology, helper functions for kSZ pipeline
# =============================================================================

from astropy.cosmology import FlatLambdaCDM
import os

# ── Simulation parameters ─────────────────────────────────────────────────
SEEDS    = [1, 2, 3, 4, 5]
BOX_LEN  = 400.0
N_SIDE   = 64
cell_size = BOX_LEN / N_SIDE   # 6.25 cMpc

# ── Cosmology ─────────────────────────────────────────────────────────────
cosmo    = FlatLambdaCDM(H0=67.66, Om0=0.3096, Ob0=0.0490)
h_little = 0.6766

# ── Physical constants ────────────────────────────────────────────────────
c_kms        = 299792.458
c_Mpc_s      = c_kms / 3.08567758e19

Omega_b      = 0.0490
rho_crit_cgs = 1.88e-29 * h_little**2
m_p_g        = 1.6726e-24
n_H0_cm3     = Omega_b * rho_crit_cgs / m_p_g
sigma_T_cm2  = 6.6524e-25
cm_per_Mpc   = 3.08568e24
tau_prefactor = n_H0_cm3 * sigma_T_cm2 * cm_per_Mpc

T_CMB_uK     = 2.7255e6   # μK

# ── 2D Fourier grid ───────────────────────────────────────────────────────
pix_size_Mpc = cell_size
area_2D      = BOX_LEN**2
dk           = 2 * np.pi / BOX_LEN
kx           = np.fft.fftshift(np.fft.fftfreq(N_SIDE)) * N_SIDE * dk
ky           = np.fft.fftshift(np.fft.fftfreq(N_SIDE)) * N_SIDE * dk
KX, KY       = np.meshgrid(kx, ky, indexing='ij')
kgrid        = np.sqrt(KX**2 + KY**2)
k_min        = dk
k_max        = kgrid.max() * 0.95
N_KBINS      = 25
k_bins       = np.logspace(np.log10(k_min), np.log10(k_max), N_KBINS + 1)
k_centers    = 0.5 * (k_bins[:-1] + k_bins[1:])
n_kbins      = len(k_centers)

# ── Output directory ──────────────────────────────────────────────────────
PLOT_DIR_4 = os.path.expanduser(
    '~/Desktop/Research/Semester 6/Plots/LAE_cross/27Jun2026_400Mpc_halo_LAE_LBG_3D'
)
os.makedirs(PLOT_DIR_4, exist_ok=True)

# ── Helper functions ──────────────────────────────────────────────────────
def make_overdensity(field_2d):
    mean = field_2d.mean()
    if mean <= 0:
        return np.zeros_like(field_2d)
    return (field_2d - mean) / mean

def cross_power_2d(field_A, field_B):
    fft_A     = np.fft.fftshift(np.fft.fft2(field_A))
    fft_B     = np.fft.fftshift(np.fft.fft2(field_B))
    norm      = (pix_size_Mpc / N_SIDE)**2
    cross_2d  = np.real(np.conj(fft_A) * fft_B) * norm
    auto_A_2d = np.abs(fft_A)**2 * norm
    auto_B_2d = np.abs(fft_B)**2 * norm
    P_cross   = np.full(n_kbins, np.nan)
    P_err     = np.full(n_kbins, np.nan)
    r_cross   = np.full(n_kbins, np.nan)
    for j in range(n_kbins):
        mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j+1])
        if mask.sum() == 0:
            continue
        cv         = cross_2d[mask]
        PA         = np.mean(auto_A_2d[mask])
        PB         = np.mean(auto_B_2d[mask])
        P_cross[j] = np.mean(cv)
        dk_bin     = k_bins[j+1] - k_bins[j]
        n_modes    = max(1, k_centers[j] * dk_bin * area_2D / (2 * np.pi))
        P_err[j]   = np.sqrt(PA * PB + P_cross[j]**2) / np.sqrt(n_modes)
        denom      = np.sqrt(PA * PB)
        if denom > 0:
            r_cross[j] = P_cross[j] / denom
    return P_cross, P_err, r_cross

def make_ell(k_centers, chi_Mpc, h=h_little):
    return k_centers * chi_Mpc/h 

def to_Cell(P_cross, P_err, chi_Mpc, h=h_little):
    factor    =  h**2 / chi_Mpc**2 
    return P_cross * factor, P_err * factor

def to_Dell(ell, C_ell, C_ell_err, T_CMB_uK=None):
    D_ell     = ell * (ell + 1) * C_ell     / (2 * np.pi)
    D_ell_err = ell * (ell + 1) * C_ell_err / (2 * np.pi)
    if T_CMB_uK is not None:
        D_ell     *= T_CMB_uK**2
        D_ell_err *= T_CMB_uK**2
    return D_ell, D_ell_err

print(f'  tau_prefactor = {tau_prefactor:.4e} Mpc⁻¹')
print(f'  c_Mpc_s       = {c_Mpc_s:.4e} Mpc/s')
print(f'  k range       : [{k_centers[0]:.4f}, {k_centers[-1]:.4f}] Mpc⁻¹')
print(f'  n_kbins       : {n_kbins}')
print('✓ CELL 3 complete')

# %%
# =============================================================================
# CELL 4 — Build projected 2D maps
#
# kSZ integrand: ΔT/T = -σ_T n_e0 ∫ (1+δ) x_e (v/c) (1/a²) e^{-τ} ds
#
# Input:  field_data, tracer_data from Cell 2 (3D lightcones, 64×64×512)
# Output: 2D projected maps (64×64) per seed:
#   kSZ_map    : ΔT/T
#   kSZ2_map   : (ΔT/T)²
#   xe2_map    : ∫ x_e² ds                 [Mpc]
#   v2_map     : ∫ v_∥² ds                 [km²/s²·Mpc]
#   vproj_map  : ∫ v_∥ ds                  [km/s·Mpc]
#   vproj2_map : (∫ v_∥ ds)²
#   halo_map   : δ_h
#   lbg_map    : δ_LBG
#   lae_map    : δ_LAE
# =============================================================================

print('='*60)
print('CELL 4 — Building projected 2D maps')
print('='*60)

PLOT_DIR_4 = os.path.expanduser(
    '~/Desktop/Research/Semester 6/Plots/LAE_cross/27Jun2026_400Mpc_halo_LAE_LBG_3D'
)
os.makedirs(PLOT_DIR_4, exist_ok=True)

# ── integration range ─────────────────────────────────────────────────────
Z_kSZ_MIN  = 5.12   # match our lightcone ZMIN
Z_kSZ_MAX  = 19.89  # match our lightcone ZMAX
Z_HALO_MIN = 5.12   # tracer range — full lightcone
Z_HALO_MAX = 19.89

kSZ_maps    = {}
kSZ2_maps   = {}
xe2_maps    = {}
v2_maps     = {}
vproj_maps  = {}
vproj2_maps = {}
halo_maps   = {}
lbg_maps    = {}
lae_maps    = {}

for SEED in SEEDS:
    if SEED not in field_data:
        print(f'  ✗ seed {SEED}: no field data')
        continue

    z_lc       = field_data[SEED]['z_lc']
    xHI_lc     = field_data[SEED]['xHI_lc']
    density_lc = field_data[SEED]['density_lc']   # (1+δ)
    v_kms      = field_data[SEED]['velocity_kms']  # km/s
    v_Mpc_s    = field_data[SEED]['velocity_lc']   # Mpc/s

    # ── trim to kSZ integration range ────────────────────────────────────
    zi_lo = int(np.searchsorted(z_lc, Z_kSZ_MIN))
    zi_hi = int(np.searchsorted(z_lc, Z_kSZ_MAX))
    z     = z_lc[zi_lo:zi_hi]
    xHI   = xHI_lc   [:, :, zi_lo:zi_hi]
    delta = density_lc[:, :, zi_lo:zi_hi]
    v_k   = v_kms     [:, :, zi_lo:zi_hi]   # km/s
    v_m   = v_Mpc_s   [:, :, zi_lo:zi_hi]   # Mpc/s
    x_e   = 1.0 - xHI

    # ── comoving distance spacing ds ──────────────────────────────────────
    d_com = cosmo.comoving_distance(z).to_value('Mpc')
    ds    = np.abs(np.gradient(d_com))   # (N_z,) Mpc/slice

    # ── scale factor a(z) = 1/(1+z) ──────────────────────────────────────
    a_arr = 1.0 / (1.0 + z)   # (N_z,)

    # ── optical depth τ(z) ───────────────────────────────────────────────
    x_e_mean = x_e.mean(axis=(0, 1))
    dtau     = tau_prefactor * x_e_mean * (1.0 + z)**2 * ds
    tau_arr  = np.cumsum(dtau)
    e_tau    = np.exp(-tau_arr)

    # ── kSZ map ───────────────────────────────────────────────────────────
    v_over_c  = v_m / c_Mpc_s   # dimensionless

    integrand = (tau_prefactor
                 * delta                                              # (1+δ)
                 * x_e
                 * v_over_c
                 * (1.0 / a_arr**2)[np.newaxis, np.newaxis, :]
                 * e_tau[np.newaxis, np.newaxis, :]
                 * ds[np.newaxis, np.newaxis, :])

    kSZ_map = -np.sum(integrand, axis=2)   # (64,64)

    # ── xe² map ──────────────────────────────────────────────────────────
    xe2_map = np.sum(x_e**2 * ds[np.newaxis, np.newaxis, :], axis=2)

    # ── v² map ───────────────────────────────────────────────────────────
    v2_map = np.sum(v_k**2 * ds[np.newaxis, np.newaxis, :], axis=2)

    # ── v_proj maps ───────────────────────────────────────────────────────
    vproj_map  = np.sum(v_k * ds[np.newaxis, np.newaxis, :], axis=2)
    vproj2_map = vproj_map**2

    # ── tracer projected maps ─────────────────────────────────────────────
    if SEED in tracer_data:
        z_nodes_t = tracer_data[SEED]['z_nodes']
        zi_lo_t   = int(np.searchsorted(z_nodes_t, Z_HALO_MIN))
        zi_hi_t   = int(np.searchsorted(z_nodes_t, Z_HALO_MAX))

        halo_proj = tracer_data[SEED]['halo_count_lc'][:,:,zi_lo_t:zi_hi_t].sum(axis=2)
        lbg_proj  = tracer_data[SEED]['lbg_count_lc'] [:,:,zi_lo_t:zi_hi_t].sum(axis=2)
        lae_proj  = tracer_data[SEED]['lae_count_lc'] [:,:,zi_lo_t:zi_hi_t].sum(axis=2)

        halo_map  = make_overdensity(halo_proj)
        lbg_map   = make_overdensity(lbg_proj)
        lae_map   = make_overdensity(lae_proj)
    else:
        print(f'  ⚠ seed {SEED}: no tracer data')
        halo_map = np.zeros((N_SIDE, N_SIDE))
        lbg_map  = np.zeros((N_SIDE, N_SIDE))
        lae_map  = np.zeros((N_SIDE, N_SIDE))

    # ── store ─────────────────────────────────────────────────────────────
    kSZ_maps   [SEED] = kSZ_map.astype(np.float32)
    kSZ2_maps  [SEED] = (kSZ_map**2).astype(np.float32)
    xe2_maps   [SEED] = xe2_map.astype(np.float32)
    v2_maps    [SEED] = v2_map.astype(np.float32)
    vproj_maps [SEED] = vproj_map.astype(np.float32)
    vproj2_maps[SEED] = vproj2_map.astype(np.float32)
    halo_maps  [SEED] = halo_map.astype(np.float32)
    lbg_maps   [SEED] = lbg_map.astype(np.float32)
    lae_maps   [SEED] = lae_map.astype(np.float32)

    print(f'  ✓ seed {SEED}:  '
          f'kSZ rms={np.sqrt(np.mean(kSZ_map**2)):.3e}  '
          f'δ_h rms={halo_map.std():.3f}  '
          f'δ_LBG rms={lbg_map.std():.3f}  '
          f'δ_LAE rms={lae_map.std():.3f}')

print(f'\n  ✓ {len(kSZ_maps)}/{len(SEEDS)} seeds done')
print(f'  kSZ range: z={Z_kSZ_MIN}-{Z_kSZ_MAX}  '
      f'tracer range: z={Z_HALO_MIN}-{Z_HALO_MAX}')

# ── visualisation ─────────────────────────────────────────────────────────
SEED_SHOW = SEEDS[0]
if SEED_SHOW in kSZ_maps:
    maps_to_show = [
        (kSZ_maps   [SEED_SHOW], r'$\Delta T/T$ (kSZ)',            'RdBu_r'),
        (kSZ2_maps  [SEED_SHOW], r'$(\Delta T/T)^2$ (kSZ$^2$)',    'hot_r'),
        (xe2_maps   [SEED_SHOW], r'$\int x_e^2\,ds$ [Mpc]',        'viridis'),
        (vproj_maps [SEED_SHOW], r'$\int v_\parallel\,ds$',         'RdBu_r'),
        (halo_maps  [SEED_SHOW], r'$\delta_h$',                     'PuOr'),
        (lbg_maps   [SEED_SHOW], r'$\delta_{\rm LBG}$',             'PuOr'),
        (lae_maps   [SEED_SHOW], r'$\delta_{\rm LAE}$',             'PuOr'),
    ]

    ncols = 4
    nrows = int(np.ceil(len(maps_to_show) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 4*nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, (m, title, cmap) in zip(axes, maps_to_show):
        vmax = np.percentile(np.abs(m[np.isfinite(m)]), 98)
        vmin = -vmax if m.min() < 0 else 0
        im   = ax.imshow(m, origin='lower', cmap=cmap,
                         vmin=vmin, vmax=vmax,
                         extent=[0, BOX_LEN, 0, BOX_LEN],
                         rasterized=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x [cMpc]')
        ax.set_ylabel('y [cMpc]')

    for ax in axes[len(maps_to_show):]:
        ax.set_visible(False)

    fig.suptitle(
        rf'Projected 2D maps — seed {SEED_SHOW}  '
        rf'(kSZ+tracers: $z={Z_kSZ_MIN:.2f}$–${Z_kSZ_MAX:.2f}$)',
        fontsize=12, fontweight='bold')

    outpath = os.path.join(PLOT_DIR_4,
                           f'projected_maps_seed{SEED_SHOW}.pdf')
    fig.savefig(outpath, dpi=300)
    print(f'\n  ✓ saved: {outpath}')
    plt.show()
    plt.close(fig)

print('\n✓ CELL 4 complete')

# %%
# =============================================================================
# CELL 5 — Auto-power spectra D_ell vs ell
# Median ± 1σ over 5 seeds
# Fields: kSZ, kSZ², xe², v², vproj², δ_h, δ_LBG, δ_LAE
# =============================================================================

print('='*60)
print('CELL 5 — Auto-power spectra')
print('='*60)

# ── reference redshift ────────────────────────────────────────────────────
z_ref   = 0.5 * (5.12 + 19.89)
chi_ref = cosmo.comoving_distance(z_ref).to_value('Mpc')
ell_ref = make_ell(k_centers, chi_ref)

print(f'  z_ref    = {z_ref:.2f}')
print(f'  chi_ref  = {chi_ref:.0f} Mpc')
print(f'  ell range = [{ell_ref.min():.0f}, {ell_ref.max():.0f}]')

# ── field definitions ──────────────────────────────────────────────────────
auto_fields = {
    'kSZ'    : {'maps': kSZ_maps,    'label': r'$\Delta T_{\rm kSZ}$',
                'color': 'steelblue'},
    'kSZ2'   : {'maps': kSZ2_maps,   'label': r'$(\Delta T_{\rm kSZ})^2$',
                'color': 'darkblue'},
    'xe2'    : {'maps': xe2_maps,    'label': r'$\int x_e^2\,ds$',
                'color': 'darkgreen'},
    'v2'     : {'maps': v2_maps,     'label': r'$v_\parallel^2$',
                'color': 'darkred'},
    'vproj2' : {'maps': vproj2_maps, 'label': r'$(v_{\rm proj})^2$',
                'color': 'darkorange'},
    'halo'   : {'maps': halo_maps,   'label': r'$\delta_h$',
                'color': 'purple'},
    'lbg'    : {'maps': lbg_maps,    'label': r'$\delta_{\rm LBG}$',
                'color': 'firebrick'},
    'lae'    : {'maps': lae_maps,    'label': r'$\delta_{\rm LAE}$',
                'color': 'crimson'},
}

yunits = {
    'kSZ'    : r'$D_\ell^{\Delta T_{\rm kSZ}}$  [$\mu$K$^2$]',
    'kSZ2'   : r'$D_\ell^{(\Delta T_{\rm kSZ})^2}$  [$\mu$K$^4$]',
    'xe2'    : r'$D_\ell^{x_e^2\,ds}$  [Mpc$^2$]',
    'v2'     : r'$D_\ell^{v_\parallel^2}$  [km$^4$s$^{-2}$Mpc$^2$]',
    'vproj2' : r'$D_\ell^{(v_{\rm proj})^2}$',
    'halo'   : r'$D_\ell^{\delta_h}$',
    'lbg'    : r'$D_\ell^{\delta_{\rm LBG}}$',
    'lae'    : r'$D_\ell^{\delta_{\rm LAE}}$',
}

# ── compute per seed ──────────────────────────────────────────────────────
auto_results = {}

for fname, fdict in auto_fields.items():
    maps  = fdict['maps']
    T_cmb = T_CMB_uK if fname in ('kSZ', 'kSZ2') else None
    auto_results[fname] = {}

    for SEED in SEEDS:
        if SEED not in maps:
            continue
        field_2d = maps[SEED].astype(np.float64)
        delta_f  = field_2d - field_2d.mean()
        P_cross, P_err, _ = cross_power_2d(delta_f, delta_f)
        C_ell, C_ell_err  = to_Cell(P_cross, P_err, chi_ref)
        D_ell, D_err = to_Dell(ell_ref, C_ell, C_ell_err, T_CMB_uK=T_cmb)

        if fname == 'kSZ2':
            D_ell = D_ell * T_CMB_uK**2
            D_err = D_err * T_CMB_uK**2

        auto_results[fname][SEED] = (D_ell, D_err)
        print(f'  ✓ {fname}: {len(auto_results[fname])} seeds')

# ── median ± 1σ over seeds ────────────────────────────────────────────────
auto_stats = {}

for fname in auto_fields:
    D_list = [auto_results[fname][s][0] for s in SEEDS
              if s in auto_results[fname]]
    if len(D_list) == 0:
        continue
    D_arr = np.array(D_list)
    auto_stats[fname] = {
        'median': np.nanmedian(D_arr, axis=0),
        'std'   : np.nanstd  (D_arr, axis=0),
        'p16'   : np.nanpercentile(D_arr, 16, axis=0),
        'p84'   : np.nanpercentile(D_arr, 84, axis=0),
    }

print(f'\n  ✓ median+1σ computed for {len(auto_stats)} fields')

# ── plot ──────────────────────────────────────────────────────────────────
n_fields  = len(auto_fields)
ncols     = 4
nrows     = int(np.ceil(n_fields / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5*ncols, 4*nrows),
                          constrained_layout=True)
axes      = np.array(axes).flatten()
seed_colors = plt.cm.tab10(np.linspace(0, 0.6, len(SEEDS)))

for ai, (fname, fdict) in enumerate(auto_fields.items()):
    if ai >= len(axes):
        break
    ax    = axes[ai]
    color = fdict['color']

    # individual seeds
    for si, SEED in enumerate(SEEDS):
        if SEED not in auto_results[fname]:
            continue
        D_ell, _ = auto_results[fname][SEED]
        valid = np.isfinite(D_ell) & (ell_ref > 10) & (D_ell > 0)
        ax.plot(ell_ref[valid], D_ell[valid],
                color=seed_colors[si], lw=1.0, alpha=0.4,
                label=f'seed {SEED}')

    # median + 1σ band (p16-p84)
    if fname in auto_stats:
        med  = auto_stats[fname]['median']
        p16  = auto_stats[fname]['p16']
        p84  = auto_stats[fname]['p84']
        valid = np.isfinite(med) & (ell_ref > 10) & (med > 0)
        ax.plot(ell_ref[valid], med[valid],
                color=color, lw=2.5, label='median')
        ax.fill_between(ell_ref[valid], p16[valid], p84[valid],
                        color=color, alpha=0.2, label=r'$1\sigma$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(yunits.get(fname, r'$D_\ell$'))
    ax.set_title(fdict['label'], fontsize=13)
    ax.legend(fontsize=7, framealpha=0.5, ncol=2)

for ai in range(n_fields, len(axes)):
    axes[ai].set_visible(False)

fig.suptitle(
    rf'Auto-power spectra — 400 Mpc/64$^3$, 5 seeds, median $\pm1\sigma$  '
    rf'($z_{{ref}}={z_ref:.1f}$)',
    fontsize=12, fontweight='bold')

outpath = os.path.join(PLOT_DIR_4, 'auto_spectra_all_fields.pdf')
fig.savefig(outpath, dpi=300)
print(f'\n  ✓ saved: {outpath}')
plt.show()
plt.close(fig)

print('\n✓ CELL 5 complete')

# %%
# =============================================================================
# CELL 6 — Cross-power spectra: {kSZ2, xe2, v2, vproj, vproj2} × {δ_h, δ_LAE, δ_LBG}
# Per redshift slice (using tracer z-bins from z_lc)
# =============================================================================

print('='*60)
print('CELL 6 — Cross-power spectra')
print('='*60)

# ── z bins for tracer projection ──────────────────────────────────────────
# We cross-correlate kSZ² (projected over full LoS) with tracers projected
# in redshift slices of width dz, centered on z_cents
DZ        = 0.5
Z_LO_CROSS = 5.12
Z_HI_CROSS = 19.89
z_edges   = np.arange(Z_LO_CROSS, Z_HI_CROSS + DZ, DZ)
z_cents   = 0.5 * (z_edges[:-1] + z_edges[1:])
print(f'  {len(z_cents)} redshift bins  Δz={DZ}  z=[{z_cents[0]:.2f},{z_cents[-1]:.2f}]')

# ── signal maps (projected over full LoS, per seed) ───────────────────────
# kSZ2, xe2, v2, vproj, vproj2 already in *_maps dicts from Cell 4

signal_maps = {
    'kSZ2'   : kSZ2_maps,
    'xe2'    : xe2_maps,
    'v2'     : v2_maps,
    'v_proj' : vproj_maps,
    'v_proj2': vproj2_maps,
}

# ── cross-correlation loop ────────────────────────────────────────────────
# cross_results_LAE[seed][z_cent][fname] = {'ell', 'D_ell', 'D_err', 'r'}
# cross_results_LBG[seed][z_cent][fname] = same
# cross_results_halo[seed][z_cent][fname] = same

cross_results_LAE  = {s: {} for s in SEEDS}
cross_results_LBG  = {s: {} for s in SEEDS}
cross_results_halo = {s: {} for s in SEEDS}

for SEED in SEEDS:
    if SEED not in kSZ2_maps:
        continue

    z_nodes_t = tracer_data[SEED]['z_nodes']

    for zi, z_c in enumerate(z_cents):
        z_lo = z_edges[zi]
        z_hi = z_edges[zi + 1]

        # ── tracer maps projected in this z slice ─────────────────────────
        zi_lo_t = int(np.searchsorted(z_nodes_t, z_lo))
        zi_hi_t = int(np.searchsorted(z_nodes_t, z_hi))
        if zi_hi_t <= zi_lo_t:
            continue

        halo_proj = tracer_data[SEED]['halo_count_lc'][:,:,zi_lo_t:zi_hi_t].sum(axis=2)
        lae_proj  = tracer_data[SEED]['lae_count_lc'] [:,:,zi_lo_t:zi_hi_t].sum(axis=2)
        lbg_proj  = tracer_data[SEED]['lbg_count_lc'] [:,:,zi_lo_t:zi_hi_t].sum(axis=2)

        delta_halo = make_overdensity(halo_proj)
        delta_lae  = make_overdensity(lae_proj)
        delta_lbg  = make_overdensity(lbg_proj)

        # skip empty slices
        if delta_lae.std() == 0 and delta_lbg.std() == 0:
            continue

        # ── Limber ell at z_c ─────────────────────────────────────────────
        chi_c = cosmo.comoving_distance(z_c).to_value('Mpc')
        ell_c = make_ell(k_centers, chi_c)

        cross_results_LAE [SEED][z_c] = {}
        cross_results_LBG [SEED][z_c] = {}
        cross_results_halo[SEED][z_c] = {}

        for fname, smaps in signal_maps.items():
            if SEED not in smaps:
                continue

            sig = smaps[SEED].astype(np.float64)
            sig = sig - sig.mean()

            T_cmb = T_CMB_uK if fname == 'kSZ2' else None

            # × LAE
            P, Pe, r = cross_power_2d(sig, delta_lae - delta_lae.mean())
            C, Ce    = to_Cell(P, Pe, chi_c)
            D, De    = to_Dell(ell_c, C, Ce, T_CMB_uK=T_cmb)
            cross_results_LAE[SEED][z_c][fname] = {
                'ell': ell_c, 'D_ell': D, 'D_err': De, 'r': r}

            # × LBG
            P, Pe, r = cross_power_2d(sig, delta_lbg - delta_lbg.mean())
            C, Ce    = to_Cell(P, Pe, chi_c)
            D, De    = to_Dell(ell_c, C, Ce, T_CMB_uK=T_cmb)
            cross_results_LBG[SEED][z_c][fname] = {
                'ell': ell_c, 'D_ell': D, 'D_err': De, 'r': r}

            # × halo
            P, Pe, r = cross_power_2d(sig, delta_halo - delta_halo.mean())
            C, Ce    = to_Cell(P, Pe, chi_c)
            D, De    = to_Dell(ell_c, C, Ce, T_CMB_uK=T_cmb)
            cross_results_halo[SEED][z_c][fname] = {
                'ell': ell_c, 'D_ell': D, 'D_err': De, 'r': r}

    n_z = len(cross_results_LAE[SEED])
    print(f'  seed {SEED}: {n_z} z-bins computed')

print(f'\n✓ CELL 6 complete — cross-correlations stored for LAE, LBG, halo')

# %%
# =============================================================================
# CELL 7 — D_ℓ vs ℓ  (rainbow over redshift)
# Cross-correlations: {kSZ2, xe2, v2, vproj, vproj2} × {δ_LAE, δ_LBG, δ_h}
# One plot per (signal × tracer) combination — color = redshift
# Median ± 1σ over 5 seeds
# =============================================================================

print('='*60)
print('CELL 7 — D_ℓ vs ℓ  (rainbow)')
print('='*60)

signal_labels = {
    'kSZ2'   : r'${\rm kSZ}^2$',
    'xe2'    : r'$x_e^2$',
    'v2'     : r'$v_\parallel^2$',
    'v_proj' : r'$v_{\rm proj}$',
    'v_proj2': r'$v_{\rm proj}^2$',
}
tracer_labels = {
    'LAE' : r'$\delta_{\rm LAE}$',
    'LBG' : r'$\delta_{\rm LBG}$',
    'halo': r'$\delta_h$',
}
tracer_results = {
    'LAE' : cross_results_LAE,
    'LBG' : cross_results_LBG,
    'halo': cross_results_halo,
}
tracer_colors = {
    'LAE' : 'plasma',
    'LBG' : 'viridis',
    'halo': 'inferno',
}

z_all  = np.array(sorted({z for s in SEEDS
                           for z in cross_results_LAE[s].keys()}))
norm_z = mcolors.Normalize(vmin=z_all.min(), vmax=z_all.max())

os.makedirs(PLOT_DIR_4, exist_ok=True)

for tracer_name, cross_res in tracer_results.items():
    cmap_z = plt.get_cmap(tracer_colors[tracer_name])

    for fname in signal_labels:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        for z_c in z_all:
            # collect D_ell across seeds
            D_seeds = []
            ell_ref = None
            for SEED in SEEDS:
                res = cross_res[SEED].get(z_c, {}).get(fname)
                if res is None:
                    continue
                ell_ref = res['ell']
                D_seeds.append(res['D_ell'])
            if len(D_seeds) == 0 or ell_ref is None:
                continue

            D_arr    = np.array(D_seeds)
            D_med    = np.nanmedian(D_arr, axis=0)
            D_p16    = np.nanpercentile(D_arr, 16, axis=0)
            D_p84    = np.nanpercentile(D_arr, 84, axis=0)
            valid    = np.isfinite(D_med) & (ell_ref > 10)
            if valid.sum() < 3:
                continue

            color = cmap_z(norm_z(z_c))
            ax.plot(ell_ref[valid], D_med[valid],
                    color=color, lw=1.5, alpha=0.85)
            ax.fill_between(ell_ref[valid], D_p16[valid], D_p84[valid],
                            color=color, alpha=0.10)

        ax.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(
            rf'$D_\ell^{{\,{signal_labels[fname].strip("$")} \times {tracer_labels[tracer_name].strip("$")}}}$')
        ax.set_title(
            rf'{signal_labels[fname]} $\times$ {tracer_labels[tracer_name]}'
            rf'  —  $D_\ell$ vs $\ell$  ({len(SEEDS)} seeds, median $\pm1\sigma$)',
            fontsize=12, fontweight='bold')

        sm = plt.cm.ScalarMappable(cmap=cmap_z, norm=norm_z)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02).set_label(r'Redshift $z$')

        outpath = os.path.join(PLOT_DIR_4,
                    f'cross_Dell_vs_ell_{fname}_{tracer_name}_rainbow.pdf')
        fig.savefig(outpath, dpi=200)
        print(f'  ✓ {fname} × {tracer_name}  saved')
        plt.show()
        plt.close(fig)

print('\n✓ CELL 7 complete')

# %%
# =============================================================================
# CELL 8 — D_ℓ vs z  at fixed ℓ
# Cross-correlations: {kSZ2, xe2, v2, vproj, vproj2} × {δ_LAE, δ_LBG, δ_h}
# One plot per (signal × tracer) — lines = ℓ targets
# Median ± 1σ over 5 seeds
# =============================================================================

print('='*60)
print('CELL 8 — D_ℓ vs z  (fixed ℓ)')
print('='*60)

ELL_TARGETS  = [500, 1000, 3000]
ELL_COLORS   = ['darkblue', 'darkgreen', 'darkred']

z_all = np.array(sorted({z for s in SEEDS
                          for z in cross_results_LAE[s].keys()}))

for tracer_name, cross_res in tracer_results.items():
    for fname in signal_labels:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 constrained_layout=True)

        # ── D_ℓ vs z ─────────────────────────────────────────────────────
        ax = axes[0]
        for ell_target, color in zip(ELL_TARGETS, ELL_COLORS):
            z_plot, D_plot, D_p16, D_p84 = [], [], [], []

            for z_c in z_all:
                D_seeds = []
                for SEED in SEEDS:
                    res = cross_res[SEED].get(z_c, {}).get(fname)
                    if res is None:
                        continue
                    ei = int(np.argmin(np.abs(res['ell'] - ell_target)))
                    if np.isfinite(res['D_ell'][ei]):
                        D_seeds.append(res['D_ell'][ei])
                if len(D_seeds) == 0:
                    continue
                D_arr = np.array(D_seeds)
                z_plot.append(z_c)
                D_plot.append(np.nanmedian(D_arr))
                D_p16 .append(np.nanpercentile(D_arr, 16))
                D_p84 .append(np.nanpercentile(D_arr, 84))

            if len(z_plot) < 3:
                continue
            z_plot = np.array(z_plot)
            D_plot = np.array(D_plot)
            D_p16  = np.array(D_p16)
            D_p84  = np.array(D_p84)

            ax.plot(z_plot, D_plot, color=color, lw=2.0,
                    label=rf'$\ell={ell_target}$')
            ax.fill_between(z_plot, D_p16, D_p84,
                            color=color, alpha=0.15)

        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(
            rf'$D_\ell^{{\,{signal_labels[fname].strip("$")} \times {tracer_labels[tracer_name].strip("$")}}}$')
        ax.set_title(r'$D_\ell$ vs $z$', fontsize=12)
        ax.legend(fontsize=10, framealpha=0.6)

        # ── r vs z ───────────────────────────────────────────────────────
        ax = axes[1]
        for ell_target, color in zip(ELL_TARGETS, ELL_COLORS):
            z_plot, r_plot, r_p16, r_p84 = [], [], [], []

            for z_c in z_all:
                r_seeds = []
                for SEED in SEEDS:
                    res = cross_res[SEED].get(z_c, {}).get(fname)
                    if res is None:
                        continue
                    ei = int(np.argmin(np.abs(res['ell'] - ell_target)))
                    if np.isfinite(res['r'][ei]) and np.abs(res['r'][ei]) < 1.5:
                        r_seeds.append(res['r'][ei])
                if len(r_seeds) == 0:
                    continue
                r_arr = np.array(r_seeds)
                z_plot.append(z_c)
                r_plot.append(np.nanmedian(r_arr))
                r_p16 .append(np.nanpercentile(r_arr, 16))
                r_p84 .append(np.nanpercentile(r_arr, 84))

            if len(z_plot) < 3:
                continue
            z_plot = np.array(z_plot)
            r_plot = np.array(r_plot)
            r_p16  = np.array(r_p16)
            r_p84  = np.array(r_p84)

            ax.plot(z_plot, r_plot, color=color, lw=2.0,
                    marker='o', markersize=3,
                    label=rf'$\ell={ell_target}$')
            ax.fill_between(z_plot, r_p16, r_p84,
                            color=color, alpha=0.15)

        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'Correlation coefficient $r$')
        ax.set_title(r'$r$ vs $z$', fontsize=12)
        ax.legend(fontsize=10, framealpha=0.6)

        fig.suptitle(
            rf'{signal_labels[fname]} $\times$ {tracer_labels[tracer_name]}'
            rf'  ({len(SEEDS)} seeds, median $\pm1\sigma$)',
            fontsize=13, fontweight='bold')

        outpath = os.path.join(PLOT_DIR_4,
                    f'cross_Dell_vs_z_{fname}_{tracer_name}.pdf')
        fig.savefig(outpath, dpi=200)
        print(f'  ✓ {fname} × {tracer_name}  saved')
        plt.show()
        plt.close(fig)

print('\n✓ CELL 8 complete')

# %%
# =============================================================================
# CELL 9a — CMB filter ingredients
# La Plante+2022 Eq. 8, 11, 14, 15
# Produces: Fl, Cl_T2T2_f  (per experiment)
# =============================================================================
print('='*60)
print('CELL 9a — CMB filter ingredients')
print('='*60)

import camb
from scipy.interpolate import interp1d as interp1d_snr

# ── ell grid (dense, used throughout 9a/9b/9c) ────────────────────────────
ell_grid = np.geomspace(100, 10000, 500)

# ── CAMB C_ell^TT  (μK², raw C_ell not D_ell) ────────────────────────────
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.66,
                   ombh2=0.0490 * 0.6766**2,
                   omch2=(0.3096 - 0.0490) * 0.6766**2)
pars.InitPower.set_params(ns=0.9665)
pars.set_for_lmax(12000, lens_potential_accuracy=0)
camb_results = camb.get_results(pars)
powers       = camb_results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
Cl_TT_raw    = powers['total'][:, 0]          # μK²
ell_camb     = np.arange(len(Cl_TT_raw))
Cl_TT        = interp1d_snr(ell_camb[2:], Cl_TT_raw[2:],
                             bounds_error=False,
                             fill_value='extrapolate')(ell_grid)

idx3 = np.argmin(np.abs(ell_grid - 3000))
print(f'  C_TT(3000)  = {Cl_TT[idx3]:.3e} uK^2  (expect ~7e-4)')

# ── C_ell^(kSZ,late)  Park+2018 (Eq. 9) ──────────────────────────────────
# D_ell^late = 1.38 * (ell/3000)^0.21  μK²
D_kSZ_late  = 1.38 * (ell_grid / 3000.0)**0.21
Cl_kSZ_late = D_kSZ_late * 2*np.pi / (ell_grid * (ell_grid + 1))   # μK²
print(f'  C_kSZ_late(3000) = {Cl_kSZ_late[idx3]:.3e} uK^2  (expect ~1e-6)')

# ── C_ell^(kSZ,reion) from our kSZ (LINEAR) auto-power ───────────────────
# La Plante Eq. 8 uses the reionisation-era kSZ *temperature* power spectrum
# That is the LINEAR kSZ field (ΔT/T), NOT kSZ2.
# auto_results['kSZ'][seed] = (D_ell, D_err)  in μK²  (set by to_Dell in Cell 5)

z_ref_ksz   = 0.5 * (5.12 + 19.89)
chi_ref_ksz = cosmo.comoving_distance(z_ref_ksz).to_value('Mpc')
ell_sim_ksz = make_ell(k_centers, chi_ref_ksz)

D_kSZ_reion_seeds = []
for SEED in SEEDS:
    if SEED not in auto_results.get('kSZ', {}):
        continue
    D_ell, _ = auto_results['kSZ'][SEED]
    D_kSZ_reion_seeds.append(D_ell)

D_kSZ_reion_med  = np.nanmedian(np.array(D_kSZ_reion_seeds), axis=0)   # μK²
Cl_kSZ_reion_sim = D_kSZ_reion_med * 2*np.pi / (ell_sim_ksz * (ell_sim_ksz + 1))

valid_r = (np.isfinite(Cl_kSZ_reion_sim) &
           (ell_sim_ksz > 10) &
           (Cl_kSZ_reion_sim > 0))
Cl_kSZ_reion = interp1d_snr(ell_sim_ksz[valid_r], Cl_kSZ_reion_sim[valid_r],
                              bounds_error=False,
                              fill_value='extrapolate')(ell_grid)
Cl_kSZ_reion = np.clip(Cl_kSZ_reion, 0, None)
print(f'  C_kSZ_reion(3000) = {Cl_kSZ_reion[idx3]:.3e} uK^2  (expect ~1e-7 to 1e-5)')

# ── Instrument noise N_ell  (Table 1, La Plante+2022) ─────────────────────
# N_ell = (Delta_N * pi/180/60)^2 * exp(theta_FWHM^2 * ell^2 / 8ln2)
# Delta_N in μK-arcmin, theta_FWHM in arcmin
experiments = {
    'SO'     : {'Delta_N': 10.0, 'theta_arcmin': 1.4     },
    'CMB-S4' : {'Delta_N':  2.0, 'theta_arcmin': 1.4     },
    'CMB-HD' : {'Delta_N':  0.6, 'theta_arcmin': 10/60   },  # 10 arcsec
}

Nl = {}
for name, exp in experiments.items():
    th_rad   = exp['theta_arcmin'] * np.pi / (180 * 60)
    Nl[name] = (exp['Delta_N'] * np.pi / 180 / 60)**2 * np.exp(
                    th_rad**2 * ell_grid**2 / (8 * np.log(2)))

print(f'  N_ell^SO(3000)     = {Nl["SO"][idx3]:.3e} uK^2  (expect ~1e-5)')
print(f'  N_ell^CMB-HD(3000) = {Nl["CMB-HD"][idx3]:.3e} uK^2')

# ── Filter F(ell) and beam b(ell)  (Eq. 8, 10) ───────────────────────────
# F(ell) = C_kSZ_reion / (C_TT + C_kSZ_reion + C_kSZ_late + N_ell)
# b(ell) = exp(-theta^2 * ell^2 / 16ln2)   Gaussian beam
# f(ell) = F(ell) * b(ell)

Fl = {}; bl = {}; fl = {}
for name, exp in experiments.items():
    th_rad     = exp['theta_arcmin'] * np.pi / (180 * 60)
    denom      = Cl_TT + Cl_kSZ_reion + Cl_kSZ_late + Nl[name]
    Fl[name]   = Cl_kSZ_reion / denom
    bl[name]   = np.exp(-th_rad**2 * ell_grid**2 / (16 * np.log(2)))
    fl[name]   = Fl[name] * bl[name]

print(f'  F_SO(3000)     = {Fl["SO"][idx3]:.4e}  (expect ~1e-3)')
print(f'  F_CMB-HD(3000) = {Fl["CMB-HD"][idx3]:.4e}')

# ── C_ell^(T̄T̄,f)  (Eq. 15) ──────────────────────────────────────────────
# C_TT_f(ell) = f^2(ell) * [C_TT + C_kSZ_reion + C_kSZ_late + N_ell]  μK²
Cl_TT_f = {}
for name in experiments:
    total         = Cl_TT + Cl_kSZ_reion + Cl_kSZ_late + Nl[name]
    Cl_TT_f[name] = fl[name]**2 * total

print(f'  C_TT_f^SO(3000)     = {Cl_TT_f["SO"][idx3]:.3e} uK^2')
print(f'  C_TT_f^CMB-HD(3000) = {Cl_TT_f["CMB-HD"][idx3]:.3e} uK^2')

# ── C_ell^(T²T²,f)  Gaussian approx  (Eq. 14) ───────────────────────────
# C_T2T2_f ≈ 2/(2pi) * ∫ L dL [C_L^(T̄T̄,f)]²   (scalar, ell-independent)
Cl_T2T2_f = {}
for name in experiments:
    C         = Cl_TT_f[name]                             # μK²
    integrand = ell_grid * C**2 / (2 * np.pi)
    integral  = np.trapz(integrand, ell_grid)             # μK⁴
    Cl_T2T2_f[name] = 2.0 * integral * np.ones_like(ell_grid)   # μK⁴ scalar

print(f'  C_T2T2_f^SO     = {Cl_T2T2_f["SO"][0]:.3e} uK^4  (expect ~1e-8 to 1e-6)')
print(f'  C_T2T2_f^CMB-HD = {Cl_T2T2_f["CMB-HD"][0]:.3e} uK^4')
print('\n✓ CELL 9a complete\n')






# %%
# =============================================================================
# CELL 9b — Apply filter to kSZ maps, compute filtered kSZ²×LAE cross-power
# Produces: Cl_signal[exp_name][z_c]  in μK²
#           Cl_lae_auto[z_c]          dimensionless
# =============================================================================
print('='*60)
print('CELL 9b — Filtered kSZ² × LAE cross-power')
print('='*60)

# ── 2D k-grid for Fourier filtering ──────────────────────────────────────
# Must match the layout used by cross_power_2d (which uses np.fft.fftfreq)
h_little = cosmo.H0.value / 100.0          # 0.6766
dk        = 2 * np.pi / BOX_LEN           # consistent with Cell 3
kx        = np.fft.fftfreq(N_SIDE) * N_SIDE * dk   # shape (N_SIDE,)
ky        = np.fft.fftfreq(N_SIDE) * N_SIDE * dk
KX, KY    = np.meshgrid(kx, ky)
k2d       = np.sqrt(KX**2 + KY**2)        # (N_SIDE, N_SIDE)

# ell_2d uses the same chi reference as the kSZ auto-power (median z)
ell2d_ref = make_ell(k2d, chi_ref_ksz)    # (N_SIDE, N_SIDE)
ell2d_ref[N_SIDE//2, N_SIDE//2] = 1e-6    # avoid k=0 issues

# Build filtered kSZ² maps per experiment (apply f(ell) in Fourier space)
filtered_kSZ2 = {}    # filtered_kSZ2[exp_name][SEED] = (N_SIDE, N_SIDE)
for name in experiments:
    filtered_kSZ2[name] = {}
    f_2d = interp1d_snr(ell_grid, fl[name],
                         bounds_error=False, fill_value=0.0)(ell2d_ref)
    for SEED in SEEDS:
        if SEED not in kSZ_maps:
            continue
        kSZ_map      = kSZ_maps[SEED].astype(np.float64)
        kSZ_k        = np.fft.fft2(kSZ_map)
        kSZ_k_filt   = kSZ_k * f_2d                       # filter in k-space
        kSZ_filt     = np.real(np.fft.ifft2(kSZ_k_filt))  # back to real space
        filtered_kSZ2[name][SEED] = kSZ_filt**2           # square -> (ΔT/T)²

    print(f'  ✓ {name}: filtered kSZ² maps for {len(filtered_kSZ2[name])} seeds')

# ── LAE auto-power C_ell^(δ_LAE δ_LAE) per z-bin ────────────────────────
# Dimensionless (no T_CMB factor)
Cl_lae_auto = {}
z_nodes_ref = tracer_data[SEEDS[0]]['z_nodes']

for zi, z_c in enumerate(z_cents):
    z_lo    = z_edges[zi];  z_hi = z_edges[zi + 1]
    zi_lo   = int(np.searchsorted(z_nodes_ref, z_lo))
    zi_hi   = int(np.searchsorted(z_nodes_ref, z_hi))
    if zi_hi <= zi_lo:
        continue

    chi_c = cosmo.comoving_distance(z_c).to_value('Mpc')
    ell_c = make_ell(k_centers, chi_c)

    Cl_seeds = []
    for SEED in SEEDS:
        if SEED not in tracer_data:
            continue
        lae_proj = (tracer_data[SEED]['lae_count_lc'][:, :, zi_lo:zi_hi]
                    .sum(axis=2))
        if lae_proj.sum() == 0:
            continue
        delta_g  = make_overdensity(lae_proj)
        P, Pe, _ = cross_power_2d(delta_g - delta_g.mean(),
                                   delta_g - delta_g.mean())
        C, Ce    = to_Cell(P, Pe, chi_c)
        Cl_seeds.append(C)

    if not Cl_seeds:
        continue
    Cl_med = np.nanmedian(np.array(Cl_seeds), axis=0)
    valid  = np.isfinite(Cl_med) & (ell_c > 10)
    if valid.sum() < 3:
        continue
    Cl_lae_auto[z_c] = interp1d_snr(ell_c[valid], Cl_med[valid],
                                      bounds_error=False,
                                      fill_value='extrapolate')(ell_grid)

print(f'  ✓ C_ell^(δ_LAE δ_LAE) for {len(Cl_lae_auto)} z-bins')

# ── Filtered kSZ² × LAE cross-power per z-bin per experiment ─────────────
# D_ell in μK²  (to_Dell with T_CMB_uK applies one T_CMB² to (ΔT/T)²×dimless)
# Then C_ell = D_ell * 2pi / (ell*(ell+1))  in μK²
Cl_signal = {name: {} for name in experiments}

for name in experiments:
    for zi, z_c in enumerate(z_cents):
        z_lo  = z_edges[zi];  z_hi = z_edges[zi + 1]
        zi_lo = int(np.searchsorted(z_nodes_ref, z_lo))
        zi_hi = int(np.searchsorted(z_nodes_ref, z_hi))
        if zi_hi <= zi_lo:
            continue

        chi_c = cosmo.comoving_distance(z_c).to_value('Mpc')
        ell_c = make_ell(k_centers, chi_c)

        D_seeds = []
        for SEED in SEEDS:
            if SEED not in filtered_kSZ2[name]:
                continue
            if SEED not in tracer_data:
                continue

            lae_proj  = (tracer_data[SEED]['lae_count_lc'][:, :, zi_lo:zi_hi]
                         .sum(axis=2))
            if lae_proj.sum() == 0:
                continue
            delta_lae = make_overdensity(lae_proj)

            sig       = filtered_kSZ2[name][SEED].astype(np.float64)
            sig       = sig - sig.mean()

            P, Pe, _  = cross_power_2d(sig, delta_lae - delta_lae.mean())
            C, Ce     = to_Cell(P, Pe, chi_c)
            # to_Dell with T_CMB_uK: (ΔT/T)² × dimless → μK²
            D, De     = to_Dell(ell_c, C, Ce, T_CMB_uK=T_CMB_uK)
            D_seeds.append(D)

        if not D_seeds:
            continue

        D_med  = np.nanmedian(np.array(D_seeds), axis=0)   # μK²
        Cl_med = D_med * 2*np.pi / (ell_c * (ell_c + 1))  # μK²  (no extra division)
        valid  = np.isfinite(Cl_med) & (ell_c > 10)
        if valid.sum() < 3:
            continue

        Cl_signal[name][z_c] = interp1d_snr(ell_c[valid], Cl_med[valid],
                                              bounds_error=False,
                                              fill_value=0.0)(ell_grid)

    n_z = len(Cl_signal[name])
    print(f'  ✓ C_ell^(T_f²×LAE) [{name}]: {n_z} z-bins')

# ── Diagnostic at ell~1000 ────────────────────────────────────────────────
idx1  = np.argmin(np.abs(ell_grid - 1000))
z_chk = z_cents[5]
print(f'\n  Diagnostic at ell~1000, z~{z_chk:.1f}:')
for name in experiments:
    CT2 = Cl_T2T2_f[name][0]
    Cs  = Cl_signal[name].get(z_chk, np.zeros_like(ell_grid))[idx1]
    Cg  = Cl_lae_auto.get(z_chk, np.zeros_like(ell_grid))[idx1]
    ratio = Cs**2 / (CT2 * np.abs(Cg) + 1e-99)
    print(f'    [{name:8s}]  Cs={Cs:.3e} uK²  CT2={CT2:.3e} uK⁴'
          f'  Cg={Cg:.3e}  Cs²/(CT2·Cg)={ratio:.3e}')
print('  (ratio << 1 → noise dominated  ✓)')
print('\n✓ CELL 9b complete\n')


# %%
# =============================================================================
# CELL 9c — S/N via discrete ell sum  (Eq. 13)  +  plot
# =============================================================================
print('='*60)
print('CELL 9c — S/N calculation and plot')
print('='*60)

# Discrete sum over integer ell (Eq. 13):
# (S/N)² = f_sky * Σ_{ell=100}^{10000} (2ell+1) * Cs² / (CT2·Cg + Cs²)
f_sky   = 0.053          # Roman HLS 2200 deg²
ell_int = np.arange(100, 10001)   # every integer ell

SN_results = {}

for name in experiments:
    SN_results[name] = {}
    CT2_val = Cl_T2T2_f[name][0]    # scalar (Gaussian approx: ell-independent)

    for z_c in z_cents:
        if z_c not in Cl_signal[name] or z_c not in Cl_lae_auto:
            continue

        Cs_int = interp1d_snr(ell_grid, Cl_signal[name][z_c],
                               bounds_error=False, fill_value=0.0)(ell_int)
        Cg_int = interp1d_snr(ell_grid, Cl_lae_auto[z_c],
                               bounds_error=False, fill_value=0.0)(ell_int)

        num   = Cs_int**2
        denom = CT2_val * np.abs(Cg_int) + Cs_int**2
        good  = denom > 0
        SN2   = f_sky * np.sum((2*ell_int[good] + 1) * num[good] / denom[good])
        SN_results[name][z_c] = np.sqrt(max(SN2, 0.0))

    z_arr  = np.array(sorted(SN_results[name].keys()))
    SN_arr = np.array([SN_results[name][z] for z in z_arr])
    total  = np.sqrt(np.sum(SN_arr**2))
    print(f'  {name:8s}: peak S/N = {SN_arr.max():.3f}'
          f'  at z = {z_arr[np.argmax(SN_arr)]:.2f}'
          f'   total S/N = {total:.3f}')

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
exp_colors = {'SO': 'steelblue', 'CMB-S4': 'darkorange', 'CMB-HD': 'darkgreen'}

for name in experiments:
    z_arr  = np.array(sorted(SN_results[name].keys()))
    SN_arr = np.array([SN_results[name][z] for z in z_arr])
    ax.plot(z_arr, SN_arr, color=exp_colors[name], lw=2.5,
            marker='o', markersize=4, label=name)

ax.axhline(1, color='gray', ls='--', lw=1, label=r'$1\sigma$')
ax.axhline(3, color='gray', ls=':',  lw=1, label=r'$3\sigma$')
ax.set_xlabel(r'Redshift $z$', fontsize=13)
ax.set_ylabel(r'S/N per $\Delta z = 0.5$ bin', fontsize=13)
ax.set_title(
    r'kSZ$^2$ $\times$ LAE  S/N vs redshift'
    f'\n(400 Mpc/64³, 5 seeds, $f_{{\\rm sky}}={f_sky}$, filtered kSZ)',
    fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(z_cents[0] - 0.3, z_cents[-1] + 0.3)

outpath = os.path.join(PLOT_DIR_4, 'SNR_vs_z_LAE_filtered_v3.pdf')
fig.savefig(outpath, dpi=300)
print(f'\n  ✓ saved → {outpath}')
plt.show()
plt.close(fig)

print('\n✓ CELL 9c complete')

# %%
# =============================================================================
# CELL 9d (v3) — Four separate one-row diagnostic figures
# All labels use pure LaTeX math, no Unicode characters
#
# Figure 1 — Maps:       kSZ raw | filtered kSZ | LAE overdensity (peak z)
# Figure 2 — Filter:     components | F(ell) | C_T2T2_f
# Figure 3 — Spectra:    kSZ2 auto | cross before/after filter | rainbow
# Figure 4 — Results:    LAE auto (full vs slice) | S/N vs z | cumulative S/N
# =============================================================================
print('='*60)
print('CELL 9d (v3) — Four one-row diagnostic figures')
print('='*60)

import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib as mpl

# use matplotlib mathtext renderer (no external LaTeX needed)
mpl.rcParams['text.usetex'] = False

SEED_SHOW  = SEEDS[0]
EXP_SHOW   = 'CMB-HD'
exp_colors = {'SO': 'steelblue', 'CMB-S4': 'darkorange', 'CMB-HD': 'darkgreen'}

# ── peak S/N z-bin ────────────────────────────────────────────────────────
z_peak_sn = max(SN_results[EXP_SHOW],
                key=lambda z: SN_results[EXP_SHOW].get(z, 0))
print(f'  Peak S/N z-bin ({EXP_SHOW}): z = {z_peak_sn:.2f}')

z_nodes_ref = tracer_data[SEEDS[0]]['z_nodes']
zi_lo_peak  = int(np.searchsorted(z_nodes_ref, z_peak_sn - 0.25))
zi_hi_peak  = int(np.searchsorted(z_nodes_ref, z_peak_sn + 0.25))
chi_peak    = cosmo.comoving_distance(z_peak_sn).to_value('Mpc')
ell_peak    = make_ell(k_centers, chi_peak)

lae_slice      = (tracer_data[SEED_SHOW]['lae_count_lc']
                  [:, :, zi_lo_peak:zi_hi_peak].sum(axis=2))
delta_lae_peak = make_overdensity(lae_slice)

kSZ_raw_map   = kSZ_maps[SEED_SHOW].astype(np.float64)
kSZ2_raw_map  = kSZ_raw_map**2
kSZ2_filt_map = filtered_kSZ2[EXP_SHOW][SEED_SHOW]

prod_raw  = ((kSZ2_raw_map  - kSZ2_raw_map.mean())
             * (delta_lae_peak - delta_lae_peak.mean()))
prod_filt = ((kSZ2_filt_map - kSZ2_filt_map.mean())
             * (delta_lae_peak - delta_lae_peak.mean()))

f_2d_show   = interp1d_snr(ell_grid, fl[EXP_SHOW],
                             bounds_error=False, fill_value=0.0)(ell2d_ref)
kSZ_filt_uK = (np.real(np.fft.ifft2(np.fft.fft2(kSZ_raw_map) * f_2d_show))
               * T_CMB_uK)
kSZ_uK      = kSZ_raw_map * T_CMB_uK

ext = [0, BOX_LEN, 0, BOX_LEN]

def symvlim(arr, pct=99):
    v = np.nanpercentile(np.abs(arr), pct)
    return -v, v

FIGW = 18    # width of each one-row figure
FIGH = 5.5  # height

# =============================================================================
# FIGURE 1 — Maps
# =============================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(FIGW, FIGH),
                            constrained_layout=True)

# panel 0: raw kSZ
ax = axes1[0]
vlo, vhi = symvlim(kSZ_uK)
im = ax.imshow(kSZ_uK, origin='lower', cmap='RdBu_r',
               vmin=vlo, vmax=vhi, extent=ext)
plt.colorbar(im, ax=ax, label=r'$\Delta T_{\rm kSZ}\ [\mu{\rm K}]$')
ax.set_title('kSZ map (raw)', fontsize=12, fontweight='bold')
ax.set_xlabel('x [Mpc/h]'); ax.set_ylabel('y [Mpc/h]')
ax.text(0.03, 0.97, f'seed {SEED_SHOW}', transform=ax.transAxes,
        va='top', fontsize=9)

# panel 1: filtered kSZ
ax = axes1[1]
vlo2, vhi2 = symvlim(kSZ_filt_uK)
im2 = ax.imshow(kSZ_filt_uK, origin='lower', cmap='RdBu_r',
                vmin=vlo2, vmax=vhi2, extent=ext)
plt.colorbar(im2, ax=ax, label=r'$[f(\ell)\,\Delta T]\ [\mu{\rm K}]$')
ax.set_title(fr'Filtered kSZ map  [{EXP_SHOW}]', fontsize=12, fontweight='bold')
ax.set_xlabel('x [Mpc/h]'); ax.set_ylabel('y [Mpc/h]')

# panel 2: LAE overdensity at peak z-bin
ax = axes1[2]
vhi3 = np.nanpercentile(delta_lae_peak, 99)
im3  = ax.imshow(delta_lae_peak, origin='lower', cmap='PuRd',
                 vmin=0, vmax=max(vhi3, 0.1), extent=ext)
plt.colorbar(im3, ax=ax, label=r'$\delta_{\rm LAE}$')
ax.set_title(fr'LAE overdensity  ($z_c = {z_peak_sn:.2f}$, $\Delta z=0.5$)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('x [Mpc/h]'); ax.set_ylabel('y [Mpc/h]')

fig1.suptitle('Figure 1 — kSZ and LAE maps', fontsize=13, fontweight='bold')
out1 = os.path.join(PLOT_DIR_4, 'diag_fig1_maps.pdf')
fig1.savefig(out1, dpi=200, bbox_inches='tight')
print(f'  ✓ Fig 1 saved → {out1}')
plt.show(); plt.close(fig1)

# =============================================================================
# FIGURE 2 — Filter
# =============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(FIGW, FIGH),
                            constrained_layout=True)

# panel 0: filter denominator components
ax = axes2[0]
D_TT    = Cl_TT        * ell_grid*(ell_grid+1)/(2*np.pi)
D_reion = Cl_kSZ_reion * ell_grid*(ell_grid+1)/(2*np.pi)
D_late  = Cl_kSZ_late  * ell_grid*(ell_grid+1)/(2*np.pi)
ax.loglog(ell_grid, D_TT,    color='black',    lw=2,   label='CMB (CAMB)')
ax.loglog(ell_grid, D_reion, color='royalblue', lw=2,
          label=r'kSZ reion (sims)')
ax.loglog(ell_grid, D_late,  color='tomato',    lw=2,
          label='kSZ late (Park+18)')
for name, col in exp_colors.items():
    D_nl = Nl[name] * ell_grid*(ell_grid+1)/(2*np.pi)
    ax.loglog(ell_grid, D_nl, color=col, ls='--', lw=1.5,
              label=f'$N_\\ell$ {name}')
ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell\ [\mu{\rm K}^2]$')
ax.set_title('Filter denominator components\n(La Plante+22 Fig. 1)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.set_xlim(100, 10000)

# panel 1: F(ell) per experiment
ax = axes2[1]
for name, col in exp_colors.items():
    ax.semilogx(ell_grid, Fl[name], color=col, lw=2.5, label=name)
ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$F(\ell)$')
ax.set_title('Optimal filter $F(\\ell)$  (Eq. 8)\nApplied to CMB only — LAE unfiltered',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim(100, 10000)
ax.axvline(1000, color='gray', ls=':', lw=1, alpha=0.6)
ax.axvline(3000, color='gray', ls=':', lw=1, alpha=0.6)

# panel 2: C_TT_f and C_T2T2_f
ax = axes2[2]
for name, col in exp_colors.items():
    D_TTf = Cl_TT_f[name] * ell_grid*(ell_grid+1)/(2*np.pi)
    ax.loglog(ell_grid, D_TTf, color=col, lw=2,
              label=fr'$D_\ell^{{T_f T_f}}$ [{name}]')
    ax.axhline(Cl_T2T2_f[name][0], color=col, ls='--', lw=1.5,
               label=fr'$C_{{T^2T^2,f}}$ = {Cl_T2T2_f[name][0]:.1e}')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell^{T_f T_f}\ [\mu{\rm K}^2]$  /  $C_{T^2T^2,f}\ [\mu{\rm K}^4]$')
ax.set_title('Filtered map power and $C_{T^2T^2,f}$\n(La Plante+22 Eq. 14-15)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(100, 10000)

fig2.suptitle('Figure 2 — CMB filter ingredients', fontsize=13, fontweight='bold')
out2 = os.path.join(PLOT_DIR_4, 'diag_fig2_filter.pdf')
fig2.savefig(out2, dpi=200, bbox_inches='tight')
print(f'  ✓ Fig 2 saved → {out2}')
plt.show(); plt.close(fig2)

# =============================================================================
# FIGURE 3 — Spectra: kSZ2 auto  +  cross before/after  +  rainbow
# =============================================================================
fig3, axes3 = plt.subplots(1, 4, figsize=(24, FIGH), constrained_layout=True)

# panel 0: kSZ2 auto-power (corrected units)
ax = axes3[0]
D_kSZ2_all = []
for SEED in SEEDS:
    if SEED not in auto_results.get('kSZ2', {}):
        continue
    D_ell, _ = auto_results['kSZ2'][SEED]
    D_kSZ2_all.append(D_ell)
if D_kSZ2_all:
    arr  = np.array(D_kSZ2_all)
    med  = np.nanmedian(arr, axis=0)
    lo   = np.nanpercentile(arr, 16, axis=0)
    hi   = np.nanpercentile(arr, 84, axis=0)
    ellp = make_ell(k_centers, chi_ref_ksz)
    vp   = np.isfinite(med) & (med > 0) & (ellp > 10)
    ax.loglog(ellp[vp], med[vp], color='darkblue', lw=2.5, label='median')
    ax.fill_between(ellp[vp], lo[vp], hi[vp],
                    color='royalblue', alpha=0.25, label=r'$\pm 1\sigma$')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell^{(\Delta T_{\rm kSZ})^2}\ [\mu{\rm K}^4]$')
ax.set_title(r'kSZ$^2$ auto-power  (units corrected: $\times T_{\rm CMB}^4$)'
             f'\n{len(SEEDS)} seeds,  $z_{{\\rm ref}}={z_ref_ksz:.1f}$',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim(100, 10000)

# panel 1: cross-power before and after filter at peak z-bin
ax = axes3[1]
D_uf_all = []
for SEED in SEEDS:
    res = cross_results_LAE[SEED].get(z_peak_sn, {}).get('kSZ2')
    if res is None:
        continue
    D_uf_all.append(res['D_ell'])
if D_uf_all:
    med_uf = np.nanmedian(np.array(D_uf_all), axis=0)
    ell_uf = cross_results_LAE[SEEDS[0]][z_peak_sn]['kSZ2']['ell']
    vuf    = np.isfinite(med_uf) & (ell_uf > 10)
    ax.plot(ell_uf[vuf], med_uf[vuf],
            color='gray', lw=2, ls='--', label='Unfiltered (raw kSZ$^2$)')
for name, col in exp_colors.items():
    if z_peak_sn not in Cl_signal[name]:
        continue
    D_s = Cl_signal[name][z_peak_sn] * ell_grid*(ell_grid+1)/(2*np.pi)
    ax.plot(ell_grid, D_s, color=col, lw=2, label=f'Filtered [{name}]')
ax.axhline(0, color='black', lw=0.7, ls=':')
ax.set_xscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell^{{\rm kSZ}^2 \times {\rm LAE}}\ [\mu{\rm K}^2]$')
ax.set_title(r'Cross-power: unfiltered vs filtered'
             f'\nPeak S/N bin  $z_c = {z_peak_sn:.2f}$',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_xlim(100, 10000)

# panel 2: rainbow over z-bins (CMB-HD)
ax = axes3[2]
z_list = sorted(Cl_signal[EXP_SHOW].keys())
cmap_z = plt.cm.plasma
znorm  = mcolors.Normalize(vmin=min(z_list), vmax=max(z_list))
for z_c in z_list:
    D_s = Cl_signal[EXP_SHOW][z_c] * ell_grid*(ell_grid+1)/(2*np.pi)
    ax.plot(ell_grid, D_s, color=cmap_z(znorm(z_c)), lw=1.3, alpha=0.85)
ax.axhline(0, color='black', lw=0.7, ls=':')
sm = plt.cm.ScalarMappable(cmap=cmap_z, norm=znorm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Redshift $z$')
ax.set_xscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell^{{\rm kSZ}^2 \times {\rm LAE}}\ [\mu{\rm K}^2]$')
ax.set_title(fr'Filtered cross-power: all $z$-bins  [{EXP_SHOW}]'
             r'  ($\Delta z = 0.5$)',
             fontsize=12, fontweight='bold')
ax.set_xlim(100, 10000)

# panel 3: LAE auto-power — full projection + multiple thin z-slices
ax = axes3[3]

# full projection (Cell 5)
D_lae_fp_all = []
for SEED in SEEDS:
    if SEED not in auto_results.get('lae', {}):
        continue
    D_ell, _ = auto_results['lae'][SEED]
    D_lae_fp_all.append(D_ell)
if D_lae_fp_all:
    arr_f = np.array(D_lae_fp_all)
    med_f = np.nanmedian(arr_f, axis=0)
    lo_f  = np.nanpercentile(arr_f, 16, axis=0)
    hi_f  = np.nanpercentile(arr_f, 84, axis=0)
    ellf  = make_ell(k_centers, chi_ref_ksz)
    vf    = np.isfinite(med_f) & (med_f > 0) & (ellf > 10)
    ax.loglog(ellf[vf], med_f[vf], color='crimson', lw=2.5,
              label=r'Full projection ($z_{\rm ref}=12.5$)')
    ax.fill_between(ellf[vf], lo_f[vf], hi_f[vf], color='crimson', alpha=0.2)

# several thin slices at representative redshifts
slice_zs  = [6.0, 7.0, 8.0, 9.0, 10.0]   # pick z-bins near peak S/N
cmap_sl   = plt.cm.viridis
sl_norm   = mcolors.Normalize(vmin=min(slice_zs), vmax=max(slice_zs))

for z_sl in slice_zs:
    # find nearest z_cent
    z_c   = z_cents[np.argmin(np.abs(np.array(z_cents) - z_sl))]
    zi_lo = int(np.searchsorted(z_nodes_ref, z_c - 0.25))
    zi_hi = int(np.searchsorted(z_nodes_ref, z_c + 0.25))
    chi_c = cosmo.comoving_distance(z_c).to_value('Mpc')
    ell_c = make_ell(k_centers, chi_c)

    D_sl_all = []
    for SEED in SEEDS:
        if SEED not in tracer_data:
            continue
        sl = (tracer_data[SEED]['lae_count_lc']
              [:, :, zi_lo:zi_hi].sum(axis=2))
        if sl.sum() == 0:
            continue
        dg       = make_overdensity(sl)
        P, Pe, _ = cross_power_2d(dg - dg.mean(), dg - dg.mean())
        C, Ce    = to_Cell(P, Pe, chi_c)
        D_sl_all.append(C * ell_c*(ell_c+1)/(2*np.pi))
    if not D_sl_all:
        continue
    med_sl = np.nanmedian(np.array(D_sl_all), axis=0)
    vs     = np.isfinite(med_sl) & (med_sl > 0) & (ell_c > 10)
    ax.loglog(ell_c[vs], med_sl[vs], color=cmap_sl(sl_norm(z_sl)),
              lw=2, ls='--', label=fr'$z_c = {z_c:.1f}$')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell^{\delta_{\rm LAE}}$  (dimensionless)')
ax.set_title('LAE auto-power\nFull projection + thin slices (no filter)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(100, 10000)

fig3.suptitle(r'Figure 3 — Power spectra: kSZ$^2$ auto, cross-power, LAE auto',
              fontsize=13, fontweight='bold')

out3 = os.path.join(PLOT_DIR_4, 'diag_fig3_spectra.pdf')
fig3.savefig(out3, dpi=200, bbox_inches='tight')
print(f'  ✓ Fig 3 saved → {out3}')
plt.show(); plt.close(fig3)

# =============================================================================
# FIGURE 4 — Results: LAE auto  +  S/N vs z  +  cumulative S/N
# =============================================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(12, FIGH), constrained_layout=True)


# panel 0: S/N vs z
ax = axes4[0]
for name, col in exp_colors.items():
    z_arr  = np.array(sorted(SN_results[name].keys()))
    SN_arr = np.array([SN_results[name][z] for z in z_arr])
    total  = np.sqrt(np.sum(SN_arr**2))
    ax.plot(z_arr, SN_arr, color=col, lw=2.5, marker='o', markersize=4,
            label=fr'{name}  (total = {total:.0f}$\sigma$)')
ax.axhline(1, color='gray', ls='--', lw=1, label=r'$1\sigma$')
ax.axhline(3, color='gray', ls=':',  lw=1, label=r'$3\sigma$')
ax.set_xlabel(r'Redshift $z$', fontsize=12)
ax.set_ylabel(r'S/N per $\Delta z = 0.5$ bin', fontsize=12)
ax.set_title(r'kSZ$^2$ $\times$ LAE  S/N vs $z$'
             f'\n$f_{{\\rm sky}} = {f_sky}$,  filtered kSZ',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# panel 1: cumulative S/N vs ell_max
ax = axes4[1]
ell_max_arr = np.arange(200, 10001, 100)
for name, col in exp_colors.items():
    CT2_val = Cl_T2T2_f[name][0]
    SN2_cum = np.zeros(len(ell_max_arr))
    for z_c in z_cents:
        if z_c not in Cl_signal[name] or z_c not in Cl_lae_auto:
            continue
        Cs_i = interp1d_snr(ell_grid, Cl_signal[name][z_c],
                             bounds_error=False, fill_value=0.0)(ell_int)
        Cg_i = interp1d_snr(ell_grid, Cl_lae_auto[z_c],
                             bounds_error=False, fill_value=0.0)(ell_int)
        num  = Cs_i**2
        den  = CT2_val * np.abs(Cg_i) + Cs_i**2
        good = den > 0
        for ei, ell_max in enumerate(ell_max_arr):
            mask = good & (ell_int <= ell_max)
            SN2_cum[ei] += f_sky * np.sum(
                (2*ell_int[mask]+1) * num[mask] / den[mask])
    ax.plot(ell_max_arr, np.sqrt(np.maximum(SN2_cum, 0)),
            color=col, lw=2.5, label=name)
ax.axhline(1, color='gray', ls='--', lw=1)
ax.axhline(3, color='gray', ls=':',  lw=1)
ax.set_xlabel(r'$\ell_{\rm max}$', fontsize=12)
ax.set_ylabel(r'Cumulative S/N  (all $z$-bins)', fontsize=12)
ax.set_title('Cumulative S/N vs $\\ell_{\\rm max}$\n'
             '(La Plante+22 Fig. 10 analogue)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

fig4.suptitle(r'Figure 4 — S/N vs $z$ and cumulative S/N',
              fontsize=13, fontweight='bold')
out4 = os.path.join(PLOT_DIR_4, 'diag_fig4_results.pdf')
fig4.savefig(out4, dpi=200, bbox_inches='tight')
print(f'  ✓ Fig 4 saved → {out4}')
plt.show(); plt.close(fig4)

print('\n✓ CELL 9d (v3) complete — four figures saved')


