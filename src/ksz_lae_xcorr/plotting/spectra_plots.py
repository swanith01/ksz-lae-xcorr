"""
plotting/spectra_plots.py
===========================
D_ell vs ell (rainbow over z), D_ell vs z (fixed ell), and S/N vs z plots.
Refactored from notebook Cells 7, 8, 9c.

LBG panels are included here for physics interpretation (the paper text
explains what LBG cross-correlation looks like) but are visually
distinguished (dashed lines / separate panel) from the LAE SNR result,
which is the headline forecast.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_dell_vs_ell(cross_results: dict, tracer: str, signal: str, seed: int, out_dir: str) -> None:
    """Rainbow-over-redshift D_ell vs ell for one seed/tracer/signal combination."""
    os.makedirs(out_dir, exist_ok=True)
    z_cents = sorted(cross_results[tracer][seed].keys())
    cmap = matplotlib.colormaps["rainbow"].resampled(max(len(z_cents), 1))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for i, z_c in enumerate(z_cents):
        entry = cross_results[tracer][seed][z_c].get(signal)
        if entry is None:
            continue
        ax.errorbar(entry["ell"], entry["D_ell"], yerr=entry["D_err"], color=cmap(i),
                    lw=1, alpha=0.8, label=f"z={z_c:.1f}" if i % 4 == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    ax.set_title(f"{signal} $\\times$ {tracer}, seed {seed}")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(out_dir, f"dell_vs_ell_{signal}_{tracer}_seed{seed}.pdf"))
    plt.close(fig)


def plot_dell_vs_z(cross_results: dict, tracer: str, signal: str, ell_target: float,
                    seeds: list[int], out_dir: str) -> None:
    """D_ell at a fixed ell, plotted vs redshift, one line per seed."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    for seed in seeds:
        if seed not in cross_results[tracer]:
            continue
        z_cents = sorted(cross_results[tracer][seed].keys())
        D_vals = []
        for z_c in z_cents:
            entry = cross_results[tracer][seed][z_c].get(signal)
            if entry is None:
                D_vals.append(np.nan)
                continue
            idx = np.argmin(np.abs(entry["ell"] - ell_target))
            D_vals.append(entry["D_ell"][idx])
        ax.plot(z_cents, D_vals, marker="o", ms=3, lw=1, label=f"seed {seed}")

    ax.set_xlabel("z")
    ax.set_ylabel(r"$D_\ell$" + f" at $\\ell \\approx {ell_target:.0f}$")
    ax.set_title(f"{signal} $\\times$ {tracer}")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, f"dell_vs_z_{signal}_{tracer}_ell{ell_target:.0f}.pdf"))
    plt.close(fig)


def plot_dell_vs_ell_median_sigma(cross_results: dict, tracer: str, signal: str, seeds: list,
                                   out_dir: str, z_targets: list | None = None) -> None:
    """
    D_ell vs ell, median +/- 1 sigma across seeds, one curve+band per redshift
    bin (rainbow over z) -- matches the paper draft's Figure 2 style
    ("kSZ^2 x h, D_ell vs ell (N seeds, median +/-1 sigma)").

    z_targets: optional list of z-bin centers to plot (subset); default plots
    every z-bin available. With many z-bins, consider passing a subset --
    plotting all of them on one panel gets crowded fast.
    """
    from ksz_lae_xcorr.correlation.seed_stats import aggregate_over_seeds

    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate_over_seeds(cross_results, tracer, signal, seeds)
    if not agg:
        raise ValueError(f"No z-bins with >=2 seeds for tracer='{tracer}' signal='{signal}' -- "
                          f"nothing to plot.")

    z_list = z_targets if z_targets is not None else sorted(agg.keys())
    z_list = [z for z in z_list if z in agg]
    cmap = matplotlib.colormaps["rainbow"].resampled(max(len(z_list), 1))

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for i, z_c in enumerate(z_list):
        d = agg[z_c]
        color = cmap(i)
        ax.plot(d["ell"], d["median"], color=color, lw=1.6,
                label=f"z={z_c:.1f} (n={d['n_seeds']})" if i % max(len(z_list) // 8, 1) == 0 else None)
        ax.fill_between(d["ell"], d["lower"], d["upper"], color=color, alpha=0.18, lw=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$D_\ell$")
    ax.set_title(f"{signal} $\\times$ {tracer}, {len(seeds)} seeds, median $\\pm1\\sigma$")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(out_dir, f"dell_vs_ell_median_sigma_{signal}_{tracer}.pdf"))
    plt.close(fig)



def plot_dell_vs_z_multi_ell(cross_results: dict, tracer: str, signal: str, seeds: list,
                              out_dir: str, ell_targets: list | None = None,
                              n_ell_targets: int = 8) -> None:
    """
    D_ell vs z, median +/- 1 sigma across seeds, one curve+band per fixed ell
    value, colored by ell via a real colorbar (matches the paper draft's
    Figure 3 style -- D_ell vs z at fixed ell -- generalized from 3 discrete
    lines to a continuous colorbar).

    IMPORTANT: ell = k*chi(z), so the actual ell value at a given k-bin INDEX
    shifts between z-bins (chi(z) changes). This function interpolates each
    z-bin's median/sigma curve (originally sampled on that z-bin's own ell
    grid) onto the SAME set of target ell values before plotting vs z --
    reading off a fixed k-bin index across z-bins would silently mix
    different actual ell values together, which is wrong.

    ell_targets: explicit list of ell values to plot. If None, auto-picks
    n_ell_targets log-spaced values spanning the ell range common to most
    z-bins (5th-95th percentile of each z-bin's ell range, to avoid picking
    targets that require extrapolating far beyond what most z-bins cover).
    """
    from scipy.interpolate import interp1d

    from ksz_lae_xcorr.correlation.seed_stats import aggregate_over_seeds

    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate_over_seeds(cross_results, tracer, signal, seeds)
    if not agg:
        raise ValueError(f"No z-bins with >=2 seeds for tracer='{tracer}' signal='{signal}' -- "
                          f"nothing to plot.")

    z_list = sorted(agg.keys())

    if ell_targets is None:
        ell_los = [np.nanpercentile(agg[z]["ell"], 5) for z in z_list]
        ell_his = [np.nanpercentile(agg[z]["ell"], 95) for z in z_list]
        common_lo, common_hi = max(ell_los), min(ell_his)
        if not (common_hi > common_lo > 0):
            # z-bins don't share a common ell range (unusual) -- fall back to
            # the overall min/max across all z-bins instead of failing.
            common_lo = min(np.nanmin(agg[z]["ell"]) for z in z_list)
            common_hi = max(np.nanmax(agg[z]["ell"]) for z in z_list)
        ell_targets = np.geomspace(common_lo, common_hi, n_ell_targets)
    else:
        ell_targets = np.array(ell_targets)

    # interpolate each z-bin's median/sigma onto the common ell_targets
    median_vs_z = np.full((len(ell_targets), len(z_list)), np.nan)
    sigma_vs_z = np.full((len(ell_targets), len(z_list)), np.nan)
    for j, z_c in enumerate(z_list):
        d = agg[z_c]
        valid = np.isfinite(d["ell"]) & np.isfinite(d["median"])
        if valid.sum() < 2:
            continue
        f_med = interp1d(d["ell"][valid], d["median"][valid], bounds_error=False, fill_value=np.nan)
        f_sig = interp1d(d["ell"][valid], d["sigma"][valid], bounds_error=False, fill_value=np.nan)
        median_vs_z[:, j] = f_med(ell_targets)
        sigma_vs_z[:, j] = f_sig(ell_targets)

    norm = matplotlib.colors.LogNorm(vmin=ell_targets.min(), vmax=ell_targets.max())
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for i, ell in enumerate(ell_targets):
        color = cmap(norm(ell))
        med = median_vs_z[i, :]
        sig = sigma_vs_z[i, :]
        ax.plot(z_list, med, color=color, lw=1.6)
        ax.fill_between(z_list, med - sig, med + sig, color=color, alpha=0.18, lw=0)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Multipole $\ell$")

    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"$D_\ell$")
    ax.set_title(f"{signal} $\\times$ {tracer}, {len(seeds)} seeds, median $\\pm1\\sigma$")
    fig.savefig(os.path.join(out_dir, f"dell_vs_z_multi_ell_{signal}_{tracer}.pdf"))
    plt.close(fig)



def plot_snr_vs_z(cfg, SN_results: dict, z_cents, out_dir: str) -> None:
    """S/N vs z, one line per CMB experiment. LAE-only (see snr/ module docstrings)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    default_colors = {"SO": "steelblue", "CMB-S4": "darkorange", "CMB-HD": "darkgreen"}

    for name in cfg.snr.experiments:
        zs = np.array(sorted(SN_results[name].keys()))
        if len(zs) == 0:
            continue
        sn = np.array([SN_results[name][z] for z in zs])
        ax.plot(zs, sn, color=default_colors.get(name), lw=2.5, marker="o", markersize=4, label=name)

    ax.axhline(1, color="gray", ls="--", lw=1, label=r"$1\sigma$")
    ax.axhline(3, color="gray", ls=":", lw=1, label=r"$3\sigma$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"S/N per $\Delta z$ bin", fontsize=13)
    ax.set_title(
        r"kSZ$^2$ $\times$ LAE S/N vs redshift"
        f"\n({cfg.box.box_len_mpc:.0f} Mpc, {len(cfg.box.seeds)} seeds, "
        f"$f_{{\\rm sky}}={cfg.snr.f_sky}$, filtered kSZ)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=11)
    fig.savefig(os.path.join(out_dir, "SNR_vs_z_LAE_filtered.pdf"), dpi=300)
    plt.close(fig)
