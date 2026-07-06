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
