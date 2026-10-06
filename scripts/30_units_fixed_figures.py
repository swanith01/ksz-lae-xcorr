#!/usr/bin/env python3
"""
scripts/30_units_fixed_figures.py
===================================
Two figures that document the 2026-10-06 field-units fix (utils/field_units.py):

 Fig 1  units_fixed_vs_camb_seed{S}.{pdf,png}
        Raw velocity_z / hires_density power spectra vs CAMB linear theory at several z:
        row 1  P_vz(k) [(km/s)^2 Mpc^3]: raw (as used before), corrected v_pec = raw/(1+z), CAMB theory
        row 2  ratio to CAMB  (raw -> (1+z)^2,  corrected -> 1, flat in k over the linear range)
        row 3  density P(k)/P_lin(k,z): raw hires_density (z=0 IC, as used before) vs D(z)/D(0)-scaled
        needs the npz from  scripts/29_field_units_power_check.py --save-npz ...

 Fig 2  ksz_auto_units_fixed_seed{S}.{pdf,png}
        kSZ auto-power D_ell of ONE seed before/after the fix (D_total solid, D_diag dashed),
        from two scripts/26 pickles (legacy fields vs physical fields).

    python scripts/30_units_fixed_figures.py --npz paper/figure_scripts/output/field_units_power_seed1.npz \\
        --legacy-pkl data/products/ksz_auto_wrapcycle_legacyfields/seed1_wo0.pkl \\
        --physical-pkl data/products/ksz_auto_wrapcycle/seed1_wo0.pkl --seed 1
"""
import argparse
import os
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.utils.figio import save_fig

KM_PER_MPC = 3.0856775814913673e19
# validated reference palette, slots 1-2 (+ neutral ink for the theory curve)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2dd"
K_LO, K_HI = 0.04, 0.4      # linear, unsmoothed range for a 300 Mpc box at 1 Mpc cells


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_velocity_density(npz_path, seed, out_dir, Om0):
    from ksz_lae_xcorr.utils.field_units import growth_factor_ratio
    d = np.load(npz_path)
    k, zs = d["k"], d["z"]
    nz = len(zs)
    fig, axes = plt.subplots(3, nz, figsize=(3.3 * nz, 8.6), sharex=True, squeeze=False)
    c2 = KM_PER_MPC ** 2
    for j, z in enumerate(zs):
        raw = d["P_vz_raw"][j] * c2
        cor = raw / (1 + z) ** 2
        th = d["P_vz_pec_theory"][j] * c2
        ax = axes[0, j]
        ax.loglog(k, raw, color=ORANGE, lw=1.6, label="raw velocity_z (as used)")
        ax.loglog(k, cor, color=BLUE, lw=1.6, label=r"corrected $v_{pec}$ = raw/(1+z)")
        ax.loglog(k, th, color=INK, lw=1.0, ls=(0, (4, 2)), label="CAMB linear theory")
        ax.set_title(f"z = {z:.2f}", fontsize=10, color=INK)
        ax2 = axes[1, j]
        ax2.axhline(1.0, color=INK, lw=0.8)
        ax2.axhline((1 + z) ** 2, color=ORANGE, lw=0.8, ls=":")
        ax2.text(k[0], (1 + z) ** 2 * 1.08, r"$(1+z)^2$", color=MUTED, fontsize=8)
        ax2.loglog(k, d["P_vz_raw"][j] / d["P_vz_pec_theory"][j], color=ORANGE, lw=1.6)
        ax2.loglog(k, d["P_vz_raw"][j] / d["P_vz_pec_theory"][j] / (1 + z) ** 2, color=BLUE, lw=1.6)
        ax3 = axes[2, j]
        Dz2 = float(growth_factor_ratio(Om0, z)) ** 2
        ax3.axhline(1.0, color=INK, lw=0.8)
        ax3.loglog(k, d["P_d"][j] / d["P_lin_z"][j], color=ORANGE, lw=1.6)
        ax3.loglog(k, d["P_d"][j] * Dz2 / d["P_lin_z"][j], color=BLUE, lw=1.6)
        for a in (ax, ax2, ax3):
            a.axvspan(K_LO, K_HI, color="#f1f0ea", zorder=0)
            _style(a)
        ax3.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=9, color=MUTED)
        if j == 0:
            ax.set_ylabel(r"$P_{v_z}(k)$ [(km/s)$^2$ Mpc$^3$]", fontsize=9, color=MUTED)
            ax2.set_ylabel("P$_{v}$ / CAMB $P_{v,pec}$", fontsize=9, color=MUTED)
            ax3.set_ylabel(r"$P_\delta$ / CAMB $P_{lin}(k,z)$", fontsize=9, color=MUTED)
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Raw py21cmfast fields vs CAMB linear theory, seed {seed} (shaded: linear range "
                 f"k={K_LO}-{K_HI} Mpc$^{{-1}}$; lowest-k bins have few modes)", fontsize=10, color=INK, y=0.995)
    fig.tight_layout()
    paths = save_fig(fig, os.path.join(out_dir, f"units_fixed_vs_camb_seed{seed}"))
    plt.close(fig)
    return paths


def fig_ksz(legacy_pkl, physical_pkl, seed, out_dir, window="patchy", ell_ref=3000.0):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    notes = []
    for path, color, lab in ((legacy_pkl, ORANGE, "raw fields (before fix)"), (physical_pkl, BLUE, "physical fields (after fix)")):
        if not path or not os.path.exists(path):
            notes.append(f"{lab}: {path} not found - skipped")
            continue
        r = pickle.load(open(path, "rb"))[window]
        ell = r["ell"]
        ax.loglog(ell, np.abs(r["D_total"]), color=color, lw=1.8, label=f"{lab}: $D_{{total}}$")
        ax.loglog(ell, np.abs(r["D_diag"]), color=color, lw=1.2, ls=(0, (4, 2)), label=f"{lab}: $D_{{diag}}$")
        i = int(np.argmin(np.abs(ell - ell_ref)))
        ax.annotate(f"{r['D_total'][i]:.3g} $\\mu K^2$", (ell[i], abs(r["D_total"][i])), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=MUTED)
    ax.plot([ell_ref], [1.0], marker="D", mfc="none", mec=INK, ms=7, ls="none",
            label=r"$\sim1\,\mu K^2$ at $\ell$=3000 (La Plante+22 SO filter, approximate)")
    ax.set_xlabel(r"$\ell$", fontsize=9, color=MUTED)
    ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi\ \ [\mu K^2]$", fontsize=9, color=MUTED)
    ax.set_title(f"kSZ auto-power, seed {seed}, {window} window", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    paths = save_fig(fig, os.path.join(out_dir, f"ksz_auto_units_fixed_seed{seed}"))
    plt.close(fig)
    return paths, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--legacy-pkl", default=None)
    ap.add_argument("--physical-pkl", default=None)
    ap.add_argument("--window", choices=("patchy", "full"), default="patchy")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    args = ap.parse_args()
    from ksz_lae_xcorr.utils.config import load_config
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.npz:
        print("fig 1:", *fig_velocity_density(args.npz, args.seed, args.out_dir, float(cfg.cosmology.Om0)))
    if args.legacy_pkl or args.physical_pkl:
        paths, notes = fig_ksz(args.legacy_pkl, args.physical_pkl, args.seed, args.out_dir, args.window)
        print("fig 2:", *paths)
        for n in notes:
            print("  note:", n)


if __name__ == "__main__":
    sys.exit(main())
