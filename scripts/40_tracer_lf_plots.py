#!/usr/bin/env python3
"""
scripts/40_tracer_lf_plots.py
===============================
Three figures from the per-seed CSVs of scripts/39 (no cluster access needed -- run it on the desktop):

  halo_mass_function_vs_ST.png        dn/dlogM of ALL haloes at z ~ 6, 7, 8, 10 (seed mean +- s.e.m.) against the Sheth-Tormen
                                      mass function, with the ratio underneath
  uv_luminosity_function_vs_lit.png   UV LF of the LBG catalogue against observed UV LFs (Bouwens+21, Donnan+24,
                                      Finkelstein+24, McLeod+24) and their Schechter / double-power-law fits
  lya_luminosity_function_vs_lit.png  Lya LF of the LAE catalogue (observed luminosity) against SILVERRUSH (Konno+18,
                                      Umeda+25), Santos+16 and the JWST census (Kageura+25)
plus lf_summary.csv (n(>cut) of the catalogue vs the observed fit at the same z).

The literature numbers are in data/reference/luminosity_functions (SOURCES.md says where each came from).

    python scripts/40_tracer_lf_plots.py --in-dir data/plots/2026-10-15_tracer_lf --out-dir data/plots/2026-10-15_tracer_lf
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io import luminosity_functions as lf
from ksz_lae_xcorr.utils.figio import save_fig

Z_PANELS = (6.0, 7.0, 8.0, 10.0)
# colour-blind-safe (Okabe-Ito) colour + a marker per reference, so identity never rests on colour alone
STYLE = {"Bouwens21": ("#0072B2", "o"), "Donnan24": ("#D55E00", "s"), "Finkelstein24": ("#009E73", "^"), "McLeod24": ("#CC79A7", "D"),
         "Umeda25": ("#0072B2", "o"), "Kageura25": ("#D55E00", "s"), "Konno18": ("#009E73", "^"), "Santos16": ("#CC79A7", "D"),
         "Itoh18": ("#E69F00", "v"), "Ouchi10": ("#56B4E9", "P"), "Konno14": ("#E69F00", "X")}
PRETTY = {"Bouwens21": "Bouwens+21", "Donnan24": "Donnan+24", "Finkelstein24": "Finkelstein+24", "McLeod24": "McLeod+24",
          "Umeda25": "Umeda+25", "Kageura25": "Kageura+25", "Konno18": "Konno+18", "Santos16": "Santos+16", "Itoh18": "Itoh+18",
          "Ouchi10": "Ouchi+10", "Konno14": "Konno+14"}
SIM = "k"


def load_all(in_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(in_dir, "lf_seed*.csv"))):
        seed = int(os.path.basename(p).split("seed")[1].split(".")[0])
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(dict(seed=seed, kind=r["kind"], z=float(r["z"]), x=float(r["x"]), dx=float(r["dx"]), n=float(r["n"]),
                                 phi=float(r["phi"]), med=float(r["median_log10M"] or "nan"), st=float(r["phi_ST"] or "nan")))
    if not rows:
        raise SystemExit(f"no lf_seed*.csv in {in_dir}")
    return rows


def aggregate(rows, kind, z_target):
    """{x, n, phi, sem, med, st, z, n_seeds} at the snapshot nearest z_target (seed mean +- s.e.m.)."""
    rr = [r for r in rows if r["kind"] == kind]
    if not rr:
        return None
    zs = np.array(sorted({r["z"] for r in rr}))
    z = float(zs[np.argmin(np.abs(zs - z_target))])
    sel = [r for r in rr if r["z"] == z]
    xs = np.array(sorted({r["x"] for r in sel}))
    out = {k: [] for k in ("n", "phi", "sem", "med", "st")}
    seeds = len({r["seed"] for r in sel})
    for x in xs:
        s = [r for r in sel if r["x"] == x]
        phi = np.array([r["phi"] for r in s])
        out["n"].append(np.mean([r["n"] for r in s]))
        out["phi"].append(phi.mean())
        out["sem"].append(phi.std(ddof=1) / np.sqrt(len(phi)) if len(phi) > 1 else np.nan)
        out["med"].append(np.nanmean([r["med"] for r in s]) if np.isfinite([r["med"] for r in s]).any() else np.nan)
        out["st"].append(s[0]["st"])
    return dict(x=xs, z=z, n_seeds=seeds, **{k: np.array(v, dtype=float) for k, v in out.items()})


def _ylim_from(a, lo_floor, hi_cap=None):
    pos = a[np.isfinite(a) & (a > 0)]
    return (max(pos.min() / 3, lo_floor), (hi_cap or pos.max() * 5)) if pos.size else (lo_floor, 1)


# ---------------------------------------------------------------------------------------- halo mass function
def fig_halo_mf(rows, min_n=10):
    fig, axes = plt.subplots(2, len(Z_PANELS), figsize=(4.2 * len(Z_PANELS), 6.2), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[3, 1.2]))
    for j, zt in enumerate(Z_PANELS):
        a = aggregate(rows, "halo_mf", zt)
        ax, rx = axes[0, j], axes[1, j]
        ok = a["n"] >= min_n
        ax.plot(a["x"], a["st"], "-", color="#D55E00", lw=2.2, label="Sheth-Tormen", zorder=1)
        ax.errorbar(a["x"][ok], a["phi"][ok], yerr=np.nan_to_num(a["sem"][ok]), fmt="o", color=SIM, ms=4, lw=1,
                    label=f"21cmFAST catalogue ({a['n_seeds']} seed{'s' if a['n_seeds'] > 1 else ''})", zorder=2)
        ax.set_yscale("log")
        ax.set_xlim(8.0, a["x"][ok].max() + 0.6)
        ax.set_ylim(a["phi"][ok].min() / 30, a["st"][np.isfinite(a["st"])].max() * 3)
        ax.set_title(f"z = {a['z']:.2f}")
        if j == 0:
            ax.set_ylabel(r"$dn/d\log_{10}M\ [{\rm Mpc^{-3}\,dex^{-1}}]$")
            ax.legend(fontsize=8, loc="lower left", frameon=False)
        rx.axhline(1.0, color="0.5", lw=0.8)
        rx.plot(a["x"][ok], a["phi"][ok] / a["st"][ok], "o-", color=SIM, ms=3, lw=1)
        rx.set_ylim(0, 2.0)
        rx.set_xlabel(r"$\log_{10}\,M_{\rm halo}\ [M_\odot]$")
        if j == 0:
            rx.set_ylabel("catalogue / ST")
    return fig


# ---------------------------------------------------------------------------------------- UV LF
def _lit_panel(ax, pts, z_sim, keys, tol, xcol, ycol, ylog_vals, err_up, err_lo, nearest=()):
    """draw observed points of the refs in `keys` whose z lies within tol of z_sim; returns legend handles."""
    handles = {}
    zz = np.asarray(pts["z"], dtype=float)
    for key in keys:
        zl_all = sorted(set(zz[(pts["ref_key"] == key) & (np.abs(zz - z_sim) <= tol)]))
        if key in nearest and zl_all:                       # one redshift bin of a coarse-binned survey: the closest
            zl_all = [min(zl_all, key=lambda q: abs(q - z_sim))]
        for zl in zl_all:
            m = (pts["ref_key"] == key) & (zz == zl)
            col, mk = STYLE[key]
            x = np.asarray(pts[xcol], dtype=float)[m]
            y = ylog_vals(m)
            up, lo = err_up(m, y), err_lo(m, y)
            ul = np.asarray(pts["upper_limit"], dtype=float)[m] > 0
            good = ~ul
            ax.errorbar(x[good], y[good], yerr=[np.nan_to_num(lo[good]), np.nan_to_num(up[good])], fmt=mk, color=col, ms=5.5,
                        mfc="none", mew=1.3, lw=0.9, capsize=0, zorder=3)
            if ul.any():
                ax.plot(x[ul], y[ul], "v", color=col, ms=5, mfc="none", mew=1.0, zorder=3)
            lab = f"{PRETTY[key]} (z={zl:g})"
            handles[lab] = plt.Line2D([], [], marker=mk, ls="", color=col, mfc="none", mew=1.3, ms=6, label=lab)
    return handles


def fig_uvlf(rows, min_n=10, faint_mass=10 ** 9.8):
    pts, fits = lf.uvlf_points(), lf.uvlf_fits()
    fig, axes = plt.subplots(1, len(Z_PANELS), figsize=(4.4 * len(Z_PANELS) + 3.2, 4.8), sharey=True, sharex=True)
    allh = {}
    for ax, zt in zip(axes, Z_PANELS):
        a = aggregate(rows, "uvlf", zt)
        ok = a["n"] >= min_n
        full = ok & ~(a["med"] < np.log10(faint_mass))
        ax.plot(a["x"][full], a["phi"][full], "-", color=SIM, lw=2.4, zorder=4)
        ax.plot(a["x"][ok], a["phi"][ok], "--", color=SIM, lw=1.3, zorder=4)
        ax.fill_between(a["x"][ok], np.clip(a["phi"] - np.nan_to_num(a["sem"]), 1e-12, None)[ok], (a["phi"] + np.nan_to_num(a["sem"]))[ok],
                        color=SIM, alpha=0.2, lw=0, zorder=3)
        tol = 0.6 if zt < 9 else 1.1
        h = _lit_panel(ax, pts, a["z"], ("Bouwens21", "Donnan24", "Finkelstein24", "McLeod24"), tol, "M_UV", "phi_Mpc3_mag",
                       lambda m: np.asarray(pts["phi_Mpc3_mag"], dtype=float)[m],
                       lambda m, y: np.asarray(pts["err_up"], dtype=float)[m],
                       lambda m, y: np.minimum(np.asarray(pts["err_lo"], dtype=float)[m], 0.9 * y))
        allh.update(h)
        Mg = np.linspace(-23, -15.5, 200)
        fz = np.asarray(fits["z"], dtype=float)
        for i in np.where(np.abs(fz - a["z"]) <= tol)[0]:
            col = STYLE[str(fits["ref_key"][i])][0]
            ax.plot(Mg, lf.uv_phi_of_row(fits, i, Mg), "-", color=col, lw=1.0, alpha=0.8, zorder=2)
        ax.set_yscale("log")
        ax.set_xlim(-14.5, -23)
        ax.set_ylim(1e-8, 0.1)
        ax.set_title(f"z = {a['z']:.2f}")
        ax.set_xlabel(r"$M_{\rm UV}$")
    axes[0].set_ylabel(r"$\phi\ [{\rm Mpc^{-3}\,mag^{-1}}]$")
    sim = [plt.Line2D([], [], color=SIM, lw=2.4, label="21cmFAST LBG catalogue"),
           plt.Line2D([], [], color=SIM, lw=1.3, ls="--", label="dashed: median halo mass < 1e9.8 Msun\n(the catalogue's mass cut shapes the faint end)"),
           plt.Line2D([], [], color="0.4", lw=1.0, label="published Schechter / DPL fit")]
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.legend(handles=sim + list(allh.values()), loc="center right", fontsize=7.5, frameon=False)
    return fig


# ---------------------------------------------------------------------------------------- Lya LF
def fig_lae_lf(rows, min_n=3):
    pts, fits = lf.lae_lf_points(), lf.lae_lf_fits()
    fig, axes = plt.subplots(1, len(Z_PANELS), figsize=(4.4 * len(Z_PANELS) + 3.0, 4.8), sharey=True, sharex=True)
    allh = {}
    for ax, zt in zip(axes, Z_PANELS):
        a = aggregate(rows, "lae_lf", zt)
        ok = (a["n"] >= min_n) & (a["x"] > 42.0)
        ax.errorbar(a["x"][ok], a["phi"][ok], yerr=np.nan_to_num(a["sem"][ok]), fmt="o-", color=SIM, ms=4, lw=1.6, zorder=4)
        tol = 3.5
        keys = ("Umeda25", "Konno18", "Santos16", "Kageura25")
        tol_hst = 0.7                                        # narrow-band surveys: only bins within 0.7 of the snapshot

        def y(m):
            return 10.0 ** np.asarray(pts["log_phi"], dtype=float)[m]

        def up(m, yy):
            return 10.0 ** (np.asarray(pts["log_phi"], dtype=float)[m] + np.nan_to_num(np.asarray(pts["err_up"], dtype=float)[m])) - yy

        def lo(m, yy):
            return yy - 10.0 ** (np.asarray(pts["log_phi"], dtype=float)[m] - np.nan_to_num(np.asarray(pts["err_lo"], dtype=float)[m]))
        allh.update(_lit_panel(ax, pts, a["z"], ("Umeda25", "Konno18", "Santos16"), tol_hst, "log_L", "log_phi", y, up, lo))
        allh.update(_lit_panel(ax, pts, a["z"], ("Kageura25",), tol, "log_L", "log_phi", y, up, lo, nearest=("Kageura25",)))
        Lg = np.linspace(42.0, 44.2, 200)
        fz = np.asarray(fits["z"], dtype=float)
        for i in np.where((np.abs(fz - a["z"]) <= tol_hst) & np.isin(fits["ref_key"], ("Umeda25", "Konno18", "Itoh18", "Santos16")))[0]:
            if fits["ref_key"][i] == "Santos16" and not np.isfinite(fits["alpha_err"][i]):
                continue
            ax.plot(Lg, lf.schechter_lya_dex(Lg, fits["log_Lstar"][i], fits["log_phistar"][i], fits["alpha"][i]), "-",
                    color=STYLE[str(fits["ref_key"][i])][0], lw=0.8, alpha=0.7, zorder=2)
        ax.set_yscale("log")
        ax.set_xlim(42.0, 44.2)
        ax.set_ylim(1e-8, 0.03)
        ax.set_title(f"z = {a['z']:.2f}")
        ax.set_xlabel(r"$\log_{10}\,L_{{\rm Ly}\alpha}\ [{\rm erg\,s^{-1}}]$")
    axes[0].set_ylabel(r"$\phi\ [{\rm Mpc^{-3}\,dex^{-1}}]$")
    sim = [plt.Line2D([], [], color=SIM, marker="o", lw=1.6, ms=4, label="21cmFAST LAE catalogue (observed $L$)"),
           plt.Line2D([], [], color="0.4", lw=0.8, label="published Schechter fit")]
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.legend(handles=sim + list(allh.values()), loc="center right", fontsize=7.5, frameon=False)
    return fig


def summary_rows(rows):
    """n(> cut) of the catalogue vs the published fits at the nearest z (LAE: L > 10^42.5, 10^43 ; UV: M < -18, -20)."""
    out = []
    uv, lae = lf.uvlf_fits(), lf.lae_lf_fits()
    for zt in Z_PANELS:
        a = aggregate(rows, "uvlf", zt)
        b = aggregate(rows, "lae_lf", zt)
        for cut in (-18.0, -20.0):
            sim = float(np.sum(a["phi"][a["x"] < cut] * 0.5))
            lit = [(str(uv["ref_key"][i]), lf.uv_cumulative(uv, i, cut)) for i in np.where(np.abs(np.asarray(uv["z"], dtype=float) - a["z"]) <= 0.7)[0]]
            out.append(dict(kind="UV", z=a["z"], cut=cut, n_catalogue=sim, n_literature=np.mean([v for _, v in lit]) if lit else np.nan,
                            source=";".join(sorted({k for k, _ in lit}))))
        for cut in (42.5, 43.0):
            sim = float(np.sum(b["phi"][b["x"] >= cut] * 0.1))
            lit = [(str(lae["ref_key"][i]), lf.lya_cumulative(lae, i, cut)) for i in np.where(np.abs(np.asarray(lae["z"], dtype=float) - b["z"]) <= 0.7)[0]
                   if str(lae["ref_key"][i]) in ("Umeda25", "Konno18", "Itoh18")]
            out.append(dict(kind="Lya", z=b["z"], cut=cut, n_catalogue=sim, n_literature=np.mean([v for _, v in lit]) if lit else np.nan,
                            source=";".join(sorted({k for k, _ in lit}))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="folder with lf_seed*.csv (scripts/39)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)
    rows = load_all(args.in_dir)
    print(f"seeds: {sorted({r['seed'] for r in rows})}")
    for fn, name in ((fig_halo_mf(rows), "halo_mass_function_vs_ST"), (fig_uvlf(rows), "uv_luminosity_function_vs_lit"),
                     (fig_lae_lf(rows), "lya_luminosity_function_vs_lit")):
        print("figure:", *save_fig(fn, os.path.join(out_dir, name)))
        plt.close(fn)
    s = summary_rows(rows)
    with open(os.path.join(out_dir, "lf_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s[0]))
        w.writeheader()
        w.writerows(s)
    print(f"\n{'kind':>4} {'z':>6} {'cut':>6} {'n catalogue [Mpc^-3]':>22} {'n observed fit [Mpc^-3]':>25} {'cat/obs':>8}  source")
    for r in s:
        print(f"{r['kind']:>4} {r['z']:6.2f} {r['cut']:6.1f} {r['n_catalogue']:22.3g} {r['n_literature']:25.3g} {r['n_catalogue'] / r['n_literature']:8.2f}  {r['source']}")


if __name__ == "__main__":
    sys.exit(main())
