#!/usr/bin/env python3
"""
scripts/36_tracer_bias_plot.py
================================
Plots + one table from the per-seed CSVs of scripts/35 (no cluster access needed -- run it on the desktop):

  tracer_bias_vs_z.png        b_lin(z) of every cut (halo mass | LAE luminosity | LBG MUV), seed mean +- s.e.m., dashed =
                              analytic Sheth-Mo-Tormen bias of the same sample, black = the La Plante / Waters+16 bias
                              b_g(z) = 2.1 (1+z) - 5.3 that their delta_g assumes.
  tracer_number_density.png   n(z) [Mpc^-3] and the number of objects in the 300 Mpc box (right axis).
  tracer_bias_vs_number.png   the pitch plot: objects in the box against bias, at z = 6, 7, 8, 10, with the Waters+16 bias
                              marked -- how many objects are left when a sample is as biased as their galaxies?
  tracer_bias_summary.csv     seed-aggregated numbers behind the plots.
Also prints, per redshift, the volume (and box side) needed to have N_TARGET objects for each cut.

    python scripts/36_tracer_bias_plot.py --in-dir data/plots/2026-10-14_tracer_bias --out-dir data/plots/2026-10-14_tracer_bias
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

from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.tracers.suite import spec_from_name
from ksz_lae_xcorr.utils.figio import save_fig

KINDS = (("halo", "halo mass cut", "Blues"), ("lae", r"LAE Ly$\alpha$ luminosity cut", "Greens"),
         ("lbg", "LBG UV magnitude cut", "Oranges"))
MARK = {"halo": "o", "lae": "s", "lbg": "^"}
N_TARGET = 1e4


def load_all(in_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(in_dir, "tracer_bias_seed*.csv"))):
        seed = int(os.path.basename(p).split("seed")[1].split(".")[0])
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(dict(seed=seed, z=float(r["z"]), name=r["name"], kind=r["kind"], cut=float(r["cut"]),
                                 n_obj=float(r["n_obj"]), n=float(r["n_per_mpc3"]), b=float(r["b_lin"]),
                                 b_st=float(r["b_ST"])))
    if not rows:
        raise SystemExit(f"no tracer_bias_seed*.csv in {in_dir}")
    return rows


def aggregate(rows):
    """{name: dict(z, b, b_sem, b_st, n_obj, n, n_seeds)} (seed mean, s.e.m.)."""
    out = {}
    for name in sorted({r["name"] for r in rows}):
        rr = [r for r in rows if r["name"] == name]
        zs = np.array(sorted({r["z"] for r in rr}))
        agg = {k: [] for k in ("b", "b_sem", "b_st", "n_obj", "n", "n_seeds")}
        for z in zs:
            sel = [r for r in rr if r["z"] == z]
            b = np.array([r["b"] for r in sel], dtype=float)
            ok = np.isfinite(b)
            agg["b"].append(np.nanmean(b) if ok.any() else np.nan)
            agg["b_sem"].append(np.nanstd(b) / np.sqrt(max(ok.sum(), 1)) if ok.sum() > 1 else 0.0)
            agg["b_st"].append(np.nanmean([r["b_st"] for r in sel]) if np.isfinite([r["b_st"] for r in sel]).any() else np.nan)
            agg["n_obj"].append(np.mean([r["n_obj"] for r in sel]))
            agg["n"].append(np.mean([r["n"] for r in sel]))
            agg["n_seeds"].append(len(sel))
        out[name] = {"z": zs, "kind": rr[0]["kind"], "cut": rr[0]["cut"], **{k: np.array(v, dtype=float) for k, v in agg.items()}}
    return out


def _colors(names, cmap):
    cm = plt.get_cmap(cmap)
    return {n: cm(0.35 + 0.6 * i / max(len(names) - 1, 1)) for i, n in enumerate(names)}


def _ordered(agg, kind):
    names = [n for n, a in agg.items() if a["kind"] == kind]
    key = (lambda n: -agg[n]["cut"]) if kind == "lbg" else (lambda n: agg[n]["cut"])     # faint -> bright / low -> high cut
    return sorted(names, key=key)


def fig_bias_vs_z(agg, min_obj=30):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    zz = np.linspace(5, 14, 100)
    for ax, (kind, title, cmap) in zip(axes, KINDS):
        names = _ordered(agg, kind)
        col = _colors(names, cmap)
        for n in names:
            a = agg[n]; ok = a["n_obj"] >= min_obj
            lab = spec_from_name(n).label
            ax.plot(a["z"][ok], a["b"][ok], "-", color=col[n], lw=1.8, label=lab)
            ax.fill_between(a["z"][ok], (a["b"] - a["b_sem"])[ok], (a["b"] + a["b_sem"])[ok], color=col[n], alpha=0.25, lw=0)
            ax.plot(a["z"][ok], a["b_st"][ok], "--", color=col[n], lw=1.0)
        ax.plot(zz, bluetides_bias_gz(zz), "-", color="k", lw=3, label=r"Waters+16 / La Plante+22: $2.1(1+z)-5.3$")
        ax.set_xlim(5, 14); ax.set_ylim(0, 26)
        ax.set_xlabel("z"); ax.set_title(title); ax.grid(alpha=0.2); ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(r"linear bias $b$ (tracer vs matter, $k<0.15$ Mpc$^{-1}$)")
    fig.suptitle("Bias of the 21cmFAST tracers (solid, seed mean) vs analytic Sheth-Mo-Tormen (dashed) and the La Plante+22 bias (black)\n"
                 "curves hidden where fewer than 30 objects are in the 300 Mpc box", fontsize=10)
    return fig


def fig_number_density(agg, boxlen=300.0):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True, constrained_layout=True)
    vol = boxlen ** 3
    for ax, (kind, title, cmap) in zip(axes, KINDS):
        names = _ordered(agg, kind)
        col = _colors(names, cmap)
        for n in names:
            a = agg[n]; ok = a["n"] > 0
            ax.plot(a["z"][ok], a["n"][ok], "-", color=col[n], lw=1.8, label=spec_from_name(n).label)
        ax.set_yscale("log"); ax.set_xlim(5, 20); ax.set_xlabel("z"); ax.set_title(title); ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7, loc="lower left")
        sec = ax.secondary_yaxis("right", functions=(lambda y: y * vol, lambda y: y / vol))
        sec.set_ylabel(f"objects in the {boxlen:.0f} Mpc box")
    axes[0].set_ylabel(r"number density [Mpc$^{-3}$]")
    return fig


def nearest_z(agg, zt):
    zs = next(iter(agg.values()))["z"]
    return float(zs[np.argmin(np.abs(zs - zt))])


def fig_bias_vs_number(agg, z_list=(6.0, 7.0, 8.0, 10.0)):
    fig, axes = plt.subplots(1, len(z_list), figsize=(4.0 * len(z_list) + 2.6, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    handles = {}
    for ax, zt in zip(axes, z_list):
        zn = nearest_z(agg, zt)
        for kind, title, cmap in KINDS:
            names = _ordered(agg, kind)
            col = _colors(names, cmap)
            for n in names:
                a = agg[n]; i = int(np.argmin(np.abs(a["z"] - zn)))
                if a["n_obj"][i] <= 0 or not np.isfinite(a["b"][i]):
                    continue
                ax.errorbar(a["b"][i], a["n_obj"][i], xerr=a["b_sem"][i], fmt=MARK[kind], color=col[n], ms=7)
                handles[n] = plt.Line2D([], [], marker=MARK[kind], ls="", color=col[n], ms=7, label=spec_from_name(n).label)
        bw = float(bluetides_bias_gz(zn))
        ax.axvline(bw, color="k", lw=2.5)
        ax.text(bw - 0.3, 0.97, "La Plante bias", rotation=90, ha="right", va="top", fontsize=8, transform=ax.get_xaxis_transform())
        ax.axhline(N_TARGET, color="0.5", ls=":", lw=1)
        ax.set_yscale("log"); ax.set_xlim(0, 26); ax.set_xlabel("linear bias b")
        ax.set_title(f"z = {zn:.1f}"); ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("objects in the 300 Mpc box")
    axes[0].text(0.3, N_TARGET * 1.15, rf"$N=10^{{{np.log10(N_TARGET):.0f}}}$", fontsize=8, color="0.4")
    order = [n for k, _, _ in KINDS for n in _ordered(agg, k) if n in handles]
    fig.tight_layout(rect=(0, 0, 0.87, 1))
    fig.legend([handles[n] for n in order], [handles[n].get_label() for n in order], loc="center right", fontsize=7, frameon=False)
    return fig


def summary_table(agg, z_list=(6.0, 7.0, 8.0, 10.0), boxlen=300.0, write=None):
    lines, out_rows = [], []
    for zt in z_list:
        zn = nearest_z(agg, zt)
        bw = float(bluetides_bias_gz(zn))
        lines.append(f"\nz = {zn:.2f}   La Plante / Waters+16 bias = {bw:.1f}")
        lines.append(f"  {'tracer':>16} {'N in box':>12} {'b':>6} {'b/b_W':>6} {'vol. factor for N=1e4':>22} {'box side [Mpc]':>15}")
        for kind, _, _ in KINDS:
            for n in _ordered(agg, kind):
                a = agg[n]; i = int(np.argmin(np.abs(a["z"] - zn)))
                N, b = a["n_obj"][i], a["b"][i]
                fac = N_TARGET / N if N > 0 else np.inf
                side = boxlen * fac ** (1 / 3) if np.isfinite(fac) else np.inf
                lines.append(f"  {n:>16} {N:12,.0f} {b:6.2f} {b / bw:6.2f} {fac:22.3g} {side:15.0f}")
                out_rows.append(dict(z=zn, name=n, n_obj=N, b=b, b_waters=bw, volume_factor_1e4=fac, box_side_mpc=side))
    if write:
        with open(write, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader(); w.writerows(out_rows)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="folder with tracer_bias_seed*.csv (from scripts/35)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--box-len", type=float, default=300.0)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate(load_all(args.in_dir))
    print(f"{len(agg)} tracers, seeds per snapshot: {int(next(iter(agg.values()))['n_seeds'].max())}")
    for fn, name in ((fig_bias_vs_z(agg), "tracer_bias_vs_z"), (fig_number_density(agg, args.box_len), "tracer_number_density"),
                     (fig_bias_vs_number(agg), "tracer_bias_vs_number")):
        print("figure:", *save_fig(fn, os.path.join(out_dir, name)))
        plt.close(fn)
    print(summary_table(agg, boxlen=args.box_len, write=os.path.join(out_dir, "tracer_bias_summary.csv")))


if __name__ == "__main__":
    sys.exit(main())
