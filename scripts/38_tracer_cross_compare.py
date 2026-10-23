#!/usr/bin/env python3
"""
scripts/38_tracer_cross_compare.py
====================================
D_ell^cross (kSZ^2 x tracer) against x_HII for every tracer of the suite, next to La Plante+22 Fig. 7 and (optionally) the
matter x Waters+16-bias run.  Reads the CSVs that scripts/28 wrote per tracer (pbs/pbs_tracer_cross.sh):

    <root>/<tracer name>/lp_cross_vs_z0_SO_wo0.csv        (columns z0, dz, x_hii, mean_l*, sigma_mean_l*, ...)

No cluster access needed -- run it on the desktop after copying the folder.

    python scripts/38_tracer_cross_compare.py --root data/plots/2026-10-16_tracer_cross \\
        --ref-csv data/plots/2026-10-12_zreion_cross_10seeds/lp_cross_vs_z0_SO_wo0.csv --ref-label "matter x Waters+16 bias"

Outputs (in --out-dir, default --root):
    tracer_cross_vs_xhii.png/.pdf   rows = ell, columns = halo | LAE | LBG, one curve per cut (seed mean), paper in orange;
                                    symmetric-log y axis by default (--yscale linear for the old look)
    panels/cross_<kind>_l<ell>.png  with --panels: each panel on its own
    tracer_cross_summary.csv        peak D_ell, x_HII at the peak, peak / s.e.m. (S/N of the seed mean), ratio to the paper peak
and prints the same table.
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.la_plante_reference import load_fig7_vs_xhii
from ksz_lae_xcorr.tracers.suite import spec_from_name
from ksz_lae_xcorr.utils.figio import save_fig

KINDS = (("halo", "halo mass cut", "Blues"), ("lae", r"LAE Ly$\alpha$ luminosity cut", "Greens"),
         ("lbg", "LBG UV magnitude cut", "Oranges"))
ORANGE = "#eb6834"


def load_csv(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return {n: np.atleast_1d(a[n]) for n in a.dtype.names}


def find_runs(root, tag="lp_cross_vs_z0_SO_wo0.csv"):
    """{tracer name: columns} for every <root>/<name>/<tag>."""
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "*", tag))):
        name = os.path.basename(os.path.dirname(p))
        try:
            spec_from_name(name)
        except Exception:
            continue
        runs[name] = load_csv(p)
    if not runs:
        raise SystemExit(f"no <name>/{tag} under {root}")
    return runs


def _order(names, kind):
    sel = [n for n in names if spec_from_name(n).kind == kind]
    return sorted(sel, key=lambda n: -spec_from_name(n).cut if kind == "lbg" else spec_from_name(n).cut)


def _colors(names, cmap):
    cm = plt.get_cmap(cmap)
    return {n: cm(0.35 + 0.6 * i / max(len(names) - 1, 1)) for i, n in enumerate(names)}


def _ells(runs):
    c = next(iter(runs.values()))
    return sorted(int(k.split("_l")[1]) for k in c if k.startswith("mean_l"))


def _linthresh(paper, ref, ell, floor=0.003):
    """Half the paper's peak D_ell: the paper curve and the reference stay in the linear core of the symlog axis."""
    pk = paper[ell][1].max() if ell in paper else 0.0
    if ref is not None and f"mean_l{ell}" in ref:
        pk = max(pk, float(np.nanmax(ref[f"mean_l{ell}"])))
    return float(f"{max(0.5 * pk, floor):.1g}")          # one significant figure -> clean tick labels


def _style_axis(ax, yscale, lin):
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=lin, linscale=2.0)
        ax.axhspan(-lin, lin, color="0.5", alpha=0.07, lw=0, zorder=0)       # shaded band = the linear core
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        t = [0.0, lin, -lin] + [sgn * d for d in (0.1, 1.0, 10.0) if d > lin for sgn in (1, -1)]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(sorted(t)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2g}"))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(1.0, 0.0)
    ax.set_ylabel(r"$D_\ell^{\rm cross}\ [\mu{\rm K}^2]$")


def _draw_panel(ax, runs, paper, ref, ref_label, dz, ell, kind, title, cmap):
    names = _order(runs, kind)
    col = _colors(names, cmap)
    if ell in paper:
        ax.plot(paper[ell][0], paper[ell][1], color=ORANGE, lw=3, label="La Plante+22 Fig. 7 (Fiducial)", zorder=1)
    if ref is not None and f"mean_l{ell}" in ref:
        s = np.isclose(ref["dz"], dz)
        ax.plot(ref["x_hii"][s], ref[f"mean_l{ell}"][s], "k--", lw=2, label=ref_label, zorder=2)
    for n in names:
        c = runs[n]; s = np.isclose(c["dz"], dz)
        ax.errorbar(c["x_hii"][s], c[f"mean_l{ell}"][s], yerr=c[f"sigma_mean_l{ell}"][s], color=col[n], marker="o",
                    ms=3, lw=1.3, elinewidth=0.7, label=spec_from_name(n).label, zorder=3)
    ax.set_title(rf"$\ell={ell}$: {title}")


def make_figure(runs, paper, ref=None, ref_label="matter x Waters+16", dz=1.0, ells=None, yscale="symlog"):
    """3 x 3 grid (rows = ell, columns = halo | LAE | LBG).  yscale 'symlog' (default) keeps the paper-sized curves readable
    and the large rare-tracer excursions on the same axis; the y range is shared along each ell row."""
    ells = ells or _ells(runs)
    fig, axes = plt.subplots(len(ells), 3, figsize=(16, 4.2 * len(ells)), sharex=True, sharey="row" if yscale == "symlog" else False,
                             squeeze=False, constrained_layout=True)
    for r, ell in enumerate(ells):
        lin = _linthresh(paper, ref, ell)
        for ax, (kind, title, cmap) in zip(axes[r], KINDS):
            _draw_panel(ax, runs, paper, ref, ref_label, dz, ell, kind, title, cmap)
            _style_axis(ax, yscale, lin)
            if r == len(ells) - 1:
                ax.set_xlabel(r"$x_{\rm HII}$")
            if r == 0:
                ax.legend(fontsize=7, loc="upper left")
    if yscale == "symlog":
        fig.suptitle(r"symmetric-log y axis: linear inside the grey band ($\pm$ half the paper peak), logarithmic outside", fontsize=10)
    return fig


def make_panel(runs, paper, ref, ref_label, dz, ell, kind, yscale="linear"):
    """One panel on its own (zoomed): linear y limited to the paper-scale range plus the errors of the abundant samples."""
    k = [q for q in KINDS if q[0] == kind][0]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    _draw_panel(ax, runs, paper, ref, ref_label, dz, ell, *k)
    _style_axis(ax, yscale, _linthresh(paper, ref, ell))
    ax.set_xlabel(r"$x_{\rm HII}$")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def summary(runs, paper, ref=None, dz=1.0, write=None):
    ells = _ells(runs)
    lines, rows = [], []
    head = f"  {'tracer':>16} {'ell':>5} {'peak D_l':>10} {'x_HII@peak':>11} {'peak/sem':>9} {'peak/paper':>11}"
    lines.append(head)
    items = [(n, runs[n]) for k, _, _ in KINDS for n in _order(runs, k)]
    if ref is not None:
        items.append(("reference", ref))
    for n, c in items:
        s = np.isclose(c["dz"], dz)
        for ell in ells:
            if f"mean_l{ell}" not in c:
                continue
            m, e, x = c[f"mean_l{ell}"][s], c[f"sigma_mean_l{ell}"][s], c["x_hii"][s]
            if not np.isfinite(m).any():
                continue
            i = int(np.nanargmax(m))
            pk = paper[ell][1].max() if ell in paper else np.nan
            rows.append(dict(name=n, ell=ell, peak=m[i], x_hii_at_peak=x[i], sem=e[i],
                             peak_over_sem=m[i] / e[i] if e[i] > 0 else np.nan, peak_over_paper=m[i] / pk))
            r = rows[-1]
            lines.append(f"  {n:>16} {ell:5d} {r['peak']:10.4f} {r['x_hii_at_peak']:11.2f} {r['peak_over_sem']:9.1f} {r['peak_over_paper']:11.2f}")
    if write and rows:
        with open(write, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with <tracer>/lp_cross_vs_z0_SO_wo0.csv")
    ap.add_argument("--tag", default="lp_cross_vs_z0_SO_wo0.csv")
    ap.add_argument("--ref-csv", default=None, help="scripts/28 CSV of matter x Waters+16 bias, drawn dashed")
    ap.add_argument("--ref-label", default="matter x Waters+16 bias")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--yscale", choices=("symlog", "linear"), default="symlog", help="y axis of the 3x3 figure")
    ap.add_argument("--panels", action="store_true", help="also write each of the 9 panels as its own file in <out-dir>/panels/")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)
    runs = find_runs(args.root, args.tag)
    ref = load_csv(args.ref_csv) if args.ref_csv else None
    paper = load_fig7_vs_xhii()
    print(f"{len(runs)} tracers: {sorted(runs)}")
    fig = make_figure(runs, paper, ref, args.ref_label, args.dz, yscale=args.yscale)
    print("figure:", *save_fig(fig, os.path.join(out_dir, "tracer_cross_vs_xhii")))
    plt.close(fig)
    if args.panels:
        os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)
        for ell in _ells(runs):
            for kind, _, _ in KINDS:
                if not _order(runs, kind):
                    continue
                fig = make_panel(runs, paper, ref, args.ref_label, args.dz, ell, kind, yscale=args.yscale)
                save_fig(fig, os.path.join(out_dir, "panels", f"cross_{kind}_l{ell}"))
                plt.close(fig)
    print(summary(runs, paper, ref, args.dz, write=os.path.join(out_dir, "tracer_cross_summary.csv")))


if __name__ == "__main__":
    sys.exit(main())
