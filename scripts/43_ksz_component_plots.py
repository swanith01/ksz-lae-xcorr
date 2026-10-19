#!/usr/bin/env python3
"""
scripts/43_ksz_component_plots.py
===================================
Plots for the component decomposition of the kSZ^2 x tracer cross-power (scripts/41 + 42).  Desktop-only: reads the
components_<tracer>.npz files copied from swarm.

    python scripts/43_ksz_component_plots.py --root data/plots/2026-10-19_components --ell 1000

Outputs in --out-dir (default --root), for the chosen ell:
    components_<kind>_l<ell>.png/.pdf   one figure per tracer kind (halo | lae | lbg), 2x2 panels:
                                        rows  = what is squared: the ionisation field x_e (top) or the velocity v_los (bottom)
                                        cols  = project then square (left) | square every slice, then project (right)
                                        solid + dots = that component, dashed = the full kSZ^2 of the same ordering (same colour);
                                        3 cuts (lowest, middle, highest) by default -- no more, to keep it readable
    components_fraction_l<ell>.png/.pdf one summary figure: for each tracer, which fraction of the full kSZ^2 x tracer cross-power
                                        each variant carries (window-pooled over x_HII in --xrange; bootstrap over seeds)
    components_fraction_l<ell>.csv      the numbers behind it
y axis: symmetric log by default (linear around zero, logarithmic outside) so the large noisy excursions of the rare tracers
do not squash the rest; --yscale linear for the plain axis.
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

from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.tracers.suite import spec_from_name
from ksz_lae_xcorr.utils.figio import save_fig

KINDS = (("halo", "halo mass cut", "Blues"), ("lae", r"LAE Ly$\alpha$ luminosity cut", "Greens"),
         ("lbg", "LBG UV magnitude cut", "Oranges"))
COMP_COLOR = {"x": "#0072B2", "v": "#E69F00"}
PICK_COLORS = {"halo": "Blues", "lae": "Greens", "lbg": "Oranges"}


def load_runs(root):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "components_*.npz"))):
        name = os.path.basename(p)[len("components_"):-len(".npz")]
        with np.load(p, allow_pickle=False) as z:
            runs[name] = {k: z[k] for k in z.files}
    if not runs:
        raise SystemExit(f"no components_*.npz under {root}")
    return runs


def ordered(runs, kind):
    names = [n for n in runs if n != "matter_Waters" and spec_from_name(n).kind == kind]
    return sorted(names, key=lambda n: -spec_from_name(n).cut if kind == "lbg" else spec_from_name(n).cut)


def pick(names, k):
    """k cuts spread over the list: lowest, ..., highest."""
    if len(names) <= k:
        return list(names)
    return [names[int(round(i * (len(names) - 1) / (k - 1)))] for i in range(k)]


def stats(run, variant, ell):
    vi = list(run["variants"]).index(variant)
    ti = int(np.argmin(np.abs(run["ell_targets"] - ell)))
    D = run["D"][:, vi, :, ti]
    n = np.sum(np.isfinite(D), axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(D, axis=0), np.nanstd(D, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))


def _linthresh(runs, names, ell):
    """One significant figure of half the largest |mean D| of the full kSZ^2 (A) among the plotted tracers."""
    pk = max(np.nanmax(np.abs(stats(runs[n], "A_full", ell)[0])) for n in names)
    return float(f"{0.5 * pk:.1g}") if np.isfinite(pk) and pk > 0 else 1e-3


def _symlog(ax, lin):
    ax.set_yscale("symlog", linthresh=lin, linscale=2.0)
    ax.axhspan(-lin, lin, color="0.5", alpha=0.07, lw=0, zorder=0)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    lo, hi = ax.get_ylim()
    ticks = [0.0, lin, -lin] + [s * lin * 10 ** j for j in range(1, 8) for s in (1, -1) if lo <= s * lin * 10 ** j <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(sorted(ticks)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2g}"))


def fig_kind(runs, kind, title, ell, ncut=3, yscale="symlog"):
    names = pick(ordered(runs, kind), ncut)
    if not names:
        return None
    cm = plt.get_cmap(PICK_COLORS[kind])
    col = {n: cm(0.4 + 0.55 * i / max(len(names) - 1, 1)) for i, n in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for r, comp in enumerate(("x", "v")):
        for c, order in enumerate(kc.ORDERINGS):
            ax = axes[r, c]
            for n in names:
                x = np.nanmean(runs[n]["x_hii"], axis=0)
                m, e = stats(runs[n], f"{order}_{comp}", ell)
                mf, _ = stats(runs[n], f"{order}_full", ell)
                ax.plot(x, mf, color=col[n], lw=1.3, ls="--", zorder=2)
                ax.errorbar(x, m, yerr=e, color=col[n], marker="o", ms=3.5, lw=1.5, elinewidth=0.7, zorder=3,
                            label=spec_from_name(n).label)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlim(1.0, 0.0)
            ax.set_title(f"{kc.LABELS[comp]}: {kc.ORDER_LABELS[order]}")
            if c == 0:
                ax.set_ylabel(r"$D^{\rm frac}_\ell$  (fractional modulation $\times\ \delta_{\rm tracer}$)")
            if r == 1:
                ax.set_xlabel(r"$x_{\rm HII}$")
    if yscale == "symlog":
        lin = _linthresh(runs, names, ell)
        for ax in axes.ravel():
            _symlog(ax, lin)
    h, lab = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="0.3", ls="--", lw=1.3)]
    lab += ["full kSZ$^2$, same ordering"]
    axes[0, 0].legend(h, lab, fontsize=8, loc="upper left")
    fig.suptitle(rf"$\ell={ell:g}$, {title}: solid = one ingredient, dashed = full kSZ$^2$", fontsize=11)
    return fig


def fraction_table(runs, ell, xrange):
    rows = []
    for name, run in runs.items():
        x = np.nanmean(run["x_hii"], axis=0)
        ti = int(np.argmin(np.abs(run["ell_targets"] - ell)))
        for order in kc.ORDERINGS:
            vf = list(run["variants"]).index(f"{order}_full")
            for comp in ("x", "v", "d"):
                vv = list(run["variants"]).index(f"{order}_{comp}")
                r = kc.fraction_of_full(run["D"][:, vv, :, ti], run["D"][:, vf, :, ti], x, xrange)
                rows.append({"tracer": name, "ordering": order, "component": comp, "ell": ell,
                             "R": r["R"], "R_lo16": r["lo"], "R_hi84": r["hi"],
                             "x_hii_lo": xrange[0], "x_hii_hi": xrange[1]})
    return rows


def _cut_label(name):
    sp = spec_from_name(name)
    return f"{sp.cut:g}" if sp.kind == "lbg" else f"{np.log10(sp.cut):.1f}"


def fig_fraction(rows, runs, ell, xrange, ymin=-1.5, ymax=3.0):
    groups = [("matter", "matter ($b_g\\delta_m$)")] + [(k, t) for k, t, _ in KINDS]
    groups = [(k, t) for k, t in groups if (k == "matter" and "matter_Waters" in runs) or (k != "matter" and ordered(runs, k))]
    nr, nc = (1, len(groups)) if len(groups) <= 2 else (2, 2)
    fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.4 * nr), constrained_layout=True, squeeze=False)
    bar_spec = [(o, c) for c in ("x", "v") for o in kc.ORDERINGS]
    by = {(r["tracer"], r["ordering"], r["component"]): r for r in rows}
    for ax, (kind, title) in zip(axes.ravel(), groups):
        names = ["matter_Waters"] if kind == "matter" else ordered(runs, kind)
        w = 0.8 / len(bar_spec)
        for gi, n in enumerate(names):
            for bi, (o, c) in enumerate(bar_spec):
                r = by[(n, o, c)]
                xpos = gi + (bi - (len(bar_spec) - 1) / 2) * w
                val = r["R"]
                ax.bar(xpos, np.clip(val, ymin, ymax) if np.isfinite(val) else 0, width=w * 0.92, color=COMP_COLOR[c],
                       hatch="" if o == "A" else "///", edgecolor="white" if o == "A" else "k", lw=0.4)
                if np.isfinite(val):
                    lo, hi = np.clip(r["R_lo16"], ymin, ymax), np.clip(r["R_hi84"], ymin, ymax)
                    ax.plot([xpos, xpos], [lo, hi], color="k", lw=0.8)
                    if val > ymax or val < ymin:
                        ax.annotate(f"{val:.0f}", (xpos, np.clip(val, ymin, ymax)), ha="center", va="bottom" if val > 0 else "top",
                                    fontsize=6)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["" if n == "matter_Waters" else _cut_label(n) for n in names],
                           rotation=45, ha="right", fontsize=8)
        ax.set_xlabel({"matter": "", "halo": r"$\log_{10} M_{\rm min}$", "lae": r"$\log_{10} L_{{\rm Ly}\alpha}$",
                       "lbg": r"$M_{\rm UV}$ limit"}[kind])
        ax.set_title(title)
        ax.set_ylabel(r"fraction of full kSZ$^2\times$tracer")
    from matplotlib.patches import Patch
    axes.ravel()[0].legend(handles=[Patch(color=COMP_COLOR["x"], label=r"$x_e$"), Patch(color=COMP_COLOR["v"], label=r"$v_{\rm los}$"),
                               Patch(facecolor="w", edgecolor="k", label="project, then square"),
                               Patch(facecolor="w", edgecolor="k", hatch="///", label="square, then project")],
                      fontsize=7, loc="upper left")
    fig.suptitle(rf"$\ell={ell:g}$; windows with ${xrange[0]:g}\leq x_{{\rm HII}}\leq {xrange[1]:g}$; bars = median, lines = 16-84% (seed bootstrap); "
                 r"dotted = 1 (the component carries all of it)", fontsize=10)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ell", type=float, default=1000.0)
    ap.add_argument("--ncut", type=int, default=3, help="cuts drawn per tracer kind in the 2x2 figures")
    ap.add_argument("--xrange", type=float, nargs=2, default=(0.1, 0.7))
    ap.add_argument("--yscale", choices=("symlog", "linear"), default="symlog")
    args = ap.parse_args(argv)
    out = args.out_dir or args.root
    os.makedirs(out, exist_ok=True)
    runs = load_runs(args.root)
    print(f"{len(runs)} tracers: {sorted(runs)}")
    for kind, title, _ in KINDS:
        fig = fig_kind(runs, kind, title, args.ell, args.ncut, args.yscale)
        if fig is not None:
            print("figure:", *save_fig(fig, os.path.join(out, f"components_{kind}_l{int(args.ell)}")))
            plt.close(fig)
    rows = fraction_table(runs, args.ell, tuple(args.xrange))
    with open(os.path.join(out, f"components_fraction_l{int(args.ell)}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    fig = fig_fraction(rows, runs, args.ell, tuple(args.xrange))
    print("figure:", *save_fig(fig, os.path.join(out, f"components_fraction_l{int(args.ell)}")))
    plt.close(fig)
    print(f"\n  {'tracer':>16} {'order':>5} {'x_e':>16} {'v_los':>16} {'density':>16}   (fraction of full kSZ^2 x tracer, x_HII {args.xrange[0]}-{args.xrange[1]})")
    for n in runs:
        for o in kc.ORDERINGS:
            cells = []
            for c in ("x", "v", "d"):
                r = [q for q in rows if q["tracer"] == n and q["ordering"] == o and q["component"] == c][0]
                cells.append(f"{r['R']:6.2f} [{r['R_lo16']:5.2f},{r['R_hi84']:5.2f}]")
            print(f"  {n:>16} {o:>5} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    sys.exit(main())
