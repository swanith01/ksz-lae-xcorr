#!/usr/bin/env python3
"""
scripts/45_coeval_vs_lightcone.py
===================================
Aggregates scripts/44 (one pickle per seed) into  D_ell^cross(x_HII)  from coeval boxes versus the stitched lightcone, for the
matter x Waters+16 reference and each tracer cut, and overlays them.  Light; run it wherever the pickles are (swarm or desktop).

    python scripts/45_coeval_vs_lightcone.py --coeval-dir data/products_tracers/coeval --out-dir data/plots/2026-10-19_coeval

Outputs (--out-dir):
    coeval_vs_lightcone.png/.pdf      rows = ell (500, 1000, 3000), columns = matter | one cut per tracer kind
                                      line + band = stitched lightcone (seed mean +- s.e.m.), orange dots = coeval slabs,
                                      green squares = coeval full-box Limber estimate (scripts/44 D_box; --estimators to choose),
                                      dashed black = La Plante+22 Fig. 7 fiducial curve (matter x their bias; --no-paper to drop)
    coeval_vs_lightcone_summary.csv   per tracer and ell: pooled ratio coeval / lightcone over --xrange (seed bootstrap, paired
                                      by seed), and the RMS over windows of the paired difference in units of its s.e.m.
    coeval_<tracer>.csv               z0, dz, x_hii, mean/sem of both estimators at every ell
Reading the numbers: ratio = 1 and an RMS z-score near 1 mean the two estimators agree within the seed noise.
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.io.la_plante_reference import load_fig7_vs_xhii
from ksz_lae_xcorr.tracers.suite import spec_from_name
from ksz_lae_xcorr.utils.figio import save_fig

MATTER = "matter_Waters"
LC_COLOR, CO_COLOR, BOX_COLOR = "#0072B2", "#D55E00", "#009E73"
EST = {"slab": ("D_coeval", "coeval slabs", CO_COLOR, "o"), "box": ("D_box", "coeval full box (Limber)", BOX_COLOR, "s")}


def load_seeds(coeval_dir, offset=0):
    out = {}
    for p in sorted(glob.glob(os.path.join(coeval_dir, f"seed*_wo{offset}_coeval.pkl"))):
        with open(p, "rb") as f:
            d = pickle.load(f)
        out[d["seed"]] = d
    if not out:
        raise SystemExit(f"no seed*_wo{offset}_coeval.pkl in {coeval_dir}")
    return out


def stack(seeds_data, name, key="D_coeval"):
    """(seeds, Dc, Dl, x_hii, z0, targets): Dc/Dl (n_seed, n_window, n_ell); windows matched by (z0, dz).
    key: which coeval estimate ('D_coeval' = slabs, 'D_box' = full-box Limber); NaN for seeds whose pickle lacks it."""
    seeds = sorted(seeds_data)
    keys = sorted({(w["dz"], w["z0"]) for s in seeds for w in seeds_data[s]["windows"]}, key=lambda k: k[1])
    shape = (len(seeds), len(keys), len(seeds_data[seeds[0]]["ell_targets"]))
    Dc, Dl = np.full(shape, np.nan), np.full(shape, np.nan)
    xh = np.full((len(seeds), len(keys)), np.nan)
    for si, s in enumerate(seeds):
        d = seeds_data[s]
        if name not in d["names"]:
            continue
        has = key in d and name in d[key]
        idx = {(w["dz"], w["z0"]): i for i, w in enumerate(d["windows"])}
        for ki, k in enumerate(keys):
            if k in idx:
                Dl[si, ki] = d["D_lightcone"][name][idx[k]]
                if has:
                    Dc[si, ki] = d[key][name][idx[k]]
                xh[si, ki] = d["windows"][idx[k]]["x_hii_lightcone"]
    return seeds, Dc, Dl, np.nanmean(xh, axis=0), np.array([k[1] for k in keys]), np.array(seeds_data[seeds[0]]["ell_targets"])


def mean_sem(D):
    n = np.sum(np.isfinite(D), axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(D, axis=0), np.nanstd(D, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))


def agreement(Dc, Dl, x, xrange, n_boot=1000, seed=0):
    """Per ell: pooled ratio coeval/lightcone (paired seed bootstrap) and the RMS paired z-score over windows."""
    sel = (x >= xrange[0]) & (x <= xrange[1])
    out = []
    rng = np.random.default_rng(seed)
    for ti in range(Dc.shape[2]):
        c, l = Dc[:, sel, ti], Dl[:, sel, ti]
        ok = np.isfinite(c).all(axis=0) & np.isfinite(l).all(axis=0)
        if not ok.any() or c.shape[0] < 2:
            out.append((np.nan, np.nan, np.nan, np.nan)); continue
        c, l = c[:, ok], l[:, ok]
        ns = c.shape[0]
        rs = []
        for _ in range(n_boot):
            ii = rng.integers(0, ns, ns)
            den = l[ii].mean(axis=0).sum()
            rs.append(c[ii].mean(axis=0).sum() / den if den != 0 else np.nan)
        diff = c - l
        z = diff.mean(axis=0) / (diff.std(axis=0, ddof=1) / np.sqrt(ns) + 1e-300)
        out.append((float(np.nanmedian(rs)), float(np.nanpercentile(rs, 16)), float(np.nanpercentile(rs, 84)),
                    float(np.sqrt(np.mean(z ** 2)))))
    return out


def pick_columns(names, cuts):
    cols = [MATTER] if MATTER in names else []
    if cuts:
        return cols + [c for c in cuts if c in names]
    for kind in ("halo", "lae", "lbg"):
        ns = [n for n in names if n != MATTER and spec_from_name(n).kind == kind]
        if ns:
            cols.append(sorted(ns)[0] if len(ns) == 1 else ns[len(ns) // 2])
    return cols


def _symlog(ax, lin):
    ax.set_yscale("symlog", linthresh=lin, linscale=2.0)
    ax.axhspan(-lin, lin, color="0.5", alpha=0.07, lw=0, zorder=0)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    lo, hi = ax.get_ylim()
    ticks = [0.0, lin, -lin] + [s * lin * 10 ** j for j in range(1, 8) for s in (1, -1) if lo <= s * lin * 10 ** j <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(sorted(ticks)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2g}"))


def make_figure(data, cols, targets, yscale="symlog", paper=None):
    """paper: {ell: (x_HII, D_ell)} (La Plante+22 Fig. 7, fiducial); drawn dashed black in every panel when given."""
    paper = paper or {}
    fig, axes = plt.subplots(len(targets), len(cols), figsize=(4.6 * len(cols), 3.6 * len(targets)), sharex=True,
                             sharey="row", squeeze=False, constrained_layout=True)
    for ti, ell in enumerate(targets):
        ref = data[cols[0]]
        pk = max(float(np.nanmax(np.abs(mean_sem(ref["Dl"])[0][:, ti]))), float(np.max(paper[int(ell)][1])) if int(ell) in paper else 0.0)
        lin = float(f"{0.5 * pk:.1g}") if yscale == "symlog" else None
        for ci, name in enumerate(cols):
            d = data[name]; ax = axes[ti, ci]
            if int(ell) in paper:
                ax.plot(*paper[int(ell)], color="k", ls="--", lw=2.0, zorder=1, label="La Plante+22 Fig. 7 (fiducial)")
            ml, sl = mean_sem(d["Dl"])
            ax.fill_between(d["x"], ml[:, ti] - sl[:, ti], ml[:, ti] + sl[:, ti], color=LC_COLOR, alpha=0.25, lw=0)
            ax.plot(d["x"], ml[:, ti], color=LC_COLOR, lw=2, label="stitched lightcone")
            for j, (est, D) in enumerate(d["coeval"].items()):
                mc, sc = mean_sem(D)
                _, lab, col, mk = EST[est]
                dx = 0.006 * (j - 0.5 * (len(d["coeval"]) - 1))
                ax.errorbar(d["x"] + dx, mc[:, ti], yerr=sc[:, ti], color=col, ls="none", marker=mk, ms=4, elinewidth=1, capsize=0,
                            label=lab)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlim(1.0, 0.0)
            if ti == 0:
                ax.set_title("matter $\\times$ Waters+16 bias" if name == MATTER else spec_from_name(name).label, fontsize=10)
            if ci == 0:
                ax.set_ylabel(rf"$\ell={ell:g}$:  $D_\ell^{{\rm cross}}\ [\mu{{\rm K}}^2]$")
            if ti == len(targets) - 1:
                ax.set_xlabel(r"$x_{\rm HII}$")
            if yscale == "symlog":
                _symlog(ax, lin)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeval-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--cuts", default=None, help="comma-separated tracer names to draw besides matter (default: one per kind)")
    ap.add_argument("--xrange", type=float, nargs=2, default=(0.1, 0.7))
    ap.add_argument("--yscale", choices=("symlog", "linear"), default="symlog")
    ap.add_argument("--no-paper", action="store_true", help="do not overplot the La Plante+22 Fig. 7 (fiducial) curve")
    ap.add_argument("--estimators", default=None, help="comma-separated: slab,box (default: those present in the pickles)")
    args = ap.parse_args(argv)
    out = args.out_dir or args.coeval_dir
    os.makedirs(out, exist_ok=True)
    sd = load_seeds(args.coeval_dir, args.wrap_offset)
    names = sd[sorted(sd)[0]]["names"]
    present = [e for e in EST if any(EST[e][0] in d for d in sd.values())]
    ests = [e for e in (args.estimators.split(",") if args.estimators else present) if e in present]
    print(f"seeds {sorted(sd)}; tracers {names}; coeval estimators {ests}")
    data, rows = {}, []
    for n in names:
        data[n] = {"coeval": {}}
        cols = {}
        for est in ests:
            seeds, Dc, Dl, x, z0, targets = stack(sd, n, EST[est][0])
            data[n]["coeval"][est] = Dc
            data[n].update({"Dl": Dl, "x": x, "z0": z0})
            if not cols:
                cols = {"z0": z0, "x_hii": x}
            mc, sc = mean_sem(Dc); ml, sl = mean_sem(Dl)
            for ti, t in enumerate(targets):
                for lab, m, s_ in ((est, mc, sc), ("lightcone", ml, sl)):
                    cols[f"mean_{lab}_l{int(t)}"], cols[f"sem_{lab}_l{int(t)}"] = m[:, ti], s_[:, ti]
            for t, (R, lo, hi, zr) in zip(targets, agreement(Dc, Dl, x, tuple(args.xrange))):
                rows.append({"tracer": n, "estimator": est, "ell": t, "ratio_coeval_over_lightcone": R, "lo16": lo, "hi84": hi,
                             "rms_paired_z": zr, "n_seeds": int(np.isfinite(Dc[:, :, 0]).any(axis=1).sum()),
                             "x_hii_lo": args.xrange[0], "x_hii_hi": args.xrange[1]})
        data[n]["Dc"] = data[n]["coeval"][ests[0]]
        with open(os.path.join(out, f"coeval_{n}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(list(cols)); w.writerows(zip(*[np.asarray(c) for c in cols.values()]))
    with open(os.path.join(out, "coeval_vs_lightcone_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cols = pick_columns(names, args.cuts.split(",") if args.cuts else None)
    paper = {} if args.no_paper else load_fig7_vs_xhii()
    fig = make_figure(data, cols, targets, args.yscale, paper)
    print("figure:", *save_fig(fig, os.path.join(out, "coeval_vs_lightcone")))
    plt.close(fig)
    if paper:
        print(f"\n  peak D_ell [uK^2] over x_HII (seed-mean curves) vs the paper's Fig. 7 peak")
        print(f"  {'tracer':>16} {'ell':>5} {'lightcone':>10} " + " ".join(f"{e:>9}" for e in ests) + f" {'paper':>8}")
        for n in names:
            for ti, t in enumerate(targets):
                pl = np.nanmax(mean_sem(data[n]["Dl"])[0][:, ti])
                pcs = [np.nanmax(mean_sem(data[n]["coeval"][e])[0][:, ti]) for e in ests]
                pp = float(np.max(paper[int(t)][1])) if int(t) in paper else np.nan
                print(f"  {n:>16} {t:5.0f} {pl:10.4f} " + " ".join(f"{v:9.4f}" for v in pcs) + f" {pp:8.4f}")
    print(f"\n  {'tracer':>16} {'est':>5} {'ell':>5} {'coeval/lightcone':>18} {'RMS z':>7}   (x_HII {args.xrange[0]}-{args.xrange[1]}; paired by seed)")
    for r in rows:
        print(f"  {r['tracer']:>16} {r['estimator']:>5} {r['ell']:5.0f} {r['ratio_coeval_over_lightcone']:7.2f} [{r['lo16']:5.2f},{r['hi84']:5.2f}] "
              f"{r['rms_paired_z']:7.2f}")


if __name__ == "__main__":
    sys.exit(main())
