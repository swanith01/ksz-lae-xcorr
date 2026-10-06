#!/usr/bin/env python3
"""
scripts/27_ksz_auto_power_aggregate.py
========================================
Combine the per-seed outputs of scripts/26 into the seed-combined kSZ
auto-power: median with the seed spread (sigma, 16-84 percentile, min-max),
plus a per-seed table, a CSV for the paper, and the overview figure.

D_total (the coherent LOS sum on the wrap-cycle lightcone) is THE kSZ
auto-power; D_diag / D_off are consistency diagnostics (D_off/D_total should
be small now).

Usage:
    python scripts/27_ksz_auto_power_aggregate.py
    python scripts/27_ksz_auto_power_aggregate.py --wrap-offset 100
    python scripts/27_ksz_auto_power_aggregate.py --legacy-pkl data/products/coherence_decomposition.pkl
"""

import argparse
import glob
import os
import pickle
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.correlation.ksz_auto_wrapcycle import summarise_at_ell
from ksz_lae_xcorr.correlation.seed_stats import aggregate_coherence_over_seeds
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig

FIELDS = ("D_total", "D_diag", "D_off")


def _load(in_dir, offset):
    res = {}
    for p in sorted(glob.glob(os.path.join(in_dir, f"seed*_wo{offset}.pkl"))):
        m = re.search(r"seed(\d+)_wo", os.path.basename(p))
        with open(p, "rb") as f:
            res[int(m.group(1))] = pickle.load(f)
    return res


def _aggregate(block_by_seed):
    seeds = sorted(block_by_seed)
    agg = {f: aggregate_coherence_over_seeds(block_by_seed, seeds, field=f) for f in FIELDS}
    for f in FIELDS:
        st = agg[f]["stack"]
        agg[f]["p16"], agg[f]["p84"] = np.nanpercentile(st, [16, 84], axis=0)
        agg[f]["min"], agg[f]["max"] = np.nanmin(st, axis=0), np.nanmax(st, axis=0)
    return agg


def _write_csv(path, agg):
    ell = agg["D_total"]["ell"]
    cols, names = [ell], ["ell"]
    for f in FIELDS:
        for k in ("median", "sigma", "p16", "p84", "min", "max"):
            cols.append(agg[f][k]); names.append(f"{f}_{k}")
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(names), comments="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--in-dir", default=None, help="default: <products_root>/ksz_auto_wrapcycle")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    ap.add_argument("--legacy-pkl", default=None,
                    help="optional old coherence_decomposition.pkl; overlays its patchy D_diag "
                         "(NOTE: legacy ne normalisation, fixed-angle lightcone)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    in_dir = args.in_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    os.makedirs(args.out_dir, exist_ok=True)
    res = _load(in_dir, args.wrap_offset)
    if len(res) < 2:
        sys.exit(f"Need >=2 seeds in {in_dir} (pattern seed*_wo{args.wrap_offset}.pkl), found {sorted(res)}")
    seeds = sorted(res)
    print(f"{len(seeds)} seeds: {seeds}  (wrap offset {args.wrap_offset})")
    metas = {(r["meta"]["mode"], r["meta"]["ne_convention"], r["meta"]["tau0_mode"]) for r in res.values()}
    if len(metas) != 1:
        sys.exit(f"Seeds were run with different settings {metas} -- refusing to combine.")
    mode, ne_conv, tau0_mode = metas.pop()
    print(f"settings: mode={mode} ne={ne_conv} tau0={tau0_mode}")

    print("\nPer-seed, ell~3000 [uK^2]:")
    print(f"{'seed':>4} {'window':>14} {'chi_eff':>8} {'D_total':>10} {'D_diag(1Mpc)':>12} {'D_off':>10} {'off/tot':>8} "
          f"{'D_diag(grp)':>11} {'off/tot(grp)':>12}")
    for s in seeds:
        r = res[s]["patchy"]; q = summarise_at_ell(r, 3000.0)
        print(f"{s:>4} {r['z_lo']:6.2f}-{r['z_hi']:<6.2f} {r['chi_eff']:8.0f} {q['D_total']:10.4g} "
              f"{q['D_diag']:12.4g} {q['D_off']:10.4g} {q['D_off_over_total']:8.1%} "
              + (f"{q['D_diag_grouped']:11.4g} {q['D_off_grouped_over_total']:12.1%}" if "D_diag_grouped" in q else
                 f"{'n/a':>11} {'n/a':>12}"))

    out = {"seeds": seeds, "wrap_offset": args.wrap_offset,
           "settings": {"mode": mode, "ne_convention": ne_conv, "tau0": tau0_mode}}
    for key in ("patchy", "full"):
        agg = _aggregate({s: res[s][key] for s in seeds})
        out[key] = {f: {k: v for k, v in agg[f].items() if k != "stack"} for f in FIELDS}
        out[key]["stack_D_total"] = agg["D_total"]["stack"]
        _write_csv(os.path.join(args.out_dir, f"ksz_auto_power_wrapcycle_{key}.csv"), agg)
        i = int(np.argmin(np.abs(agg["D_total"]["ell"] - 3000)))
        tot, dg, of = (agg[f]["median"][i] for f in FIELDS)
        print(f"\n[{key}] ell~{agg['D_total']['ell'][i]:.0f}, {len(seeds)} seeds, median [sigma] [16-84%]:")
        print(f"   D_total = {tot:.4g} [{agg['D_total']['sigma'][i]:.3g}] "
              f"[{agg['D_total']['p16'][i]:.3g}, {agg['D_total']['p84'][i]:.3g}] uK^2")
        print(f"   D_diag  = {dg:.4g} [{agg['D_diag']['sigma'][i]:.3g}]   D_off = {of:.4g}   "
              f"D_off/D_total = {of / tot:.1%}")
        out[key]["_agg"] = agg

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, key, ttl in ((axes[0, 0], "patchy", "patchy window"), (axes[0, 1], "full", "full z-range")):
        agg = out[key]["_agg"]; ell = agg["D_total"]["ell"]
        for s, row in zip(agg["D_total"]["seeds"], agg["D_total"]["stack"]):
            ax.plot(ell, row, color="0.75", lw=0.8, zorder=1)
        ax.fill_between(ell, agg["D_total"]["min"], agg["D_total"]["max"], color="C0", alpha=0.12, label="min-max")
        ax.fill_between(ell, agg["D_total"]["p16"], agg["D_total"]["p84"], color="C0", alpha=0.3, label="16-84%")
        ax.plot(ell, agg["D_total"]["median"], "o-", color="C0", lw=2, label=r"median $D_{\rm total}$")
        ax.plot(ell, agg["D_diag"]["median"], "s--", color="C3", lw=1.2, ms=3, label=r"median $D_{\rm diag}$")
        if key == "patchy" and args.legacy_pkl:
            with open(args.legacy_pkl, "rb") as f:
                leg = pickle.load(f)
            lp = {s: r["patchy"] for s, r in leg.items() if "patchy" in r}
            if len(lp) >= 2:
                la = aggregate_coherence_over_seeds(lp, list(lp), field="D_diag")
                ax.plot(la["ell"], la["median"], "^:", color="k", ms=3,
                        label="legacy fixed-angle $D_{\\rm diag}$ (old norm.)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$D_\ell^{\rm kSZ}$ [$\mu$K$^2$]")
        z_lo = np.median([res[s][key]["z_lo"] for s in seeds]); z_hi = np.median([res[s][key]["z_hi"] for s in seeds])
        ax.set_title(f"kSZ auto-power, {ttl} (z~{z_lo:.1f}-{z_hi:.1f}), {len(seeds)} seeds")
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    for key, c in (("patchy", "C0"), ("full", "C2")):
        agg = out[key]["_agg"]; ell = agg["D_total"]["ell"]
        ratio = agg["D_off"]["stack"] / agg["D_total"]["stack"]
        for row in ratio:
            ax.plot(ell, row, color=c, lw=0.6, alpha=0.4)
        ax.plot(ell, np.nanmedian(ratio, axis=0), color=c, lw=2, label=f"median ({key})")
    ax.axhline(0, color="gray", lw=0.5); ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$D_{\rm off}/D_{\rm total}$")
    ax.set_title("periodicity diagnostic (small => coherent sum ~ incoherent)"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for s in seeds:
        r = res[s]
        ax.plot(r["z_lc"], r["xHI_mean"], lw=1, label=f"seed {s}")
        ax.axvspan(r["patchy"]["z_lo"], r["patchy"]["z_hi"], color="C0", alpha=0.03)
    ax.set_xlabel("z"); ax.set_ylabel(r"$\bar{x}_{\rm HI}$"); ax.set_title("reionization histories & patchy windows")
    ax.legend(fontsize=6, ncol=2)

    fig.suptitle(f"kSZ auto-power from wrap-cycle lightcone | mode={mode}, n_e={ne_conv}, tau0={tau0_mode}")
    base = os.path.join(args.out_dir, f"ksz_auto_power_wrapcycle_wo{args.wrap_offset}")
    print("\nfigure:", *save_fig(fig, base)); plt.close(fig)

    pkl = os.path.join(cfg.paths.products_root, f"ksz_auto_power_wrapcycle_agg_wo{args.wrap_offset}.pkl")
    for key in ("patchy", "full"):
        out[key].pop("_agg", None)
    os.makedirs(cfg.paths.products_root, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(out, f)
    print("aggregate:", pkl)
    print("CSV:", os.path.join(args.out_dir, "ksz_auto_power_wrapcycle_{patchy,full}.csv"))


if __name__ == "__main__":
    sys.exit(main())
