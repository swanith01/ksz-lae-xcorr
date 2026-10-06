#!/usr/bin/env python3
"""
scripts/28_lp_cross_vs_z0.py
==============================
MAP-BASED La Plante+2022 reproduction: D_l^cross(z0) = l(l+1)/2pi T_CMB^2
C_l[(filtered kSZ)^2 x delta_g] at l = 500, 1000, 3000, from the wrap-cycle
lightcones' projected maps (correlation/lp_maps.py has the recipe + caveats).
This is the "D_stitched" arm; no bispectrum / Limber shortcut.

Inputs : <products_root>/ksz_auto_wrapcycle/seed*_wo{K}_lpmaps.pkl
         (written by scripts/26 --save-maps, i.e. submit with SAVE_MAPS=1)
Outputs: CSVs + figure under --out-dir, a pkl with the arrays, and a console
         table of the LP-band comparison.

C_TT: CAMB (as the rest of the SNR code) or, if CAMB is not installed in the
active env, --cltt-file CSV with columns `ell,Cl_uK2` (RAW C_l in uK^2, not D_l).

Usage:
    python scripts/28_lp_cross_vs_z0.py
    python scripts/28_lp_cross_vs_z0.py --wrap-offset 100 --experiment CMB-S4
    python scripts/28_lp_cross_vs_z0.py --cltt-file data/reference/cltt_raw.csv
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

from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.io.la_plante_reference import load_dell_vs_z0_bands
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig


def _load_products(in_dir, offset):
    out = {}
    for p in sorted(glob.glob(os.path.join(in_dir, f"seed*_wo{offset}_lpmaps.pkl"))):
        m = re.search(r"seed(\d+)_wo", os.path.basename(p))
        with open(p, "rb") as f:
            out[int(m.group(1))] = pickle.load(f)
    return out


def _cl_tt_provider(cfg, cltt_file):
    if cltt_file:
        tab = np.loadtxt(cltt_file, delimiter=",", skiprows=1)
        ell_t, cl_t = tab[:, 0], tab[:, 1]
        return lambda ell: np.interp(ell, ell_t, cl_t)
    from ksz_lae_xcorr.snr.cmb_filter import camb_cl_tt
    return lambda ell: camb_cl_tt(cfg, ell)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--in-dir", default=None, help="default: <products_root>/ksz_auto_wrapcycle")
    ap.add_argument("--out-dir", default="paper/figure_scripts/output")
    ap.add_argument("--experiment", default="SO", help="key of cfg.snr.experiments (LP Figs 4/5/7 use SO)")
    ap.add_argument("--cltt-file", default=None, help="CSV ell,Cl_uK2 (raw C_l) instead of CAMB")
    args = ap.parse_args()

    cfg = load_config(args.config)
    in_dir = args.in_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    prods = _load_products(in_dir, args.wrap_offset)
    if len(prods) < 2:
        sys.exit(f"Need >=2 seeds with _lpmaps.pkl in {in_dir} (wo{args.wrap_offset}); found {sorted(prods)}. "
                 f"Run scripts/26 with --save-maps (qsub env SAVE_MAPS=1).")
    os.makedirs(args.out_dir, exist_ok=True)

    p0 = prods[sorted(prods)[0]]
    print(f"{len(prods)} seeds {sorted(prods)}; kSZ map z>={p0['kSZ_z_min']}, "
          f"ne_scale={p0['ne_scale']}, tau0={p0['tau0']}")
    res = lp_maps.run_lp_analysis(cfg, prods, _cl_tt_provider(cfg, args.cltt_file), args.experiment)
    kg, filt = res["kg"], res["filter"]

    # ---- diagnostics: is the kSZ map sane, is the filter sane ---------------
    rms = [float(np.std(p["kSZ_map"])) for p in prods.values()]
    print(f"\nchi_ref (kSZ-map chi_eff, seed median) = {res['chi_ref']:.0f} Mpc; "
          f"ell_min = k_f*chi = {2 * np.pi / kg.box_len * res['chi_ref']:.0f}")
    print(f"kSZ map rms(dT/T) = {np.median(rms):.3g}  (per seed: {', '.join(f'{r:.2g}' for r in rms)})")
    ell_s, C_s = lp_maps.kSZ_map_Cl(prods[sorted(prods)[0]]["kSZ_map"], kg, res["chi_ref"])
    i3 = int(np.nanargmin(np.abs(ell_s - 3000)))
    D3 = ell_s[i3] * (ell_s[i3] + 1) * C_s[i3] / (2 * np.pi)
    print(f"seed {sorted(prods)[0]} map auto D_l at l~{ell_s[i3]:.0f}: {D3:.3g} uK^2 "
          f"(LP SO-era kSZ_reion ~ 1 uK^2 at l=3000)")
    for e in (500.0, 1000.0, 3000.0):
        j = int(np.argmin(np.abs(filt["ell_grid"] - e)))
        print(f"filter {args.experiment} @ l={e:.0f}: F={filt['Fl'][j]:.3g} b={filt['bl'][j]:.3g} f={filt['fl'][j]:.3g}")

    # ---- table --------------------------------------------------------------
    T = res["ell_targets"]
    hdr = "z0   dz   x_HII | " + " | ".join(f"l={int(t)}: mean +- sem (median)  n_modes" for t in T)
    print("\n" + hdr)
    for wi in range(len(res["z0"])):
        cells = [f"{res['mean'][wi, ti]:9.3e} +- {res['sigma_mean'][wi, ti]:8.2e} ({res['median'][wi, ti]:9.3e}) "
                 f"{res['n_modes'][wi, ti]:5.0f}" for ti in range(len(T))]
        print(f"{res['z0'][wi]:4.1f} {res['dz'][wi]:3.1f}  {res['x_hii'][wi]:5.3f} | " + " | ".join(cells))

    # repo-relative first: cfg.paths.project_root is the DATA dir on the cluster, which does
    # not contain data/reference/ (that lives in the repo checkout)
    try:
        bands = load_dell_vs_z0_bands(None)
    except FileNotFoundError:
        try:
            bands = load_dell_vs_z0_bands(cfg)
        except FileNotFoundError as e:
            print(f"\n(!) digitized La Plante bands not found ({e}); skipping the comparison")
            bands = {}
    rows = lp_maps.compare_to_la_plante(res, bands)
    print("\nvs digitized La Plante+22 Fig 5 bands (first-look; bands = spread of THEIR histories):")
    for ell in T:
        sub = [r for r in rows if r["ell"] == ell and np.isfinite(r["ours_mean"])]
        if not sub:
            print(f"  l={int(ell)}: no windows with a finite D_l (l outside the box's covered range?)")
            continue
        ratios = np.array([r["ratio_to_mid"] for r in sub])
        print(f"  l={int(ell)}: {sum(r['inside'] for r in sub)}/{len(sub)} windows inside the band; "
              f"ours/band-mid median {np.nanmedian(ratios):.2f} (range {np.nanmin(ratios):.2f}-{np.nanmax(ratios):.2f})")

    # ---- outputs ------------------------------------------------------------
    tag = f"{args.experiment}_wo{args.wrap_offset}"
    csv_path = os.path.join(args.out_dir, f"lp_cross_vs_z0_{tag}.csv")
    cols, names = [res["z0"], res["dz"], res["x_hii"]], ["z0", "dz", "x_hii"]
    for ti, t in enumerate(T):
        for k in ("mean", "median", "sigma_mean", "p16", "p84", "n_modes"):
            cols.append(res[k][:, ti]); names.append(f"{k}_l{int(t)}")
    np.savetxt(csv_path, np.column_stack(cols), delimiter=",", header=",".join(names), comments="")
    pkl_path = os.path.join(cfg.paths.products_root, f"lp_cross_vs_z0_{tag}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({k: v for k, v in res.items() if k != "kg"}, f)

    fig, axes = plt.subplots(1, len(T), figsize=(5 * len(T), 4), sharex=True)
    for ti, (ax, t) in enumerate(zip(np.atleast_1d(axes), T)):
        b = bands.get(int(t))
        if b is not None:
            zz = np.linspace(max(b["z0_lo"].min(), b["z0_hi"].min()), min(b["z0_lo"].max(), b["z0_hi"].max()), 100)
            ax.fill_between(zz, np.interp(zz, b["z0_lo"], b["lo"]), np.interp(zz, b["z0_hi"], b["hi"]),
                            color="C1", alpha=0.3, label="La Plante+22 (digitized)")
        ax.plot(res["z0"], res["D"].transpose(1, 0, 2)[:, :, ti], color="0.8", lw=0.6)
        ax.errorbar(res["z0"], res["mean"][:, ti], yerr=res["sigma_mean"][:, ti], color="C0", marker="o", ms=3,
                    label=f"ours: mean of {len(res['seeds'])} seeds (grey: seeds)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(rf"$\ell={int(t)}$  ({args.experiment})")
        ax.set_xlabel(r"$z_0$  ($\Delta z=1$)")
        ax.set_ylabel(r"$D_\ell^{\rm cross}\ [\mu{\rm K}^2]$")
        if ti == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    print("\nfigure:", *save_fig(fig, os.path.join(args.out_dir, f"lp_cross_vs_z0_{tag}")))
    plt.close(fig)
    print("CSV:", csv_path, "\npkl:", pkl_path)


if __name__ == "__main__":
    sys.exit(main())
