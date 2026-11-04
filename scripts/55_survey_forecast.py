#!/usr/bin/env python3
"""
scripts/55_survey_forecast.py
===============================
The CMB-experiment x LAE-survey forecast in one figure: S/N for every LAE survey (rows) and CMB experiment (columns), from
  (a) the toy signal calibrated to La Plante Table 2 (scripts/52 recipe),
  (b) the simulation, La Plante's galaxy model (matter x Waters+16 bias) rescaled to the survey's bias,
  (c) the simulation, the simulated LAEs' own signal per unit bias (default lae_L42.5) rescaled to the survey's bias.
All three use the same survey numbers (snr/feasibility.SURVEYS):
  * shot noise from the LAEs actually in the survey catalogue where known (n_obs_deg2, e.g. SILVERRUSH: Ouchi+18), else from the
    observed Lya LF above L_min (--counts lf forces the LF everywhere);
  * bias of the population above L_min by Sheth-Tormen abundance matching (completeness thins the sample at random, so it does not
    change the bias);
  * f_sky = area x footprint overlap (feasibility.OVERLAP: HSC / PFS fields are not in SPT-3G Main);
  * l_min = max(100, 2 pi / sqrt(Omega)) of the field.
The simulation starts at z = 6, so only shells with z >= 6 enter (b) and (c), and (a) is shown for the same shells (all shells in
the CSV).  S/N of a survey = quadrature sum over its shells.  Signal: fiducial 21cmFAST history.  Debiased for the seed scatter.

    python scripts/55_survey_forecast.py --spectra-dir data/products_tracers/sim_spectra --out-dir data/plots/2026-11-03_survey_forecast

Outputs: survey_forecast.csv, survey_forecast.png/pdf
"""
import argparse
import csv
import dataclasses
import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.snr import sim_snr as ss
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.figio import save_fig

SQ_DEG = 41252.96
SHORT = {"HSC SILVERRUSH (NB816+NB921)": "HSC SILVERRUSH\n(current)", "Subaru PFS (SSP, repo value)": "Subaru PFS\n(L_min assumed)",
         "LAGER (DECam NB964)": "LAGER z~6.9\n(current)",
         "Roman grism deep z=7.5-10, no evolution from z=7": "Roman deep z~7.5-10\n(no LF evolution)",
         "Roman grism deep z=7.5-10, LF as observed at z=7.3": "Roman deep z~7.5-10\n(LF as observed)"}


def experiment_variants():
    out = []
    for name, e in fe.EXPERIMENTS.items():
        out.append((name + (" (post-ILC)" if e.ilc_key else " (white noise)"), e))
        if e.ilc_key:
            out.append((name + " (white noise)", dataclasses.replace(e, ilc_key=None)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--spectra-dir", default="data/products_tracers/sim_spectra")
    ap.add_argument("--out-dir", default="data/plots/survey_forecast")
    ap.add_argument("--lae-tracer", default="lae_L42.5")
    ap.add_argument("--counts", default="observed", choices=("observed", "lf"))
    ap.add_argument("--bias-csv", default="data/plots/2026-10-16_tracer_suite/bias/tracer_bias_summary.csv")
    ap.add_argument("--ell-max", type=float, default=10000.0)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    cosmo = get_cosmology(cfg)
    chi = lambda z: float(cosmo.comoving_distance(z).to_value("Mpc"))

    spectra = {}
    for nm in ("matter_waters", args.lae_tracer):
        p = os.path.join(args.spectra_dir, f"{nm}.npz")
        if not os.path.exists(p):
            sys.exit(f"missing {p} (scripts/53 output)")
        z = np.load(p, allow_pickle=False)
        spectra[nm] = {k: z[k] for k in z.files}
    btab = ss.load_bias_table(args.bias_csv)

    comp = fe.lp_components(cfg)
    sig = fe.SignalModel(cfg)
    scale, _ = fe.calibrate_to_lp_table2(sig, comp)
    ref = fe.experiment_filter(fe.EXPERIMENTS["SO"], comp)
    variants = experiment_variants()
    filters = {lab: fe.experiment_filter(e, comp) for lab, e in variants}
    hb = fe.HaloBias(cfg)

    rows = []
    surveys = list(fe.SURVEYS.values())
    labs = [lab for lab, _ in variants]
    grids = {k: np.full((len(surveys), len(labs)), np.nan) for k in ("toy", "sim_LP", "sim_LAE")}
    for si, sv in enumerate(surveys):
        for li, (lab, e) in enumerate(variants):
            ov = fe.overlap(sv.name, e.name)
            acc = {"toy": 0.0, "toy_all": 0.0, "sim_LP": 0.0, "sim_LAE": 0.0}
            used, notes = [], []
            for sh in sv.shells:
                area = sh.area_deg2 or sv.area_deg2
                f_sky = ov * area / SQ_DEG
                n3 = fe.lae_number_density(sh.L_min, sh.lf)
                c, lo, hi = chi(sh.z), chi(sh.z - sh.dz / 2), chi(sh.z + sh.dz / 2)
                if args.counts == "observed" and sh.n_obs_deg2:
                    n2d = sh.n_obs_deg2 / (np.pi / 180) ** 2
                    notes.append(f"z={sh.z}: N/deg2 {sh.n_obs_deg2:.0f} observed")
                else:
                    n2d = n3 * c ** 2 * (hi - lo)
                    notes.append(f"z={sh.z}: N/deg2 {n2d * (np.pi / 180) ** 2:.0f} from LF")
                b_sv, _ = hb.bias_for_density(n3, sh.z)
                if f_sky <= 0:
                    continue
                bias_fn = lambda zz, b=b_sv: np.full_like(np.asarray(zz, float), b)
                t2 = fe.snr_window(sig, filters[lab], ref, sh.z, sh.dz, bias_fn, f_sky, n2d) ** 2
                acc["toy_all"] += t2
                wi = ss.shell_window(spectra["matter_waters"], sh.z)
                if wi is None:
                    continue
                used.append(sh.z)
                acc["toy"] += t2
                lmin = fe.field_ell_min(area)
                for key, nm in (("sim_LP", "matter_waters"), ("sim_LAE", args.lae_tracer)):
                    sp = spectra[nm]
                    wj = ss.shell_window(sp, sh.z)
                    b_sim = ss.sim_bias(nm, sh.z, btab)
                    r = b_sv / b_sim if np.isfinite(b_sim) and b_sim > 0 else np.nan
                    ei = [str(x) for x in sp["experiments"]].index(lab)
                    dchi_w = chi(sp["z_hi"][wj]) - chi(sp["z_lo"][wj])
                    acc[key] += ss.survey_shell_snr2(sp, ei, wj, r, f_sky, n2d, dchi_w, hi - lo, lmin, args.ell_max) if np.isfinite(r) else np.nan
            res = {k: float(np.sqrt(v)) for k, v in acc.items()}
            for k in grids:
                grids[k][si, li] = res[k] if ov > 0 else 0.0
            rows.append({"survey": sv.name, "cmb": lab, "overlap": ov, "shells_in_sim": " ".join(f"{z:g}" for z in used),
                         "SN_toy_same_shells": round(res["toy"], 4), "SN_toy_all_shells": round(res["toy_all"], 4),
                         "SN_sim_LaPlante_model": round(res["sim_LP"], 4), f"SN_sim_{args.lae_tracer}": round(res["sim_LAE"], 4),
                         "counts": "; ".join(notes)})
    with open(os.path.join(args.out_dir, "survey_forecast.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- figure -----------------------------------------------------------------------------------------------------------
    vals = np.concatenate([g[g > 0] for g in grids.values()])
    norm = LogNorm(vmin=max(1e-3, float(np.nanmin(vals))), vmax=max(1.0, float(np.nanmax(vals))))
    titles = {"toy": "(a) toy, La Plante galaxy model", "sim_LP": "(b) simulation, La Plante galaxy model",
              "sim_LAE": f"(c) simulation, simulated LAE signal ({args.lae_tracer})"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True, sharey=True)
    for ax, k in zip(axes, grids):
        g = grids[k]
        im = ax.imshow(np.where(g > 0, g, np.nan), cmap="viridis", norm=norm, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g[i, j]
                txt = "no\noverlap" if v == 0 else (f"{v:.2g}" if np.isfinite(v) else "-")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="0.35" if v == 0 else ("k" if np.isfinite(v) and v > 0.3 * norm.vmax else "w"))
        ax.set_xticks(range(len(labs)), [l.replace(" (", "\n(").replace("SPT-3G Main", "SPT-3G") for l in labs], fontsize=6.5, rotation=0)
        ax.set_yticks(range(len(surveys)), [SHORT.get(s.name, s.name) for s in surveys], fontsize=8)
        ax.set_title(titles[k], fontsize=9)
        ax.set_facecolor("0.93")
    fig.colorbar(im, ax=axes, label="S/N", shrink=0.9)
    fig.suptitle("kSZ$^2$ x LAE forecast: LAE survey x CMB experiment (fiducial history; z $\\geq$ 6 shells; "
                 f"{'observed catalogue' if args.counts == 'observed' else 'LF'} counts; ACT/SPT-3G white noise only)", fontsize=10)
    print("figure:", *save_fig(fig, os.path.join(args.out_dir, "survey_forecast")))
    plt.close(fig)

    print(f"\n{'survey':>34} {'cmb':>26} | toy   sim(LP model)   sim({args.lae_tracer})")
    for r in rows:
        print(f"{r['survey'][:34]:>34} {r['cmb'][:26]:>26} | {r['SN_toy_same_shells']:<6g} {r['SN_sim_LaPlante_model']:<14g} {r[f'SN_sim_{args.lae_tracer}']:g}"
              + ("   (no overlap)" if r["overlap"] == 0 else ""))


if __name__ == "__main__":
    sys.exit(main())
