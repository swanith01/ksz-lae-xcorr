#!/usr/bin/env python3
"""
scripts/52_feasibility.py
===========================
Observational feasibility of the kSZ^2 x LAE cross-spectrum (see snr/feasibility.py for the model and its limits).

Outputs under --out-dir:
  feasibility_validation.csv     our S/N vs La Plante+22 Table 2 (Roman HLS, no shot noise)
  feasibility_surveys.csv        per LAE survey shell: n, N/deg2, abundance-matched halo mass and bias, signal, shot noise vs clustering
  feasibility_snr.csv            S/N for every (survey, CMB experiment, history, bias) combination
  feasibility_map.png/pdf        S/N contours in the (survey area, LAEs per deg2) plane for CMB-HD / S4 / SO
  halo_luminosity_mapping.csv    simulation: LAE luminosity cut vs halo mass cut at matched number density and matched bias
  halo_luminosity_mapping.png/pdf
Runs in seconds on a laptop (no simulation maps needed; reads only data/reference and the committed tracer-bias CSVs).

Usage:
    python scripts/52_feasibility.py
    python scripts/52_feasibility.py --out-dir data/plots/2026-11-02_feasibility
"""

import argparse
import csv
import dataclasses
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig

OKABE = {"vermillion": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9", "purple": "#CC79A7", "ink": "#222222"}
SQ_DEG = 41252.96
BIAS_LOW_FACTOR = 0.45      # sim LAE measured bias / n-matched halo bias at z = 6 (L >= 10^42.5): 2.33 / 5.37


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def experiment_variants():
    """(label, Experiment) for every experiment; the ILC-curve ones also appear with white noise only, so ACT / SPT (white only)
    can be compared like with like."""
    out = []
    for name, e in fe.EXPERIMENTS.items():
        out.append((name + (" (post-ILC)" if e.ilc_key else " (white noise)"), e))
        if e.ilc_key:
            out.append((name + " (white noise)", dataclasses.replace(e, ilc_key=None)))
    return out


# ------------------------------------------------------------------------------------------------------------------------
def validation(sig, comp):
    rows = []
    for white, ref in ((False, fe.LP_TABLE2_ILC_NO_SHOT), (True, fe.LP_TABLE2_WHITE_NO_SHOT)):
        got = fe.lp_benchmark(sig, comp, white=white)
        for k in ref:
            rows.append({"noise": "white (instrument)" if white else "post-ILC", "experiment": k, "La_Plante_Table2": ref[k],
                         "ours": round(got[k], 2), "ratio": round(got[k] / ref[k], 2),
                         "used_for_calibration": "no" if white else "yes (one common factor %.2f)" % sig.scale})
    return rows


def survey_rows(sig_fid, sig_late, hb, cfg):
    """One row per survey shell with everything that sets the noise, plus the signal at ell = 1000."""
    rows = []
    for sv in fe.SURVEYS.values():
        for sh in sv.shells:
            n = fe.lae_number_density(sh.L_min, sh.lf)
            chi, dchi, vol_per_sr = fe.shell_geometry(sig_fid.toy, sh.z, sh.dz)
            n_per_sr = n * vol_per_sr
            n_per_deg2 = n_per_sr * (np.pi / 180) ** 2
            b_hi, M = hb.bias_for_density(n, sh.z)
            b_lo = BIAS_LOW_FACTOR * b_hi
            Cmm = fe.limber_cmm(sig_fid.toy, sh.z, sh.dz, 1000.0)
            rows.append({"survey": sv.name, "z": sh.z, "dz": sh.dz, "L_min_erg_s": "%.2e" % sh.L_min, "LF": "%s z=%.1f" % sh.lf,
                         "area_deg2": sh.area_deg2 or sv.area_deg2, "n_Mpc3": "%.2e" % n, "N_per_deg2": round(n_per_deg2, 1),
                         "N_in_survey": round(n_per_deg2 * (sh.area_deg2 or sv.area_deg2)),
                         "M_min_ST_Msun": "%.2e" % M, "b_ST_matched": round(b_hi, 2), "b_low": round(b_lo, 2),
                         "x_HII_fiducial": round(float(sig_fid.xb(sh.z)), 2), "x_HII_late": round(float(sig_late.xb(sh.z)), 2),
                         "shot_noise_C_SN": "%.2e" % (1.0 / n_per_sr),
                         "clustering_b2Cmm_ell1000_bST": "%.2e" % (b_hi ** 2 * Cmm),
                         "shot_over_clustering": round((1.0 / n_per_sr) / (b_hi ** 2 * Cmm), 1),
                         "D_ell1000_fiducial_uK2": "%.2e" % sig_fid.D(sh.z, sh.dz, 1000.0, lambda z, b=b_hi: np.full_like(np.asarray(z, float), b)),
                         "D_ell1000_late_uK2": "%.2e" % sig_late.D(sh.z, sh.dz, 1000.0, lambda z, b=b_hi: np.full_like(np.asarray(z, float), b))})
    return rows


def snr_rows(sig_fid, sig_late, comp, hb, variants, overlap=1.0):
    ref = fe.experiment_filter(fe.EXPERIMENTS["SO"], comp)
    filters = {lab: fe.experiment_filter(e, comp) for lab, e in variants}
    rows = []
    for sv in fe.SURVEYS.values():
        for hist, sig in (("fiducial", sig_fid), ("late (z50~6.7)", sig_late)):
            for bias_lab, bfac in (("b = ST-matched", 1.0), ("b = %.2f x ST" % BIAS_LOW_FACTOR, BIAS_LOW_FACTOR)):
                for lab, filt in filters.items():
                    sn2 = {"shot": 0.0, "clust": 0.0}
                    for sh in sv.shells:
                        n = fe.lae_number_density(sh.L_min, sh.lf)
                        _, _, vol = fe.shell_geometry(sig.toy, sh.z, sh.dz)
                        b, _ = hb.bias_for_density(n, sh.z)
                        b *= bfac
                        bias_fn = lambda z, b=b: np.full_like(np.asarray(z, float), b)
                        f_sky = overlap * (sh.area_deg2 or sv.area_deg2) / SQ_DEG
                        s_shot = fe.snr_window(sig, filt, ref, sh.z, sh.dz, bias_fn, f_sky, n * vol)
                        s_clu = fe.snr_window(sig, filt, ref, sh.z, sh.dz, bias_fn, f_sky, None)
                        sn2["shot"] += s_shot ** 2
                        sn2["clust"] += s_clu ** 2
                    sn = float(np.sqrt(sn2["shot"]))
                    area = max((sh.area_deg2 or sv.area_deg2) for sh in sv.shells)
                    rows.append({"survey": sv.name, "history": hist, "bias": bias_lab, "cmb": lab,
                                 "SN_with_shot_noise": round(sn, 4),
                                 "SN_clustering_only": round(float(np.sqrt(sn2["clust"])), 3),
                                 # S/N^2 is proportional to the overlapping area at fixed density, so the area for 3 sigma is area x (3/SN)^2
                                 "area_for_3sigma_deg2": "%.2e" % (area * (3.0 / max(sn, 1e-12)) ** 2) if sn > 0 else "inf",
                                 "needs_more_than_whole_sky": "yes" if sn > 0 and area * (3.0 / sn) ** 2 > SQ_DEG else "no"})
    return rows


# ------------------------------------------------------------------------------------------------------------------------
def feasibility_map(sig, comp, out_base, z0=8.5, dz=1.0, bias=5.0):
    """S/N contours in (survey area, LAEs per deg2 in a dz = 1 window at z0)."""
    ref = fe.experiment_filter(fe.EXPERIMENTS["SO"], comp)
    areas = np.geomspace(1.0, 2000.0, 9)
    dens = np.geomspace(10.0, 1e5, 9)
    _, _, vol_per_sr = fe.shell_geometry(sig.toy, z0, dz)
    bias_fn = lambda z: np.full_like(np.asarray(z, float), bias)
    panels = [("CMB-HD", "CMB-HD post-ILC"), ("CMB-S4", "CMB-S4 post-ILC"), ("SO", "SO post-ILC")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True, sharey=True)
    for ax, (key, lab) in zip(axes, panels):
        filt = fe.experiment_filter(fe.EXPERIMENTS[key], comp)
        Z = np.zeros((len(dens), len(areas)))
        for j, A in enumerate(areas):
            for i, d in enumerate(dens):
                Z[i, j] = fe.snr_window(sig, filt, ref, z0, dz, bias_fn, A / SQ_DEG, d / (np.pi / 180) ** 2, n_ell=40)
        cs = ax.contour(areas, dens, Z, levels=[0.3, 1, 3, 5, 10], colors=[OKABE["blue"]], linewidths=1.2)
        ax.clabel(cs, fmt=lambda v: f"{v:g}$\\sigma$", fontsize=7)
        ax.contour(areas, dens, Z, levels=[3], colors=[OKABE["vermillion"]], linewidths=2.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("overlapping survey area [deg$^2$]")
        ax.set_title(f"{lab}", fontsize=9)
    axes[0].set_ylabel(f"LAEs per deg$^2$ in a $\\Delta z$ = {dz:g} window at z = {z0:g}")
    fig.suptitle(f"Detection forecast: kSZ$^2$ x LAE, window z = {z0:g}, $\\Delta z$ = {dz:g}, bias {bias:g} (orange = 3$\\sigma$)", fontsize=10)
    return save_fig(fig, out_base), (areas, dens)


# ------------------------------------------------------------------------------------------------------------------------
def halo_luminosity_mapping(bias_dir, out_dir, zs=(6.0, 7.1)):
    """Per-seed tracer_bias CSVs (scripts/35): for each LAE luminosity cut, the halo-mass cut with the same number density and the
    halo-mass cut with the same bias.  Values are seed means; errors are the error of the mean over the 10 seeds."""
    V = 300.0 ** 3
    R = {}
    for p in sorted(glob.glob(os.path.join(bias_dir, "tracer_bias_seed*.csv"))):
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    R.setdefault((round(float(r["z"]), 1), r["name"]), []).append(
                        (float(r["b_lin"]), float(r["n_obj"]) / V, float(r["b_ST"])))
                except ValueError:
                    continue
    rows = []
    for z in zs:
        halos = sorted([k[1] for k in R if k[0] == z and k[1].startswith("halo")])
        laes = sorted([k[1] for k in R if k[0] == z and k[1].startswith("lae")])
        hn = np.array([np.mean([v[1] for v in R[(z, h)]]) for h in halos])
        hb = np.array([np.mean([v[0] for v in R[(z, h)]]) for h in halos])
        hm = np.array([float(h.split("M")[1]) for h in halos])
        o = np.argsort(hn)
        for nm in laes:
            v = np.array(R[(z, nm)])
            n, b = v[:, 1].mean(), v[:, 0].mean()
            b_err = v[:, 0].std() / np.sqrt(len(v))
            logM_n = float(np.interp(np.log(n), np.log(hn[o]), hm[o]))        # halo cut with the same number density
            b_at_n = float(np.interp(np.log(n), np.log(hn[o]), hb[o]))
            ob = np.argsort(hb)
            logM_b = float(np.interp(b, hb[ob], hm[ob])) if hb.min() < b < hb.max() else float("nan")
            rows.append({"z": z, "LAE_cut": nm, "n_Mpc3": "%.2e" % n, "b_LAE": round(b, 2), "b_LAE_err": round(b_err, 2),
                         "b_hosts_ST": round(v[:, 2].mean(), 2),
                         "logM_same_n": round(logM_n, 2), "b_halo_same_n": round(b_at_n, 2),
                         "logM_same_b": round(logM_b, 2) if logM_b == logM_b else "b outside halo range"})
    return rows


def plot_mapping(bias_dir, rows, out_base, zs=(6.0, 7.1)):
    V = 300.0 ** 3
    fig, axes = plt.subplots(1, len(zs), figsize=(4.6 * len(zs), 3.8), constrained_layout=True, sharey=True)
    for ax, z in zip(np.atleast_1d(axes), zs):
        hn, hb, hm, ln, lb, le, ll = [], [], [], [], [], [], []
        data = {}
        for p in sorted(glob.glob(os.path.join(bias_dir, "tracer_bias_seed*.csv"))):
            with open(p) as f:
                for r in csv.DictReader(f):
                    if abs(float(r["z"]) - z) < 0.05 or round(float(r["z"]), 1) == z:
                        try:
                            data.setdefault(r["name"], []).append((float(r["b_lin"]), float(r["n_obj"]) / V))
                        except ValueError:
                            pass
        for nm, v in sorted(data.items(), key=lambda kv: float(kv[0].split("M")[1]) if kv[0].startswith("halo") else 0.0):
            v = np.array(v)
            if nm.startswith("halo"):
                hn.append(v[:, 1].mean()); hb.append(v[:, 0].mean()); hm.append(nm.split("M")[1])
            elif nm.startswith("lae"):
                ln.append(v[:, 1].mean()); lb.append(v[:, 0].mean()); le.append(v[:, 0].std() / np.sqrt(len(v))); ll.append(nm.split("L")[1])
        ax.plot(hn, hb, "-o", color=OKABE["blue"], lw=1.4, ms=4, label="halo mass cuts")
        for x, y, m in zip(hn, hb, hm):
            ax.annotate(f"$10^{{{m}}}$", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=OKABE["blue"])
        ax.errorbar(ln, lb, le, fmt="s", color=OKABE["vermillion"], ms=6, capsize=2, label="LAE luminosity cuts")
        for x, y, m in zip(ln, lb, ll):
            ax.annotate(f"$L\\geq10^{{{m}}}$", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=OKABE["vermillion"])
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xscale("log"); ax.set_xlabel("number density [Mpc$^{-3}$]"); ax.set_title(f"z = {z:g}", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("linear bias (measured, 10-seed mean)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="lower right")
    return save_fig(fig, out_base)


# ------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--out-dir", default="data/plots/feasibility")
    ap.add_argument("--bias-dir", default="data/plots/2026-10-16_tracer_suite/bias")
    ap.add_argument("--overlap", type=float, default=1.0, help="fraction of each LAE survey inside the CMB map (default 1: assumes full overlap)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)

    comp = fe.lp_components(cfg)
    sig_fid = fe.SignalModel(cfg)
    scale, raw = fe.calibrate_to_lp_table2(sig_fid, comp)
    sig_late = fe.SignalModel(cfg, z_shift=1.0, scale=scale)
    hb = fe.HaloBias(cfg)
    print(f"toy signal rescaled by {scale:.2f} to match La Plante Table 2 (raw LP/ours: " + ", ".join(f"{k} {v:.2f}" for k, v in raw.items()) + ")")

    rows = validation(sig_fid, comp)
    write_csv(os.path.join(args.out_dir, "feasibility_validation.csv"), rows)
    print("\nvalidation (Roman HLS, f_sky 0.053, no shot noise):")
    for r in rows:
        print(f"  {r['noise']:>18s} {r['experiment']:>7s}  La Plante {r['La_Plante_Table2']:6.1f}   ours {r['ours']:6.2f}   ratio {r['ratio']:.2f}   [{r['used_for_calibration']}]")

    srows = survey_rows(sig_fid, sig_late, hb, cfg)
    write_csv(os.path.join(args.out_dir, "feasibility_surveys.csv"), srows)
    print("\nLAE survey shells:")
    for r in srows:
        print(f"  {r['survey'][:30]:30s} z={r['z']:<4} dz={r['dz']:<5} n={r['n_Mpc3']} Mpc^-3  N/deg2={r['N_per_deg2']:<8} N={r['N_in_survey']:<7} "
              f"M_min={r['M_min_ST_Msun']} b_ST={r['b_ST_matched']}  x_HII fid/late {r['x_HII_fiducial']}/{r['x_HII_late']}  shot/clust(ell=1000)={r['shot_over_clustering']}")

    variants = experiment_variants()
    nrows = snr_rows(sig_fid, sig_late, comp, hb, variants, overlap=args.overlap)
    write_csv(os.path.join(args.out_dir, "feasibility_snr.csv"), nrows)
    print("\nS/N (with LAE shot noise | clustering-only ceiling), b = ST-matched:")
    for hist in ("fiducial", "late (z50~6.7)"):
        print(f"  history: {hist}")
        for sv in fe.SURVEYS:
            line = []
            for lab, _ in variants:
                r = [x for x in nrows if x["survey"] == sv and x["history"] == hist and x["cmb"] == lab and x["bias"] == "b = ST-matched"][0]
                line.append(f"{lab}: {r['SN_with_shot_noise']:.3f}|{r['SN_clustering_only']:.2f}")
            print(f"    {sv}\n      " + "\n      ".join(line))

    paths, _ = feasibility_map(sig_fid, comp, os.path.join(args.out_dir, "feasibility_map"))
    print("\nfigure:", *paths)

    if os.path.isdir(args.bias_dir):
        mrows = halo_luminosity_mapping(args.bias_dir, args.out_dir)
        write_csv(os.path.join(args.out_dir, "halo_luminosity_mapping.csv"), mrows)
        print("\nLAE luminosity cut vs halo mass cut (simulation, 10-seed means):")
        for r in mrows:
            print(f"  z={r['z']}  {r['LAE_cut']:>10s}  n={r['n_Mpc3']}  b_LAE={r['b_LAE']}+-{r['b_LAE_err']}  hosts(ST)={r['b_hosts_ST']}  "
                  f"same-n halo: logM={r['logM_same_n']} b={r['b_halo_same_n']}   same-b halo: logM={r['logM_same_b']}")
        print("figure:", *plot_mapping(args.bias_dir, mrows, os.path.join(args.out_dir, "halo_luminosity_mapping")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
