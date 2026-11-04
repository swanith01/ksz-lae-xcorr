#!/usr/bin/env python3
"""
scripts/60_snr_report_tables.py
=================================
LaTeX tables for docs/snr_report/snr_report.tex, written from the code's own inputs (snr/feasibility.py) and the CSVs of
scripts/55, 57 and 58, so the numbers in the report are exactly the ones in the figures.

    python scripts/60_snr_report_tables.py --plots data/plots/2026-11-03_ksz_lae_snr --out docs/snr_report/tables

Writes: cmb_experiments.tex, lae_surveys.tex, lp_validation.tex, pairs.tex, survey_forecast.tex
"""
import argparse
import csv
import os
import sys

import numpy as np

from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology

SRC_EXP = {
    "ACT DR6": r"Naess+25 (2503.14451); beam 1.4$'$ at 150\,GHz",
    "SPT-3G Main": r"SPT-3G (2403.17925) Tab.\,1; beam 1.2$'$ assumed",
    "SO": r"SO science goals (1808.07445); post-ILC: LP22 Fig.\,1",
    "CMB-S4": r"LP22 Tab.\,1; post-ILC: LP22 Fig.\,1; project ended 2025",
    "CMB-HD": r"LP22 Tab.\,1; post-ILC: LP22 Fig.\,1",
}
SRC_LF = {"Umeda25": "Umeda+25 Tab.\\,3", "Wold22": "Wold+22 (4-field, $\\alpha=-2.5$)"}
SRC_LMIN = {"HSC SILVERRUSH (NB816+NB921)": "Weinberger+20 Tab.\\,1", "Subaru PFS (SSP, repo value)": "assumed",
            "LAGER (DECam NB964)": "Wold+22", "Roman grism deep z=7.5-10, no evolution from z=7": "Wold+23",
            "Roman grism deep z=7.5-10, LF as observed at z=7.3": "Wold+23",
            "Roman grism 16 deg2 at deep-field depth (Maitra+25 area)": "Wold+23 depth",
            "Roman HLWAS deep tier (19.2 deg2, grism 6e-17)": "ROTAC 2025, derived"}
SRC_NOBS = {"HSC SILVERRUSH (NB816+NB921)": "Ouchi+18", "LAGER (DECam NB964)": "Wold+22"}
SHORT = {"HSC SILVERRUSH (NB816+NB921)": "HSC SILVERRUSH", "Subaru PFS (SSP, repo value)": "Subaru PFS",
         "LAGER (DECam NB964)": "LAGER", "Roman grism deep z=7.5-10, no evolution from z=7": "Roman deep (no LF evol.)",
         "Roman grism deep z=7.5-10, LF as observed at z=7.3": "Roman deep (LF as obs.)",
         "Roman grism 16 deg2 at deep-field depth (Maitra+25 area)": "Roman 16 deg$^2$ (upper)",
         "Roman HLWAS deep tier (19.2 deg2, grism 6e-17)": "Roman HLWAS deep tier"}


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def sci(x, d=1):
    if x == 0 or not np.isfinite(x):
        return "--"
    e = int(np.floor(np.log10(abs(x))))
    return rf"${x / 10 ** e:.{d}f}\times10^{{{e}}}$"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--plots", default="data/plots/2026-11-03_ksz_lae_snr")
    ap.add_argument("--out", default="docs/snr_report/tables")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    cosmo = get_cosmology(cfg)
    chi = lambda z: float(cosmo.comoving_distance(z).to_value("Mpc"))
    hb = fe.HaloBias(cfg)

    # ---- CMB experiments
    L = [r"\begin{tabular}{lcccl}", r"\toprule",
         r"experiment & $\Delta_T$ [$\mu$K-arcmin] & FWHM & area [deg$^2$] & source \\", r"\midrule"]
    for n, e in fe.EXPERIMENTS.items():
        fw = f"{e.fwhm_arcmin * 60:.0f}$''$" if e.fwhm_arcmin < 1 else f"{e.fwhm_arcmin:g}$'$"
        L.append(rf"{tex(n)} & {e.white_uK_arcmin:g} & {fw} & {e.area_deg2:,.0f} & {SRC_EXP.get(n, '')} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.out, "cmb_experiments.tex"), L)

    # ---- LAE surveys
    L = [r"\begin{tabular}{llcccccccc}", r"\toprule",
         r"survey & $z$ & $\Delta z$ & area & $L_{\min}$ [erg\,s$^{-1}$] & LF & $n(>L_{\min})$ [Mpc$^{-3}$] & "
         r"$N$/deg$^2$ (LF) & $N$/deg$^2$ used & $b_{\rm ST}$ \\", r"\midrule"]
    for sn, sv in fe.SURVEYS.items():
        for sh in sv.shells:
            n3 = fe.lae_number_density(sh.L_min, sh.lf)
            lo, hi = chi(sh.z - sh.dz / 2), chi(sh.z + sh.dz / 2)
            nlf = n3 * chi(sh.z) ** 2 * (hi - lo) * (np.pi / 180) ** 2
            b, M = hb.bias_for_density(n3, sh.z)
            used = f"{sh.n_obs_deg2:.0f} ({SRC_NOBS.get(sn, 'obs.')})" if sh.n_obs_deg2 else f"{nlf:.0f} (LF)" if nlf >= 10 else f"{nlf:.2g} (LF)"
            L.append(rf"{SHORT.get(sn, tex(sn))} & {sh.z:g} & {sh.dz:g} & {sh.area_deg2 or sv.area_deg2:g} & "
                     rf"{sci(sh.L_min)} ({SRC_LMIN.get(sn, '')}) & {SRC_LF.get(sh.lf[0], sh.lf[0])} $z={sh.lf[1]:g}$ & {sci(n3)} & "
                     rf"{(f'{nlf:.0f}' if nlf >= 10 else f'{nlf:.2g}')} & {used} & {b:.1f} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.out, "lae_surveys.tex"), L)

    # ---- validation vs La Plante Table 2
    rows = list(csv.DictReader(open(os.path.join(args.plots, "lp_validation", "lp_validation.csv"))))
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"CMB noise & shot noise & \multicolumn{2}{c}{SO} & \multicolumn{2}{c}{CMB-S4} & \multicolumn{2}{c}{CMB-HD} \\",
         r" & & LP22 & ours & LP22 & ours & LP22 & ours \\", r"\midrule"]
    for noise in ("post-ILC", "instrument (white)"):
        for shot in ("with", "without"):
            cells = []
            for e in ("SO", "CMB-S4", "CMB-HD"):
                r = [x for x in rows if x["tracer"] == "matter_waters" and x["experiment"] == e and x["noise"] == noise and x["shot_noise"] == shot][0]
                cells += [f"{float(r['SN_LaPlante_Table2']):g}", f"{float(r['SN_ours']):.2g}"]
            L.append(rf"{noise} & {shot} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.out, "lp_validation.tex"), L)

    # ---- Girish's pairs
    rows = list(csv.DictReader(open(os.path.join(args.plots, "detectability_pairs", "detectability_pairs.csv"))))
    L = [r"\begin{tabular}{lllcccc}", r"\toprule",
         r"pair & CMB (noise) & LAE survey & LAEs/deg$^2$ & S/N, LP22 galaxy model & S/N, sim.\ LAEs & area for 3$\sigma$ [deg$^2$] \\",
         r"\midrule"]
    seen = {}
    for r in rows:
        k = (r["pair"], r["cmb"], r["lae_survey"])
        seen.setdefault(k, {})[r["signal"].split(" (")[0]] = r
    for (p, cmb, sv), d in seen.items():
        a = d.get("La Plante's galaxy model", {}); s = d.get("simulated LAEs", {})
        svs = SHORT.get(sv.split(" [")[0], tex(sv.split(" [")[0])) + (r" $^\dagger$" if "as if" in sv else "")
        area = a.get("area_for_3sigma_deg2", "")
        area = r"$>$ full sky" if "whole sky" in area else area
        L.append(rf"{p} & {tex(cmb)} & {svs} & {a.get('LAEs_per_deg2', '').replace(' ', ' / ')} & {a.get('SN', '')} & {s.get('SN', '')} & {area} \\")
    L += [r"\bottomrule", r"\multicolumn{7}{l}{\footnotesize $^\dagger$ SPT-3G Main and SILVERRUSH do not overlap on the sky; computed as if they did (real S/N = 0).} \\",
          r"\end{tabular}"]
    write(os.path.join(args.out, "pairs.tex"), L)

    # ---- survey x CMB (script 55)
    rows = list(csv.DictReader(open(os.path.join(args.plots, "survey_forecast", "survey_forecast.csv"))))
    cmbs = list(dict.fromkeys(r["cmb"] for r in rows))
    L = [r"\begin{tabular}{l" + "c" * len(cmbs) + "}", r"\toprule",
         "survey / signal & " + " & ".join(r"\shortstack{" + tex(c).replace("SPT-3G Main", "SPT-3G").replace(" (", r"\\(") + "}" for c in cmbs) + r" \\", r"\midrule"]
    for sv in dict.fromkeys(r["survey"] for r in rows):
        for col, lab in (("SN_sim_LaPlante_model", "LP22 model"), ("SN_sim_lae_L42.5", "sim. LAEs")):
            cells = []
            for c in cmbs:
                r = [x for x in rows if x["survey"] == sv and x["cmb"] == c][0]
                cells.append("--" if float(r["overlap"]) == 0 else f"{float(r[col]):.2g}")
            L.append(rf"{SHORT.get(sv, tex(sv))}, {lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.out, "survey_forecast.tex"), L)


if __name__ == "__main__":
    sys.exit(main())
