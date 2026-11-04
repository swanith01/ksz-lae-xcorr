#!/usr/bin/env python3
"""
scripts/58_detectability_pairs.py
===================================
The detectability / S/N discussion for the CMB x LAE survey pairs Girish asked for (3 Nov 2026):

  1   ACT DR6          x  Subaru/HSC SILVERRUSH, z ~ 6.6        what existing data can do
  2a  Simons Obs. LAT  x  SILVERRUSH, z ~ 6.6                   better CMB, same LAE data
  2b  Simons Obs. LAT  x  LAGER, z ~ 6.9                        better CMB, same quality LAE data
  3   Simons Obs. LAT  x  deep Roman grism survey, z ~ 7.5-10    better LAE data
  4   CMB-HD           x  deep Roman grism survey, z ~ 7.5-10    both much better
(CMB-S4 left out: DOE and NSF ended the project in July 2025.)

Everything as in scripts/55-56: S/N = La Plante+22 Eq. 13 summed over l = l_min(field)..10000 and over the survey's shells in
quadrature; real survey area, observed catalogue counts for the shot noise where known (SILVERRUSH: Ouchi+18; LAGER: Wold+22),
else the LF above L_min; survey bias by Sheth-Tormen abundance matching; footprint overlap from snr/feasibility.OVERLAP.
Two signals, both from the simulation, rescaled to the survey's bias:
  * La Plante's galaxy model (matter x Waters+16 bias): the setup validated against La Plante (scripts/57; per bin our signal is
    ~2x below theirs at l >~ 1000, so these may be low by up to ~2x);
  * the simulated LAEs (lae_L42.5): includes the LAE excess, NOT yet confirmed by a null test.
Noise: ACT DR6 white noise only (optimistic, no foreground residual); SO and CMB-HD post-ILC (La Plante Fig. 1) and white noise.
Roman: the LAE density is bracketed by the z = 7.0 LF (no evolution, optimistic) and the z = 7.3 LF (observed decline).

    python scripts/58_detectability_pairs.py --out-dir data/plots/2026-11-03_ksz_lae_snr/detectability_pairs

Outputs: detectability_pairs.csv, detectability_pairs.png/pdf
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

from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.snr import sim_snr as ss
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.figio import save_fig

SQ_DEG = 41252.96
HSC = "HSC SILVERRUSH (NB816+NB921)"
LAGER = "LAGER (DECam NB964)"
ROMAN = "Roman grism deep z=7.5-10, no evolution from z=7"
ROMAN_LO = "Roman grism deep z=7.5-10, LF as observed at z=7.3"
ROMAN_W = "Roman grism 16 deg2 at deep-field depth (Maitra+25 area)"
ROMAN_W_LO = "Roman HLWAS deep tier (19.2 deg2, grism 6e-17)"
LOW = {ROMAN: ROMAN_LO, ROMAN_W: ROMAN_W_LO}     # survey -> its pessimistic variant (drawn as the small marker)
# (key, label, CMB experiment, LAE survey, shells to use (None = all), colour)
PAIRS = [
    ("1", "ACT DR6 x HSC SILVERRUSH z~6.6\n(existing data)", "ACT DR6", HSC, (6.6,), "#CC79A7"),
    ("1s", "SPT-3G x HSC SILVERRUSH z~6.6\n(existing data; NO field overlap:\nhypothetical, as if overlapping)", "SPT-3G Main", HSC, (6.6,), "#D55E00"),
    ("2a", "SO LAT x HSC SILVERRUSH z~6.6\n(better CMB)", "SO", HSC, (6.6,), "#56B4E9"),
    ("2b", "SO LAT x LAGER z~6.9\n(better CMB)", "SO", LAGER, None, "#0072B2"),
    ("3", "SO LAT x Roman deep z~7.5-10\n(better LAEs)", "SO", ROMAN, None, "#E69F00"),
    ("4", "CMB-HD x Roman deep z~7.5-10\n(both better)", "CMB-HD", ROMAN, None, "#009E73"),
    ("3w", "SO LAT x Roman wider, 16-19 deg2\n(better LAEs, Maitra+25 area)", "SO", ROMAN_W, None, "#B8860B"),
    ("4w", "CMB-HD x Roman wider, 16-19 deg2\n(both better, Maitra+25 area)", "CMB-HD", ROMAN_W, None, "#006D4C"),
]
HYPOTHETICAL = {"1s"}     # SPT-3G Main x SILVERRUSH: the fields do not overlap (Dec -42..-70 vs > -10); shown as if they did
SIGNALS = (("matter_waters", "La Plante's galaxy model (validated)"), ("lae_L42.5", "simulated LAEs (excess not yet confirmed)"))


def noise_labels(exp):
    return [("white", f"{exp} (white noise)")] if exp in ("ACT DR6", "SPT-3G Main") else [("ilc", f"{exp} (post-ILC)"), ("white", f"{exp} (white noise)")]


def area3(area, sn):
    """Survey area giving 3 sigma at the same LAE density and depth (S/N^2 ~ area); flagged when beyond the whole sky."""
    if not sn > 0:
        return ""
    a = area * (3 / sn) ** 2
    return f"{a:.3g}" + ("  (> whole sky)" if a > SQ_DEG else "")


def survey_terms(sv, shells, exp_name, lab, sp, btab, hb, chi, ell_max):
    """Per-shell (terms, edges) and the summed (S/N)^2 for one survey, CMB noise label and simulated signal."""
    ov = fe.overlap(sv.name, exp_name)
    out = []
    for sh in sv.shells:
        if shells and not any(abs(sh.z - z) < 1e-6 for z in shells):
            continue
        wi = ss.shell_window(sp, sh.z)
        if wi is None or ov <= 0:
            continue
        area = sh.area_deg2 or sv.area_deg2
        n3 = fe.lae_number_density(sh.L_min, sh.lf)
        lo, hi = chi(sh.z - sh.dz / 2), chi(sh.z + sh.dz / 2)
        n2d = sh.n_obs_deg2 / (np.pi / 180) ** 2 if sh.n_obs_deg2 else n3 * chi(sh.z) ** 2 * (hi - lo)
        b_sv, _ = hb.bias_for_density(n3, sh.z)
        b_sim = ss.sim_bias(str(sp["name"]), sh.z, btab)
        r = b_sv / b_sim
        ei = [str(x) for x in sp["experiments"]].index(lab)
        Cg = ss.survey_cg(np.nanmean(sp["Cg"][:, wi, :], axis=0), float(sp["shot_sim"][wi]), r,
                          chi(sp["z_hi"][wi]) - chi(sp["z_lo"][wi]), hi - lo, n2d)
        t = ss.snr_terms(ss.signal_sq(sp["Cs"][:, ei, wi, :] * r, True), Cg, float(sp["CT2T2"][ei]), sp["edges"][wi],
                         ov * area / SQ_DEG, fe.field_ell_min(area), ell_max)
        out.append({"z": sh.z, "area": area, "n2d_deg2": n2d * (np.pi / 180) ** 2, "b_survey": b_sv, "b_sim": b_sim, "r": r,
                    "terms": t, "edges": sp["edges"][wi]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--spectra-dir", default="data/products_tracers/sim_spectra")
    ap.add_argument("--out-dir", default="data/plots/detectability_pairs")
    ap.add_argument("--bias-csv", default="data/plots/2026-10-16_tracer_suite/bias/tracer_bias_summary.csv")
    ap.add_argument("--ell-max", type=float, default=10000.0)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    cosmo = get_cosmology(cfg)
    chi = lambda z: float(cosmo.comoving_distance(z).to_value("Mpc"))
    sp = {}
    for nm, _ in SIGNALS:
        z = np.load(os.path.join(args.spectra_dir, f"{nm}.npz"), allow_pickle=False)
        sp[nm] = {k: z[k] for k in z.files}
    btab = ss.load_bias_table(args.bias_csv)
    hb = fe.HaloBias(cfg)
    lgrid = np.geomspace(150, args.ell_max, 80)

    rows, res = [], {}
    for key, label, exp, sname, shells, col in PAIRS:
        for svn in ((sname, LOW[sname]) if sname in LOW else (sname,)):
            sv = fe.SURVEYS[svn]
            if key in HYPOTHETICAL:          # the real overlap is zero; compute as if the fields overlapped
                sv = dataclasses.replace(sv, name=sv.name + " [as if overlapping]")
            for nkey, lab in noise_labels(exp):
                for sig, sig_label in SIGNALS:
                    parts = survey_terms(sv, shells, exp, lab, sp[sig], btab, hb, chi, args.ell_max)
                    sn = float(np.sqrt(sum(p["terms"].sum() for p in parts))) if parts else 0.0
                    cum = ss.cumulative_snr([p["terms"] for p in parts], [p["edges"] for p in parts], lgrid, 100.0) if parts else np.zeros_like(lgrid)
                    area = max(p["area"] for p in parts) if parts else np.nan
                    res[(key, svn, nkey, sig)] = (sn, cum)
                    rows.append({"pair": key, "cmb": lab, "lae_survey": svn + (" [as if overlapping: real S/N = 0]" if key in HYPOTHETICAL else ""), "signal": sig_label,
                                 "shells_z": " ".join(f"{p['z']:g}" for p in parts), "area_deg2": area,
                                 "LAEs_per_deg2": " ".join(f"{p['n2d_deg2']:.0f}" if p['n2d_deg2'] >= 10 else f"{p['n2d_deg2']:.2g}" for p in parts),
                                 "b_survey": " ".join(f"{p['b_survey']:.1f}" for p in parts),
                                 "SN": round(sn, 4),
                                 "area_for_3sigma_deg2": area3(area, sn)})
    with open(os.path.join(args.out_dir, "detectability_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"{'pair':>4} {'CMB':>22} {'LAE survey':>40} | La Plante model | sim LAEs | 3sigma area (La Plante model)")
    for key, label, exp, sname, shells, col in PAIRS:
        for svn in ((sname, LOW[sname]) if sname in LOW else (sname,)):
            for nkey, lab in noise_labels(exp):
                a = [r for r in rows if r["pair"] == key and r["lae_survey"].startswith(svn) and r["cmb"] == lab]
                print(f"{key:>4} {lab:>22} {svn[:40]:>40} | {a[0]['SN']:<15} | {a[1]['SN']:<8} | {a[0]['area_for_3sigma_deg2']} deg2")

    # ---- figure ---------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(23, 6.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1, 1, 1]})
    a1, cax = axs[0], axs[1:]
    ypos = np.arange(len(PAIRS))[::-1]
    for y, (key, label, exp, sname, shells, col) in zip(ypos, PAIRS):
        for dy, (sig, _), mk in zip((0.13, -0.13), SIGNALS, ("o", "D")):
            for nkey, lab in noise_labels(exp):
                v = res[(key, sname, nkey, sig)][0]
                a1.plot(v, y + dy, mk, ms=9, color=col, mfc=col if nkey == "ilc" or exp in ("ACT DR6", "SPT-3G Main") else "white", mew=1.6,
                        zorder=3, alpha=0.55 if key in HYPOTHETICAL else 1)
                if sname in LOW:
                    vlo = res[(key, LOW[sname], nkey, sig)][0]
                    a1.plot([vlo, v], [y + dy, y + dy], color=col, lw=1, alpha=0.6, zorder=2)
                    a1.plot(vlo, y + dy, mk, ms=5, color=col, mfc="white", mew=1, alpha=0.8, zorder=3)
    a1.axvline(1, color="0.6", lw=0.8, ls=":"); a1.axvline(3, color="0.6", lw=0.8, ls="--")
    a1.set_xscale("log"); a1.set_xlim(1e-3, 10)
    a1.set_yticks(ypos, [f"{k}  {l}" for k, l, *_ in PAIRS], fontsize=8.5)
    a1.set_xlabel("total S/N (all l, all shells of the survey)")
    a1.set_title("S/N per CMB x LAE pair", fontsize=10)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color="0.3", ms=8, label="La Plante's galaxy model (validated; may be low by up to 2x)"),
         Line2D([], [], marker="D", ls="", color="0.3", ms=7, label="simulated LAEs (excess not yet confirmed)"),
         Line2D([], [], marker="o", ls="", color="0.3", mfc="0.3", ms=8, label="filled: post-ILC noise (ACT / SPT-3G: white noise only)"),
         Line2D([], [], marker="o", ls="", color="0.3", mfc="white", ms=8, label="open: white (instrument) noise"),
         Line2D([], [0], marker="o", ls="-", color="0.3", mfc="white", ms=4, lw=1,
                label="small marker: Roman deep = z = 7.3 LF (observed decline); Roman wider = HLWAS deep-tier depth"),
         Line2D([], [], color="0.6", ls=":", label="1$\\sigma$"), Line2D([], [], color="0.6", ls="--", label="3$\\sigma$")]
    a1.legend(handles=h, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)

    # cumulative S/N vs l_max: one panel per CMB experiment (ACT | SO LAT | CMB-HD)
    for ax, cmb, ttl in zip(cax, (("ACT DR6", "SPT-3G Main"), ("SO",), ("CMB-HD",)),
                            ("ACT DR6 / SPT-3G (white noise only)", "Simons Obs. LAT (post-ILC)", "CMB-HD (post-ILC)")):
        for key, label, exp, sname, shells, col in PAIRS:
            if exp not in cmb:
                continue
            nkey = "white" if exp in ("ACT DR6", "SPT-3G Main") else "ilc"
            for (sig, _), ls in zip(SIGNALS, ("-", "--")):
                sn, cum = res[(key, sname, nkey, sig)]
                lae = label.splitlines()[0].split(" x ")[1]
                ax.plot(lgrid, cum, ls, color=col, lw=2.2 if ls == "-" else 1.4,
                        alpha=0.6 if key in HYPOTHETICAL else 1,
                        label=f"{key}: {label.splitlines()[0]}" + (" (no overlap; as if)" if key in HYPOTHETICAL else "") + ", "
                              + ("La Plante's galaxy model" if ls == "-" else "simulated LAEs") + f": {sn:.2g}")
        ax.axhline(1, color="0.6", lw=0.8, ls=":", label="1$\\sigma$"); ax.axhline(3, color="0.6", lw=0.8, ls="--", label="3$\\sigma$")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 10); ax.set_xlim(150, 10000)
        ax.set_xlabel(r"$\ell_{\max}$"); ax.set_title(f"{ttl}: cumulative S/N vs l_max", fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1)
    cax[0].set_ylabel("cumulative S/N")
    fig.suptitle("kSZ$^2$ x LAE detectability for the survey pairs: existing data, better CMB, better LAEs, both "
                 "(fiducial reionisation history; real areas, counts and footprints)", fontsize=10.5)
    print("\nfigure:", *save_fig(fig, os.path.join(args.out_dir, "detectability_pairs")))
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
