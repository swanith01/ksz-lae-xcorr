#!/usr/bin/env python3
"""
scripts/59_detectability_pair_curves.py
=========================================
One figure per CMB x LAE pair of scripts/58 (Girish's list, 3 Nov 2026), in the style of scripts/56:

  left  : S/N if this survey (same area, LAE density, depth and shell width) sat at each simulated window z0 (dz = 1 windows);
          stars = its real shell(s).  A "where in z would this pair work best" curve, not a measurement.
  right : cumulative S/N vs l_max for the real shell(s).
  colour = the pair; solid = simulated LAE signal (lae_L42.5, excess not yet confirmed), dashed = La Plante's galaxy model
  (matter x W16 bias, the validated setup); thick = post-ILC noise, thin = white (instrument) noise (ACT / SPT-3G: white only).
Same noise model, survey numbers and caveats as scripts/58.  Extra pair 1s: SPT-3G Main x HSC SILVERRUSH, computed AS IF the
fields overlapped -- they do not (SILVERRUSH fields are at Dec > -10 deg, SPT-3G Main at Dec -42 to -70 deg), so the real
S/N of that pair is zero; shown only to compare the CMB data sets on the same LAE field.

    python scripts/59_detectability_pair_curves.py --out-dir data/plots/2026-11-03_ksz_lae_snr/detectability_pairs

Outputs: pair_<key>.png/pdf for each pair, pair_curves.csv
"""
import argparse
import csv
import dataclasses
import importlib.util
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

_spec = importlib.util.spec_from_file_location("pairs58", os.path.join(os.path.dirname(os.path.abspath(__file__)), "58_detectability_pairs.py"))
p58 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p58)

HYPO = " [as if overlapping]"
PAIRS = [
    ("1", "ACT DR6 x HSC SILVERRUSH z~6.6 (existing data)", "ACT DR6", p58.HSC, (6.6,), "#CC79A7", False),
    ("1s", "SPT-3G Main x HSC SILVERRUSH z~6.6 (existing data; fields do NOT overlap - hypothetical)", "SPT-3G Main", p58.HSC, (6.6,),
     "#D55E00", True),
    ("2a", "SO LAT x HSC SILVERRUSH z~6.6 (better CMB)", "SO", p58.HSC, (6.6,), "#56B4E9", False),
    ("2b", "SO LAT x LAGER z~6.9 (better CMB)", "SO", p58.LAGER, None, "#0072B2", False),
    ("3", "SO LAT x Roman deep z~7.5-10 (better LAEs)", "SO", p58.ROMAN, None, "#E69F00", False),
    ("4", "CMB-HD x Roman deep z~7.5-10 (both better)", "CMB-HD", p58.ROMAN, None, "#009E73", False),
]


def noise_labels(exp):
    return [("white", f"{exp} (white noise)")] if exp in ("ACT DR6", "SPT-3G Main") else [("ilc", f"{exp} (post-ILC)"), ("white", f"{exp} (white noise)")]


def moved_survey(sv, shells, z0, chi, hypo):
    """The survey with its (first used) shell moved to centre z0: same area, dz, L_min, LF and 3D density (observed density
    converted at the real z and held fixed)."""
    sh = [s for s in sv.shells if not shells or any(abs(s.z - z) < 1e-6 for z in shells)][0]
    n_obs = None
    if sh.n_obs_deg2:
        vol = lambda z: chi(z) ** 2 * (chi(z + sh.dz / 2) - chi(z - sh.dz / 2)) * (np.pi / 180) ** 2
        n_obs = sh.n_obs_deg2 / vol(sh.z) * vol(z0)
    new = dataclasses.replace(sh, z=float(z0), n_obs_deg2=n_obs, area_deg2=sh.area_deg2 or sv.area_deg2)
    return dataclasses.replace(sv, name=sv.name + (HYPO if hypo else ""), shells=[new])


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
    for nm, _ in p58.SIGNALS:
        z = np.load(os.path.join(args.spectra_dir, f"{nm}.npz"), allow_pickle=False)
        sp[nm] = {k: z[k] for k in z.files}
    btab = ss.load_bias_table(args.bias_csv)
    hb = fe.HaloBias(cfg)
    lgrid = np.geomspace(150, args.ell_max, 80)
    z0s = [float(z) for z, d in zip(sp["matter_waters"]["z0"], sp["matter_waters"]["dz"]) if abs(d - 1) < 1e-6]
    zb = btab.get("lae_L42.5", (np.array([np.nan]),))[0]
    rows = []

    for key, label, exp, sname, shells, col, hypo in PAIRS:
        sv0 = fe.SURVEYS[sname]
        sv = dataclasses.replace(sv0, name=sv0.name + HYPO) if hypo else sv0
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
        for nkey, lab in noise_labels(exp):
            lw = 2.2 if (nkey == "ilc" or exp in ("ACT DR6", "SPT-3G Main")) else 1.1
            nlab = "post-ILC" if nkey == "ilc" else "white noise"
            for (sig, _), ls in zip(p58.SIGNALS, ("--", "-")):
                slab = "La Plante's galaxy model" if sig == "matter_waters" else "simulated LAEs"
                # left: slide the survey over z
                y = []
                for z0 in z0s:
                    parts = p58.survey_terms(moved_survey(sv0, shells, z0, chi, hypo), None, exp, lab, sp[sig], btab, hb, chi, args.ell_max)
                    y.append(float(np.sqrt(sum(p["terms"].sum() for p in parts))) if parts else np.nan)
                a1.plot(z0s, y, ls, color=col, lw=lw, marker="o" if ls == "-" else None, ms=3)
                # real shells: stars + cumulative
                parts = p58.survey_terms(sv, shells, exp, lab, sp[sig], btab, hb, chi, args.ell_max)
                tot = float(np.sqrt(sum(p["terms"].sum() for p in parts))) if parts else 0.0
                for p in parts:
                    a1.plot(p["z"], np.sqrt(p["terms"].sum()), "*", ms=14 if lw > 2 else 10, color=col, mec="k", mew=0.6, zorder=5)
                extra = ""
                if sname == p58.ROMAN:
                    lo = p58.survey_terms(fe.SURVEYS[p58.ROMAN_LO], None, exp, lab, sp[sig], btab, hb, chi, args.ell_max)
                    extra = f" (z=7.3 LF: {np.sqrt(sum(p['terms'].sum() for p in lo)):.2g})"
                if parts:
                    cum = ss.cumulative_snr([p["terms"] for p in parts], [p["edges"] for p in parts], lgrid, 100.0)
                    a2.plot(lgrid, cum, ls, color=col, lw=lw, label=f"{slab}, {nlab}: {tot:.2g}{extra}")
                rows.append({"pair": key, "cmb": lab, "lae_survey": sname, "hypothetical_overlap": hypo, "signal": slab,
                             "SN_real_shells": round(tot, 4), "SN_z0_curve": " ".join(f"{z:g}:{v:.3g}" for z, v in zip(z0s, y))})
        for a in (a1, a2):
            a.set_yscale("log"); a.set_ylim(1e-4, 5)
            a.axhline(1, color="0.6", lw=0.8, ls=":"); a.axhline(3, color="0.6", lw=0.8, ls="--")
        if np.isfinite(zb).all():
            a1.axvspan(zb.max() + 0.25, max(z0s) + 0.3, color="0.93", zorder=0)
            a1.text(zb.max() + 0.35, 2.5, "LAE bias held constant\n(beyond measured z)", fontsize=7, color="0.4", va="top")
        a1.set_xlabel(r"window centre $z_0$  ($\Delta z = 1$)"); a1.set_ylabel("S/N")
        a1.set_title("S/N if this survey sat at each z (stars: its real shells)", fontsize=9.5)
        a2.set_xscale("log"); a2.set_xlim(150, args.ell_max)
        a2.set_xlabel(r"$\ell_{\max}$"); a2.set_ylabel("cumulative S/N (real shells)")
        a2.set_title("cumulative S/N vs l_max (real shells)", fontsize=9.5)
        from matplotlib.lines import Line2D
        key_h = [Line2D([], [], color="0.3", ls="-", lw=2, label="solid: simulated LAEs (excess not yet confirmed)"),
                 Line2D([], [], color="0.3", ls="--", lw=2, label="dashed: La Plante's galaxy model (validated; may be low by up to 2x)"),
                 Line2D([], [], color="0.3", lw=2.2, label="thick: post-ILC noise" + (" (white only for ACT / SPT-3G)" if True else "")),
                 Line2D([], [], color="0.3", lw=1.1, label="thin: white (instrument) noise"),
                 Line2D([], [], color="0.6", ls=":", label="1$\\sigma$"), Line2D([], [], color="0.6", ls="--", label="3$\\sigma$")]
        a1.legend(handles=key_h, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
        a2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
        sv_ = fe.SURVEYS[sname]
        fig.suptitle(f"pair {key}: {label}\nLAE survey: {sv_.name}, {max(s.area_deg2 or sv_.area_deg2 for s in sv_.shells):g} deg$^2$",
                     fontsize=10.5, color=col)
        print("figure:", *save_fig(fig, os.path.join(args.out_dir, f"pair_{key}")))
        plt.close(fig)
    with open(os.path.join(args.out_dir, "pair_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\npair  cmb  signal: S/N (real shells)")
    for r in rows:
        print(f"  {r['pair']:>3} {r['cmb']:>26} {r['signal']:>25}: {r['SN_real_shells']}" + ("   [hypothetical overlap]" if r["hypothetical_overlap"] else ""))


if __name__ == "__main__":
    sys.exit(main())
