#!/usr/bin/env python3
"""
scripts/56_survey_snr_curves.py
=================================
The "Fig. 4" view (S/N vs z, cumulative S/N vs l_max) for the REAL LAE surveys, one figure per survey, every CMB experiment.

  left  : S/N per Delta z = 1 window IF the survey (its area, its LAE density and selection) were placed at each window z0
          (n_3D held at the survey's value, bias re-matched with Sheth-Tormen at each z).  Big markers = the survey's real shells
          (inside the simulated range z >= 6).  A "where in z would this survey work best" curve, not a measurement.
  right : cumulative S/N vs l_max for the survey's real shells (quadrature over shells), i.e. which scales carry the signal.
  solid = simulated LAE signal per unit bias (default lae_L42.5); dashed = La Plante's galaxy model (matter x Waters+16), both
  rescaled to the survey's bias.  Same noise as scripts/55 (observed catalogue counts where known, small-field l_min, footprint
  overlap; ACT / SPT-3G white noise only).  Debiased for the seed scatter.  Needs only the committed sim_spectra .npz.

    python scripts/56_survey_snr_curves.py --out-dir data/plots/2026-11-03_ksz_lae_snr/survey_curves
"""
import argparse
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
EXPS = (("ACT DR6 (white noise)", "ACT DR6", "#CC79A7"), ("SPT-3G Main (white noise)", "SPT-3G Main", "#D55E00"),
        ("SO (post-ILC)", "SO", "#56B4E9"), ("CMB-S4 (post-ILC)", "CMB-S4", "#E69F00"), ("CMB-HD (post-ILC)", "CMB-HD", "#009E73"))


def terms(sp, lab, wi, r, f_sky, n2d, dchi_w, dchi_sh, lmin, lmax):
    ei = [str(x) for x in sp["experiments"]].index(lab)
    Cg = ss.survey_cg(np.nanmean(sp["Cg"][:, wi, :], axis=0), float(sp["shot_sim"][wi]), r, dchi_w, dchi_sh, n2d)
    return ss.snr_terms(ss.signal_sq(sp["Cs"][:, ei, wi, :] * r, True), Cg, float(sp["CT2T2"][ei]), sp["edges"][wi], f_sky, lmin, lmax)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--spectra-dir", default="data/products_tracers/sim_spectra")
    ap.add_argument("--out-dir", default="data/plots/survey_curves")
    ap.add_argument("--lae-tracer", default="lae_L42.5")
    ap.add_argument("--bias-csv", default="data/plots/2026-10-16_tracer_suite/bias/tracer_bias_summary.csv")
    ap.add_argument("--ell-max", type=float, default=10000.0)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    cosmo = get_cosmology(cfg)
    chi = lambda z: float(cosmo.comoving_distance(z).to_value("Mpc"))
    sp = {}
    for nm in ("matter_waters", args.lae_tracer):
        z = np.load(os.path.join(args.spectra_dir, f"{nm}.npz"), allow_pickle=False)
        sp[nm] = {k: z[k] for k in z.files}
    btab = ss.load_bias_table(args.bias_csv)
    hb = fe.HaloBias(cfg)
    zb = btab.get(args.lae_tracer, (np.array([np.nan]),))[0]
    lmax_grid = np.geomspace(150, args.ell_max, 80)

    for sv in fe.SURVEYS.values():
        sh0 = sv.shells[-1]                                    # selection/density template for the sliding curve: the highest-z shell
        area0 = sh0.area_deg2 or sv.area_deg2
        n3_lf0 = fe.lae_number_density(sh0.L_min, sh0.lf)
        c0, lo0, hi0 = chi(sh0.z), chi(sh0.z - sh0.dz / 2), chi(sh0.z + sh0.dz / 2)
        n3_0 = (sh0.n_obs_deg2 / (np.pi / 180) ** 2 / (c0 ** 2 * (hi0 - lo0))) if sh0.n_obs_deg2 else n3_lf0
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        z0 = sp["matter_waters"]["z0"]
        for lab, ename, col in EXPS:
            ov = fe.overlap(sv.name, ename)
            if ov <= 0:
                a1.plot([], [], color=col, label=f"{ename}: no footprint overlap")
                continue
            # ---- left: slide the survey over the windows
            for key, ls, mk in ((args.lae_tracer, "-", "o"), ("matter_waters", "--", None)):
                s = sp[key]
                y = []
                for wi in range(len(z0)):
                    zc = float(s["z0"][wi])
                    dsh = sh0.dz
                    c, lo, hi = chi(zc), chi(zc - dsh / 2), chi(zc + dsh / 2)
                    n2d = n3_0 * c ** 2 * (hi - lo)
                    b_sv, _ = hb.bias_for_density(n3_lf0, zc)
                    b_sim = ss.sim_bias(key, zc, btab)
                    r = b_sv / b_sim if b_sim > 0 else np.nan
                    dw = chi(s["z_hi"][wi]) - chi(s["z_lo"][wi])
                    t = terms(s, lab, wi, r, ov * area0 / SQ_DEG, n2d, dw, hi - lo, fe.field_ell_min(area0), args.ell_max)
                    y.append(np.sqrt(t.sum()))
                a1.plot(s["z0"], y, ls, color=col, marker=mk, ms=3, lw=1.6 if ls == "-" else 1.1,
                        label=ename if ls == "-" else None)
            # ---- real shells: markers on the left, cumulative on the right
            for key, ls in ((args.lae_tracer, "-"), ("matter_waters", "--")):
                s = sp[key]
                tl, el, tot = [], [], 0.0
                for sh in sv.shells:
                    wi = ss.shell_window(s, sh.z)
                    if wi is None:
                        continue
                    area = sh.area_deg2 or sv.area_deg2
                    n3 = fe.lae_number_density(sh.L_min, sh.lf)
                    c, lo, hi = chi(sh.z), chi(sh.z - sh.dz / 2), chi(sh.z + sh.dz / 2)
                    n2d = sh.n_obs_deg2 / (np.pi / 180) ** 2 if sh.n_obs_deg2 else n3 * c ** 2 * (hi - lo)
                    b_sv, _ = hb.bias_for_density(n3, sh.z)
                    b_sim = ss.sim_bias(key, sh.z, btab)
                    r = b_sv / b_sim if b_sim > 0 else np.nan
                    dw = chi(s["z_hi"][wi]) - chi(s["z_lo"][wi])
                    lmin = fe.field_ell_min(area)
                    t = terms(s, lab, wi, r, ov * area / SQ_DEG, n2d, dw, hi - lo, lmin, args.ell_max)
                    tl.append(t); el.append(s["edges"][wi])
                    tot += t.sum()
                    if ls == "-":
                        a1.plot([sh.z], [np.sqrt(t.sum())], "*", color=col, ms=12, mec="k", mew=0.6, zorder=5)
                if tl:
                    cum = ss.cumulative_snr(tl, el, lmax_grid, 100.0)
                    a2.plot(lmax_grid, cum, ls, color=col, lw=1.6 if ls == "-" else 1.1,
                            label=f"{ename}: {np.sqrt(tot):.2g}" + ("" if ls == "-" else " (La Plante model)"))
        a1.set_yscale("log"); a2.set_yscale("log"); a2.set_xscale("log")
        for a in (a1, a2):
            a.axhline(1, color="0.6", lw=0.6, ls=":"); a.axhline(3, color="0.6", lw=0.6, ls="--")
            a.legend(fontsize=7, frameon=False, ncol=1)
        lo_y = min(a1.get_ylim()[0], a2.get_ylim()[0])
        a1.set_ylim(max(lo_y, 1e-4), 5); a2.set_ylim(max(lo_y, 1e-4), 5)
        if np.isfinite(zb).all():
            a1.axvspan(zb.max() + 0.25, z0.max() + 0.3, color="0.92", zorder=0)
            a1.text(zb.max() + 0.35, 2.5, "LAE bias held constant\n(beyond measured z)", fontsize=6.5, color="0.4", va="top")
        a1.set_xlabel(r"window centre $z_0$  ($\Delta z = 1$)"); a1.set_ylabel("S/N")
        a1.set_title(f"S/N if this survey sat at each z  (stars: its real shells)", fontsize=9)
        a2.set_xlabel(r"$\ell_{\max}$"); a2.set_ylabel("cumulative S/N (real shells)")
        a2.set_title("cumulative S/N vs $\\ell_{max}$", fontsize=9)
        dens = f"{sh0.n_obs_deg2:.0f}/deg$^2$ observed" if sh0.n_obs_deg2 else f"n(>L_min) from LF"
        fig.suptitle(f"{sv.name}: {area0:g} deg$^2$, {dens}   |   solid: simulated LAE signal ({args.lae_tracer}), "
                     f"dashed: La Plante galaxy model", fontsize=9.5)
        tag = sv.name.split(" (")[0].split(",")[0].replace(" ", "_").replace("/", "")
        if "LF as observed" in sv.name:
            tag += "_LF_observed"
        print("figure:", *save_fig(fig, os.path.join(args.out_dir, f"snr_curves_{tag}")))
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
