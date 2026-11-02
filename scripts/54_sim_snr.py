#!/usr/bin/env python3
"""
scripts/54_sim_snr.py
=======================
Step 2 of the SIMULATION-based S/N (snr/sim_snr.py): La Plante+22 Eq. 13 on the binned spectra of scripts/53.
Desktop, seconds.  The old exploratory notebook's Figs. 4 and 5, on the 300 Mpc box with physical field units.

  box mode    : Cs and Cg straight from the simulation (the tracer's own clustering AND its own shot noise), one f_sky for
                every window (default 0.053 = Roman HLS, as in La Plante Table 2 and the old Fig. 4).
  survey mode : each LAE survey shell of snr.feasibility.SURVEYS: the simulated window that contains the shell, f_sky = survey
                area, the survey's shot noise, and the simulated clustering and signal rescaled to the survey's bias
                (ST abundance matching, as in scripts/52):  r = b_survey / b_sim,  Cs -> r Cs,
                Cg = r^2 (Cg_sim - N_sim) dchi_win/dchi_shell + 1/n_2D.   --bias-rescale none keeps r = 1.
Cs^2 is reported raw (mean over seeds, squared) and debiased (mean^2 - sem^2); quote the debiased one.

Outputs in --out-dir:
  sim_snr_windows.csv     per tracer, experiment, window: x_HII, S/N raw and debiased (box mode)
  sim_snr_totals.csv      per tracer, experiment: total over non-overlapping windows (box mode), raw and debiased
  sim_snr_survey.csv      per signal tracer, survey shell, experiment (survey mode), with the scripts/52 number next to it
  fig4_<tracer>.png/pdf   S/N vs z0 and cumulative S/N vs l_max (SO / CMB-S4 / CMB-HD post-ILC), debiased solid, raw dashed
  sim_snr_totals.png/pdf  total S/N, every tracer x experiment (debiased)

    python scripts/54_sim_snr.py --spectra-dir data/products_tracers/sim_spectra --out-dir data/plots/2026-11-03_sim_snr \\
        --feasibility-csv data/plots/2026-11-02_feasibility/feasibility_snr.csv
"""
import argparse
import csv
import glob
import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.snr import feasibility as fe
from ksz_lae_xcorr.snr import sim_snr as ss
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.figio import save_fig

SQ_DEG = 41252.96
FIG_EXPS = (("SO (post-ILC)", "#56B4E9"), ("CMB-S4 (post-ILC)", "#E69F00"), ("CMB-HD (post-ILC)", "#009E73"))


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def load_spectra(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p, allow_pickle=False)
        out[str(z["name"])] = {k: z[k] for k in z.files}
    return out


def window_terms(sp, ei, wi, f_sky, debias, ell_min, ell_max, Cs_scale=1.0, Cg=None):
    Cs2 = ss.signal_sq(sp["Cs"][:, ei, wi, :] * Cs_scale, debias)
    Cg = np.nanmean(sp["Cg"][:, wi, :], axis=0) if Cg is None else Cg
    return ss.snr_terms(Cs2, Cg, float(sp["CT2T2"][ei]), sp["edges"][wi], f_sky, ell_min, ell_max)


def load_bias_table(path):
    """{tracer name: (z array, b array)} from tracer_bias_summary.csv (scripts/36)."""
    if not path or not os.path.exists(path):
        return {}
    tab = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            tab.setdefault(r["name"], []).append((float(r["z"]), float(r["b"])))
    return {k: (np.array([a for a, _ in sorted(v)]), np.array([b for _, b in sorted(v)])) for k, v in tab.items()}


def sim_bias(name, z, btab):
    """Large-scale bias of the simulated tracer at z: Waters+16 for the matter run, else the measured one (scripts/35-36),
    linearly interpolated in z and held constant outside the measured range (flagged)."""
    if name.startswith("matter"):
        return float(bluetides_bias_gz(np.array([z]))[0]), "Waters+16"
    if name not in btab:
        return np.nan, "no bias"
    zz, bb = btab[name]
    flag = "measured" if zz[0] - 0.05 <= z <= zz[-1] + 0.05 else "measured, held const outside z range"
    return float(np.interp(z, zz, bb)), flag


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--spectra-dir", default="data/products_tracers/sim_spectra")
    ap.add_argument("--out-dir", default="data/plots/sim_snr")
    ap.add_argument("--f-sky", type=float, default=0.053, help="box mode f_sky (default Roman HLS 0.053)")
    ap.add_argument("--ell-min", type=float, default=100.0)
    ap.add_argument("--ell-max", type=float, default=10000.0)
    ap.add_argument("--bias-csv", default="data/plots/2026-10-16_tracer_suite/bias/tracer_bias_summary.csv")
    ap.add_argument("--bias-rescale", default="st", choices=("st", "none"))
    ap.add_argument("--feasibility-csv", default=None, help="scripts/52 feasibility_snr.csv, to print the toy number next to ours")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # empty l-bins are NaN by design
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    spectra = load_spectra(args.spectra_dir)
    if not spectra:
        sys.exit(f"no .npz in {args.spectra_dir} (run scripts/53 on swarm first)")
    cosmo = get_cosmology(cfg)
    chi = lambda z: float(cosmo.comoving_distance(z).to_value("Mpc"))

    # ------------------------------------------------------------------ box mode
    rows_w, rows_t = [], []
    lmax_grid = np.geomspace(max(args.ell_min, 150), args.ell_max, 80)
    for name, sp in spectra.items():
        exps = [str(e) for e in sp["experiments"]]
        sel = ss.non_overlapping(sp["z0"], sp["dz"])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ei, lab in enumerate(exps):
            tot = {}
            for debias in (True, False):
                terms = [window_terms(sp, ei, wi, args.f_sky, debias, args.ell_min, args.ell_max) for wi in range(len(sp["z0"]))]
                sn_w = np.sqrt([t.sum() for t in terms])
                tot[debias] = float(np.sqrt(sum(terms[i].sum() for i in sel)))
                if debias:
                    for wi in range(len(sp["z0"])):
                        rows_w.append({"tracer": name, "cmb": lab, "z0": sp["z0"][wi], "dz": sp["dz"][wi], "x_hii": round(float(sp["x_hii"][wi]), 3),
                                       "SN_debiased": round(float(sn_w[wi]), 4),
                                       "SN_raw": round(float(np.sqrt(window_terms(sp, ei, wi, args.f_sky, False, args.ell_min, args.ell_max).sum())), 4)})
                for flab, col in FIG_EXPS:
                    if lab != flab:
                        continue
                    ls = "-" if debias else "--"
                    a1.plot(sp["z0"], sn_w, ls, color=col, marker="o" if debias else None, ms=3,
                            label=f"{lab}: total {tot[debias]:.3g}" + ("" if debias else " (raw)"))
                    cum = ss.cumulative_snr([terms[i] for i in sel], [sp["edges"][i] for i in sel], lmax_grid, args.ell_min)
                    a2.plot(lmax_grid, cum, ls, color=col, label=flab + ("" if debias else " (raw)"))
            rows_t.append({"tracer": name, "cmb": lab, "f_sky": args.f_sky, "windows": " ".join(f"{z:g}" for z in sp["z0"][sel]),
                           "SN_total_debiased": round(tot[True], 4), "SN_total_raw": round(tot[False], 4)})
        a1.set_xlabel(r"$z_0$  (window $\Delta z$ = %g)" % sp["dz"][0]); a1.set_ylabel("S/N per window")
        a1.set_title(f"{name}: S/N vs z  (f_sky = {args.f_sky:g}, box shot noise)", fontsize=9)
        a2.set_xscale("log"); a2.set_xlabel(r"$\ell_{\max}$"); a2.set_ylabel("cumulative S/N (non-overlapping windows)")
        a2.set_title("cumulative S/N vs $\\ell_{max}$", fontsize=9)
        for a in (a1, a2):
            a.axhline(1, color="0.6", lw=0.6, ls=":"); a.axhline(3, color="0.6", lw=0.6, ls="--")
            a.legend(fontsize=7, frameon=False)
        save_fig(fig, os.path.join(args.out_dir, f"fig4_{name}"))
        plt.close(fig)
    write_csv(os.path.join(args.out_dir, "sim_snr_windows.csv"), rows_w)
    write_csv(os.path.join(args.out_dir, "sim_snr_totals.csv"), rows_t)

    # amplitude check against scripts/28 (lp_cross_vs_z0_SO_wo0.csv): seed-mean D_l of the SO post-ILC cross at l = 500/1000/3000
    rows_d = []
    for name, sp in spectra.items():
        exps = [str(e) for e in sp["experiments"]]
        if "SO (post-ILC)" not in exps:
            continue
        ei = exps.index("SO (post-ILC)")
        for wi in range(len(sp["z0"])):
            e = sp["edges"][wi]
            lc = np.sqrt(e[1:] * e[:-1])
            C = np.nanmean(sp["Cs"][:, ei, wi, :], axis=0)
            D = lc * (lc + 1) * C / (2 * np.pi)
            ok = np.isfinite(D)
            row = {"tracer": name, "z0": sp["z0"][wi], "x_hii": round(float(sp["x_hii"][wi]), 3)}
            for L in (500, 1000, 3000):
                row[f"D_l{L}_uK2"] = float(np.interp(np.log(L), np.log(lc[ok]), D[ok])) if ok.sum() > 1 else np.nan
            rows_d.append(row)
    write_csv(os.path.join(args.out_dir, "sim_dell_check_SO.csv"), rows_d)

    print(f"box mode (f_sky = {args.f_sky}, simulated shot noise; total over non-overlapping windows), S/N debiased | raw:")
    names = list(spectra)
    exps = [str(e) for e in spectra[names[0]]["experiments"]]
    print(f"{'tracer':>18} | " + " | ".join(f"{e[:18]:>18}" for e in exps))
    grid = np.full((len(names), len(exps)), np.nan)
    for i, n in enumerate(names):
        cells = []
        for j, e in enumerate(exps):
            r = [x for x in rows_t if x["tracer"] == n and x["cmb"] == e]
            if r:
                grid[i, j] = r[0]["SN_total_debiased"]
                cells.append(f"{r[0]['SN_total_debiased']:8.3g} |{r[0]['SN_total_raw']:7.3g}")
            else:
                cells.append(" " * 18)
        print(f"{n:>18} | " + " | ".join(cells))

    fig, ax = plt.subplots(figsize=(1.2 * len(exps) + 3, 0.35 * len(names) + 1.8), constrained_layout=True)
    im = ax.imshow(np.log10(np.maximum(grid, 1e-3)), cmap="viridis", aspect="auto")
    for i in range(len(names)):
        for j in range(len(exps)):
            ax.text(j, i, f"{grid[i, j]:.2g}", ha="center", va="center", fontsize=7, color="w" if grid[i, j] < 3 else "k")
    ax.set_xticks(range(len(exps)), exps, rotation=35, ha="right", fontsize=7); ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title(f"simulation S/N, f_sky = {args.f_sky:g}, box shot noise (debiased)", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10 S/N")
    save_fig(fig, os.path.join(args.out_dir, "sim_snr_totals"))
    plt.close(fig)

    # ------------------------------------------------------------------ survey mode
    btab = load_bias_table(args.bias_csv)
    hb = fe.HaloBias(cfg)
    toy52 = {}
    if args.feasibility_csv and os.path.exists(args.feasibility_csv):
        with open(args.feasibility_csv) as f:
            for r in csv.DictReader(f):
                if r["history"] == "fiducial" and r["bias"].startswith("b = ST"):
                    toy52[(r["survey"], r["cmb"])] = float(r["SN_with_shot_noise"])
    rows_s = []
    for sname, sv in fe.SURVEYS.items():
        for sh in sv.shells:
            area = sh.area_deg2 or sv.area_deg2
            n3 = fe.lae_number_density(sh.L_min, sh.lf)
            c, lo, hi = chi(sh.z), chi(sh.z - sh.dz / 2), chi(sh.z + sh.dz / 2)
            n2d = n3 * c ** 2 * (hi - lo)
            b_sv, _ = hb.bias_for_density(n3, sh.z)
            for name, sp in spectra.items():
                inside = [i for i in range(len(sp["z0"])) if sp["z_lo"][i] <= sh.z <= sp["z_hi"][i]]
                if not inside:
                    rows_s.append({"signal_tracer": name, "survey": sname, "shell_z": sh.z, "cmb": "-", "note": "no simulated window contains this z"})
                    continue
                wi = min(inside, key=lambda i: abs(sp["z0"][i] - sh.z))
                b_sim, bflag = sim_bias(name, sh.z, btab)
                r = b_sv / b_sim if (args.bias_rescale == "st" and np.isfinite(b_sim) and b_sim > 0) else 1.0
                dchi_w = chi(sp["z_hi"][wi]) - chi(sp["z_lo"][wi])
                Cg = ss.survey_cg(np.nanmean(sp["Cg"][:, wi, :], axis=0), float(sp["shot_sim"][wi]), r, dchi_w, hi - lo, n2d)
                for ei, lab in enumerate(str(e) for e in sp["experiments"]):
                    sn = {d: float(np.sqrt(window_terms(sp, ei, wi, area / SQ_DEG, d, args.ell_min, args.ell_max, Cs_scale=r, Cg=Cg).sum()))
                          for d in (True, False)}
                    rows_s.append({"signal_tracer": name, "survey": sname, "shell_z": sh.z, "window_z0": sp["z0"][wi], "cmb": lab,
                                   "area_deg2": area, "n2d_per_deg2": round(n2d * (np.pi / 180) ** 2, 2),
                                   "b_survey": round(b_sv, 2), "b_sim": round(b_sim, 2) if np.isfinite(b_sim) else "nan", "b_sim_source": bflag,
                                   "r": round(r, 3), "SN_debiased": round(sn[True], 4), "SN_raw": round(sn[False], 4),
                                   "SN_script52_toy_all_shells": toy52.get((sname, lab), "")})
    write_csv(os.path.join(args.out_dir, "sim_snr_survey.csv"), rows_s)
    print(f"\nsurvey mode (bias rescale: {args.bias_rescale}); post-ILC experiments, S/N debiased (script 52 toy, all shells):")
    for r in rows_s:
        if r["cmb"] in ("CMB-HD (post-ILC)", "CMB-S4 (post-ILC)", "SO (post-ILC)") or r["cmb"] == "-":
            if r["cmb"] == "-":
                print(f"  {r['signal_tracer']:>16} {r['survey'][:28]:>28} z={r['shell_z']}: {r['note']}")
            else:
                print(f"  {r['signal_tracer']:>16} {r['survey'][:28]:>28} z={r['shell_z']} win {r['window_z0']:.1f} {r['cmb']:>18}: "
                      f"{r['SN_debiased']:.3g}  (raw {r['SN_raw']:.3g}; toy {r['SN_script52_toy_all_shells']}) r={r['r']} [{r['b_sim_source']}]")
    print(f"\noutputs in {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
