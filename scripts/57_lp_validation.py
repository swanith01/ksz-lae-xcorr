#!/usr/bin/env python3
"""
scripts/57_lp_validation.py
=============================
Like-for-like check of our simulation-based S/N against La Plante, Sipple & Lidz 2022 (ApJ 928, 162), Table 2 and Figs 9-10.

La Plante's setup (Sec. 4 of the paper):
  * galaxies = matter x Waters+16 bias (our 'matter_waters' run), Roman HLS, f_sky = 0.053, CMB covers the whole HLS field;
  * ONE redshift bin per number, (z0, dz) chosen to maximise the S/N, window floored at z = 6 (Table 2 note, Fig. 9);
  * S/N = Eq. 13 summed over l >= 100 (Fig. 10);
  * shot noise C_SN = 1/N per sr with N from W16 (BlueTides, 5 sigma, geometric mean of intrinsic and dust-corrected): we use
    the dotted lines of their own Fig. 8, vector-digitised (data/reference/la_plante_2022/fig8_shot_noise_digitized.csv),
    log-linear in z between z = 7 and 10 and extrapolated outside.
Our spectra exist for dz = 1 windows; a wider bin is built from consecutive non-overlapping dz = 1 windows (galaxy map = mean of
the unit maps, so Cs = mean of the unit Cs and, in the Limber approximation, Cg = sum of the unit Cg / N^2), on a common l grid.
The same is done with the SIGNAL of simulated LBGs, rescaled from their own measured large-scale bias to the Waters+16 bias
(r = b_W16 / b_sim; bins limited to z <= 10.5, where b_sim is measured), keeping La Plante's clustering and shot noise: does a real galaxy sample give the same kSZ^2 response per unit
bias as biased matter?

Outputs (--out-dir): lp_validation.csv, lp_validation.png/pdf (cumulative S/N vs l_max over La Plante Fig. 10, Table 2 bars,
S/N(z0, dz) for CMB-S4 like their Fig. 9).  Needs only the committed sim_spectra .npz and data/reference.

    python scripts/57_lp_validation.py --out-dir data/plots/2026-11-03_ksz_lae_snr/lp_validation
"""
import argparse
import csv
import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_lae_xcorr.snr import sim_snr as ss
from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.figio import save_fig

F_SKY = 0.053
REF = "data/reference/la_plante_2022"
LP_TABLE2 = {  # fiducial history: (ILC, instrument) with shot noise, (ILC, instrument) without
    "SO": {("ilc", True): 1.4, ("white", True): 1.7, ("ilc", False): 2.4, ("white", False): 2.9},
    "CMB-S4": {("ilc", True): 2.8, ("white", True): 11, ("ilc", False): 4.8, ("white", False): 18},
    "CMB-HD": {("ilc", True): 13, ("white", True): 35, ("ilc", False): 23, ("white", False): 64},
}
LABEL = {("SO", "ilc"): "SO (post-ILC)", ("SO", "white"): "SO (white noise)", ("CMB-S4", "ilc"): "CMB-S4 (post-ILC)",
         ("CMB-S4", "white"): "CMB-S4 (white noise)", ("CMB-HD", "ilc"): "CMB-HD (post-ILC)", ("CMB-HD", "white"): "CMB-HD (white noise)"}
COL = {"SO": "#56B4E9", "CMB-S4": "#E69F00", "CMB-HD": "#009E73"}
SQ = (np.pi / 180) ** 2


def lp_dndz():
    """La Plante's Roman HLS galaxies per deg^2 per unit z, from their Fig. 8 (dz = 1 windows): callable z -> N/deg2/dz."""
    with open(os.path.join(REF, "fig8_shot_noise_digitized.csv")) as f:
        r = list(csv.DictReader(f))
    z = np.array([float(x["z0"]) for x in r]); n = np.array([float(x["N_per_deg2"]) for x in r])
    p = np.polyfit(z, np.log10(n), 1)
    return lambda zz: 10 ** np.where((np.asarray(zz) >= z[0]) & (np.asarray(zz) <= z[-1]), np.interp(zz, z, np.log10(n)), np.polyval(p, zz))


def tilings(sp):
    """Consecutive non-overlapping dz=1 windows: lists of window indices for every start (6.0 or 6.5 grid) and length."""
    z0 = sp["z0"]; dz = sp["dz"]
    unit = [i for i in range(len(z0)) if abs(dz[i] - 1) < 1e-6]
    out = []
    for start in (6.5, 7.0):
        chain = [i for i in unit if abs(((z0[i] - start) % 1.0)) < 1e-6 or abs(((z0[i] - start) % 1.0) - 1) < 1e-6]
        chain.sort(key=lambda i: z0[i])
        for a in range(len(chain)):
            for b in range(a + 1, len(chain) + 1):
                out.append(chain[a:b])
    return out


def on_grid(sp, wi, arr, centres):
    """Interpolate a binned spectrum (n_bin,) or (n_seed, n_bin) of window wi onto common l bin centres (NaN outside)."""
    e = sp["edges"][wi]; lc = np.sqrt(e[1:] * e[:-1])
    arr = np.atleast_2d(arr)
    out = np.full((arr.shape[0], len(centres)), np.nan)
    for s in range(arr.shape[0]):
        ok = np.isfinite(arr[s])
        if ok.sum() > 1:
            v = np.interp(np.log(centres), np.log(lc[ok]), arr[s][ok], left=np.nan, right=np.nan)
            out[s] = v
    return out


def wide_window_terms(sp, wins, lab, edges, dndz, shot, r_of_z, sp_clust):
    """(S/N)^2 per common l bin for the wide window made of unit windows `wins`.  Signal from `sp` (rescaled by r), galaxy
    clustering from `sp_clust` (always the matter x W16 run: La Plante's clustering, and a sparse tracer's own clustering minus
    its shot noise is too noisy to divide by), shot noise from La Plante Fig. 8."""
    ei = [str(x) for x in sp["experiments"]].index(lab)
    centres = np.sqrt(edges[1:] * edges[:-1])
    n = len(wins)
    Cs = np.zeros((sp["Cs"].shape[0], len(centres))); Cg = np.zeros(len(centres)); N_sr = 0.0
    for wi in wins:
        zc = float(sp["z0"][wi]); r = r_of_z(zc)
        Cs += on_grid(sp, wi, sp["Cs"][:, ei, wi, :] * r, centres) / n
        wc = int(np.argmin(np.abs(sp_clust["z0"] - zc) + 10 * np.abs(sp_clust["dz"] - sp["dz"][wi])))
        clust = np.nanmean(sp_clust["Cg"][:, wc, :], axis=0)
        Cg += on_grid(sp_clust, wc, clust, centres)[0] / n ** 2
        N_sr += dndz(zc) * (sp["z_hi"][wi] - sp["z_lo"][wi]) / SQ
    if shot:
        Cg = Cg + 1.0 / N_sr
    return ss.snr_terms(ss.signal_sq(Cs, True), Cg, float(sp["CT2T2"][ei]), edges, F_SKY, 100.0, 10000.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--spectra-dir", default="data/products_tracers/sim_spectra")
    ap.add_argument("--out-dir", default="data/plots/lp_validation")
    ap.add_argument("--lbg", default="lbg_MUV-21.0,lbg_MUV-19.0")
    ap.add_argument("--bias-csv", default="data/plots/2026-10-16_tracer_suite/bias/tracer_bias_summary.csv")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    dndz = lp_dndz()
    btab = ss.load_bias_table(args.bias_csv)
    edges = np.geomspace(100, 10000, 49)
    tracers = ["matter_waters"] + [t for t in args.lbg.split(",") if t]
    sp = {}
    for t in tracers:
        z = np.load(os.path.join(args.spectra_dir, f"{t}.npz"), allow_pickle=False)
        sp[t] = {k: z[k] for k in z.files}

    def r_fn(t):
        if t == "matter_waters":
            return lambda z: 1.0
        return lambda z: float(bluetides_bias_gz(np.array([z]))[0]) / ss.sim_bias(t, z, btab)

    rows, best = [], {}
    for t in tracers:
        s = sp[t]
        tl = tilings(s)
        if t != "matter_waters":      # the LBG bias is measured at z = 6.0-10.1 only: no bins beyond z = 10.5
            tl = [w for w in tl if s["z_hi"][w[-1]] <= 10.5 + 1e-6]
        for exp in LP_TABLE2:
            for noise in ("ilc", "white"):
                for shot in (True, False):
                    lab = LABEL[(exp, noise)]
                    cand = []
                    for wins in tl:
                        tt = wide_window_terms(s, wins, lab, edges, dndz, shot, r_fn(t), sp["matter_waters"])
                        cand.append((float(np.sqrt(tt.sum())), wins, tt))
                    sn, wins, tt = max(cand, key=lambda c: c[0])
                    lo, hi = float(s["z_lo"][wins[0]]), float(s["z_hi"][wins[-1]])
                    lp = LP_TABLE2[exp][(noise, shot)]
                    best[(t, exp, noise, shot)] = (sn, wins, tt, cand)
                    rows.append({"tracer": t, "experiment": exp, "noise": "post-ILC" if noise == "ilc" else "instrument (white)",
                                 "shot_noise": "with" if shot else "without", "SN_ours": round(sn, 3), "SN_LaPlante_Table2": lp,
                                 "ratio_ours_over_LP": round(sn / lp, 2), "best_bin_z": f"{lo:g}-{hi:g}", "best_bin_z0": round(0.5 * (lo + hi), 2),
                                 "best_bin_dz": round(hi - lo, 2)})
    with open(os.path.join(args.out_dir, "lp_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'tracer':>14} {'exp':>7} {'noise':>18} {'shot':>7} | ours   La Plante  ratio  best bin")
    for r in rows:
        print(f"{r['tracer']:>14} {r['experiment']:>7} {r['noise']:>18} {r['shot_noise']:>7} | {r['SN_ours']:<6} {r['SN_LaPlante_Table2']:<10} {r['ratio_ours_over_LP']:<6} z={r['best_bin_z']}")

    # clustering check against La Plante Fig. 8 (dashed lines; read by eye at l = 1000: ~0.15, 0.19, 0.24, 0.27 for z0 = 7..10)
    s = sp["matter_waters"]
    print("\nmatter x Waters clustering, D_l(1000) = l(l+1)C_gg/2pi of the dz = 1 window (La Plante Fig. 8 dashed, by eye: 0.15 / 0.19 / 0.24 / 0.27):")
    for z0 in (7.0, 8.0, 9.0, 10.0):
        wi = int(np.argmin(np.abs(s["z0"] - z0)))
        c = on_grid(s, wi, np.nanmean(s["Cg"][:, wi, :], axis=0), np.array([1000.0]))[0, 0]
        print(f"   z0 = {z0:4.1f}: {1000 * 1001 * c / (2 * np.pi):.3f}")

    # where the gap comes from: our box has no modes below l ~ 260 (k_f chi); La Plante's own S/N above that l, from their Fig. 10
    lp10c = {}
    with open(os.path.join(REF, "fig10_cumulative_snr_fiducial_digitized.csv")) as f:
        for r in csv.DictReader(f):
            lp10c.setdefault(r["experiment"], []).append((float(r["ell_max"]), float(r["cumulative_SN"])))
    print("\npost-ILC, with shot noise: La Plante's S/N from l > l_min of our box (quadrature remainder of their Fig. 10) vs ours")
    for exp, v in lp10c.items():
        x, y = np.array(sorted(v)).T
        sn, wins, tt, _ = best[("matter_waters", exp, "ilc", True)]
        lmin = float(edges[:-1][np.argmax(tt > 0)])
        rem = np.sqrt(max(y[-1] ** 2 - np.interp(lmin, x, y) ** 2, 0.0))
        print(f"   {exp:>7}: our first l with signal {lmin:.0f};  La Plante total {y[-1]:.2f}, of which below that l {np.interp(lmin, x, y):.2f}, "
              f"above {rem:.2f};  ours {sn:.2f}  (ratio above {sn / rem:.2f})")

    # ---- figure ---------------------------------------------------------------------------------------------------------
    lp10 = {}
    with open(os.path.join(REF, "fig10_cumulative_snr_fiducial_digitized.csv")) as f:
        for r in csv.DictReader(f):
            lp10.setdefault(r["experiment"], []).append((float(r["ell_max"]), float(r["cumulative_SN"])))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    lgrid = edges[1:]
    for exp in LP_TABLE2:
        sn, wins, tt, _ = best[("matter_waters", exp, "ilc", True)]
        a1.plot(lgrid, np.sqrt(np.cumsum(tt)), color=COL[exp], lw=2, label=f"{exp}: ours {sn:.2g}")
        if exp in lp10:
            x, y = np.array(sorted(lp10[exp])).T
            a1.plot(x, y, color=COL[exp], ls=":", lw=1.6, label=f"{exp}: La Plante Fig. 10 ({LP_TABLE2[exp][('ilc', True)]:g})")
        for t, ls in zip(tracers[1:], ("--", "-.")):
            sn2, _, tt2, _ = best[(t, exp, "ilc", True)]
            a1.plot(lgrid, np.sqrt(np.cumsum(tt2)), color=COL[exp], ls=ls, lw=1, alpha=0.8, label=f"   {t} (bias -> W16): {sn2:.2g}" if exp == "CMB-HD" else None)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(100, 10000); a1.set_ylim(0.05, 40)
    a1.axhline(1, color="0.6", lw=0.6, ls=":"); a1.axhline(3, color="0.6", lw=0.6, ls="--")
    a1.set_xlabel(r"$\ell_{\max}$"); a1.set_ylabel("cumulative S/N (best single bin)")
    a1.set_title("post-ILC noise, with shot noise: solid ours (matter x W16 bias), dotted La Plante", fontsize=8.5)
    a1.legend(fontsize=6.5, frameon=False, loc="upper left")

    # Table 2 bars
    cases = [(n, s_) for n in ("ilc", "white") for s_ in (True, False)]
    xl = [f"{'ILC' if n == 'ilc' else 'instr.'}\n{'with SN' if s_ else 'no SN'}" for n, s_ in cases]
    wbar = 0.13
    for k, exp in enumerate(LP_TABLE2):
        x = np.arange(len(cases)) + (k - 1) * 2.2 * wbar
        a2.bar(x - wbar / 2, [LP_TABLE2[exp][c] for c in cases], wbar, color=COL[exp], alpha=0.45, edgecolor=COL[exp], label=f"{exp} La Plante")
        a2.bar(x + wbar / 2, [best[("matter_waters", exp) + c][0] for c in cases], wbar, color=COL[exp], label=f"{exp} ours")
    a2.set_yscale("log"); a2.set_xticks(range(len(cases)), xl, fontsize=8); a2.set_ylabel("S/N (best single bin)")
    a2.set_title("La Plante Table 2 (light) vs ours (dark), fiducial history", fontsize=8.5)
    a2.legend(fontsize=6.5, frameon=False, ncol=2)

    # S/N(z0, dz) for CMB-S4, ILC, with shot noise (their Fig. 9)
    _, _, _, cand = best[("matter_waters", "CMB-S4", "ilc", True)]
    s = sp["matter_waters"]
    zc = [0.5 * (s["z_lo"][w[0]] + s["z_hi"][w[-1]]) for _, w, _ in cand]
    dzs = [s["z_hi"][w[-1]] - s["z_lo"][w[0]] for _, w, _ in cand]
    sc = a3.scatter(zc, dzs, c=[c[0] for c in cand], cmap="viridis", s=55, marker="s")
    i = int(np.argmax([c[0] for c in cand]))
    a3.plot(zc[i], dzs[i], "o", mfc="none", mec="k", ms=12, label=f"ours: max {cand[i][0]:.2g} at z0={zc[i]:.1f}, dz={dzs[i]:.0f}")
    a3.plot(9.2, 4.2, "x", color="#D55E00", ms=10, mew=2, label="La Plante Fig. 9: max 2.8 at z0~9.2, dz~4.2")
    fig.colorbar(sc, ax=a3, label="S/N")
    a3.set_xlabel("bin centre $z_0$"); a3.set_ylabel(r"bin width $\Delta z$"); a3.legend(fontsize=7, frameon=False, loc="upper right")
    a3.set_title("CMB-S4 post-ILC with shot noise: S/N per single bin", fontsize=8.5)
    fig.suptitle("Validation against La Plante+22: same galaxy model (matter x Waters+16 bias), Roman HLS f_sky = 0.053, "
                 "their Fig. 8 shot noise, one bin maximising S/N", fontsize=10)
    print("\nfigure:", *save_fig(fig, os.path.join(args.out_dir, "lp_validation")))
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
