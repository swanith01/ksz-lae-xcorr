#!/usr/bin/env python3
"""
scripts/51_pedagogy_gif.py
============================
A teaching animation of the kSZ^2 x tracer cross-correlation (default tracer: the LAE cut), in the spirit of Guo et al. (2026) Fig. 9,
which shows the 21-cm slice at the epoch where their kSZ^2 x 21cm^2 cross-correlation peaks.

Three panels, one GIF:
  left   the filtered kSZ^2 map of the whole reionisation lightcone (the map the estimator cross-correlates; the same in every frame)
  middle the tracer map of one redshift window (dz = 1), stepping from high z (little ionisation) to low z, labelled with z, x_HII and
         the window's D_ell^cross(ell)
  right  D_ell^cross(x_HII) of this seed (all windows), with a marker on the window shown in the other two panels, in sync with the animation
         (windows already shown are filled, the peak window is starred; --paper adds La Plante+22 Fig. 7 for matter x Waters)
Both maps are smoothed at the soft-mode scale chi/ell and shown as fractional fluctuations (red = above the mean).
An arrow (on both panels, same pixel) marks where the LOCAL product  (kSZ^2 fluctuation) x (tracer overdensity)  is largest in that window:
the product of the two smoothed maps, normalised so that its map mean is the Pearson correlation of the two maps; D_ell^cross is the band-limited
version of that mean, so the arrow is the patch that contributes most to it.  In the window where D_ell^cross(x_HII) of this seed peaks
(or --peak-z), the arrow gets large and flashes, with a banner.

    python scripts/51_pedagogy_gif.py --seed 1 --cltt-file <ell,Cl_uK2.csv> --out-dir data/plots/2026-10-28_gif
    python scripts/51_pedagogy_gif.py --chunks data/products_tracers/chunks/seed1_wo0_chunks.pkl --tracer lae_L42.5   # from scripts/48 instead

Inputs (default mode): the kSZ products of >= 2 seeds (scripts/26 --save-maps, for the filter) and the tracer maps of --seed (scripts/37), as
scripts/44/48.  Light (a minute); run it on usha after copying the pickles.   Note one seed is noisy: the flashing window is that seed's peak.
Outputs (--out-dir): cross_correlation_<tracer>.gif and cross_correlation_<tracer>_windows.csv (z0, x_hii, D_ell, Pearson r, arrow pixel).
"""
import argparse
import csv
import os
import pickle
import sys

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ksz_lae_xcorr.correlation import ksz_components as kc
from ksz_lae_xcorr.correlation import lp_maps
from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.utils.config import load_config

TRACER_LABELS = {"lae_L42.5": "LAE  $L_{\\rm Ly\\alpha}>10^{42.5}$ erg/s", "halo_M10.0": "halo  $M>10^{10}M_\\odot$",
                 "lbg_MUV-19.0": "LBG  $M_{\\rm UV}<-19$", "matter_Waters": "matter $\\times$ Waters+16 bias"}


def smooth_zero_mean(m, sigma_pix):
    """Periodic Gaussian smoothing, mean removed."""
    s = gaussian_filter(np.asarray(m, float), sigma_pix, mode="wrap")
    return s - s.mean()


def kSZ2_fraction(S, sigma_pix):
    S = np.asarray(S, float)
    return smooth_zero_mean(S / S.mean() - 1.0, sigma_pix)


def local_product(a, b):
    """a*b / (std a std b): its map mean is the Pearson correlation of a and b."""
    a, b = a - a.mean(), b - b.mean()
    sa, sb = a.std(), b.std()
    return a * b / (sa * sb) if sa > 0 and sb > 0 else np.zeros_like(a)


def arrow_pixel(prod, sigma_pix):
    """(row, col) of the largest local product after a further smoothing of the same width (so one noisy pixel cannot win)."""
    sm = gaussian_filter(prod, sigma_pix, mode="wrap")
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(sm)), sm.shape))


def d_ell_at(S, g, kg, chi_eff, ell):
    c = lp_maps.lp_cross_dell(S, g, kg, chi_eff)
    return float(lp_maps.interp_at_ell(c["ell"], c["D_ell"], [ell])[0])


def pick_peak(x_hii, D, xrange=(0.05, 0.9)):
    """Index of the largest D among windows with x_HII in xrange (None if none)."""
    x, D = np.asarray(x_hii, float), np.asarray(D, float)
    ok = np.isfinite(D) & (x >= xrange[0]) & (x <= xrange[1])
    return int(np.flatnonzero(ok)[np.argmax(D[ok])]) if ok.any() else None


def load_inputs(args, cfg):
    """-> kg, S (filtered, squared, whole lightcone), chi_ref, windows [{z0, x_hii, chi_eff, g}] (ascending z0), tracer label."""
    kg = KGrid(cfg)
    if args.chunks:
        with open(args.chunks, "rb") as f:
            d = pickle.load(f)
        if args.tracer not in d["gal"]:
            sys.exit(f"tracer {args.tracer} not in {args.chunks}; has {sorted(d['gal'])}")
        wins = [{"z0": w["z0"], "x_hii": w["x_hii"], "chi_eff": w["chi_eff"], "g": np.asarray(d["gal"][args.tracer][i], float)}
                for i, w in enumerate(d["windows"])]
        return kg, np.asarray(d["S_global"], float), float(d["chi_ref"]), sorted(wins, key=lambda w: w["z0"])
    in_dir = args.ksz_dir or os.path.join(cfg.paths.products_root, "ksz_auto_wrapcycle")
    prods = kc.load_lpmaps_products(in_dir, args.wrap_offset)
    if len(prods) < 2 or args.seed not in prods:
        sys.exit(f"need >= 2 seeds incl. seed {args.seed} with _lpmaps.pkl in {in_dir}; found {sorted(prods)}")
    kg, filt, chi_ref = kc.build_filter_from_products(cfg, prods, kc.cl_tt_provider(cfg, args.cltt_file), args.experiment)
    prod = prods[args.seed]
    S = lp_maps.filter_and_square(prod["kSZ_map"], filt["ell_grid"], filt["fl"], kg, chi_ref)
    if args.tracer == "matter_Waters":
        gm = {(w["dz"], w["z0"]): w["gal_map"] for w in prod["windows"]}
    else:
        p = os.path.join(args.maps_root, args.tracer, f"seed{args.seed}_wo{args.wrap_offset}_lpmaps.pkl")
        if not os.path.exists(p):
            sys.exit(f"no tracer maps: {p}")
        with open(p, "rb") as f:
            gm = {(w["dz"], w["z0"]): w["gal_map"] for w in pickle.load(f)["windows"]}
    wins = [{"z0": w["z0"], "x_hii": w["x_hii"], "chi_eff": w["chi_eff"], "g": np.asarray(gm[(w["dz"], w["z0"])], float)}
            for w in prod["windows"] if abs(w["dz"] - 1.0) < 1e-9 and (w["dz"], w["z0"]) in gm]
    return kg, np.asarray(S, float), float(chi_ref), sorted(wins, key=lambda w: w["z0"])


def _vmax(img, q=98):
    return float(np.percentile(np.abs(img), q)) or 1.0


def draw_curve(ax, curve, ell, flash_on):
    """Right panel: D_ell(x_HII) of every window, the shown window highlighted."""
    x, D, k, ip = curve["x"], curve["D"], curve["k"], curve["ip"]
    o = np.argsort(x)
    if curve.get("paper") is not None:
        ax.plot(*curve["paper"], color="k", ls="--", lw=1.6, label="La Plante+22 Fig. 7 (matter)")
    ax.plot(x[o], D[o], color="0.7", lw=1.3, zorder=1)
    shown = np.zeros(len(x), bool)
    shown[curve["shown"]] = True
    ax.plot(x[~shown], D[~shown], "o", mfc="w", mec="0.5", ms=6, zorder=2)
    ax.plot(x[shown], D[shown], "o", color="#0072B2", ms=6, zorder=3)
    if ip is not None:
        ax.plot(x[ip], D[ip], "*", color="#FFD400", mec="k", mew=1.0, ms=17 if shown[ip] else 12, zorder=4)
    ax.axvline(x[k], color="#D55E00", lw=1.5, zorder=0)
    ax.plot(x[k], D[k], "o", color="#D55E00", mec="k", ms=11, zorder=5)
    if ip == k and flash_on:
        ax.plot(x[k], D[k], "o", mfc="none", mec="#FFD400", mew=3.5, ms=24, zorder=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, max(0.95, float(np.nanmax(x)) + 0.03))
    ax.set_xlabel(r"$x_{\rm HII}$  (reionisation progresses $\rightarrow$)")
    ax.set_ylabel(rf"$D_{{{ell:g}}}^{{\rm cross}}\ [\mu{{\rm K}}^2]$")
    ax.set_title("cross-correlation vs ionised fraction", fontsize=11)
    ax.text(0.97, 0.97, f"$z$ = {curve['z0'][k] - 0.5:.1f}-{curve['z0'][k] + 0.5:.1f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))
    if curve.get("paper") is not None:
        ax.legend(fontsize=7, loc="lower right", frameon=False)


def render_frame(Ssm, gsm, win, rec, cell, tracer, ell, is_peak, flash_on, curve=None, size=None, dpi=80):
    """One RGB frame (uint8 array)."""
    size = size or ((15.5, 5.6) if curve is not None else (10.0, 5.6))
    if curve is not None:
        fig, axes3 = plt.subplots(1, 3, figsize=size, dpi=dpi, constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.05]})
        axes, axc = axes3[:2], axes3[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=size, dpi=dpi, constrained_layout=True)
        axc = None
    n = Ssm.shape[0]
    r0, c0 = rec["arrow_pix"]
    v = np.array([0.5 * n - c0, 0.5 * n - r0], float)                     # arrows come in from the side of the map centre
    v = v / (np.hypot(*v) or 1.0)
    for ax, img, title in ((axes[0], Ssm, "filtered kSZ$^2$ map of the reionisation lightcone"),
                           (axes[1], gsm, f"{TRACER_LABELS.get(tracer, tracer)} map,  $z$ = {win['z0'] - 0.5:.1f}-{win['z0'] + 0.5:.1f}")):
        ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-_vmax(img), vmax=_vmax(img), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
        if is_peak:
            if flash_on:
                ax.add_patch(plt.Circle((c0, r0), 0.075 * n, fill=False, ec="#FFD400", lw=3.5))
                ax.add_patch(plt.Circle((c0, r0), 0.075 * n, fill=False, ec="k", lw=1.0))
                ax.annotate("", xy=(c0 + 0.085 * n * v[0], r0 + 0.085 * n * v[1]), xytext=(c0 + 0.26 * n * v[0], r0 + 0.26 * n * v[1]),
                            arrowprops=dict(arrowstyle="simple,head_width=1.6,head_length=1.4,tail_width=0.6", fc="#FFD400", ec="k", lw=1.6, mutation_scale=22))
        else:
            ax.annotate("", xy=(c0 + 0.025 * n * v[0], r0 + 0.025 * n * v[1]), xytext=(c0 + 0.13 * n * v[0], r0 + 0.13 * n * v[1]),
                        arrowprops=dict(arrowstyle="simple,head_width=1.4,head_length=1.2,tail_width=0.45", fc="w", ec="k", lw=1.2, mutation_scale=14))
        if ax is axes[0]:
            ax.plot([0.06 * n, 0.06 * n + 100.0 / cell], [0.92 * n] * 2, color="k", lw=3)
            ax.text(0.06 * n, 0.935 * n, "100 Mpc", fontsize=8, color="k", va="bottom")
    d = rec["D"]
    axes[1].text(0.02, 0.02, f"$x_{{\\rm HII}}$ = {win['x_hii']:.2f}\n$D_{{{ell:g}}}$ = {d:.3g} $\\mu$K$^2$\nr = {rec['r']:+.2f}",
                 transform=axes[1].transAxes, va="bottom", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    if is_peak:
        axes[1].text(0.5, 0.97, "cross-correlation peaks here", transform=axes[1].transAxes, ha="center", va="top", fontsize=12,
                     weight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#FFD400", ec="k", alpha=0.95))
    if axc is not None:
        draw_curve(axc, curve, ell, flash_on)
    fig.suptitle(f"red = above the mean, blue = below (smoothed at $\\ell$ = {ell:g}).   Arrow: where (kSZ$^2$ fluctuation) $\\times$ "
                 f"(tracer overdensity) is largest in this window", fontsize=9)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def save_gif(frames, durations_ms, path):
    ims = [Image.fromarray(f).quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=durations_ms, loop=0, optimize=False, disposal=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wrap-offset", type=int, default=0)
    ap.add_argument("--tracer", default="lae_L42.5")
    ap.add_argument("--ell", type=float, default=1000.0)
    ap.add_argument("--ksz-dir", default=None, help="kSZ products (default <products_root>/ksz_auto_wrapcycle)")
    ap.add_argument("--maps-root", default="data/products_tracers/maps")
    ap.add_argument("--chunks", default=None, help="use scripts/48's pickle instead (6 windows)")
    ap.add_argument("--experiment", default="SO")
    ap.add_argument("--cltt-file", default=None)
    ap.add_argument("--peak-z", type=float, default=None, help="z0 of the window to flash (default: this seed's D_ell peak)")
    ap.add_argument("--xrange", type=float, nargs=2, default=(0.05, 0.9), help="x_HII range searched for the peak")
    ap.add_argument("--zmax", type=float, default=None, help="drop windows above this z0")
    ap.add_argument("--no-curve", action="store_true", help="two panels only (no D_ell(x_HII) panel)")
    ap.add_argument("--paper", action="store_true", help="overplot La Plante+22 Fig. 7 (fiducial, matter x Waters+16) on the curve panel")
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--flashes", type=int, default=4)
    ap.add_argument("--out-dir", default="data/plots/gif")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cell = float(cfg.box.box_len_mpc) / int(cfg.box.hii_dim)
    kg, S, chi_ref, wins = load_inputs(args, cfg)
    if args.zmax is not None:
        wins = [w for w in wins if w["z0"] <= args.zmax]
    sg = (chi_ref / args.ell) / cell
    Ssm = kSZ2_fraction(S, sg)
    recs = []
    for w in wins:
        gsm = smooth_zero_mean(w["g"], sg)
        prod = local_product(Ssm, gsm)
        recs.append({"gsm": gsm, "r": float(prod.mean()), "arrow_pix": arrow_pixel(prod, sg),
                     "D": d_ell_at(S, w["g"], kg, w["chi_eff"], args.ell)})
    if args.peak_z is not None:
        ip = int(np.argmin([abs(w["z0"] - args.peak_z) for w in wins]))
    else:
        ip = pick_peak([w["x_hii"] for w in wins], [r["D"] for r in recs], tuple(args.xrange))
    print(f"{len(wins)} windows, z0 {wins[0]['z0']}..{wins[-1]['z0']}; peak window: "
          + (f"z0={wins[ip]['z0']} x_HII={wins[ip]['x_hii']:.2f} D={recs[ip]['D']:.4f}" if ip is not None else "none"))
    order = list(range(len(wins)))[::-1]                       # high z -> low z: x_HII grows
    xs = np.array([w["x_hii"] for w in wins], float)
    Ds = np.array([r["D"] for r in recs], float)
    paper = None
    if args.paper:
        from ksz_lae_xcorr.io.la_plante_reference import load_fig7_vs_xhii
        paper = load_fig7_vs_xhii().get(int(round(args.ell)))

    def curve_for(i):
        if args.no_curve:
            return None
        shown = [j for j in order[:order.index(i) + 1]]
        return {"x": xs, "D": Ds, "k": i, "ip": ip, "shown": shown, "z0": [w["z0"] for w in wins], "paper": paper}

    frames, dur = [], []
    base = int(round(1000 / args.fps))
    for i in order:
        if i == ip:
            for k in range(2 * args.flashes):
                frames.append(render_frame(Ssm, recs[i]["gsm"], wins[i], recs[i], cell, args.tracer, args.ell, True, k % 2 == 0, curve_for(i)))
                dur.append(380)
            frames.append(render_frame(Ssm, recs[i]["gsm"], wins[i], recs[i], cell, args.tracer, args.ell, True, True, curve_for(i)))
            dur.append(1500)
        else:
            frames.append(render_frame(Ssm, recs[i]["gsm"], wins[i], recs[i], cell, args.tracer, args.ell, False, False, curve_for(i)))
            dur.append(base)
    dur[-1] += 1200                                             # linger on the last frame before looping
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tracer
    path = os.path.join(args.out_dir, f"cross_correlation_{tag}.gif")
    save_gif(frames, dur, path)
    with open(os.path.join(args.out_dir, f"cross_correlation_{tag}_windows.csv"), "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["z0", "x_hii", "D_ell_uK2", "pearson_r", "arrow_row", "arrow_col", "is_peak"])
        for i, (w, r) in enumerate(zip(wins, recs)):
            w_.writerow([w["z0"], w["x_hii"], r["D"], r["r"], r["arrow_pix"][0], r["arrow_pix"][1], int(i == ip)])
    print(f"saved {path}  ({len(frames)} frames)")


if __name__ == "__main__":
    sys.exit(main())
