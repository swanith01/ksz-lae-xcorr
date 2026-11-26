#!/usr/bin/env python3
"""
scripts/63_lae_lightcone_compare.py  (26 Nov 2026)
====================================================
Old vs new LAE catalogue as a lightcone picture (run on swarm, where the boxes and catalogues are).

  lae_lightcone_seed<S> : two panels (old catalogue / new catalogue, same seed): a thin slab of the box (transverse x vs
                          comoving distance) built from the coeval snapshots -- each snapshot fills the distance range
                          between its neighbours' midpoints, line-of-sight cells taken periodically along the box z axis --
                          with the neutral fraction as background and the observed LAEs (L_obs > --lmin) in the slab as dots.
                          Old: catalogue of 26 Aug 2026 (/user1/jahaan/swanith/obs_prop) at the saved (Lagrangian) halo
                          positions; new: scripts/61 catalogue on the repaired boxes at the displaced positions.
  lae_lightcone_counts  : seed-averaged comoving number density of the same LAEs vs z, old vs new (band: range of seeds).

    python scripts/63_lae_lightcone_compare.py --seed 1 --out-dir data/plots/2026-11-26_lae_lightcone
"""
import argparse
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10", help="for the seed-averaged counts")
ap.add_argument("--boxes-root", default="/user1/swanith/kSZ2_LAE_project_22Jun2026")
ap.add_argument("--old-root", default="/user1/jahaan/swanith/obs_prop")
ap.add_argument("--new-root", default="/user1/swanith/lae_catalogue_v2/obs_prop")
ap.add_argument("--lmin", type=float, default=10**42.5, help="observed Lya luminosity cut (lae_L42.5 tracer)")
ap.add_argument("--zmin", type=float, default=5.8)
ap.add_argument("--zmax", type=float, default=9.5)
ap.add_argument("--slab", type=float, default=20.0, help="slab thickness in the second transverse direction [cMpc]")
ap.add_argument("--box-len", type=float, default=300.0)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out-dir", default="data/plots/2026-11-26_lae_lightcone")
args = ap.parse_args()
L, n = args.box_len, args.n
dx = L / n
tag = lambda z, s: f"z{z:.6f}_{n}_{int(L)}_s{s}.npy"
chi = lambda z: Planck18.comoving_distance(z).to_value("Mpc")


def snapshots(seed):
    zs = sorted(float(p.split("coeval_z")[1]) for p in glob.glob(os.path.join(args.boxes_root, f"seed_{seed}", "coeval_z*")))
    return [z for z in zs if args.zmin - 0.3 <= z <= args.zmax + 0.3]


def catalogue(root, z, seed):
    """(halo_ids, L_obs) above the cut, or None if missing."""
    try:
        ids = np.load(os.path.join(root, "halo_ids_obs", f"halo_ids_obs_{tag(z, seed)}"))
        lum = np.load(os.path.join(root, "lya_lum_obs", f"lya_lum_obs_{tag(z, seed)}"))
    except FileNotFoundError:
        return None
    keep = lum > args.lmin
    return ids[keep].astype(np.int64), lum[keep]


def coords_of(seed, z, ids, which):
    f = "halo_coords.npy" if which == "old" else "halo_coords_eulerian.npy"
    c = np.load(os.path.join(args.boxes_root, f"seed_{seed}", f"coeval_z{z:.6f}", f), mmap_mode="r")
    order = np.argsort(ids)
    out = np.empty((ids.size, 3), np.float64)
    out[order] = np.asarray(c[ids[order]], dtype=np.float64)
    return out


def lightcone(seed):
    zs = snapshots(seed)
    ch = np.array([chi(z) for z in zs])
    edges = np.concatenate([[ch[0] - (ch[1] - ch[0]) / 2], (ch[1:] + ch[:-1]) / 2, [ch[-1] + (ch[-1] - ch[-2]) / 2]])
    lo_c, hi_c = chi(args.zmin), chi(args.zmax)
    ny = int(round(args.slab / dx))
    xh_strips, pts = [], {"old": [], "new": []}
    for i, z in enumerate(zs):
        a, b = max(edges[i], lo_c), min(edges[i + 1], hi_c)
        if b <= a:
            continue
        cells = np.arange(np.floor(a / dx), np.ceil(b / dx)).astype(int)
        xh = np.load(os.path.join(args.boxes_root, f"seed_{seed}", f"coeval_z{z:.6f}", "neutral_fraction.npy"), mmap_mode="r")
        strip = np.asarray(xh[:, :ny, :][:, :, cells % n], dtype=np.float32).mean(axis=1)     # (x, chi cells)
        xh_strips.append((cells * dx, strip))
        for which, root in (("old", args.old_root), ("new", args.new_root)):
            cat = catalogue(root, z, seed)
            if cat is None or cat[0].size == 0:
                continue
            c = coords_of(seed, z, cat[0], which)
            in_slab = (c[:, 1] % L) < args.slab
            c = c[in_slab]
            # the copy of the halo's LOS position that falls in [a, b)
            los = c[:, 2] % L
            k0 = np.floor(a / L)
            for k in (k0, k0 + 1):
                d = k * L + los
                sel = (d >= a) & (d < b)
                pts[which].append(np.column_stack([d[sel], c[sel, 0] % L]))
    return xh_strips, {k: (np.concatenate(v) if v else np.empty((0, 2))) for k, v in pts.items()}


def counts(seeds):
    zs = snapshots(seeds[0])
    out = {"old": [], "new": []}
    for s in seeds:
        for which, root in (("old", args.old_root), ("new", args.new_root)):
            row = []
            for z in zs:
                cat = catalogue(root, z, s)
                row.append(np.nan if cat is None else cat[0].size / L**3)
            out[which].append(row)
    return np.array(zs), {k: np.array(v) for k, v in out.items()}


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    strips, pts = lightcone(args.seed)
    zt = [z for z in (6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5) if args.zmin <= z <= args.zmax]
    fig, axes = plt.subplots(2, 1, figsize=(16, 6.2), sharex=True, constrained_layout=True)
    titles = {"old": "old catalogue (26 Aug 2026): frozen z = 0 density, start (Lagrangian) halo positions, old IGM code",
              "new": "new catalogue (25 Nov 2026): repaired boxes (evolved density, displaced haloes), Jahaan's current IGM code"}
    for ax, which in zip(axes, ("old", "new")):
        for x0, strip in strips:
            ext = [x0[0], x0[-1] + dx, 0, L]
            im = ax.imshow(strip, origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        p = pts[which]
        ax.plot(p[:, 0], p[:, 1], ".", ms=2.2, color="#D55E00", alpha=0.8)
        ax.set_xlim(chi(args.zmin), chi(args.zmax)); ax.set_ylim(0, L)
        ax.set_ylabel("x [cMpc]")
        ax.set_title(f"{titles[which]}  --  {len(p):,} LAEs in this {args.slab:g} cMpc slab", fontsize=9.5, loc="left")
        top = ax.secondary_xaxis("top", functions=(lambda c: c, lambda c: c))
        top.set_xticks([chi(z) for z in zt]); top.set_xticklabels([f"z={z:g}" for z in zt], fontsize=8)
    axes[-1].set_xlabel("comoving distance [cMpc]")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01); cb.set_label("neutral fraction (slab average)")
    fig.legend(handles=[Line2D([], [], marker=".", ls="", color="#D55E00", ms=8,
                               label=f"observed LAEs, L(Ly$\\alpha$) > {args.lmin:.2g} erg/s (tracer lae_L42.5), seed {args.seed}")],
               loc="lower center", bbox_to_anchor=(0.45, -0.06), frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"lae_lightcone_seed{args.seed}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure:", os.path.join(args.out_dir, f"lae_lightcone_seed{args.seed}.png"))

    seeds = [int(s) for s in args.seeds.split(",")]
    zs, cnt = counts(seeds)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for which, col, lab in (("old", "0.45", "old catalogue"), ("new", "#0072B2", "new catalogue (repaired boxes)")):
        v = cnt[which]
        m = np.nanmean(v, 0)
        ax.plot(zs, np.clip(m, 1e-12, None), "-o", ms=3, color=col, lw=1.8, label=f"{lab}, mean of {len(seeds)} seeds")
        ax.fill_between(zs, np.clip(np.nanmin(v, 0), 1e-12, None), np.clip(np.nanmax(v, 0), 1e-12, None), color=col, alpha=0.2, lw=0)
    ax.set_yscale("log"); ax.set_ylim(1e-9, None)
    ax.set_xlabel("z"); ax.set_ylabel(r"$n$(observed LAEs) [cMpc$^{-3}$]")
    ax.set_title(f"observed LAEs with L(Ly$\\alpha$) > {args.lmin:.2g} erg/s (band: range of seeds)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9, ncol=1)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"lae_lightcone_counts.{ext}"), dpi=200, bbox_inches="tight")
    print("figure:", os.path.join(args.out_dir, "lae_lightcone_counts.png"))
    print("\n   z     n_old [cMpc^-3]   n_new [cMpc^-3]")
    for z, a, b in zip(zs, np.nanmean(cnt["old"], 0), np.nanmean(cnt["new"], 0)):
        print(f"  {z:6.3f}   {a:12.3g}   {b:12.3g}")


if __name__ == "__main__":
    sys.exit(main())
