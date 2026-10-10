"""
zreion/convert.py
====================
Apply the zreion model to the density field of an EXISTING py21cmfast box (same seed, same initial conditions),
so the 21cmFAST ionisation field and the zreion one can be compared on identical matter, and so the existing
wrap-cycle / La Plante cross-power pipeline can be run with ONLY the ionisation field swapped.

Density used for zreion: hires_density.npy is the z=0-normalised linear IC field (utils/field_units.py), so
    delta_lin(z_mean) = D(z_mean)/D(0) * delta_IC                              (lpt='linear')
or, closer to the paper's 2LPT density, the Zel'dovich-displaced, TSC-deposited version of it (lpt='za').
"""
from __future__ import annotations

import os

import numpy as np

from ksz_lae_xcorr.utils.field_units import growth_factor_ratio
from ksz_lae_xcorr.zreion.box import apply_zreion, zeldovich_density


def zre_from_ic(delta_ic: np.ndarray, boxsize_mpc: float, Om0: float, h: float, zmean: float = 8.0,
                alpha: float = 0.2, k0_hmpc: float = 0.9, rsmooth_hmpc: float = 1.0, deconvolve: bool = True,
                lpt: str = "za") -> np.ndarray:
    """z_re field (float32, same grid) from the z=0-normalised linear IC overdensity of a 21cmFAST box."""
    if lpt not in ("za", "linear"):
        raise ValueError("lpt must be 'za' or 'linear'")
    n = delta_ic.shape[0]
    D = np.float32(growth_factor_ratio(Om0, zmean))
    dlin = (delta_ic.astype(np.float32) - np.float32(delta_ic.mean())) * D
    if lpt == "za":
        dm = zeldovich_density(np.fft.rfftn(dlin), n, boxsize_mpc)
    else:
        dm = dlin
    return apply_zreion(dm, boxsize_mpc, zmean=zmean, alpha=alpha, k0_hmpc=k0_hmpc, h=h,
                        rsmooth_hmpc=rsmooth_hmpc, deconvolve=deconvolve)


def xhi_from_zre(z_re: np.ndarray, z: float) -> np.ndarray:
    """Neutral fraction at redshift z: 0 where the cell reionised earlier (z_re > z), else 1."""
    return (z_re <= z).astype(np.float32)


def xhi_quantile_matched(z_re: np.ndarray, x_hii_target: float) -> np.ndarray:
    """Neutral-fraction map with EXACTLY the requested volume-averaged ionised fraction: the earliest-
    reionised `x_hii_target` of the cells are ionised.  Removes the (model-dependent) history so only the
    MORPHOLOGY of the zreion field is compared with another model at the same mean x_HII."""
    x = float(np.clip(x_hii_target, 0.0, 1.0))
    flat = z_re.reshape(-1)
    k = int(round((1.0 - x) * flat.size))                  # number of neutral cells
    if k <= 0:
        return np.zeros(z_re.shape, np.float32)
    if k >= flat.size:
        return np.ones(z_re.shape, np.float32)
    thr = np.partition(flat, k - 1)[k - 1]                 # k-th smallest z_re
    return (z_re <= thr).astype(np.float32)


def _kmag(n, boxsize):
    k1 = np.fft.fftfreq(n, d=boxsize / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=boxsize / n) * 2 * np.pi
    return np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)


def binned_spectra(a: np.ndarray, b: np.ndarray, boxsize: float, nbins: int = 20):
    """Spherically binned P_aa, P_bb, P_ab (Mpc^3) of two cubic fields (means removed) and r = P_ab/sqrt(P_aa P_bb).
    Returns dict(k, Paa, Pbb, Pab, r)."""
    n = a.shape[0]
    fa = np.fft.rfftn(a - a.mean()); fb = np.fft.rfftn(b - b.mean())
    kk = _kmag(n, boxsize)
    wt = np.full(kk.shape, 2.0); wt[..., 0] = 1.0
    if n % 2 == 0:
        wt[..., -1] = 1.0
    kf = 2 * np.pi / boxsize
    edges = np.geomspace(kf * 0.99, kk.max() * 1.001, nbins + 1)
    idx = np.digitize(kk.reshape(-1), edges) - 1
    ok = (idx >= 0) & (idx < nbins) & (kk.reshape(-1) > 0)
    w = wt.reshape(-1)[ok]; ii = idx[ok]
    norm = boxsize ** 3 / n ** 6
    def avg(x):
        s = np.bincount(ii, weights=w * x.reshape(-1)[ok], minlength=nbins)
        c = np.bincount(ii, weights=w, minlength=nbins)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan) * norm
    Paa, Pbb = avg(np.abs(fa) ** 2), avg(np.abs(fb) ** 2)
    Pab = avg((fa * np.conj(fb)).real)
    kb = avg(kk)  # mean k per bin (norm factor removed below)
    kb = kb / norm
    with np.errstate(all="ignore"):
        r = Pab / np.sqrt(Paa * Pbb)
    return dict(k=kb, Paa=Paa, Pbb=Pbb, Pab=Pab, r=r)


def write_zreion_coeval_root(src_root: str, dst_root: str, seed: int, xhi_for_z, link: bool = True,
                             z_range: tuple | None = None, log=print) -> list:
    """Create dst_root/seed_<seed>/coeval_z*/ mirroring src_root: hires_density.npy and velocity_z.npy are
    SYMLINKS to the originals (no copy), neutral_fraction.npy is replaced by xhi_for_z(z) (a float32 cube).
    Refuses to overwrite an existing destination seed folder.  Returns the list of snapshot redshifts."""
    src = os.path.join(src_root, f"seed_{seed}"); dst = os.path.join(dst_root, f"seed_{seed}")
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} already exists -- remove it or choose another --out-root")
    zs = []
    for d in sorted(os.listdir(src)):
        if not d.startswith("coeval_z"):
            continue
        z = float(d.replace("coeval_z", ""))
        if z_range and not (z_range[0] <= z <= z_range[1]):
            continue
        os.makedirs(os.path.join(dst, d))
        for f in ("hires_density.npy", "velocity_z.npy"):
            s = os.path.join(src, d, f)
            if os.path.exists(s):
                (os.symlink if link else __import__("shutil").copy2)(os.path.abspath(s), os.path.join(dst, d, f))
        np.save(os.path.join(dst, d, "neutral_fraction.npy"), xhi_for_z(z).astype(np.float32))
        zs.append(z)
    log(f"wrote {len(zs)} snapshots to {dst}")
    return zs
