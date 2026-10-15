"""
tracers/suite.py
=================
A SUITE of galaxy-like tracers cut out of the 21cmFAST halo catalogue, for comparing their bias, number density
and kSZ^2 cross-power with the La Plante+22 / Waters+16 galaxy bias.

Tracer kinds (all are subsets of the FULL halo catalogue; LAE/LBG ids index the full array, see
Stitcher.load_lae_grid):
  halo  mass cut                  M_halo > cut [Msun]
  lae   Lya-luminosity cut        lya_lum_obs >= cut [erg/s]       (Jahaan's catalogue)
  lbg   UV-magnitude cut          MUV_lbg < cut  (brighter than)   (Jahaan's catalogue)

What lives here (no cluster I/O except `load_catalogues`, so everything else is unit-testable on arrays):
  * TracerSpec / default cut lists / name <-> spec round trip
  * count_grid       vectorised CIC-free NGP count grid (replaces the per-halo python loop of Stitcher._bin_to_grid)
  * select_members   the (coords, halo masses) of one spec
  * halo_bias_from_grids   b(k) = P_tm / P_mm and the large-scale average b_lin (cross-spectrum, so no shot noise bias)
  * sheth_tormen_bias / mean_bias_of_masses   analytic (Sheth, Mo & Tormen 2001) reference for the SAME sample
  * snapshot_tracers   all specs for one (seed, z) in a single pass over the catalogue

Units: Mpc (not h^-1) everywhere, halo masses in Msun, like the rest of the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ksz_lae_xcorr.utils.field_units import growth_factor_ratio

DELTA_C = 1.686
SIGMA8_21CMFAST = 0.8102           # 21cmFAST default; the config cosmology block has no sigma8
RHO_CRIT0_H2 = 2.775e11            # Msun / Mpc^3 per h^2

DEFAULT_HALO_CUTS = (1e9, 10 ** 9.5, 1e10, 10 ** 10.5, 1e11, 10 ** 11.5, 1e12)
DEFAULT_LAE_CUTS = (1e42, 10 ** 42.5, 1e43)             # erg/s (SILVERRUSH ~2e42-8e42, Roman ~7e42-1e43)
DEFAULT_LBG_CUTS = (-17.0, -18.0, -19.0, -20.0, -21.0)  # MUV (fiducial -17)


@dataclass(frozen=True)
class TracerSpec:
    kind: str          # 'halo' | 'lae' | 'lbg'
    cut: float

    def __post_init__(self):
        if self.kind not in ("halo", "lae", "lbg"):
            raise ValueError(f"kind must be halo/lae/lbg, got {self.kind!r}")

    @property
    def name(self) -> str:
        if self.kind == "lbg":
            return f"lbg_MUV{self.cut:+.1f}"
        tag = "M" if self.kind == "halo" else "L"
        return f"{self.kind}_{tag}{np.log10(self.cut):.1f}"

    @property
    def label(self) -> str:
        if self.kind == "halo":
            return rf"halo $M>10^{{{np.log10(self.cut):.1f}}}\,M_\odot$"
        if self.kind == "lae":
            return rf"LAE $L_{{\rm Ly\alpha}}>10^{{{np.log10(self.cut):.1f}}}$ erg/s"
        return rf"LBG $M_{{\rm UV}}<{self.cut:.0f}$"


def spec_from_name(name: str) -> TracerSpec:
    kind, rest = name.split("_", 1)
    if kind == "lbg":
        return TracerSpec("lbg", float(rest.replace("MUV", "")))
    return TracerSpec(kind, 10 ** float(rest[1:]))


def default_specs(halo=DEFAULT_HALO_CUTS, lae=DEFAULT_LAE_CUTS, lbg=DEFAULT_LBG_CUTS) -> list[TracerSpec]:
    return ([TracerSpec("halo", c) for c in halo] + [TracerSpec("lae", c) for c in lae]
            + [TracerSpec("lbg", c) for c in lbg])


def parse_specs(text: str) -> list[TracerSpec]:
    """'halo:1e10,halo:11,lae:42.5,lbg:-19'  or  'default'.  ';' or ',' between specs, spaces ignored.
    halo/lae values below 100 are read as log10 (halo:10.5 = 1e10.5 Msun, lae:42.5 = 10^42.5 erg/s); lbg is MUV."""
    text = text.replace(" ", "")
    if text in ("", "default"):
        return default_specs()
    out = []
    for tok in text.replace(";", ",").split(","):
        kind, val = tok.split(":")
        v = float(val)
        if kind in ("halo", "lae") and v < 100.0:
            v = 10.0 ** v
        out.append(TracerSpec(kind, v))
    return out


# ------------------------------------------------------------------ gridding
def count_grid(coords: np.ndarray, ngrid: int, cell: float, chunk: int = 8_000_000) -> np.ndarray:
    """Nearest-grid-point counts on an (ngrid)^3 periodic grid (float32).  Vectorised; ~1 s per 50M objects."""
    n3 = ngrid ** 3
    acc = np.zeros(n3, dtype=np.int64)
    for i in range(0, len(coords), chunk):
        c = np.asarray(coords[i:i + chunk], dtype=np.float64)
        idx = np.floor(c / cell).astype(np.int64) % ngrid          # floor, not truncation: negatives wrap correctly
        flat = (idx[:, 0] * ngrid + idx[:, 1]) * ngrid + idx[:, 2]
        acc += np.bincount(flat, minlength=n3)
    return acc.reshape(ngrid, ngrid, ngrid).astype(np.float32)


def select_members(spec: TracerSpec, coords, masses, lae=None, lbg=None):
    """(coords, masses) of the objects of `spec`.  lae = (ids, lya_lum), lbg = (ids, MUV) as in Jahaan's files
    (ids index the FULL halo arrays).  Returns empty arrays when the catalogue is not available."""
    empty = (np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32))
    if spec.kind == "halo":
        sel = np.asarray(masses) > spec.cut
        return np.asarray(coords)[sel], np.asarray(masses)[sel]
    cat = lae if spec.kind == "lae" else lbg
    if cat is None:
        return empty
    ids, val = np.asarray(cat[0]), np.asarray(cat[1])
    keep = val >= spec.cut if spec.kind == "lae" else val < spec.cut
    ids = ids[keep]
    if len(ids) == 0:
        return empty
    ids = np.sort(ids)                       # sequential reads on a memory-mapped catalogue
    return np.asarray(coords[ids]), np.asarray(masses[ids])


# ------------------------------------------------------------------ bias
def _k_shell_sums(f_a, f_b, boxsize: float, edges: np.ndarray):
    """sum over rfft modes in shells of Re(f_a conj f_b) with the correct mode multiplicity.  Returns (sum, nmodes)."""
    n = f_a.shape[0]
    kf = 2 * np.pi / boxsize
    k1 = np.fft.fftfreq(n, d=1.0 / n) * kf
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kf
    kmag = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.full(f_a.shape[-1], 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    w = np.broadcast_to(w[None, None, :], f_a.shape)
    cross = np.real(f_a * np.conj(f_b)) * w
    which = np.digitize(kmag.ravel(), edges) - 1
    ok = (which >= 0) & (which < len(edges) - 1) & (kmag.ravel() > 0)
    s = np.bincount(which[ok], weights=cross.ravel()[ok], minlength=len(edges) - 1)
    m = np.bincount(which[ok], weights=w.ravel()[ok], minlength=len(edges) - 1)
    return s, m


def halo_bias_from_grids(n_grid: np.ndarray, delta_m: np.ndarray, boxsize: float, kmax: float = 0.15,
                         n_kbins: int = 14, k_plot_max: float = 1.0) -> dict:
    """Linear bias of a tracer count grid relative to the matter field:  b(k) = P_tm / P_mm  (cross-spectrum, so the
    tracer shot noise does not bias it).  b_lin = sum_{k<kmax} P_tm / sum_{k<kmax} P_mm.
    b_lin_err: scatter of b(k) over the bins below kmax, divided by sqrt(n bins) (a rough number; with only ~10 bins
    and 10 seeds we report the seed scatter in the plots instead).
    n_grid, delta_m: (n,n,n); delta_m must be the SAME grid and zero-mean overdensity."""
    n = n_grid.shape[0]
    nobj = float(n_grid.sum())
    out = {"n_obj": nobj, "nbar": nobj / n ** 3}
    kf = 2 * np.pi / boxsize
    edges = np.linspace(0.5 * kf, k_plot_max, n_kbins + 1)
    if nobj <= 0:
        out.update(k=0.5 * (edges[1:] + edges[:-1]), b_k=np.full(n_kbins, np.nan), n_modes=np.zeros(n_kbins),
                   b_lin=np.nan, b_lin_err=np.nan)
        return out
    dh = n_grid / (nobj / n ** 3) - 1.0
    fh = np.fft.rfftn(dh.astype(np.float32))
    fm = np.fft.rfftn((delta_m - np.mean(delta_m)).astype(np.float32))
    s_hm, nm = _k_shell_sums(fh, fm, boxsize, edges)
    s_mm, _ = _k_shell_sums(fm, fm, boxsize, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_k = s_hm / s_mm
    kc = 0.5 * (edges[1:] + edges[:-1])
    low = kc <= kmax
    out.update(k=kc, b_k=b_k, n_modes=nm,
               b_lin=float(s_hm[low].sum() / s_mm[low].sum()),
               b_lin_err=float(np.nanstd(b_k[low]) / np.sqrt(max(low.sum(), 1))))
    return out


class SigmaOfM:
    """sigma(M) at z=0 from the linear spectrum (top-hat), tabulated once; interpolation in log M."""

    def __init__(self, Om: float, Ob: float, h: float, ns: float, sigma8: float = SIGMA8_21CMFAST,
                 backend: str = "auto", log10m_range=(6.0, 14.5), n: int = 60):
        from ksz_lae_xcorr.zreion.box import linear_pk, sigma_R
        pk = linear_pk(dict(Om=Om, Ob=Ob, h=h, ns=ns, sigma8=sigma8), backend=backend)
        rho_m = Om * RHO_CRIT0_H2 * h ** 2                 # comoving Msun / Mpc^3
        self.Om = Om
        self.rho_m = rho_m
        self.logm = np.linspace(*log10m_range, n)
        R = (3.0 * 10 ** self.logm / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
        self.sig = np.array([sigma_R(pk, r) for r in R])

    def __call__(self, m):
        return np.interp(np.log10(m), self.logm, self.sig)

    def dlnsigma_dlnm(self, m):
        """d ln sigma / d ln M (negative), from the tabulated curve."""
        g = np.gradient(np.log(self.sig), self.logm * np.log(10.0))
        return np.interp(np.log10(m), self.logm, g)


def sheth_tormen_bias(m, z: float, sigma_of_m: SigmaOfM, a: float = 0.707, p: float = 0.3) -> np.ndarray:
    """Sheth, Mo & Tormen (2001) halo bias b(M, z)."""
    nu = DELTA_C / (sigma_of_m(m) * float(growth_factor_ratio(sigma_of_m.Om, z)))
    anu2 = a * nu ** 2
    return 1.0 + (anu2 - 1.0) / DELTA_C + 2.0 * p / (DELTA_C * (1.0 + anu2 ** p))


def sheth_tormen_dndlnm(m, z: float, sigma_of_m: SigmaOfM, a: float = 0.707, p: float = 0.3, A: float = 0.3222) -> np.ndarray:
    """Sheth & Tormen (1999) halo mass function dn/dlnM [Mpc^-3] (comoving, per unit ln M):
    dn/dlnM = (rho_m / M) nu f(nu) |dln sigma / dln M|,  f(nu) = A sqrt(2a/pi) [1 + (a nu^2)^-p] exp(-a nu^2 / 2)."""
    m = np.asarray(m, dtype=float)
    D = float(growth_factor_ratio(sigma_of_m.Om, z))
    nu = DELTA_C / (sigma_of_m(m) * D)
    f = A * np.sqrt(2.0 * a / np.pi) * (1.0 + (a * nu ** 2) ** (-p)) * np.exp(-0.5 * a * nu ** 2)
    rho_m = sigma_of_m.rho_m
    return rho_m / m * nu * f * np.abs(sigma_of_m.dlnsigma_dlnm(m))


def sheth_tormen_n_gt(m_cut, z: float, sigma_of_m: SigmaOfM, m_max: float = 10 ** 15.5, n: int = 600) -> float:
    """n(> M_cut) [Mpc^-3] = integral of the Sheth-Tormen dn/dlnM."""
    lnm = np.linspace(np.log(m_cut), np.log(m_max), n)
    f = sheth_tormen_dndlnm(np.exp(lnm), z, sigma_of_m)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(lnm)))


def mean_bias_of_masses(masses, z: float, sigma_of_m: SigmaOfM, max_n: int = 200_000, seed: int = 0) -> float:
    """Number-weighted mean ST bias of a sample (subsampled to max_n for speed)."""
    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[masses > 0]
    if len(masses) == 0:
        return float("nan")
    if len(masses) > max_n:
        masses = np.random.default_rng(seed).choice(masses, max_n, replace=False)
    return float(np.mean(sheth_tormen_bias(masses, z, sigma_of_m)))


# ------------------------------------------------------------------ catalogue access (cluster)
def load_catalogues(stitcher, seed: int, z: float) -> dict:
    """Memory-mapped catalogue pieces for one (seed, z): coords, masses, and (ids, value) pairs for LAE/LBG (None when
    the external files are missing)."""
    from ksz_lae_xcorr.utils.external_catalogue import external_catalogue_filename
    cfg = stitcher.cfg
    coords, masses = stitcher._halo_coords_masses(seed, z)

    def _pair(root, id_prefix, val_prefix):
        ip = os.path.join(root, id_prefix, external_catalogue_filename(id_prefix, z, cfg, seed))
        vp = os.path.join(root, val_prefix, external_catalogue_filename(val_prefix, z, cfg, seed))
        if not (os.path.exists(ip) and os.path.exists(vp)):
            return None
        return np.load(ip, mmap_mode="r"), np.load(vp, mmap_mode="r")

    return {"coords": coords, "masses": masses,
            "lae": _pair(stitcher.root_lae, "halo_ids_obs", "lya_lum_obs"),
            "lbg": _pair(stitcher.root_lbg, "halo_ids_lbg", "MUV_lbg")}


def snapshot_tracers(cat: dict, specs, ngrid: int, cell: float, max_mass_sample: int = 200_000, rng_seed: int = 0) -> dict:
    """All specs for one snapshot in one pass.  Returns {spec.name: {'grid', 'n', 'mass_sample'}}."""
    rng = np.random.default_rng(rng_seed)
    out = {}
    for spec in specs:
        c, m = select_members(spec, cat["coords"], cat["masses"], cat["lae"], cat["lbg"])
        ms = np.asarray(m, dtype=np.float32)
        if len(ms) > max_mass_sample:
            ms = rng.choice(ms, max_mass_sample, replace=False)
        out[spec.name] = {"grid": count_grid(c, ngrid, cell), "n": int(len(m)), "mass_sample": ms}
    return out
