"""
snr/feasibility.py
===================
Observational feasibility of the kSZ^2 x LAE cross-spectrum: S/N for CMB experiments (current and planned) crossed with LAE surveys
(current and planned), for any galaxy window (z0, dz) -- including narrow-band shells.

Why this module does not need simulation maps (the raw lightcone products live on usha/swarm, not in the repo):

  signal   D_ell(z0, dz) = A(ell) * R_exp * Toy.cross(...)['D']        [uK^2]
           Toy = correlation/toy_model.py (eps = 8.25, shape fitted to La Plante+22 Fig. 7); A(ell) is its amplitude fitted to
           that same figure (SO ILC filter) at ell = 500, 1000, 3000 and interpolated in log-log.
           Linear in the tracer bias: the toy takes b(z) as a function, so a tracer with bias b_t(z) is b_t/b_Waters times
           La Plante's galaxy sample (same window).  This is the part that should be replaced by the simulation's measured D_ell
           (scripts 28/44) once the LAE lightcone excess is trusted.
  R_exp    = sigma2_f(exp) / sigma2_f(SO ILC): ratio of the filtered reionisation-kSZ variance, int ell dell/2pi f^2 C_kSZ,reion.
           In the squeezed limit the cross-power is the response of the LOCAL filtered power, so it scales with that variance if
           the response has the same ell-shape as C_kSZ.
  noise    La Plante+22 Eq. 13 (Gaussian):  (S/N)^2 = f_sky sum_ell (2 ell + 1) Cs^2 / (C^{T2T2}_f C^{gg} + Cs^2),
           C^{T2T2}_f = 2 int ell dell/2pi (C^{TT}_f)^2  (white),  C^{gg} = b^2 C^{mm}_Limber(window) + 1/n_2D.
           CMB noise: white (N_ell = (Delta_T * arcmin)^2 exp(ell^2 theta^2 / 8 ln 2)) or post-ILC digitised (SO, S4, HD).

Validation: scripts/52_feasibility.py reproduces La Plante's Table 2 'without shot noise' column (Roman HLS, f_sky = 0.053).

NOT in the forecast (all would lower the S/N): foreground residuals other than those inside La Plante's digitised ILC curves,
lensing leakage, mask/mode-coupling, redshift-space distortions of the tracer map, and the cosmic-variance covariance between
windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ksz_lae_xcorr.utils import constants


def _trap(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# =====================================================================================================================
# experiments and surveys (numbers: see SOURCES in docs/feasibility_note.md)
# =====================================================================================================================
@dataclass
class Experiment:
    name: str
    white_uK_arcmin: float          # white temperature noise of the map used for the filter
    fwhm_arcmin: float
    area_deg2: float
    ilc_key: str | None = None      # La Plante Fig. 1 post-ILC noise curve to use instead of the white noise (SO, CMB-S4, CMB-HD)
    status: str = "current"
    note: str = ""


EXPERIMENTS = {e.name: e for e in [
    # --- current -------------------------------------------------------------------------------------------------------
    Experiment("ACT DR6", 9.6, 1.4, 19000, status="current",
               note="3-band combined depth 9.6 uK-arcmin (day+night, Naess+25 DR6 maps, 19,000 deg2); 150 GHz beam 1.4'. White noise only: "
                    "no foreground residual, so optimistic."),
    Experiment("SPT-3G Main", 1.9, 1.2, 1500, status="current",
               note="coadd 1.9 uK-arcmin over 1500 deg2 for the 2019-2023 data (SPT-3G forecast paper, arXiv 2403.17925 Table 1); "
                    "150 GHz beam 1.2' not re-verified. White noise only."),
    # --- planned -------------------------------------------------------------------------------------------------------
    Experiment("SO", 10.0, 1.4, 16000, ilc_key="SO_post_ILC_noise", status="planned",
               note="LAT baseline 145 GHz: 10 uK-arcmin, 1.4', f_sky 0.4 (SO science goals, arXiv 1808.07445); noise curve = La Plante Fig. 1 post-ILC."),
    Experiment("CMB-S4", 2.0, 1.4, 1600, ilc_key="CMB-S4_post_ILC_noise", status="planned",
               note="La Plante+22 Table 1 values (2 uK-arcmin, 1.4'); the area is the deep-field scale used here, not the La Plante f_sky. "
                    "Project status after the 2025 agency decision not checked."),
    Experiment("CMB-HD", 0.6, 1.0 / 6.0, 20000, ilc_key="CMB-HD_post_ILC_noise", status="planned",
               note="La Plante+22 Table 1 values (0.6 uK-arcmin, 10'' beam); area scale as proposed (~half sky)."),
]}


@dataclass
class Shell:
    z: float                 # centre
    dz: float                # full width
    L_min: float             # erg/s (observed Lya luminosity limit)
    lf: tuple                # (ref_key, z) row of the LAE LF fits to use for the number density
    area_deg2: float | None = None
    n_obs_deg2: float | None = None   # LAEs per deg2 actually in the survey's catalogue (realistic shot noise); None = use the LF
    n_obs_src: str = ""


@dataclass
class Survey:
    name: str
    area_deg2: float
    shells: list = field(default_factory=list)
    status: str = "current"
    note: str = ""


SURVEYS = {s.name: s for s in [
    Survey("HSC SILVERRUSH (NB816+NB921)", 14.0, status="current", shells=[
        Shell(5.7, 0.10, 6.3e42, ("Umeda25", 5.7), 14.0, n_obs_deg2=959 / 13.8,
              n_obs_src="Ouchi+18 (SILVERRUSH I, arXiv 1704.07455) homogeneous sample: 959 LAEs in 13.8 deg2"),
        Shell(6.6, 0.10, 7.9e42, ("Umeda25", 6.6), 21.0, n_obs_deg2=873 / 21.2,
              n_obs_src="Ouchi+18 (SILVERRUSH I, arXiv 1704.07455) homogeneous sample: 873 LAEs in 21.2 deg2"),
    ], note="Konno+18: 14 deg2 (z=5.7) and 21 deg2 (z=6.6), 1,266 LAEs in total; narrow bands ~120 A wide (dz ~ 0.1); L_min from "
            "Weinberger+19 Table 1. NB973/NB101 (z=7.0, 7.3) are CHORUS/Subaru pencil beams of ~1-2 deg2 and are left out."),
    Survey("Subaru PFS (SSP, repo value)", 15.0, status="planned", shells=[
        Shell(5.7, 0.20, 3.0e42, ("Umeda25", 5.7), 15.0),
        Shell(6.6, 0.20, 3.0e42, ("Umeda25", 6.6), 15.0),
    ], note="Area and z-range from configs/lae_surveys.yaml; the L_min = 3e42 erg/s is an ASSUMPTION (the repo has no flux-cut for PFS). "
            "Spectroscopic redshifts, so dz is a chosen slab, not the survey depth."),
    Survey("LAGER (DECam NB964)", 10.19, status="current", shells=[
        Shell(6.9, 0.076, 10 ** 42.8, ("Wold22", 6.9), 10.19, n_obs_deg2=174 / 10.19,
              n_obs_src="Wold+22 (arXiv 2105.12191): 174 LAE candidates in four fields, 10.19 deg2"),
    ], note="Wold+22: WIDE12 3.24, GAMA15A 2.91, COSMOS 1.90, CDFS 2.14 deg2; NB964 FWHM 92 A (dz ~ 0.076); faintest LAEs ~10^42.8 "
            "erg/s; LF = their four-field Schechter fit with alpha fixed at -2.5 (L* 8.95e42, phi* 2.19e-4). All four fields are at "
            "Dec > -30, inside the ACT / SO footprints."),
    Survey("Roman grism deep z=7.5-10, no evolution from z=7", 1.0, status="planned", shells=[
        Shell(8.0, 1.0, 1.0e43, ("Umeda25", 7.0), 1.0),
        Shell(9.0, 1.0, 1.0e43, ("Umeda25", 7.0), 1.0),
        Shell(9.75, 0.5, 1.0e43, ("Umeda25", 7.0), 1.0),
    ], note="1 deg2 grism deep field, Lya at z = 7.5-10 (Girish, 3 Nov 2026), L > 1e43 erg/s (Wold+23 arXiv 2305.01562: 50% "
            "completeness at 0.7-1.4e-17 erg/s/cm2, L_lim 7.4-9.9e42 at z = 7.5-10.5; they predict ~395 LAEs/deg2 at z = 7.25-8.75 "
            "assuming no evolution from z~7). Number density from the z=7.0 LF fit at every z (optimistic)."),
    Survey("Roman grism deep z=7.5-10, LF as observed at z=7.3", 1.0, status="planned", shells=[
        Shell(8.0, 1.0, 1.0e43, ("Umeda25", 7.3), 1.0),
        Shell(9.0, 1.0, 1.0e43, ("Umeda25", 7.3), 1.0),
        Shell(9.75, 0.5, 1.0e43, ("Umeda25", 7.3), 1.0),
    ], note="Same survey with the z=7.3 LF fit (the observed decline of the LAE density into reionisation) at every z."),
    Survey("Roman grism 16 deg2 at deep-field depth (Maitra+25 area)", 16.0, status="planned", shells=[
        Shell(8.0, 1.0, 1.0e43, ("Umeda25", 7.0), 16.0),
        Shell(9.0, 1.0, 1.0e43, ("Umeda25", 7.0), 16.0),
        Shell(9.75, 0.5, 1.0e43, ("Umeda25", 7.0), 16.0),
    ], note="Area ~16 deg2 for the Roman grism from Maitra+25 (arXiv 2505.17188) Table 1, at the depth of the Wold+23 1 deg2 deep "
            "field (L > 1e43 erg/s) and the z=7.0 LF: an UPPER estimate, since no ~16 deg2 Roman grism survey reaches that depth."),
    Survey("Roman HLWAS deep tier (19.2 deg2, grism 6e-17)", 19.2, status="planned", shells=[
        Shell(8.0, 1.0, 4.86e43, ("Umeda25", 7.0), 19.2),
        Shell(9.0, 1.0, 6.37e43, ("Umeda25", 7.0), 19.2),
        Shell(9.75, 0.5, 7.64e43, ("Umeda25", 7.0), 19.2),
    ], note="Roman core community survey: High Latitude Wide Area Survey deep tier, 19.2 deg2 imaging + slitless spectroscopy "
            "(ROTAC report, arXiv 2505.10574). Its grism line-flux limit is quoted as 2.5x deeper than the medium tier's 1.5e-16 "
            "erg/s/cm2 (5 sigma), i.e. ~6e-17 (our derivation); L_min = 4 pi d_L^2 F at the shell centres. z=7.0 LF (no evolution)."),
]}


# Footprint overlap of each LAE survey with each CMB map (fraction of the LAE area inside the CMB map).  Default 1 (not checked).
# HSC SSP Deep/UltraDeep fields (SXDS/XMM-LSS, COSMOS, DEEP2-3, ELAIS-N1) all lie at Dec > -10 deg; SPT-3G Main covers
# Dec ~ -42 to -70 deg, so the two do not overlap.  ACT DR6 (Dec ~ -63 to +23) covers all of them but ELAIS-N1 (Dec +54):
# left at 1 (the ELAIS-N1 share of the SILVERRUSH area was not checked).  PFS SSP Galaxy Evolution fields are the same HSC
# Deep fields.  LAGER fields (WIDE12, GAMA15A, COSMOS, CDFS) are at Dec > -30: inside ACT / SO.  Roman deep-field location:
# not fixed here -> 1.
# CMB-S4 is not used in the forecasts: DOE and NSF ended support for the project in July 2025.
FORECAST_EXPERIMENTS = ("ACT DR6", "SPT-3G Main", "SO", "CMB-HD")
OVERLAP = {
    ("HSC SILVERRUSH (NB816+NB921)", "SPT-3G Main"): 0.0,
    ("Subaru PFS (SSP, repo value)", "SPT-3G Main"): 0.0,
    ("LAGER (DECam NB964)", "SPT-3G Main"): 0.0,
}


def overlap(survey_name: str, experiment_name: str) -> float:
    return OVERLAP.get((survey_name, experiment_name), 1.0)


def field_ell_min(area_deg2: float, ell_min: float = 100.0) -> float:
    """Lowest usable multipole of a field of this area: max(ell_min, 2 pi / sqrt(Omega))."""
    omega = area_deg2 * (np.pi / 180) ** 2
    return max(ell_min, 2 * np.pi / np.sqrt(omega))


# =====================================================================================================================
# CMB side
# =====================================================================================================================
@dataclass
class FilterSet:
    ell: np.ndarray
    f: np.ndarray                 # F(ell) * b(ell)
    C_f: np.ndarray               # filtered total CMB power  f^2 (TT + kSZ + N)
    C_T2T2: float                 # white power of the filtered map squared (Gaussian, La Plante Eq. 14)
    sigma2_reion: float           # int ell dell/2pi f^2 C_kSZ,reion


def white_noise_Cl(ell, uK_arcmin: float, fwhm_arcmin: float):
    th = np.deg2rad(fwhm_arcmin / 60.0)
    return (np.deg2rad(uK_arcmin / 60.0)) ** 2 * np.exp(th ** 2 * np.asarray(ell, float) ** 2 / (8 * np.log(2)))


def build_filter(ell, Cl_TT, Cl_ksz_reion, Cl_ksz_late, N_ell, fwhm_arcmin: float) -> FilterSet:
    """f = F b with F = C_kSZ,reion / (TT + kSZ_reion + kSZ_late + N) -- the same recipe as snr/cmb_filter.build_filters, but
    normalised to f_max = 1 (S/N does not depend on the normalisation as long as signal and noise use the same f; the calibrated
    signal D_ell is for the unit-peak filter)."""
    ell = np.asarray(ell, float)
    th = np.deg2rad(fwhm_arcmin / 60.0)
    total = Cl_TT + Cl_ksz_reion + Cl_ksz_late + N_ell
    F = Cl_ksz_reion / total
    b = np.exp(-th ** 2 * ell ** 2 / (16 * np.log(2)))
    f = F * b
    f = f / np.max(f)            # La Plante Fig. 2: each curve is normalised to 1 at its peak; the signal D_ell is quoted for that normalisation
    C_f = f ** 2 * total
    C_T2T2 = 2.0 * _trap(ell * C_f ** 2 / (2 * np.pi), ell)
    sigma2 = _trap(ell * f ** 2 * Cl_ksz_reion / (2 * np.pi), ell)
    return FilterSet(ell, f, C_f, float(C_T2T2), float(sigma2))


def lp_components(cfg=None, ell=None):
    """La Plante+22 Fig. 1 curves on a common ell grid, as C_ell [uK^2]: dict with SO/CMB-S4/CMB-HD ILC noise, lensed primary CMB,
    kSZ_reion (30-sim mean), kSZ_late."""
    from ksz_lae_xcorr.io.la_plante_reference import load_fig1_components
    comp = load_fig1_components(cfg)
    ell = np.geomspace(100, 10000, 700) if ell is None else np.asarray(ell, float)
    out = {"ell": ell}
    for key, (le, D) in comp.items():
        C = D * 2 * np.pi / (le * (le + 1))
        out[key] = np.exp(np.interp(np.log(ell), np.log(le), np.log(C)))
    return out


def experiment_filter(exp: Experiment, comp: dict, ksz_reion=None) -> FilterSet:
    """Filter + noise of one experiment.  ksz_reion: C_ell on comp['ell'] to use for the reionisation kSZ (default: La Plante's)."""
    ell = comp["ell"]
    N = comp[exp.ilc_key] if exp.ilc_key else white_noise_Cl(ell, exp.white_uK_arcmin, exp.fwhm_arcmin)
    kr = comp["kSZ_reion_30sim_mean"] if ksz_reion is None else ksz_reion
    return build_filter(ell, comp["lensed_primary_CMB"], kr, comp["kSZ_late_Park18"], N, exp.fwhm_arcmin)


# =====================================================================================================================
# signal
# =====================================================================================================================
class SignalModel:
    """Toy-model D_ell(z0, dz, ell) for the SO-ILC filter, in uK^2, with the tracer bias supplied as a function.

    `scale` multiplies the signal; `calibrate_to_lp_table2` sets it so that the S/N of La Plante+22 Table 2 (Roman HLS, ILC noise,
    no shot noise) is reproduced (the raw toy is ~30% low at its peak).  `z_shift` > 0 delays reionisation: xbar(z) -> xbar_fid(z + z_shift)
    (z_shift = 1 puts the 50% point at z ~ 6.7 instead of ~7.7); the toy amplitude A(ell) stays the one fitted to the fiducial history."""

    CAL_ELLS = (500, 1000, 3000)

    def __init__(self, cfg, eps: float = 8.25, zmean: float = 8.0, z_shift: float = 0.0, scale: float = 1.0):
        from ksz_lae_xcorr.correlation import toy_model as tm
        from ksz_lae_xcorr.io.la_plante_reference import load_fig7_vs_xhii, load_reionization_histories
        self.cfg, self.tm = cfg, tm
        h = load_reionization_histories(cfg)["Fiducial"]
        self.h = h
        xb_fid = tm.history_fn(h["z"], h["x_HII"])
        self.eps, self.zmean, self.z_shift, self.scale = eps, zmean, z_shift, scale
        toy_fid = tm.Toy(cfg, xb_fid, eps=eps, zmean=zmean)          # bias_fn = Waters+16 (La Plante's galaxies)
        paper = load_fig7_vs_xhii(cfg)
        grid = np.linspace(0.1, 0.9, 40)
        z0 = np.arange(5.6, 14.01, 0.2)
        A = []
        for ell in self.CAL_ELLS:
            c = toy_fid.curve(z0, 1.0, ell)
            o = np.argsort(c["x_hii"])
            m = np.maximum(np.interp(grid, c["x_hii"][o], c["D"][o]), 0)
            t = np.interp(grid, *paper[ell])
            m = np.maximum(m, 1e-3 * m.max())
            ok = (m > 0) & (t > 0)
            A.append(float(np.exp(np.mean(np.log(t[ok]) - np.log(m[ok])))))
        self.A_cal = np.array(A)
        if z_shift:
            self.xb = lambda z, f=xb_fid, d=z_shift: f(np.asarray(z, float) + d)
            self.toy = tm.Toy(cfg, self.xb, eps=eps, zmean=zmean)
        else:
            self.xb, self.toy = xb_fid, toy_fid

    def A(self, ell):
        """Amplitude at ell (log-log interpolation, end slopes extrapolated)."""
        le, lA = np.log(self.CAL_ELLS), np.log(self.A_cal)
        x = np.log(np.asarray(ell, float))
        out = np.interp(x, le, lA)
        lo, hi = x < le[0], x > le[-1]
        if np.any(lo):
            out = np.where(lo, lA[0] + (lA[1] - lA[0]) / (le[1] - le[0]) * (x - le[0]), out)
        if np.any(hi):
            out = np.where(hi, lA[-1] + (lA[-1] - lA[-2]) / (le[-1] - le[-2]) * (x - le[-1]), out)
        return np.exp(out)

    def D(self, z0, dz, ell, bias_fn=None):
        """D_ell [uK^2] of the window [z0 - dz/2, z0 + dz/2] for a tracer with bias_fn(z) (default Waters+16), SO-ILC filter."""
        saved = self.toy.bias_fn
        try:
            if bias_fn is not None:
                self.toy.bias_fn = bias_fn
            return float(self.scale * self.A(ell) * self.toy.cross(z0, dz, float(ell))["D"])
        finally:
            self.toy.bias_fn = saved

    def x_hii(self, z):
        return self.xb(z)


# La Plante+22 Table 2, fiducial history, Roman HLS (f_sky = 0.053), post-ILC noise, NO galaxy shot noise
LP_TABLE2_ILC_NO_SHOT = {"SO": 2.4, "CMB-S4": 4.8, "CMB-HD": 23.0}
LP_TABLE2_WHITE_NO_SHOT = {"SO": 2.9, "CMB-S4": 18.0, "CMB-HD": 64.0}


def lp_benchmark(sig, comp, windows=((10.0, 5.0),), f_sky=0.053, white=False, n_ell=60):
    """S/N of our pipeline for the three La Plante experiments (best of `windows`), Roman HLS bias, no shot noise."""
    import dataclasses

    from ksz_lae_xcorr.snr.roman_hls_benchmark import bluetides_bias_gz
    ref = experiment_filter(EXPERIMENTS["SO"], comp)
    out = {}
    for n in ("SO", "CMB-S4", "CMB-HD"):
        e = dataclasses.replace(EXPERIMENTS[n], ilc_key=None) if white else EXPERIMENTS[n]
        f = experiment_filter(e, comp)
        out[n] = max(snr_window(sig, f, ref, z0, dz, bluetides_bias_gz, f_sky, None, n_ell=n_ell) for z0, dz in windows)
    return out


def calibrate_to_lp_table2(sig, comp, n_iter=4):
    """Set sig.scale so the geometric-mean ratio of (La Plante Table 2 ILC, no shot noise) / (our S/N) is 1. Returns (scale, raw ratios)."""
    sig.scale = 1.0
    raw = None
    for _ in range(n_iter):
        got = lp_benchmark(sig, comp)
        ratio = np.array([LP_TABLE2_ILC_NO_SHOT[k] / got[k] for k in got])
        if raw is None:
            raw = dict(zip(got, ratio))
        sig.scale *= float(np.exp(np.mean(np.log(ratio))))
    return sig.scale, raw


def limber_cmm(toy, z0, dz, ell):
    """C_ell of the matter field in a top-hat window in comoving distance (Limber):  (1/Dchi^2) int dchi P(ell/chi, z)/chi^2."""
    z = toy.z
    m = (z >= z0 - 0.5 * dz) & (z <= z0 + 0.5 * dz)
    if m.sum() < 3:                                    # window narrower than the grid: evaluate at the centre
        zc = np.array([z0 - 0.5 * dz, z0 + 0.5 * dz])
        chi = np.interp(zc, z, toy.chi)
        Dg = np.interp(zc, z, toy.D)
        k = ell / chi.mean()
        return float((Dg.mean() ** 2) * toy.pk(k) / chi.mean() ** 2 / (chi[1] - chi[0]))
    chi, Dg = toy.chi[m], toy.D[m]
    k = ell / chi
    integrand = Dg ** 2 * toy.pk(k) / chi ** 2
    dchi = chi[-1] - chi[0]
    return float(_trap(integrand, chi) / dchi ** 2)


# =====================================================================================================================
# S/N
# =====================================================================================================================
def snr_window(sig: SignalModel, filt: FilterSet, filt_ref: FilterSet, z0, dz, bias_fn, f_sky, n2d_per_sr=None,
               ell_min=100, ell_max=10000, n_ell=300, return_parts=False):
    """(S/N) of one galaxy window with one CMB filter.  n2d_per_sr = None -> no shot noise (clustering-only).
    f_sky = overlap of the CMB and tracer surveys.  ell_min is raised to 2 pi / sqrt(Omega) for small fields."""
    R = filt.sigma2_reion / filt_ref.sigma2_reion
    omega = 4 * np.pi * f_sky
    ell_min = max(ell_min, 2 * np.pi / np.sqrt(omega))
    ell = np.unique(np.round(np.geomspace(ell_min, ell_max, n_ell)).astype(int))
    zc = np.asarray([z0], float)
    b = float(np.asarray(bias_fn(zc)).ravel()[0])
    out_s, out_n = [], []
    for el in ell:
        D = sig.D(z0, dz, el, bias_fn) * R
        Cs = D * 2 * np.pi / (el * (el + 1))                   # uK^2
        Cgg = b ** 2 * limber_cmm(sig.toy, z0, dz, el) + (0.0 if n2d_per_sr is None else 1.0 / n2d_per_sr)
        out_s.append(Cs)
        out_n.append(filt.C_T2T2 * Cgg)
    Cs, Nn = np.array(out_s), np.array(out_n)
    # integrate over ell as a sum over integers: interpolate the per-ell terms to every integer ell
    ei = np.arange(int(ell[0]), int(ell[-1]) + 1)
    term = np.interp(ei, ell, (2 * ell + 1) * Cs ** 2 / (Nn + Cs ** 2))
    sn2 = f_sky * float(np.sum(term))
    if return_parts:
        return float(np.sqrt(max(sn2, 0.0))), {"ell": ell, "Cs": Cs, "noise": Nn, "R": R, "b": b}
    return float(np.sqrt(max(sn2, 0.0)))


def shell_geometry(toy, z, dz):
    """(chi, dchi, chi^2 dchi) of a redshift shell [Mpc]."""
    chi = float(np.interp(z, toy.z, toy.chi))
    lo, hi = np.interp([z - 0.5 * dz, z + 0.5 * dz], toy.z, toy.chi)
    return chi, float(hi - lo), chi ** 2 * float(hi - lo)


# =====================================================================================================================
# LAE number density and bias (abundance matching)
# =====================================================================================================================
def lae_number_density(L_min: float, lf_key: tuple, cfg=None) -> float:
    """n(> L_min) [Mpc^-3] from an observed LF fit in data/reference/luminosity_functions/lae_lf_fits.csv."""
    from ksz_lae_xcorr.io.luminosity_functions import lae_lf_fits, lya_cumulative
    fits = lae_lf_fits()
    ref, z = lf_key
    i = [k for k in range(len(fits["z"])) if str(fits["ref_key"][k]) == ref and abs(float(fits["z"][k]) - z) < 1e-6][0]
    return lya_cumulative(fits, i, float(np.log10(L_min)))


class HaloBias:
    """Sheth-Tormen n(>M) and Sheth-Mo-Tormen bias, for abundance matching (tracers/suite.py)."""

    def __init__(self, cfg, backend: str = "eh"):
        from ksz_lae_xcorr.tracers.suite import SigmaOfM
        c = cfg.cosmology
        self.cfg = cfg
        self.sig = SigmaOfM(c.Om0, c.Ob0, c.H0 / 100.0, c.ns, backend=backend)
        self.Om = c.Om0

    def n_gt(self, M, z):
        from ksz_lae_xcorr.tracers.suite import sheth_tormen_n_gt
        return sheth_tormen_n_gt(M, z, self.sig)

    def bias(self, M, z):
        from ksz_lae_xcorr.tracers.suite import sheth_tormen_bias
        return float(sheth_tormen_bias(np.asarray([M], float), z, self.sig)[0])

    def mass_for_density(self, n_target, z, log10m_range=(8.0, 13.5), n=80):
        """M [Msun] with n(> M) = n_target (ST)."""
        lm = np.linspace(*log10m_range, n)
        ng = np.array([self.n_gt(10 ** x, z) for x in lm])
        return float(10 ** np.interp(np.log(n_target), np.log(ng[::-1]), lm[::-1]))

    def bias_for_density(self, n_target, z):
        M = self.mass_for_density(n_target, z)
        return self.bias(M, z), M
