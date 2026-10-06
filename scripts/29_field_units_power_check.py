#!/usr/bin/env python3
"""
scripts/29_field_units_power_check.py
=======================================
READ-ONLY diagnostic (no pipeline output): compare the raw coeval velocity_z and
hires_density power spectra against CAMB linear theory at several redshifts and
report the ratios and their k-dependence (correlation/field_units_check.py).
Needs CAMB. Memory ~4 GB; run on a compute node (interactive qsub is fine).

    python scripts/29_field_units_power_check.py --seed 1 --z 6 9 12 15
"""
import argparse
import glob
import os
import sys

import numpy as np

from ksz_lae_xcorr.correlation.field_units_check import band_median, compare_to_linear_theory
from ksz_lae_xcorr.lightcone.stitch import Stitcher, _growth_rate_linder
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.field_units import growth_factor_ratio


def camb_pk(cfg, sigma8):
    import camb
    c = cfg.cosmology
    h = c.H0 / 100.0
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=c.H0, ombh2=c.Ob0 * h ** 2, omch2=(c.Om0 - c.Ob0) * h ** 2)
    pars.InitPower.set_params(ns=c.ns, As=2.1e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    res = camb.get_results(pars)
    s8_0 = res.get_sigma8_0()
    Omega0 = (sigma8 / s8_0) ** 2
    PK = camb.get_matter_power_interpolator(pars, zmin=0, zmax=0.5, nz_step=3, kmax=20.0,
                                            nonlinear=False, hubble_units=False, k_hunit=False)
    Om0 = float(c.Om0)

    def Pk(k, z):                       # P_lin(k, z) = D(z)^2 P_lin(k, 0), sigma8-rescaled
        return Omega0 * PK.P(0.0, k) * float(growth_factor_ratio(Om0, z)) ** 2
    return Pk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fiducial.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--z", type=float, nargs="+", default=[6.0, 9.0, 12.0, 15.0])
    ap.add_argument("--sigma8", type=float, default=0.8102, help="21cmFAST default; +-1%% in sigma8 = +-2%% in ratios")
    args = ap.parse_args()

    cfg = load_config(args.config)
    st = Stitcher(cfg)
    cosmo = get_cosmology(cfg)
    Pk = camb_pk(cfg, args.sigma8)
    root = os.path.join(cfg.paths.coeval_root, f"seed_{args.seed}")
    zs_avail = np.array(sorted(float(os.path.basename(d)[8:]) for d in glob.glob(os.path.join(root, "coeval_z*"))))
    print(f"seed {args.seed}; sigma8={args.sigma8}; ratios over k=0.04-0.4 /Mpc: median [rms dex-scatter in ln]\n")
    print("   z      R_v=Pv/Pv_pec   (1+z)^2    sqrt(R_v)   R_d(z=0 norm)   R_d(already d(z))  D(z)^-2")
    rows = []
    for zt in args.z:
        z = float(zs_avail[np.argmin(np.abs(zs_avail - zt))])
        vz = np.load(os.path.join(root, f"coeval_z{z:.6f}", "velocity_z.npy"), mmap_mode="r")
        vz = np.array(vz, dtype=np.float32)
        dl = np.asarray(st.load_field_box(args.seed, z, "density"), dtype=np.float32)
        r = compare_to_linear_theory(dl, vz, cfg.box.box_len_mpc, z, Pk, float(cosmo.H(z).to_value("km/s/Mpc")),
                                     float(_growth_rate_linder(cosmo, z)))
        rv, sv = band_median(r, "R_v"); rd0, sd0 = band_median(r, "R_d_z0"); rdz, sdz = band_median(r, "R_d_z")
        print(f"{z:7.3f}  {rv:9.3f} [{sv:.2f}]  {(1+z)**2:8.2f}  {np.sqrt(rv):8.3f}   {rd0:9.3f} [{sd0:.2f}]   "
              f"{rdz:9.3f} [{sdz:.2f}]   {float(growth_factor_ratio(cfg.cosmology.Om0, z))**-2:8.2f}", flush=True)
        rows.append(r)
    print("\nk-dependence of R_v (should be flat if it is a pure units factor):")
    print("      k   " + "  ".join(f"z={r['z']:.1f}" for r in rows))
    for i, k in enumerate(rows[0]["k"]):
        print(f"{k:8.3f}  " + "  ".join(f"{r['R_v'][i]:7.2f}" for r in rows))
    print("\nk-dependence of R_d (z=0-normalised hypothesis; flat and ~1 => hires_density is the z=0 linear IC):")
    for i, k in enumerate(rows[0]["k"]):
        print(f"{k:8.3f}  " + "  ".join(f"{r['R_d_z0'][i]:7.3f}" for r in rows))


if __name__ == "__main__":
    sys.exit(main())
