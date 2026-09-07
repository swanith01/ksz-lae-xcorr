#!/usr/bin/env python3
"""
scripts/09_coherence_decomposition.py
========================================
P_diag / P_off decomposition of each seed's own stitched kSZ auto-power
(correlation.coherence_decomposition), per Girish's 2026-09-07 steer:
before spending compute on an independent coeval-direct/Limber
recomputation, check how much of the current stitched-vs-expectation gap
is already explained by periodicity, using the SAME technique already
validated in ksz-pipeline (P_diag matched their independent direct calc
to 8.5%).

This does NOT decompose the kSZ2 x tracer cross-power -- see the module
docstring in correlation/coherence_decomposition.py for why that's a
separate, harder problem. This script only addresses the kSZ auto-power
that also feeds snr.cmb_filter.kSZ_reion_from_sim.

Usage:
    python scripts/09_coherence_decomposition.py
    python scripts/09_coherence_decomposition.py --config configs/variants/quicktest.yaml
"""

import argparse
import os
import pickle
import sys

import numpy as np

from ksz_lae_xcorr.correlation.coherence_decomposition import (
    compute_ksz_slices,
    cross_power_by_dchi,
    decompose_p_total_diag_off,
    group_slices_by_snapshot,
    random_shift_slices,
)
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.lightcone.stitch import Stitcher, setup_logger
from ksz_lae_xcorr.plotting.spectra_plots import plot_coherence_decomposition, plot_dchi_periodicity
from ksz_lae_xcorr.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--out-dir", type=str, default="paper/figure_scripts/output")
    parser.add_argument("--run-shift-control", action="store_true",
                         help="Also run the random_shift_slices control "
                              "(destroys periodicity, should collapse P_off "
                              "while leaving P_diag unchanged) -- slower, "
                              "off by default.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.box.seeds
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(cfg.paths.products_root, exist_ok=True)

    print("Loading stitched lightcone products...")
    field_data, _tracer_data = load_lightcone_products(cfg, seeds)
    print(f"  {len(field_data)}/{len(seeds)} seeds have field data")

    stitcher = Stitcher(cfg)
    results = {}

    for seed in seeds:
        if seed not in field_data:
            print(f"  seed {seed}: no field data, skipping")
            continue

        print(f"\nSeed {seed}: computing per-pixel kSZ integrand...")
        theta_slices, chi_mpc, z = compute_ksz_slices(cfg, field_data[seed])
        chi_eff = float(np.interp(0.5 * (cfg.box.z_min + cfg.box.z_max), z, chi_mpc))

        print(f"  theta_slices shape={theta_slices.shape}  "
              f"decomposing at full n_lc_pix resolution...")
        ell, D_total, D_diag, D_off = decompose_p_total_diag_off(cfg, theta_slices, chi_eff)

        frac_off_hi_ell = np.nan
        if len(ell) > 0:
            i3000 = int(np.argmin(np.abs(ell - 3000)))
            denom = D_total[i3000] if D_total[i3000] != 0 else np.nan
            frac_off_hi_ell = D_off[i3000] / denom
            print(f"  ell~{ell[i3000]:.0f}: D_total={D_total[i3000]:.4g}  "
                  f"D_diag={D_diag[i3000]:.4g}  D_off/D_total={frac_off_hi_ell:.2%}")

        plot_coherence_decomposition(cfg, ell, D_total, D_diag, D_off, seed, args.out_dir)

        print("  grouping by real coeval snapshot for the Delta-chi diagnostic...")
        logger = setup_logger(seed, cfg.paths.lightcone_root)
        snap_z = stitcher.get_snapshot_redshifts(seed, logger)
        theta_grouped, chi_grouped = group_slices_by_snapshot(cfg, theta_slices, chi_mpc, snap_z)
        dchi_c, dchi_mean, dchi_std, dchi_n = cross_power_by_dchi(
            theta_grouped, chi_grouped, cfg.box.box_len_mpc, n_dchi_bins=20
        )
        plot_dchi_periodicity(cfg, dchi_c, dchi_mean, dchi_std, seed, args.out_dir)

        seed_result = {
            "ell": ell, "D_total": D_total, "D_diag": D_diag, "D_off": D_off,
            "chi_eff": chi_eff, "frac_off_at_ell3000": frac_off_hi_ell,
            "dchi_centers": dchi_c, "dchi_mean": dchi_mean,
            "dchi_std": dchi_std, "dchi_n": dchi_n,
        }

        if args.run_shift_control:
            print("  running random-shift control (breaks periodicity, "
                  "should collapse P_off while D_diag stays fixed)...")
            shifted = random_shift_slices(theta_slices, seed=seed)
            _, D_total_s, D_diag_s, D_off_s = decompose_p_total_diag_off(cfg, shifted, chi_eff)
            max_diag_drift = float(np.nanmax(np.abs(D_diag_s - D_diag)))
            print(f"    max |D_diag drift| after shift = {max_diag_drift:.3e} "
                  f"(should be ~0, floating-point only)")
            seed_result["D_off_after_shift"] = D_off_s

        results[seed] = seed_result

    out_path = os.path.join(cfg.paths.products_root, "coherence_decomposition.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_path}")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
