#!/usr/bin/env python3
"""
scripts/08_compute_realistic_snr.py
======================================
Realistic LAE survey-selection S/N forecast, layered on top of the
"optimistic" forecast from scripts/05_compute_snr.py. Refactored from
Cells 10a-10d and 11a-11h.

For every survey in configs/lae_surveys.yaml:
  - "optimistic-with-shot-noise": full simulated LAE field, but with the
    auto-power's shot-noise term corrected for the survey's real
    (flux-cut-reduced) number density.
  - "realistic" (SILVERRUSH and Roman-Grism only, per their cut_type):
    a genuinely new spatial field keeping only LAEs that pass the survey's
    flux/REW cut at each redshift, both signal and noise recomputed from
    that field.

Requires, in order:
    scripts/02_stitch_lightcones.py       (core lightcones)
    scripts/04_compute_xcorr.py           (maps + auto_results.pkl)
    scripts/05_compute_snr.py             (filt ingredients + filtered kSZ^2)
    scripts/07_stitch_lae_value_fields.py (lc_lae_lum_3d.npz, lc_lae_rew_3d.npz)

Usage:
    python scripts/08_compute_realistic_snr.py
"""

import argparse
import os
import pickle
import sys

import numpy as np

from ksz_lae_xcorr.correlation.power_spectra import KGrid
from ksz_lae_xcorr.io.loaders import load_lightcone_products
from ksz_lae_xcorr.snr.snr_forecast import (
    build_filtered_kSZ2_maps, compute_filtered_signal, compute_lae_auto_power,
)
from ksz_lae_xcorr.snr.survey_selection import (
    apply_shot_noise_correction, build_survey_cut_field, compute_duty_cycle,
    compute_n_bar_sim, compute_realistic_power_spectra, compute_survey_snr,
    roman_grism_lmin, silverrush_lmin, silverrush_rewmin, total_snr,
)
from ksz_lae_xcorr.utils.config import load_config
from ksz_lae_xcorr.utils.cosmology import get_cosmology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--surveys-config", type=str, default="configs/lae_surveys.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    surveys_cfg = load_config(args.surveys_config)
    seeds = cfg.box.seeds
    cosmo = get_cosmology(cfg)

    print("Loading tracer data and CMB-filter/kSZ products...")
    _, tracer_data = load_lightcone_products(cfg, seeds)

    xcorr_path = f"{cfg.paths.products_root}/cross_results.pkl"
    snr_path = f"{cfg.paths.products_root}/snr_results.pkl"
    for p in (xcorr_path, snr_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run scripts 04 and 05 first.")
    with open(xcorr_path, "rb") as f:
        maps = pickle.load(f)["maps"]
    with open(snr_path, "rb") as f:
        base_snr = pickle.load(f)
    filt = base_snr["filt"]

    kg = KGrid(cfg)
    filtered_kSZ2 = build_filtered_kSZ2_maps(cfg, kg, maps["kSZ"], filt, seeds)

    dz = cfg.correlation.dz_tracer_bin
    z_edges = np.arange(cfg.box.z_min, cfg.box.z_max + dz, dz)
    z_cents = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_nodes = tracer_data[seeds[0]]["z_nodes"]

    Cl_lae_auto = compute_lae_auto_power(cfg, kg, tracer_data, seeds, z_edges, z_cents, filt["ell_grid"])
    n_bar_sim = compute_n_bar_sim(cfg, tracer_data, seeds, z_edges, z_cents, z_nodes, cosmo)

    results = {"optimistic_shot_noise": {}, "realistic": {}}

    # -- Level 1: optimistic-with-shot-noise, every named survey ------------
    print("\n--- Optimistic-with-shot-noise, per survey ---")
    lmin_fns = {
        "SILVERRUSH": lambda z: silverrush_lmin(surveys_cfg, z),
        "Roman-Grism": lambda z: roman_grism_lmin(surveys_cfg, cosmo, z),
    }
    for survey_name, spec in surveys_cfg.surveys.items():
        f_sky = spec.area_deg2 / 41253.0
        duty_cycle = {}
        if spec.has_flux_cut and survey_name in lmin_fns:
            duty_cycle = compute_duty_cycle(cfg, spec, cfg.paths.lightcone_root, seeds,
                                             z_edges, z_cents, z_nodes, lmin_fns[survey_name])
        cl_lae_auto_survey = apply_shot_noise_correction(Cl_lae_auto, n_bar_sim, duty_cycle,
                                                           spec.has_flux_cut)
        z_lo_s, z_hi_s = spec.z_range
        # signal power spectrum is the same filtered kSZ2 x LAE cross-power used in
        # the optimistic (script 05) pipeline -- restrict to this survey's z-range only.
        cl_signal_full = compute_filtered_signal(cfg, kg, filtered_kSZ2, tracer_data, seeds,
                                                   z_edges, z_cents, filt["ell_grid"])
        cl_signal_survey = {name: {z: v for z, v in d.items() if z_lo_s <= z <= z_hi_s}
                             for name, d in cl_signal_full.items()}
        cl_lae_auto_restricted = {z: v for z, v in cl_lae_auto_survey.items() if z_lo_s <= z <= z_hi_s}

        sn = compute_survey_snr(cfg, filt, cl_signal_survey, cl_lae_auto_restricted, f_sky)
        results["optimistic_shot_noise"][survey_name] = sn
        for name in cfg.snr.experiments:
            print(f"  {survey_name:12s} / {name:8s}: total S/N = {total_snr(sn.get(name, {})):.2f}"
                  f"  ({len(sn.get(name, {}))} z-bins, f_sky={f_sky:.4e})")

    # -- Level 2: realistic joint-cut field, SILVERRUSH + Roman-Grism -------
    print("\n--- Realistic (joint flux/REW cut), SILVERRUSH + Roman-Grism ---")
    realistic_specs = [
        ("SILVERRUSH", surveys_cfg.surveys["SILVERRUSH"],
         lambda z: silverrush_lmin(surveys_cfg, z), lambda z: silverrush_rewmin(surveys_cfg, z)),
        ("Roman-Grism", surveys_cfg.surveys["Roman-Grism"],
         lambda z: roman_grism_lmin(surveys_cfg, cosmo, z), None),
    ]
    for survey_name, spec, lmin_fn, rewmin_fn in realistic_specs:
        f_sky = spec.area_deg2 / 41253.0
        z_lo_s, z_hi_s = spec.z_range
        cut_fields = build_survey_cut_field(cfg.paths.lightcone_root, seeds, z_nodes, lmin_fn, rewmin_fn)
        if not cut_fields:
            print(f"  {survey_name}: no cut fields available (missing lc_lae_lum_3d.npz? "
                  f"run scripts/07_stitch_lae_value_fields.py first) -- skipping")
            continue
        cl_lae_auto_r, cl_signal_r = compute_realistic_power_spectra(
            cfg, cut_fields, filtered_kSZ2, seeds, z_lo_s, z_hi_s, z_edges, z_cents, z_nodes, filt["ell_grid"])
        sn_r = compute_survey_snr(cfg, filt, cl_signal_r, cl_lae_auto_r, f_sky)
        results["realistic"][survey_name] = sn_r
        for name in cfg.snr.experiments:
            opt_total = total_snr(results["optimistic_shot_noise"][survey_name].get(name, {}))
            real_total = total_snr(sn_r.get(name, {}))
            print(f"  {survey_name:12s} / {name:8s}: optimistic={opt_total:.2f}  realistic={real_total:.2f}")

    os.makedirs(cfg.paths.products_root, exist_ok=True)
    out_path = f"{cfg.paths.products_root}/realistic_snr_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
