"""
tests/test_lae_id_convention.py
=================================
Validation requested by G. Kulkarni (Slack, Jul 2026): establish, for one
seed and one redshift, exactly which array Jahaan's LAE/LBG `halo_ids_*`
indices refer to -- the full halo catalogue, or the mass-cut
(M_halo > tracers.lae_lbg_mass_cut_msun) subset -- and confirm the loader
in lightcone/stitch.py follows that convention.

This CANNOT run until real LAE/LBG catalogue files exist (paths.lae_catalogue_root
/ lbg_catalogue_root are still "TBD" in configs/fiducial.yaml as of this writing).
Run this as the very first thing once Jahaan hands over catalogues for even a
single seed/redshift -- before trusting any LAE cross-correlation result.

WHAT THIS CHECKS:
1. Index-bounds discrimination: if max(ids) fits within the mass-cut subset's
   length but NOT within a value that would make sense for the full array (or
   vice versa), that alone is near-conclusive.
2. Direct comparison against Jahaan's own output, IF he provides real-space
   LAE positions (not just indices) for the same seed/redshift -- the
   ultimate ground truth. Set --reference-positions if available.
3. Reports candidate position arrays under BOTH conventions side by side so
   a mismatch is immediately visible even without a ground-truth file.

Usage:
    python tests/test_lae_id_convention.py --seed 1 --z 6.5 --config configs/fiducial.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.utils.config import load_config  # noqa: E402


def round_trip_check(cfg, seed: int, z: float, reference_positions_path: str | None = None):
    halo_dir = os.path.join(cfg.paths.halo_root, f"seed_{seed}", "halo_catalogs")
    coords_path = os.path.join(halo_dir, f"halo_coords_z{z:.6f}.npy")
    masses_path = os.path.join(halo_dir, f"halo_masses_z{z:.6f}.npy")
    id_path = os.path.join(cfg.paths.lae_catalogue_root, "halo_ids_obs", f"halo_ids_obs_z{z:.4f}_s{seed}.npy")

    for p, label in [(coords_path, "halo coords"), (masses_path, "halo masses"), (id_path, "LAE ids")]:
        if not os.path.exists(p):
            print(f"MISSING {label}: {p}")
            print("Cannot run round-trip check yet -- this file doesn't exist.")
            return

    coords = np.load(coords_path)
    masses = np.load(masses_path)
    ids = np.load(id_path)
    mass_cut = float(cfg.tracers.lae_lbg_mass_cut_msun)

    mass_cut_sel = masses > mass_cut
    mass_cut_coords = coords[mass_cut_sel]

    print(f"seed={seed}  z={z}")
    print(f"  full halo catalogue      : {len(coords):>10,} halos")
    print(f"  mass-cut subset (>{mass_cut:.3e} Msun): {len(mass_cut_coords):>10,} halos")
    print(f"  LAE ids array            : {len(ids):>10,} entries, "
          f"min={ids.min()}, max={ids.max()}")
    print()

    # -- Bounds discrimination --------------------------------------------
    fits_full = ids.max() < len(coords)
    fits_mass_cut = ids.max() < len(mass_cut_coords)
    print("Bounds check:")
    print(f"  ids fit within full catalogue ({len(coords):,})?      {fits_full}")
    print(f"  ids fit within mass-cut subset ({len(mass_cut_coords):,})? {fits_mass_cut}")
    if fits_mass_cut and not fits_full:
        print("  -> inconclusive from bounds alone (mass-cut is smaller, so fitting")
        print("     it doesn't rule out the full array too -- need position comparison below)")
    elif fits_full and not fits_mass_cut:
        print("  -> STRONG EVIDENCE: ids only fit the FULL array. The mass-cut assumption")
        print("     in lightcone/stitch.py's load_lae_grid is WRONG for this data -- ids")
        print("     must index the full halo catalogue, not the mass-cut subset.")
    print()

    # -- Candidate positions under both conventions ------------------------
    candidate_full = coords[ids]
    candidate_mass_cut = mass_cut_coords[ids] if fits_mass_cut else None

    print("Candidate LAE positions under each convention:")
    print(f"  [A] ids -> full halo array      : N={len(candidate_full)}, "
          f"mean=({candidate_full[:,0].mean():.2f}, {candidate_full[:,1].mean():.2f}, "
          f"{candidate_full[:,2].mean():.2f})")
    if candidate_mass_cut is not None:
        print(f"  [B] ids -> mass-cut subset       : N={len(candidate_mass_cut)}, "
              f"mean=({candidate_mass_cut[:,0].mean():.2f}, {candidate_mass_cut[:,1].mean():.2f}, "
              f"{candidate_mass_cut[:,2].mean():.2f})")
    else:
        print("  [B] ids -> mass-cut subset       : ids out of bounds, not computable")
    print()

    # -- Ground truth comparison, if available ------------------------------
    if reference_positions_path and os.path.exists(reference_positions_path):
        ref = np.load(reference_positions_path)
        match_A = np.allclose(np.sort(candidate_full, axis=0), np.sort(ref, axis=0), atol=1e-3)
        match_B = (candidate_mass_cut is not None and
                   np.allclose(np.sort(candidate_mass_cut, axis=0), np.sort(ref, axis=0), atol=1e-3))
        print(f"Ground truth comparison against {reference_positions_path}:")
        print(f"  [A] full-array convention matches ground truth? {match_A}")
        print(f"  [B] mass-cut convention matches ground truth?   {match_B}")
        if match_B and not match_A:
            print("  CONCLUSION: mass-cut convention is correct. Current code (post-fix) is right.")
        elif match_A and not match_B:
            print("  CONCLUSION: full-array convention is correct. REVERT the mass-cut fix in")
            print("  lightcone/stitch.py's load_lae_grid/load_lbg_grid.")
        elif not match_A and not match_B:
            print("  CONCLUSION: NEITHER convention matches. Something else is going on --")
            print("  check ordering (sorted by mass? by original array position?) with Jahaan directly.")
    else:
        print("No ground-truth position file provided (--reference-positions).")
        print("Ask Jahaan whether he can export real-space LAE positions for this exact")
        print("seed/z alongside the ids, purely for this one-time validation -- that's the")
        print("only way to fully resolve this without ambiguity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z", type=float, default=6.5)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--reference-positions", type=str, default=None,
                         help="Path to Jahaan-provided ground-truth LAE positions, if available")
    args = parser.parse_args()

    cfg = load_config(args.config)
    round_trip_check(cfg, args.seed, args.z, args.reference_positions)
