"""
tests/test_lae_id_convention.py
=================================
Validation originally requested by G. Kulkarni (Slack, Jul 2026): establish
which array Jahaan's LAE/LBG `halo_ids_*` indices refer to -- the full halo
catalogue, or a mass-cut subset.

RESOLVED (Aug 2026, against real 300 Mpc data): ids index the FULL halo
array. Verified directly at z=10.085533, seed=1: the mass-cut subset
(>3.162e9 Msun) has only 350,749 halos, while LAE ids reach as high as
52,032,970 and LBG ids as high as 73,224,724 -- both far exceed the
mass-cut subset but fit exactly within the full catalogue (73,224,732
halos at that z). lightcone/stitch.py's load_lae_grid/load_lbg_grid were
corrected to match (see git history -- an earlier version incorrectly
applied the mass cut before indexing, based on an unverified assumption).

Kept as an ongoing regression check: run this against any new (seed, z)
pair (e.g. once new snapshots are added, or missing ones are backfilled)
to confirm the convention still holds and the loader is reading the
correct array.

Usage:
    python tests/test_lae_id_convention.py --seed 1 --z 10.085533 --config configs/fiducial.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.utils.config import load_config  # noqa: E402
from ksz_lae_xcorr.utils.external_catalogue import external_catalogue_filename  # noqa: E402


def round_trip_check(cfg, seed: int, z: float, reference_positions_path: str | None = None):
    halo_dir = os.path.join(cfg.paths.halo_root, f"seed_{seed}", "halo_catalogs")
    coords_path = os.path.join(halo_dir, f"halo_coords_z{z:.6f}.npy")
    masses_path = os.path.join(halo_dir, f"halo_masses_z{z:.6f}.npy")
    id_fname = external_catalogue_filename("halo_ids_obs", z, cfg, seed)
    id_path = os.path.join(cfg.paths.lae_catalogue_root, "halo_ids_obs", id_fname)

    for p, label in [(coords_path, "halo coords"), (masses_path, "halo masses"), (id_path, "LAE ids")]:
        if not os.path.exists(p):
            print(f"MISSING {label}: {p}")
            print("Cannot run round-trip check -- this file doesn't exist.")
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

    fits_full = ids.max() < len(coords)
    fits_mass_cut = ids.max() < len(mass_cut_coords)
    print("Bounds check (expect: fits full=True, fits mass-cut=False):")
    print(f"  ids fit within full catalogue ({len(coords):,})?      {fits_full}")
    print(f"  ids fit within mass-cut subset ({len(mass_cut_coords):,})? {fits_mass_cut}")
    if fits_full and not fits_mass_cut:
        print("  -> OK, matches the confirmed convention (full-array indexing).")
    elif fits_mass_cut:
        print("  -> UNEXPECTED: ids fit within the mass-cut subset too. Worth a closer look --")
        print("     this doesn't match the previously-confirmed pattern for this dataset.")
    else:
        print("  -> UNEXPECTED: ids don't fit either array. Something else is going on --")
        print("     check with Jahaan directly.")
    print()

    candidate_full = coords[ids]
    print(f"Candidate LAE positions (full-array convention): N={len(candidate_full)}, "
          f"mean=({candidate_full[:,0].mean():.2f}, {candidate_full[:,1].mean():.2f}, "
          f"{candidate_full[:,2].mean():.2f})")

    if reference_positions_path and os.path.exists(reference_positions_path):
        ref = np.load(reference_positions_path)
        match = np.allclose(np.sort(candidate_full, axis=0), np.sort(ref, axis=0), atol=1e-3)
        print(f"\nGround truth comparison against {reference_positions_path}: match={match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z", type=float, default=10.085533)
    parser.add_argument("--config", type=str, default="configs/fiducial.yaml")
    parser.add_argument("--reference-positions", type=str, default=None,
                         help="Path to Jahaan-provided ground-truth LAE positions, if available")
    args = parser.parse_args()

    cfg = load_config(args.config)
    round_trip_check(cfg, args.seed, args.z, args.reference_positions)
