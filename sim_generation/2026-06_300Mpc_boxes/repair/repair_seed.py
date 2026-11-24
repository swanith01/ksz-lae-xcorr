#!/usr/bin/env python3
"""
repair_seed.py  (24 Nov 2026)
===============================
Repair one seed of the 300 Mpc boxes (codes/run_coeval_seed.py, 30 Jun 2026) without rerunning the simulation.

The production run saved (i) the initial-conditions density instead of the evolved density and (ii) the Lagrangian
(undisplaced) halo positions.  diagnostics/test_halo_consistency.py (job 1787492) showed that both can be regenerated
bit-for-bit from the initial conditions.  For every snapshot this script

  1. rebuilds the initial conditions with the production inputs and checks them against the saved hires_density.npy;
  2. reruns the production halo chain (determine_halo_catalog, low z -> high z) and checks masses and coordinates
     bit-for-bit against the saved halo_masses.npy / halo_coords.npy;
  3. displaces the haloes with 21cmFAST's perturb_halo_catalog  ->  halo_coords_eulerian.npy  (cMpc, same row order,
     so existing halo_ids stay valid);
  4. recomputes the perturbed field, checks velocity_z against the saved velocity_z.npy, and writes the evolved
     density  ->  density_evolved.npy  (delta on the 300^3 HII grid).

Existing files are never modified; new files are only written if every check at that snapshot passes, and the run
stops at the first failure (unless --force).
A per-snapshot summary goes to seed_<N>/repair_summary.csv.

    python repair_seed.py --seed 1            # under PBS: repair/pbs_repair.sh (needs 32 threads, as production)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--root", default="/user1/swanith/kSZ2_LAE_project_22Jun2026")
ap.add_argument("--force", action="store_true", help="write outputs even if a check fails (not recommended)")
ap.add_argument("--zmax", type=float, default=None, help="stop after this redshift (for a quick trial)")
args = ap.parse_args()

NTHREADS = int(os.environ.get("OMP_NUM_THREADS", "0"))
if NTHREADS != 32:
    sys.exit(f"OMP_NUM_THREADS={NTHREADS}: the production run used 32 threads and the halo sampler's random numbers "
             "are per thread, so run with OMP_NUM_THREADS=32.")

import py21cmfast as p21c  # noqa: E402

seed_dir = os.path.join(args.root, f"seed_{args.seed}")
if not os.path.isdir(seed_dir):
    sys.exit(f"no such directory: {seed_dir}")
t0 = time.time()
log = lambda *a: print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)
log(f"py21cmfast {p21c.__version__}, seed {args.seed}, {NTHREADS} threads, root {args.root}")

# ---- production inputs, copied from codes/run_coeval_seed.py ----
p21c.config["HALO_CATALOG_MEM_FACTOR"] = 3.0
node_redshifts = np.array(p21c.get_logspaced_redshifts(min_redshift=5.0, max_redshift=20.0, z_step_factor=1.02))
inputs = p21c.InputParameters(
    node_redshifts=node_redshifts,
    random_seed=args.seed,
    simulation_options=p21c.SimulationOptions(
        BOX_LEN=300.0, HII_DIM=300, DIM=600, N_THREADS=NTHREADS, Z_HEAT_MAX=20.0,
        SAMPLER_MIN_MASS=1e8, SAMPLER_BUFFER_FACTOR=2.0,
    ),
    matter_options=p21c.MatterOptions(KEEP_3D_VELOCITIES=True, USE_INTERPOLATION_TABLES="hmf-interpolation"),
    astro_options=p21c.AstroOptions(INHOMO_RECO=True, USE_TS_FLUCT=True),
)
L = inputs.simulation_options.BOX_LEN


def zdir(z):
    return os.path.join(seed_dir, f"coeval_z{z:.6f}")


def trim(cat):
    m = np.asarray(cat.get("halo_masses")).astype(np.float32)
    c = np.asarray(cat.get("halo_coords")).astype(np.float32)
    ok = m > 0
    return m[ok], c[ok]


missing = [z for z in node_redshifts if not os.path.isdir(zdir(z))]
if missing:
    sys.exit(f"{len(missing)} snapshot folders missing, e.g. {zdir(missing[0])}")

# ---- 1. initial conditions ----
ics = p21c.compute_initial_conditions(inputs=inputs)
saved_ic = np.load(os.path.join(zdir(node_redshifts.min()), "hires_density.npy"))
ic_ok = np.array_equal(np.asarray(ics.get("hires_density")).astype(np.float32), saved_ic)
del saved_ic
log(f"initial conditions rebuilt; identical to saved hires_density.npy: {ic_ok}")
if not ic_ok and not args.force:
    sys.exit("initial conditions differ from the production run: stopping, nothing written.")

# ---- 2-4. per snapshot, low z -> high z (the production halo-chain order) ----
rows, prev, n_fail = [], None, 0
for z in np.sort(node_redshifts):
    if args.zmax is not None and z > args.zmax:
        break
    d = zdir(z)
    hc = p21c.determine_halo_catalog(redshift=z, initial_conditions=ics, descendant_halos=prev,
                                    inputs=inputs)
    prev = hc
    m_new, c_new = trim(hc)
    m_old = np.load(os.path.join(d, "halo_masses.npy"))
    c_old = np.load(os.path.join(d, "halo_coords.npy"))
    halos_ok = m_new.shape == m_old.shape and np.array_equal(m_new, m_old) and np.array_equal(c_new, c_old)
    del m_old, c_old

    ph = p21c.perturb_halo_catalog(initial_conditions=ics, halo_catalog=hc, inputs=inputs)
    m_p, c_p = trim(ph)
    del ph
    order_ok = np.array_equal(m_p, m_new)
    dvec = (c_p - c_new + L / 2) % L - L / 2
    disp = np.sqrt((dvec.astype(np.float64) ** 2).sum(1))
    del dvec

    pf = p21c.perturb_field(redshift=z, initial_conditions=ics, inputs=inputs)
    dens = np.asarray(pf.get("density")).astype(np.float32)
    vz_new = np.asarray(pf.get("velocity_z")).astype(np.float64)
    del pf
    vz_ok = np.array_equal(vz_new, np.load(os.path.join(d, "velocity_z.npy")))
    del vz_new

    ok = halos_ok and order_ok and vz_ok
    if ok or args.force:
        np.save(os.path.join(d, "halo_coords_eulerian.npy"), c_p.astype(np.float32))
        np.save(os.path.join(d, "density_evolved.npy"), dens)
    n_fail += not ok
    row = dict(z=f"{z:.6f}", n_halos=m_new.size, halos_match_saved=halos_ok, perturbed_order_ok=order_ok,
               velocity_z_match_saved=vz_ok, written=ok or args.force,
               disp_median_cMpc=round(float(np.median(disp)), 3) if disp.size else "",
               disp_p99_cMpc=round(float(np.percentile(disp, 99)), 3) if disp.size else "",
               density_std=round(float(dens.std()), 4), density_min=round(float(dens.min()), 4))
    rows.append(row)
    log(" ".join(f"{k}={v}" for k, v in row.items()))
    del m_new, c_new, m_p, c_p, disp, dens
    if not ok and not args.force:
        log("a check failed: stopping here (nothing written for this snapshot); see the line above")
        break

with open(os.path.join(seed_dir, "repair_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
log(f"seed {args.seed}: {len(rows)} snapshots, {n_fail} failed a check; summary {seed_dir}/repair_summary.csv")
sys.exit(1 if n_fail else 0)
