#!/usr/bin/env python3
"""
test_halo_consistency.py  (24 Nov 2026)
=========================================
Small-box test (50 cMpc, same cell sizes and options as the 300 Mpc production run, py21cmfast 4.1.0) of three
questions about codes/run_coeval_seed.py:

  Q1  Are our pass-1 haloes (determine_halo_catalog chained low z -> high z, as in production) identical to the
      haloes generate_coeval itself used as ionising sources?
  Q2  Are pass-1 halo coordinates the initial (Lagrangian) positions?  How far does 21cmFAST's own
      perturb_halo_catalog move them at z ~ 6-9?
  Q3  Is perturbed_field.density the evolved density (grows with time, delta >= -1), and does a stand-alone
      perturb_field(ICs) reproduce generate_coeval's density and velocity_z exactly?  (This is the cheap repair path.)

Read-only for production data: everything is written to a temporary directory that is deleted at the end.
    python test_halo_consistency.py            (run under PBS: diagnostics/pbs_test_halo_consistency.sh)
"""
import os
import sys
import tempfile
import time

import numpy as np
import py21cmfast as p21c

print(f"py21cmfast {p21c.__version__}, N_THREADS {os.environ.get('OMP_NUM_THREADS')}", flush=True)
TARGETS = [6.03, 7.08, 9.04]
L = 50.0

node_redshifts = np.array(p21c.get_logspaced_redshifts(min_redshift=5.0, max_redshift=20.0, z_step_factor=1.02))
inputs = p21c.InputParameters(            # production settings, box 300 -> 50 cMpc at the same cell sizes
    node_redshifts=node_redshifts,
    random_seed=1,
    simulation_options=p21c.SimulationOptions(
        BOX_LEN=L, HII_DIM=50, DIM=100,
        N_THREADS=int(os.environ.get("OMP_NUM_THREADS", 8)),
        Z_HEAT_MAX=20.0, SAMPLER_MIN_MASS=1e8, SAMPLER_BUFFER_FACTOR=2.0,
    ),
    matter_options=p21c.MatterOptions(KEEP_3D_VELOCITIES=True, USE_INTERPOLATION_TABLES="hmf-interpolation"),
    astro_options=p21c.AstroOptions(INHOMO_RECO=True, USE_TS_FLUCT=True),
)
p21c.config["HALO_CATALOG_MEM_FACTOR"] = 3.0
zt = [float(node_redshifts[np.argmin(abs(node_redshifts - t))]) for t in TARGETS]
print("target redshifts:", [f"{z:.6f}" for z in zt], flush=True)


def trim(hc):
    m = np.asarray(hc.get("halo_masses"))
    c = np.asarray(hc.get("halo_coords"))
    ok = m > 0
    return m[ok], c[ok]


def wrapdiff(a, b):
    return (a - b + L / 2) % L - L / 2


with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmp:
    t0 = time.time()
    # ---------------- pass 1, exactly as production (no cache, ascending z, chained) ----------------
    ics = p21c.compute_initial_conditions(inputs=inputs)
    pass1, pass1_obj, prev = {}, {}, None
    for z in np.sort(node_redshifts):
        hc = p21c.determine_halo_catalog(redshift=z, initial_conditions=ics, descendant_halos=prev, inputs=inputs)
        prev = hc
        if any(abs(z - t) < 1e-9 for t in zt):
            pass1[float(z)] = trim(hc)
            pass1_obj[float(z)] = hc
    print(f"pass 1 done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- full generate_coeval in its own cache ----------------
    cache = p21c.OutputCache(os.path.join(tmp, "cache"))
    fields = {}
    for coeval, _ in p21c.generate_coeval(inputs=inputs, cache=cache, write=True):
        z = float(coeval.redshift)
        if any(abs(z - t) < 1e-9 for t in zt):
            fields[z] = dict(density=np.array(coeval.perturbed_field.get("density")),
                             vz=np.array(coeval.perturbed_field.get("velocity_z")),
                             ic=np.array(coeval.initial_conditions.get("hires_density")))
    print(f"generate_coeval done ({time.time() - t0:.0f} s)", flush=True)
    run = p21c.RunCache.from_inputs(inputs, cache)

    print("\n================ Q1: pass-1 haloes vs generate_coeval's own haloes ================")
    for z in zt:
        m1, c1 = pass1[z]
        m2, c2 = trim(run.get_output_struct_at_z("HaloCatalog", z=z))
        same_n = m1.size == m2.size
        exact = same_n and np.array_equal(m1, m2) and np.array_equal(c1, c2)
        line = f"z={z:7.4f}: N pass1 {m1.size:,}  N internal {m2.size:,}  identical (masses+coords, same order): {exact}"
        if not exact:
            bins = np.logspace(8, 13, 11)
            h1, _ = np.histogram(m1, bins); h2, _ = np.histogram(m2, bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                line += f"\n           mass-function ratio pass1/internal per dex/2: {np.round(h1 / h2, 3)}"
            if same_n:
                o1, o2 = np.lexsort(c1.T), np.lexsort(c2.T)
                line += f"\n           same set in a different order: {np.array_equal(m1[o1], m2[o2]) and np.array_equal(c1[o1], c2[o2])}"
        print(line)

    print("\n================ Q2: pass-1 coordinates vs 21cmFAST's displaced (perturbed) positions ================")
    for z in zt:
        m1, c1 = pass1[z]
        ph = p21c.perturb_halo_catalog(initial_conditions=ics, halo_catalog=pass1_obj[z], inputs=inputs)
        mp, cp = trim(ph)
        if mp.size != m1.size or not np.array_equal(mp, m1):
            print(f"z={z:7.4f}: perturbed catalogue has {mp.size:,} haloes vs {m1.size:,}; masses in same order: "
                  f"{mp.size == m1.size and np.array_equal(mp, m1)}")
            continue
        d = wrapdiff(cp, c1)
        r = np.sqrt((d**2).sum(1))
        big = m1 > 1e10
        print(f"z={z:7.4f}: coords range [{c1.min():.2f}, {c1.max():.2f}] (box {L} cMpc)  "
              f"displacement |d|: median {np.median(r):.2f}, rms {np.sqrt((r**2).mean()):.2f}, 99% {np.percentile(r, 99):.2f} cMpc;"
              f"  M>1e10: median {np.median(r[big]) if big.any() else np.nan:.2f} cMpc ({big.sum():,} haloes)")

    print("\n================ Q3: densities ================")
    stds = []
    for z in zt:
        f = fields[z]
        dens, ic = f["density"], f["ic"]
        pf = p21c.perturb_field(redshift=z, initial_conditions=ics, inputs=inputs)
        same_d = np.array_equal(np.asarray(pf.get("density")), dens)
        same_v = np.array_equal(np.asarray(pf.get("velocity_z")), f["vz"])
        stds.append(dens.std())
        print(f"z={z:7.4f}: perturbed density std {dens.std():.4f}, min {dens.min():.3f} (>= -1: {dens.min() >= -1}); "
              f"IC hires_density std {ic.std():.4f}, min {ic.min():.3f}; "
              f"stand-alone perturb_field reproduces density: {same_d}, velocity_z: {same_v}")
    print("perturbed density std ratio relative to the lowest z:", np.round(np.array(stds) / stds[0], 3),
          "(must fall with z; the IC field is the same at every z)")
    print(f"\nTotal {time.time() - t0:.0f} s")
