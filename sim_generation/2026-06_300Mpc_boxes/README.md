# 300 Mpc coeval boxes (June–August 2026): code as run

Archived copy of the scripts that made the ten 300 Mpc boxes used by this repo, taken from
`/user1/swanith/kSZ2_LAE_project_22Jun2026/` on swarm (24 Nov 2026). Kept unchanged for provenance; the maintained
version of the generator is `scripts/01_run_coeval_seed.py` + `src/ksz_lae_xcorr/halos/coeval_pipeline.py`.

## What was run
- `codes/run_coeval_seed.py` (py21cmfast **4.1.0**, env `p21c_v41`), one PBS job per seed via `codes/submit_all_seeds.sh` →
  `codes/pbs_coeval.sh` (`select=1:ncpus=32:mem=40gb`, 32 OpenMP threads; the header comments in that file are stale).
- Settings: BOX_LEN 300 cMpc, HII_DIM 300 (1 cMpc cells), DIM 600 (0.5 cMpc), 65 snapshots z = 5.0–20.31
  (log-spaced, step factor 1.02), Z_HEAT_MAX 20, SAMPLER_MIN_MASS 1e8 Msun, SAMPLER_BUFFER_FACTOR 2,
  HALO_CATALOG_MEM_FACTOR 3, INHOMO_RECO, USE_TS_FLUCT, KEEP_3D_VELOCITIES, hmf-interpolation, 21cmFAST default cosmology,
  SAMPLE_METHOD MASS-LIMITED, SOURCE_MODEL CHMF-SAMPLER.
- Two passes: (1) halo catalogues with `determine_halo_catalog`, chained low z → high z; (2) `generate_coeval` for the fields.
- Production: seeds 1–10, jobs 1627636–1627647, 30 Jun 2026; 152–180 min per seed (pass 1 ≈ 12 min). 230 M haloes at
  z = 6, 267 M at z = 5 (seed 3). Jobs ran within a 40 GB request (peak memory was not logged).
- `codes/downsample_density_300.py` (17–18 Aug 2026, `codes/pbs_downsample_*.sh`): block-averages `hires_density.npy`
  (600³) to `density_300.npy` (300³) in every snapshot folder; this is the density Jahaan's LAE code read.
- `codes/verify_outputs.py`: file/shape checks; `codes/quick_lightcone_check.py`, `replot_lightcone.py`: quick-look plots.
- `diagnostics/`: small 50 Mpc test jobs (29–30 Jun) used to find the v4.1 field names and the halo-chaining order; seeds
  96, 97, 99 are test seeds. `logs/`: the PBS output of all of the above.

## Output per snapshot (`seed_<N>/coeval_z<z:.6f>/`)
| file | grid | what it really is |
|---|---|---|
| `hires_density.npy` | 600³ float32 | **initial-conditions density (linear, z = 0 normalised), the same at every z** |
| `density_300.npy` | 300³ float32 | block average of the above, so also the z = 0 initial field |
| `velocity_z.npy` | 300³ float64 | `perturbed_field.velocity_z`: comoving-coordinate rate in Mpc/s; peculiar velocity = value/(1+z) |
| `neutral_fraction.npy`, `kinetic_temperature.npy` | 300³ float32 | from `ionized_box`, correct at z |
| `halo_masses.npy`, `halo_coords.npy` | N, N×3 float32 | pass-1 `HaloCatalog` (Msun; coords 0–300) |

## Known issues (24 Nov 2026)
1. **Density is not the density at z.** Checked on seed 6 (`codes/check_density.py`): identical at z = 5–20.3, std 3.41, min −18.4,
   38% of cells with δ < −1, no growth. The evolved density is `perturbed_field.density` (HII_DIM grid); it was never saved.
   In this repo `utils/field_units.py` uses δ(z) = D(z)/D(0)·hires_density as a linear stand-in.
2. **Velocity units:** raw `velocity_z` needs 1/(1+z) (handled by `utils/field_units.py`).
3. **To check:** pass-1 halo coordinates come from `HaloCatalog`, which may be the undisplaced (Lagrangian) positions rather
   than the displaced ones of `PerturbHaloField`.
4. **To check:** pass-1 haloes are generated separately from the haloes `generate_coeval` used as ionising sources; whether
   the two catalogues are identical has not been verified.
5. The LAE catalogue built on these boxes (Jahaan, `/user1/jahaan/swanith/`, 26 Aug 2026) read `density_300.npy` and the raw
   `velocity_z.npy`; its observed (IGM-attenuated) luminosities need recomputing.
