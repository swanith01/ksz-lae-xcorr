# ksz-lae-xcorr

kSZ² × LAE cross-correlation during the Epoch of Reionization: forecasting
detectability with next-generation CMB experiments, using 21cmFAST coeval
boxes stitched into lightcones.

LBG cross-correlation is also computed alongside LAE, for physics
interpretation (comparing how the kSZ² signal correlates with a different,
UV-selected tracer population) — but the S/N forecast itself is **LAE-only**.
See "Scope: LAE vs LBG" below.

## Pipeline

```
scripts/01_run_coeval_seed.py   py21cmfast coeval boxes + two-pass halo catalogs
            |
scripts/02_stitch_lightcones.py  stitch coeval boxes + external LAE/LBG
            |                    catalogues into 3D lightcones
            |
scripts/03_make_type_b_grids.py  (optional) physical-value grids (halo
            |                    mass, Lya luminosity, MUV) for diagnostic plots
            |
scripts/04_compute_xcorr.py      projected 2D maps, cross-power spectra
            |                    (halo/LAE/LBG x kSZ2/xe2/v2/...), auto-power
            |
scripts/05_compute_snr.py        CMB filter, filtered kSZ2 x LAE, S/N vs z
            |
scripts/06_make_figures.py       all paper figures, non-interactive
```

Run from the repo root with the `ksz-lae-xcorr` conda environment active:

```bash
python scripts/01_run_coeval_seed.py --seed 1     # repeat per seed, or use pbs/submit_all_seeds.sh
python scripts/02_stitch_lightcones.py
python scripts/04_compute_xcorr.py
python scripts/05_compute_snr.py
python scripts/06_make_figures.py
```

Every script takes `--config configs/fiducial.yaml` by default; use
`configs/variants/` for alternative box sizes, seed counts, or CMB
experiment assumptions without touching the fiducial config.

Scripts `07`/`08` extend this with realistic LAE survey selection — see
"Realistic LAE survey selection" below; they're optional, not part of the
core `01`-`06` chain above.

## Scope: LAE vs LBG

Both LAE and LBG catalogues come from an external pipeline (see
`data/README.md`) and both are cross-correlated with the kSZ² signal
(`src/ksz_lae_xcorr/correlation/`). The production S/N forecast
(`src/ksz_lae_xcorr/snr/`) defaults to LAE — `configs/fiducial.yaml`'s
`snr.tracer: lae` is the single place this default is set, and
`scripts/05_compute_snr.py` is the LAE-only headline-forecast entry point.

As of the Stage 1 literature-benchmark work (see below), the SNR module
itself is **tracer-generic**: `snr_forecast.py`'s `run_snr_pipeline` and
friends take a `tracer_key` argument (default `'lae_count_lc'`, so
`scripts/05`'s behavior is unchanged) and also work with
`'lbg_count_lc'`. `scripts/10_stage1_lbg_benchmark.py` uses this to run
the same estimator against the real LBG catalogue. LBG cross-power
figures from `scripts/06` remain for the paper's physics discussion, not
the headline LAE detectability claim — that hasn't changed.

## Realistic LAE survey selection (extension)

Beyond the core "optimistic" S/N forecast (`scripts/05`, which assumes every
simulated LAE above the halo mass cut is observed), `scripts/07` and
`scripts/08` add a realistic-survey-selection layer, refactored from the
original analysis notebook's Cells 10-11:

```
scripts/07_stitch_lae_value_fields.py   full-3D LAE luminosity + REW fields
            |                           (lc_lae_lum_3d.npz, lc_lae_rew_3d.npz)
            |
scripts/08_compute_realistic_snr.py     for each survey in configs/lae_surveys.yaml:
                                           - "optimistic-with-shot-noise": full
                                             simulated field, shot-noise term
                                             corrected for the survey's real
                                             (flux-cut-reduced) density
                                           - "realistic" (SILVERRUSH, Roman-Grism):
                                             a genuinely new field keeping only
                                             LAEs passing the survey's flux/REW
                                             cut at each redshift
```

This requires a **new external input from Jahaan not needed by the core
pipeline**: rest-frame Lya equivalent width (`lya_rew_obs`), alongside the
luminosity his pipeline already provides -- see `data/README.md`.

Survey definitions, flux/REW cut tables, and the Roman-Grism luminosity-
distance-based flux ceiling all live in `configs/lae_surveys.yaml`, kept
separate from `configs/fiducial.yaml` since these are observational/survey
parameters, not simulation parameters.

`src/ksz_lae_xcorr/snr/survey_selection.py` implements this generically --
one code path reused across every named survey, rather than duplicated
per-survey blocks as in the original notebook.

## Periodicity diagnostic (P_diag/P_off)

`stitch.py`'s lightcone construction repeats the same finite 300 Mpc box
periodically along the line of sight to cover the full z=5-20 range --
the same mechanism the companion `ksz-pipeline` repo found inflates their
stitched kSZ auto-power spectrum relative to an independent direct/Limber
calculation. `scripts/09_coherence_decomposition.py` (backed by
`src/ksz_lae_xcorr/correlation/coherence_decomposition.py`) checks for
this here, WITHOUT needing a second independent calculation: it
decomposes the existing stitched kSZ map's power into P_diag (sum of each
LOS pixel's own auto-power -- periodicity-curbed) and P_off (the
cross-pixel term -- P_total minus P_diag, the periodicity artifact
itself). Measured result on the real 10-seed fiducial run: median
D_off/D_total = 68% at ell~3000 (10/10 seeds agree, range 64-75%),
i.e. stitched shows roughly D_total/D_diag ~ 3x more power than the
periodicity-curbed estimate at that scale -- worse than the companion
repo's 2.3x at their larger 800 Mpc box, consistent with their own
finding that the artifact's amplitude grows as the box shrinks.

This decomposes the kSZ AUTO-power only (the ingredient feeding
`snr/cmb_filter.py`'s `kSZ_reion_from_sim`, for which
`kSZ_reion_from_sim_diag` is the periodicity-curbed drop-in alternative)
-- not the kSZ2 x tracer CROSS-power itself, which is a different
(bispectrum-type) statistic this decomposition doesn't directly address.
See the module docstrings for the full reasoning and caveats.

```
scripts/09_coherence_decomposition.py    P_diag/P_off per seed + Delta-chi
            |                            periodicity check, from real data
            |                            already on disk (no new sim needed)
scripts/12_plot_coherence_summary.py     one seed-averaged summary plot
                                          (reads the seed_agg pickle 09 saves)
```

## Stage 1 literature benchmark (La Plante, Sipple & Lidz 2022)

Before trusting this repo's kSZ2 x galaxy cross-correlation for the
paper's own LAE forecast, `scripts/10`/`11` reproduce La Plante, Sipple &
Lidz 2022 (ApJ 928, 162; arXiv:2111.13717) as closely as this simulation
allows, as a pass/fail sanity check on the estimator itself:

```
scripts/10_stage1_lbg_benchmark.py       runs the existing SNR pipeline
            |                            (tracer_key='lbg_count_lc') against
            |                            this repo's REAL LBG catalogue --
            |                            the paper's actual target population
            |                            (Roman HLS Lyman-break galaxies),
            |                            not LAEs
scripts/11_roman_hls_dell_comparison.py  builds the galaxy field the SAME
                                          way the paper does (Eq.6-7: a
                                          linear-bias-weighted density
                                          field, bg(z)=2.1(1+z)-5.3 from
                                          Waters et al. 2016), decoupling
                                          the check from this repo's own
                                          (separately evolving) LBG
                                          catalogue, and plots D_ell
                                          against a hand-read reference
                                          point from the paper's Figure 4
```

Both live in `src/ksz_lae_xcorr/snr/roman_hls_benchmark.py` +
`snr_forecast.py`'s tracer-generic refactor above. Known, stated
limitations (see that module's docstring): the bias curve and box
periodicity caveat above both apply; shot noise is not yet included
(compare against the paper's own "without shot noise" Table 2 column);
and this repo's `instrument_noise()` implements the paper's Eq. 11 (naive
instrument-only noise), not their post-ILC residual-foreground model --
compare against Table 2's "Instrument Noise" column specifically, not
the abstract's headline sigma (that's the "ILC Noise" column).

## Cosmology

This repo standardizes on the **21cmFAST-default cosmology**
(`H0=67.77, Om0=0.3086, Ob0=0.0489`) everywhere — coeval box generation,
lightcone stitching (comoving-distance-to-pixel mapping), and the
correlation/SNR analysis (Limber `ell`, CAMB `C_ell^TT`). This matches the
`ksz2-21cm` repo and the underlying py21cmfast simulation itself, and is a
deliberate departure from `ksz-pipeline`'s astropy Planck18 preset. Do not
construct a second cosmology object anywhere in this repo — import
`ksz_lae_xcorr.utils.cosmology.get_cosmology(cfg)`.

(Earlier scratch code split this into two different cosmologies between
the stitching step and the analysis step; that inconsistency has been
resolved here — see git history for the fix.)

## Fiducial simulation

300 cMpc box, HII_DIM=300³ (velocity, xHI, kinetic temperature),
DIM=600³ (density, halos), seeds 1–10, z=5–20. See `configs/fiducial.yaml`
for the full parameter set, including the two-pass halo-catalog fix
(`src/ksz_lae_xcorr/halos/coeval_pipeline.py` docstring has the full
diagnostic writeup for why the two-pass design is necessary).

An earlier 400 Mpc / 64³ / 5-seed exploratory run exists in
`notebooks/exploratory/` for reference, but produced under-resolved,
unreliable LAE catalogues at that grid resolution and is not used for the
paper.

## Environment

```bash
conda env create -f environment.yml
conda activate ksz-lae-xcorr
```

## Data

See `data/README.md` for the full manifest: what's generated by this
pipeline vs. what comes from the external LAE/LBG catalogue pipeline, and
where each product lives (not committed to GitHub — see `.gitignore`).

## Repository layout

```
configs/         box/cosmology/path parameters -- single source of truth
src/ksz_lae_xcorr/
  halos/         py21cmfast coeval + two-pass halo catalog generation
  lightcone/     3D lightcone stitching
  tracers/       physical-value (mass/luminosity/MUV) grids for diagnostics
  correlation/   projected maps, cross-power, auto-power (halo/LAE/LBG),
                 periodicity decomposition (coherence_decomposition.py)
  snr/           CMB filter + S/N forecast (LAE-default, tracer-generic),
                 Stage 1 literature benchmark (roman_hls_benchmark.py)
  io/            product loaders
  plotting/      all figure-generating code
  utils/         config loader, cosmology, physical constants
scripts/         numbered, executable pipeline steps (see above)
notebooks/exploratory/   the original analysis notebook, kept for reference
paper/figure_scripts/    output figures for the paper live here
pbs/             cluster job scripts
tests/           smoke tests
```

## Quicktest (halo pipeline + stitching + cross-correlation sanity check)

Before committing to a full cluster run, `configs/variants/quicktest.yaml`
runs the exact same code path (`halos/` → `lightcone/` → `correlation/`) at
a tiny size (50 Mpc, HII_DIM=32, DIM=64, 1 seed, z=6–10) so it finishes in
under a minute on a desktop. It deliberately runs **halo tracer only** —
LAE/LBG are left out via `lightcone.fields.discrete: [halos]` and
`correlation.tracers: [halo]` until Jahaan's catalogues are available (see
`data/README.md`).

### One-time environment setup

21cmFAST v4 moves fast enough that conda-forge and PyPI can lag behind
whatever's actually installed in a working env. Confirm what you actually
have before assuming `environment.yml` is right:

```bash
conda activate <your-working-21cmfast-env>
python -c "import py21cmfast as p21c; print(p21c.__version__)"
python -c "import py21cmfast as p21c; print('determine_halo_catalog' in dir(p21c), 'generate_coeval' in dir(p21c))"
```

Both should print `True`/`True` for `determine_halo_catalog` and
`generate_coeval` — those are the two calls `halos/coeval_pipeline.py`
depends on. This repo is built against the official **21cmFAST v4.1.0**
PyPI release (`pip install 21cmFAST==4.1.0`), confirmed to match. If your
working env has a different version, check whether these two functions
exist under those exact names before assuming the code will just work —
an earlier dev-snapshot build we tried (`4.0.0b1.dev...`) had renamed them
(`compute_halo_grid`, `determine_halo_list`) and would have needed a
different `halos/coeval_pipeline.py` to match.

```bash
conda env create -f environment.yml
conda activate ksz-lae-xcorr
pip install -e .          # editable install of src/ksz_lae_xcorr -- required,
                           # scripts import it as a package, not via PYTHONPATH
python -c "import ksz_lae_xcorr; print('OK')"
```

### Running the quicktest

```bash
export OMP_NUM_THREADS=4   # a handful of cores is plenty at this size
python scripts/01_run_coeval_seed.py --seed 1 --config configs/variants/quicktest.yaml
python scripts/02_stitch_lightcones.py --seed 1 --config configs/variants/quicktest.yaml
python scripts/04_compute_xcorr.py --config configs/variants/quicktest.yaml
```

Expect ~1,000,000 halos at z=6 falling to a few hundred thousand by z=10.27
(monotonic decrease with z is the expected structure-formation trend), and
`cross_results.pkl` / `auto_results.pkl` under `quicktest_data/products/`.

### Inspecting results

```bash
python -c "
import pickle, numpy as np
with open('quicktest_data/products/cross_results.pkl', 'rb') as f:
    d = pickle.load(f)
cross = d['cross_results']
halo = cross['halo'][1]
for signal in ['kSZ2', 'xe2', 'v2']:
    for z in sorted(halo.keys()):
        D = halo[z][signal]['D_ell']
        print(f'{signal} z={z:.2f}: D_ell range [{np.nanmin(D):.4g}, {np.nanmax(D):.4g}]')
"
```

Large swings and sign changes in `D_ell` at this box size are **expected**,
not a bug — 32³ cells gives very few independent Fourier modes per k-bin,
so sample variance dominates. The 300 Mpc/300³ fiducial run averaged over
10 seeds is what actually beats that down; this test only confirms the
code path runs correctly end to end, not that the numbers mean anything
physically.

Plots (`plotting/lightcone_panels.py`, `plotting/spectra_plots.py`) can be
called directly on the quicktest products the same way `scripts/06` calls
them on real products — see git history around this section for worked
examples, or just ask.

### Known gotchas hit during this validation (fixed, but worth knowing)

- **PyYAML silently turns unsigned scientific notation into a string.**
  `1.0e10` parses as the string `'1.0e10'`, not the float `1e10` --
  `1.0e+10` (explicit sign) is required. All configs in this repo use the
  signed form; `lightcone/stitch.py` also defensively casts these values
  with `float(...)` so a future slip doesn't fail silently deep inside a
  numpy comparison.
- **`matplotlib.cm.get_cmap` was removed** in newer matplotlib --
  `matplotlib.colormaps["name"].resampled(n)` is the current API, used in
  `plotting/spectra_plots.py`.
- **`np.trapz` was removed** in newer NumPy (renamed `np.trapezoid` in
  2.0+) -- `snr/cmb_filter.py`'s `filtered_noise_power` uses a manual
  trapezoidal sum instead, so it works regardless of which NumPy version
  a given conda env happens to have.
- 21cmFAST's exact installed build matters more than usual right now (see
  environment setup above) -- check the two function names before
  assuming any dev/beta build matches this repo's API.

## Releases

- `v0.1` — tagged when the first full pipeline (steps 01–06) runs end to end.
- `submitted-v1` — tagged at journal submission.
- `accepted-v1` — tagged at acceptance.
