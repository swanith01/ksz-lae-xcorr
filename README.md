# ksz-lae-xcorr

kSZ² × LAE cross-correlation during the Epoch of Reionization: forecasting
detectability with next-generation CMB experiments, using 21cmFAST coeval
boxes stitched into lightcones.

LBG cross-correlation is also computed alongside LAE, for physics
interpretation (comparing how the kSZ² signal correlates with a different,
UV-selected tracer population) — but the S/N forecast itself is **LAE-only**.
See "Scope: LAE vs LBG" below.

## Start here: current status (2026-10-01)

**Trusted right now:**
- Lightcone construction and tracer counting (halo/LAE/LBG spatial
  distributions relative to xHI) — real, correctly-populated data
  confirmed for seed 1 (LAE max count 3, LBG max count 15, both
  nonzero); re-stitch for the remaining 9 seeds was in progress as of
  2026-09-09 (see the checkpoint gotcha below for why this needed
  fixing at all) — confirm all 10 landed before citing this as fully done.
- The velocity field feeding the **direct/coeval** kSZ construction —
  fixed 2026-09-09 (see "Velocity conversion" below), verified with a
  first-principles physics check on real data, not just an eyeball
  sanity check.
- The **direct/coeval** kSZ² × galaxy estimator (`correlation/direct_bispectrum.py`,
  `scripts/15`/`18`) — first real-data results land within ~15-20x of
  La Plante et al. 2022's published amplitude (down from ~1000-8000x
  before the 2026-09-09 velocity fix). Not a validated match, but the most
  trustworthy number this repo currently has for the cross-correlation.
- **Linear (continuity-equation) LOS velocity reconstruction** —
  `correlation/velocity_reconstruction.py` / `scripts/25`, built 2026-09-30
  per Girish's request, now wired into the direct/coeval estimator
  (`scripts/15 --velocity {native,halo_reconstructed,both}`). Validated on
  real data for both the matter field and the halo tracer catalogue — see
  "Velocity reconstruction" below for the shot-noise/smoothing gotcha this
  surfaced for sparse discrete tracers.

**Major finding, fix shipped, full verification still pending:**
- **`io/loaders.py` had the SAME velocity-unit bug as the 2026-09-09
  `lightcone/stitch.py` fix, in a second code path that fix never
  touched** — found 2026-10-01 while chasing an unrelated kSZ auto-power
  crash. `load_lightcone_products` (feeding the **stitched-lightcone**
  pathway: `scripts/04/05/06/08/09/11/13/14`) assumed the stitched
  `lc_vz.npz` product was in km/s and converted it to Mpc/s anyway — but
  it's already Mpc/s on disk (confirmed directly against real data,
  `check_vz_units.py`), so every kSZ map built from it was crushed by a
  spurious extra ~1/3.086e19 factor (linear in the kSZ map, quadratic in
  kSZ²/`D_diag`). Fixed in `io/loaders.py`; 134/134 tests pass.
  **This is the leading candidate explanation for the "stitched pathway
  returns ~0" mystery reported below on 2026-09-09** — `scripts/11`/`13`
  both go through this exact loader — but that is not yet confirmed: the
  first rerun of `scripts/09` with the fix was interrupted before
  finishing (it needs to go back to a background PBS batch job, like the
  original `1733359.swarm` run, not an interactive foreground run). Until
  `scripts/09`/`11`/`13` (and ideally `04`/`05`/`06`/`08`/`14`) are rerun
  and actually checked, treat every absolute D_ell/SNR number from the
  stitched pathway as unconfirmed, not merely "known broken" as before.

**Known broken, actively being investigated:**
- The **stitched** kSZ² × galaxy pathway (`scripts/11`/`13`, feeding off
  `scripts/04`'s `cross_results.pkl`) returned numbers indistinguishable
  from zero as of 2026-09-09, even though the underlying velocity data
  checked out fine at the time and the *direct* method (same corrected
  velocity, no stitching) worked. See the finding directly above — this
  now has a strong candidate root cause, pending the reruns described
  there.

**Not yet done:**
- `scripts/05` (the actual production LAE S/N forecast) has not been run
  with the 2026-09-09 velocity fix, nor the 2026-10-01 loaders.py fix.
- Stage 2 (realistic LAE survey selection replacing the Roman-HLS LBG
  proxy, `scripts/07`/`08`) — untouched, blocked on Stage 1 settling first.

See "Velocity conversion", "Velocity reconstruction", "Direct/coeval kSZ2
x galaxy estimator", "Periodicity diagnostic", and "Known gotchas" below
for the full story on each of these.

## Wrap-cycle kSZ auto-power (2026-10-06, supersedes scripts/09 + 22 once validated)

Per-seed kSZ auto-power from a **wrap-cycle stitched lightcone**, adapted from
ksz-pipeline (`stitch_from_coeval.py`, commit 0c3b578; algorithm copied,
interfaces adapted). Each wrap cycle (one `BOX_LEN` of comoving LOS distance)
gets its own transverse rotation angle (`default_rng((wrap_seed, cycle))`), applied
to *all* snapshots and all three fields at that LOS position, with bilinear periodic
sampling. A fixed angle does not break periodicity (the revisited slab is bit-identical).

* `lightcone/wrap_cycle.py` - geometry + streaming stitcher (only the two bracketing snapshots in memory).
* `correlation/ksz_auto_wrapcycle.py` - stitch -> `field_data_seed` -> patchy-window and full-range D_total/D_diag/D_off.
* `scripts/26_ksz_auto_power_wrapcycle.py --seed S` - one seed -> `data/products/ksz_auto_wrapcycle/seed{S}_wo{K}.pkl`.
* `scripts/27_ksz_auto_power_aggregate.py` - median + sigma + 16-84% + min-max over seeds, CSVs, figure.
* `pbs/submit_ksz_wrapcycle.sh` - one job per seed + a dependent aggregation job.

**Deliberate differences from ksz-pipeline**
1. Our cosmology (`get_cosmology(cfg)`), not Planck18.
2. **Velocity untouched.** ksz-pipeline is py21cmfast v3 (Zel'dovich displacement, needs D f H/(1+z)); ours is v4 and already Mpc/s. Tests fail on any rescaling.
3. `mode='grid-wrap'` (period n) instead of scipy `'wrap'` (period n-1). `--mode wrap` reproduces ksz-pipeline bit-for-bit (verified on 479 slabs).
4. **LOS grid uniform in comoving distance, spacing = cell size (1 Mpc)** (3022 slabs over z=5-20). The old lightcone used 512 pixels uniform in z, i.e. 16 Mpc/pixel at z=5 down to 2.4 Mpc at z=20; since P_diag = sum|theta_i|^2 scales as ds^2 per pixel, that alone biased the old D_diag independent of periodicity.

**Two normalisation issues found while porting (both on by default in scripts/26; `--ne-convention legacy --tau0 none` restores the old numbers)**
* `constants.tau_prefactor` uses n = Ob0 rho_c / m_p, i.e. all baryons as hydrogen. Helium-inclusive n_e is 0.82x that (2.07e-7 vs 2.52e-7 cm^-3; ksz-pipeline: 2.064e-7) => legacy D_ell is ~1.49x too high. The old `build_projected_maps`/scripts 04-08/11/13-15 still use the legacy value.
* `compute_ksz_slices` started tau at 0 at z_min; the real tau below z=5 is ~0.031 (`utils/optical_depth.analytic_tau_below`), worth ~6% in D_ell.
Both enter `compute_ksz_slices` through optional `field_data_seed['ne_scale'|'tau0']`; absent => legacy behaviour unchanged.

Run: `bash pbs/submit_ksz_wrapcycle.sh` (try `SEEDS="1 2"` first; `WRAP_OFFSET=100` repeats with fresh angles).

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
itself). Measured result on the real 10-seed fiducial run, post
velocity-conversion-fix (2026-09-09): median D_off/D_total = 72% at
ell~3000 (10/10 seeds agree, range 64-78%) -- consistent with, though not
identical to, the pre-fix figure (68%), since the RATIO is dimensionless
and largely insensitive to the overall velocity scaling that changed.
Worse than the companion repo's 2.3x (equivalent framing) at their larger
800 Mpc box, consistent with their own finding that the artifact's
amplitude grows as the box shrinks.

**Patchy-window rewrite (2026-09-30), per Girish's follow-up** ("why is
it till z=12.5? ... we want it for the full patchy window"):
`compute_patchy_window_diag_power` restricts the LOS sum to each seed's
OWN actual patchy-reionization window (z>=6 floor, x_HI strictly between
0 and 1, via `clamp_window_to_patchy_regime`) instead of the full
box.z_min-z_max range, and uses a power-weighted `chi_eff`
(`chi_eff_power_weighted`) for the ell=k*chi Limber conversion instead of
an arbitrary box-midpoint reference. Because `chi_eff` is now genuinely
seed-specific (real spread ~0.1% across seeds, e.g. 9168.4/9164.3/9161.8/
9170.2 Mpc), each seed's `ell` grid is a slightly different rescaling of
the same `k_centers` -- `seed_stats.aggregate_coherence_over_seeds` was
updated (2026-10-01) to interpolate onto a common reference grid rather
than requiring an exact match (still raises if the deviation exceeds 5%,
which would indicate an actual config mismatch rather than this harmless
scatter).

CAVEAT as of 2026-10-01: `scripts/22_ksz_auto_power_plot.py` (the plot
built on top of this) ran successfully against the full 10-seed patchy
window for the first time, but `D_diag` came back at ~1e-36 uK^2 --
nowhere near the expected O(1) uK^2 scale, and the same order of
magnitude as the OLD pre-periodicity-fix cancellation artifact this
module was built to cure. Root cause: the separate `io/loaders.py`
velocity-unit bug described in "Start here" above -- `D_diag` is
quadratic in velocity, so the ~1/3.086e19 crush shows up squared here.
`coherence_decomposition.pkl` needs regenerating (`scripts/09`, as a
background PBS job, not interactively) with the `io/loaders.py` fix
before `scripts/22`'s numbers mean anything. The D_off/D_total RATIO
figure quoted above predates both the patchy-window rewrite and this
bug and should be re-measured, not assumed unaffected.

This decomposes the kSZ AUTO-power only (the ingredient feeding
`snr/cmb_filter.py`'s `kSZ_reion_from_sim`, for which
`kSZ_reion_from_sim_diag` is the periodicity-curbed drop-in alternative)
-- not the kSZ2 x tracer CROSS-power itself, which is a different
(bispectrum-type) statistic this decomposition doesn't directly address.
See the module docstrings for the full reasoning and caveats.

```
scripts/09_coherence_decomposition.py    P_diag/P_off per seed, each over
            |                            its own patchy window + power-
            |                            weighted chi_eff, from real data
            |                            already on disk (no new sim needed)
scripts/12_plot_coherence_summary.py     one seed-averaged summary plot
            |                            (reads the seed_agg pickle 09 saves)
scripts/22_ksz_auto_power_plot.py        kSZ + xe2/v2/v_proj/v_proj2 auto-
                                          power overview grid, median +/-
                                          sigma across seeds (see caveat above)
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

## Velocity conversion (fixed 2026-09-09)

`lightcone/stitch.py`'s handling of the raw `velocity_z` field from
py21cmfast coeval boxes was wrong, and had been wrong all along --
found while chasing the (still partially open) Stage 1 amplitude
mismatch. The fix and the reasoning behind it:

`halos/coeval_pipeline.py` saves `velocity_z` completely raw --
`coeval.perturbed_field.get("velocity_z")`, no conversion at save time.
The stitching step used to apply `box/(1+z)*3.086e19`, a formula carried
over from the companion `ksz-pipeline` repo, where it's correct: that
repo runs **py21cmfast v3**, whose raw velocity field really is an
unconverted linear-theory Zel'dovich *displacement*, needing a full
`D(z)*f(z)*H(z)/(1+z)` reconstruction (their own validated formula) to
become a genuine velocity.

This repo runs **py21cmfast v4**. Its coeval `velocity_z` is *already* a
genuine comoving peculiar velocity in Mpc/s -- confirmed two ways: (1)
raw `velocity_z`, used completely as-is, gives $v/c \sim 4\times10^{-3}$,
squarely physical; (2) a rigorous check --
`correlation/velocity_convention_check.py`, comparing the raw field's own
power spectrum against the density field's via the linear-theory
continuity equation $P_v(k) = \frac{1}{3}(faH/k)^2 P_\delta(k)$ on the
same real snapshot -- gives a correction factor of ~0.8-0.9 across nearly
two decades of $k$ (only drifting at $k>0.7\,{\rm Mpc}^{-1}$, exactly
where linear theory is expected to break down), vs ~$10^{19}$ for the old
formula, which no unit reinterpretation could rescue.

**Fix**: `Stitcher.load_field_box`'s `vz` case now returns the raw value
completely unconverted. `velocity_z_to_mpc_per_s` (the v3-style
Zel'dovich reconstruction, kept for reference/comparison) is **not**
used anywhere in this repo's live pipeline.

**Consequence, not yet fully resolved**: this changes every kSZ-dependent
number in the repo. The direct/coeval estimator (below) improved
dramatically once re-run with the fix. The stitched pathway did not --
see "Start here" above.

```
scripts/16_velocity_conversion_check.py           cheap (one file load) --
            |                                     old vs new conversion,
            |                                     side by side, on real data
scripts/17_velocity_convention_definitive_check.py the rigorous version --
                                                    continuity-equation
                                                    physics check (see
                                                    correlation/velocity_convention_check.py),
                                                    real compute (3D FFTs),
                                                    needs qsub
```

### Second velocity-unit bug, same class, different file (fixed 2026-10-01)

The fix above only touched `lightcone/stitch.py`'s `load_field_box` --
the raw per-snapshot coeval loader used by the **direct/coeval**
pathway. The **stitched-lightcone** pathway loads velocity through a
completely separate function, `io/loaders.py`'s
`load_lightcone_products`, which reads an already-stitched product
(`lc_vz.npz`) rather than raw coeval snapshots. That function still had
the old assumption baked in: it treated `lc_vz.npz`'s raw array as km/s
and divided by `c_kms` then multiplied by `c_mpc_s` to get Mpc/s. An old
exploratory-notebook comment claimed the external stitching script
already converts to km/s -- evidently not (or no longer) true of the
product actually on disk.

Found 2026-10-01 while diagnosing why `scripts/22`'s patchy-window
`D_diag` came back at ~1e-36 uK^2 instead of O(1). Confirmed directly
against real data with a small diagnostic
(`check_vz_units.py`, kept in the repo root): `lc_vz.npz`'s raw std
(~4.2e-17) matches this repo's own independently-reconstructed Mpc/s
velocity fields (see "Velocity reconstruction" below) almost exactly --
not the hundreds-of-km/s scale a genuine km/s array would show.

**Fix**: `velocity_lc` (Mpc/s, feeds the kSZ integrand directly via
`coherence_decomposition.compute_ksz_slices` and
`projected_maps.build_projected_maps`) is now the raw array as-is, no
conversion. `velocity_kms` is now properly DERIVED from that corrected
value (an actual Mpc/s -> km/s conversion) rather than being the
untouched raw array under a misleading name.

**Scope**: every script that calls `load_lightcone_products` --
`04/05/06/08/09/11/13/14` -- computed its kSZ map (and therefore every
D_ell/SNR number) with velocity crushed by a spurious ~1/3.086e19 factor
per power of velocity, until this fix. Because it's one uniform
multiplicative constant applied identically everywhere, this should only
have corrupted absolute amplitude, not shapes or correlation
coefficients (r(k), whether a cross-correlation looks coherent) -- but
no absolute number from those scripts should be trusted or shown until
they're rerun with the fix. The direct/coeval pathway
(`scripts/15`/`18`/`25`, via `lightcone/stitch.py`) is unrelated and
unaffected. `v2`/`v_proj`/`xe2`/`v_proj2` auto-power panels were
mislabeled by this bug but not amplitude-corrupted (no extra conversion
was ever applied to the `velocity_kms` key specifically).

As of this writing, only the diagnostic has been run and the fix has
been merged and unit-tested (`tests/test_loaders.py`, 134/134 suite
passing) -- `scripts/09` has not yet successfully completed a rerun with
the fix (see "Start here" above), so there is no confirmed post-fix
number yet for any of the 8 affected scripts.

## Lightcone rendering diagnostics

`scripts/14_lightcone_fluke_demo.py` -- two things, per `--tracer`
(halo/lae/lbg): (1) the same tracer at four aggregation levels (single
seed/slice through fully averaged/summed), confirming the faithful
rendering behaves sensibly at every level; (2) the actual point --
today's faithful `imshow`-based rendering next to the OLD (removed)
scatter-based rendering, on IDENTICAL data, demonstrating directly that
an earlier apparent "the lightcone looks suspiciously dense" concern was
a rendering artifact of the old plotting code, not real structure. The
old-rendering function is deliberately kept ONLY inside this script, not
restored to `plotting/lightcone_panels.py`, so it can't accidentally
find its way back into the real pipeline.

## Direct/coeval kSZ2 x galaxy estimator (bypasses stitching entirely)

`correlation/direct_bispectrum.py` + `scripts/15`/`18` implement the
kSZ2 x galaxy cross-power the way La Plante+2022's own Eq. 12 reduces to
in the squeezed-triangle limit (essentially all the real S/N, per their
own Fig. 10): filter -> square -> cross-correlate, applied PER COEVAL
SNAPSHOT directly (no lightcone stitching, no periodicity risk) and
summed across snapshots with the proper Limber/visibility weighting.
Built and synthetic-tested earlier; first touched real data 2026-09-09,
same day as the velocity fix above.

```
scripts/15_direct_bispectrum_vs_stitched.py   D_ell vs ell at one (z0,dz)
            |                                 window, real La Plante+2022
            |                                 band overlaid (digitized,
            |                                 see below), plus the
            |                                 stitched-pathway number
            |                                 for direct comparison
scripts/18_direct_dell_vs_z0.py               D_ell vs z0 sweep at three
                                               fixed ell (500/1000/3000,
                                               matching the digitized
                                               La Plante band exactly),
                                               non-overlapping windows so
                                               no snapshot's expensive
                                               FFT work repeats
```

Both are real compute (per-snapshot 3D FFTs) -- run via `qsub`, not
interactively; `scripts/18` in particular sweeps this cost across many
z0 windows.

First real result (seed 1, z0=9.5, dz=1.0, post-velocity-fix): D_ell ~
0.0003-0.0004 uK^2 across ell=400-5000, vs the paper's ~0.02 uK^2 at
ell~1000 -- roughly 15-20x too high. Substantially better than the
stitched pathway's pre-fix ~1000-8000x, though not (yet) a validated
match -- treat as the current best-available number for this
cross-correlation, not a settled result.

## Velocity reconstruction (linear continuity-equation, not ML)

`correlation/velocity_reconstruction.py` / `scripts/25`, built 2026-09-30
per Girish's request: reconstructs the LOS velocity field from an
overdensity field via the linear continuity equation,
$v_{\rm los}(k) = i\,a H f\,(k_{\rm los}/k^2)\,W_G(k)\,\delta(k)/b$ (a
Gaussian smoothing kernel $W_G$ and a $1/b$ bias correction are both
optional), rather than any ML-based approach -- arXiv:2609.36355 Eqs.
11-12, adapted here to a real-space coeval box. `scripts/25` is the CLI
check: reconstructs from a chosen tracer field and compares to the
native simulation velocity (`correlation.velocity_reconstruction.
compare_reconstructed_to_native`: r(k), transfer(k), pixel Pearson r).

**Matter field** (`--tracer matter`, bias=1 by construction): near-exact
recovery on real data, as expected for linear theory applied to the
field it was derived from.

**Halo tracer** (`--tracer halo`, real halo catalogue at a mass cut --
not the matter field): large-scale bias isn't known a priori for an
arbitrary mass cut, so `measure_large_scale_bias` (cross/auto power
ratio between tracer and matter overdensity, averaged over the lowest-k
bins) measures it empirically from the box itself rather than borrowing
`roman_hls_benchmark.bluetides_bias_gz` (a luminosity-selected galaxy fit
that doesn't apply to an arbitrary mass cut). Measured halo bias at the
1e10 Msun cut, z~9-10: b~0.7-0.73 (two independent real-data
measurements) -- below 1, which is physically surprising for such a
rare/massive population this early and not yet investigated further.

**Shot-noise gotcha, found and fixed on real data**: an unsmoothed halo
reconstruction blows up -- the sparse discrete tracer field (occupied in
~0.08% of cells) has huge fractional overdensity in its few occupied
cells, and pushing that through the reconstruction's $1/k^2$ kernel
amplifies it into unphysical real-space outliers (~100x the native
field's scale). Fixed with Gaussian smoothing at `--r-smooth 18.4`
(the paper's $R_s=12.5\,h^{-1}$Mpc, converted with this repo's
$h=0.6777$) -- confirmed on real data: amplitude back to native's scale,
pixel Pearson r improved 0.50 -> 0.78. `scripts/25`'s 4-panel figure
(native | unsmoothed | smoothed | residual) makes this comparison
explicit. Output filenames bake in `--tracer`/`--r-smooth` so different
settings never silently overwrite each other.

**Wired into the direct/coeval kSZ2 x galaxy estimator** (2026-10-01):
`scripts/15 --velocity {native,halo_reconstructed,both}` substitutes a
per-snapshot halo-reconstructed velocity (same bias-measurement +
18.4 Mpc smoothing as above) for the simulation's native v_z, computing
both from the identical snapshots/delta_g so the comparison isn't
confounded by anything else differing between runs. First real result
(seed 1, 5 snapshots, z=9.0-9.8): bias measured per-snapshot in the
0.688-0.727 range; reconstructed-velocity D_ell tracks native closely at
the largest scales (near 100% signal retention) and falls off at smaller
scales (roughly 15-40% retention), a physically sensible pattern for a
tracer that carries less information than the full density field. Not
yet scaled beyond 1-2 seeds.

```
scripts/25_velocity_reconstruction_check.py   standalone reconstruction
            |                                 check, any tracer, 4-panel
            |                                 smoothing diagnostic
scripts/15_direct_bispectrum_vs_stitched.py   --velocity flag substitutes
                                               reconstructed velocity into
                                               the real kSZ2 x LAE estimator
```

## Digitized La Plante+2022 reference data

`data/reference/la_plante_2022/` + `io/la_plante_reference.py` --
real digitized points from the paper's own published figures (their
Fig. 4/5 uncertainty bands, and their reionization-history figure),
not a single hand-read peak value. See that directory's own README.md
for exact provenance (manually digitized vs. automated pixel-extraction,
which files are which, and known digitization uncertainty). Loaded
directly into `scripts/15`/`18`'s overlay plots.

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
data/reference/  digitized La Plante+2022 reference data (see its own README)
src/ksz_lae_xcorr/
  halos/         py21cmfast coeval + two-pass halo catalog generation
  lightcone/     3D lightcone stitching
  tracers/       physical-value (mass/luminosity/MUV) grids for diagnostics
  correlation/   projected maps, cross-power, auto-power (halo/LAE/LBG),
                 periodicity decomposition (coherence_decomposition.py,
                 now with per-seed patchy-window + chi_eff_power_weighted),
                 seed aggregation (seed_stats.py, tolerant of seed-specific
                 ell grids), direct/coeval bispectrum estimator
                 (direct_bispectrum.py, now with a --velocity source
                 argument), linear continuity-equation LOS velocity
                 reconstruction (velocity_reconstruction.py), velocity
                 unit-convention check (velocity_convention_check.py)
  snr/           CMB filter + S/N forecast (LAE-default, tracer-generic),
                 Stage 1 literature benchmark (roman_hls_benchmark.py,
                 now with chi_eff and patchy-window clamping, ported
                 from ksz-pipeline)
  io/            product loaders (loaders.py -- velocity units fixed
                 2026-10-01, see "Velocity conversion") + digitized
                 reference data loader (la_plante_reference.py)
  plotting/      all figure-generating code
  utils/         config loader, cosmology, physical constants,
                 figio.py (save_fig: PDF+PNG together, every plot script
                 should use this rather than calling fig.savefig directly)
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

- **Per-field checkpointing in `scripts/02` doesn't know when underlying
  code or data changed.** `Stitcher` writes one `.done` checkpoint file
  per (seed, field) at `{lightcone_root}/checkpoints/seed_{N}_{field}.done`
  and silently SKIPS re-stitching that field if the checkpoint exists --
  regardless of whether the code that builds it has since changed. Hit
  THREE separate times in one session (2026-09-09): after the velocity
  conversion fix above, after clearing only the `vz` checkpoint the fix
  had no effect until that was found; separately, LAE and LBG tracer
  grids sat all-zero across every seed for most of the same session,
  traced to `tracers/type_b_grids.py` silently returning an empty grid
  whenever the external catalogue file is missing (only a `logger.warning`,
  easy to miss) -- combined with the SAME stale-checkpoint issue, meaning
  the code path that would have surfaced the warning had never actually
  run. **Any time a fix touches something `scripts/02` stitches, manually
  delete the relevant `seed_*_{field}.done` checkpoint files before
  re-running** -- `scripts/02` will not detect the need on its own.
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
