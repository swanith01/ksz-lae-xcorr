# kSZ² × tracer: which ingredient carries it, and does the coeval-box estimator agree with the lightcone?

Both analyses sit on top of the tracer suite (`docs/tracer_suite.md`): they reuse the kSZ products of scripts/26 (`--save-maps`),
the tracer window maps of scripts/37, and the filter of scripts/28.

## 1. Component decomposition (scripts 41 → 42 → 43)

The kSZ integrand of a LOS pixel is θᵢ = −Wᵢ (1+δᵢ) xₑ,ᵢ vᵢ/c, Wᵢ = τ′(1+zᵢ)² e^{−τᵢ} dsᵢ.  We build the
filtered-and-squared field from one ingredient at a time (`correlation/ksz_components.py`):

| component | Xᵢ | what it tests |
|---|---|---|
| x | Wᵢ xₑ,ᵢ | ionisation field (patchiness), velocity not in it |
| v | Wᵢ x̄ₑ(zᵢ) vᵢ/c | velocity field, xₑ and δ at their slice mean |
| d | Wᵢ x̄ₑ(zᵢ)(1+δᵢ) | density (for closure; not plotted) |
| full | θᵢ | the real kSZ² (reproduces scripts/26's map exactly; checked in scripts/41) |

and two orderings: **A** = [F Σᵢ Xᵢ]² (project, then square: what a CMB map does) and **B** = Σᵢ [F Xᵢ]² (square every slice, then
project: no pairs of different LOS pixels).  A − B is the cross-slice part.  Because the maps have different units the cross-power is
taken with the fractional modulation s = S/⟨S⟩ − 1:  D^frac_ℓ = ℓ(ℓ+1)/2π C_ℓ(s × δ_tracer).  The summary figure shows
D^frac(component) / D^frac(full), pooled over windows with 0.1 ≤ x_HII ≤ 0.7, with a seed bootstrap.

    SEEDS="1 2 3 4 5 6 7 8 9 10" bash pbs/submit_ksz_components.sh          # scripts/41 per seed, then scripts/42
    python scripts/43_ksz_component_plots.py --root <folder with components_*.npz> --ell 1000

## 2. Coeval-box D_ℓ^cross vs the stitched lightcone (scripts 44 → 45)

For each window, every snapshot whose redshift range overlaps it contributes a slab of its own box (as many 1 Mpc cells as the
window has LOS pixels nearest that snapshot): Θ_s = Σ Wᵢ θ-cell, g_s = Σ wᵢ bᵢ δ_tracer-cell, S_s = [F Θ_s]², and
D_ℓ = ℓ(ℓ+1)/2π T²_CMB Σ_s C_ℓ(S_s × g_s) (`correlation/coeval_cross.py`).  Same filter (k⊥ only), same weights, same window top hat
as the lightcone; no interpolation between snapshots, no wrap-cycle rotations; averaged over all non-overlapping slab positions in the
box.  It is a "diagonal" estimate (pairs of pixels in different snapshots dropped); agreement with the lightcone, which contains
those pairs, means they are negligible.  Scripts/44 also computes the lightcone estimate of the same seed, so the comparison is paired.

    SEEDS="1 2 3 4 5 6 7 8 9 10" bash pbs/submit_coeval_cross.sh            # scripts/44 per seed (matter + 3 cuts)
    python scripts/45_coeval_vs_lightcone.py --coeval-dir <folder> --out-dir <plots>

Reading the table: `ratio` = pooled coeval/lightcone (1 = agree), `RMS z` = RMS over windows of the paired difference in units of its
s.e.m. (≈ 1 = consistent within the seed noise).

Limits: the slab thickness is set by the snapshot spacing (Δz ≈ 0.14–0.2, 50–70 Mpc), so for the k∥ = 0 modes the slab is thinner than
the box; the 300 Mpc box gives ℓ_min ≈ 190, ℓ = 500 has only a few modes (as in the lightcone).

### 2b. Full-box Limber estimate (`D_box`, added 2026-10-26)

The slab estimate is the Limber / "integrated bispectrum" sum of Kramer et al. (2025, arXiv 2501.07623, Eqs. 9–10, with a tracer in
place of τ): C_L = ∫dχ W_g g² T(L/χ, χ)/χ², T = the filtered hybrid bispectrum tracer × momentum × momentum at one distance.  With
50–70 Mpc slabs, LOS pairs that straddle a slab edge are lost.  `D_box` removes the edges: per snapshot the WHOLE periodic box is
projected, filtered, squared and crossed with the box's tracer map, T_s = C_ℓ([F Σ_box θ]² × Σ_box δ_g)/N_box (per LOS cell), and the
window gets D = Σ_i Wᵢ² wᵢ T_s(i) over its pixels (`coeval_cross.snapshot_box_T`, `limber_weight`).  Scripts/44 saves it next to the
slab estimate; scripts/45 draws both (green squares = box, orange = slabs; `--estimators slab,box`).
Fixed at the same time: scripts/44 interpolated every window onto the ℓ grid of the LAST window computed (ℓ = k χ_eff is
window-dependent), shifting the lower-z windows by up to ~15% in ℓ.  Re-run scripts/44 for all seeds.
