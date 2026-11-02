# Observational feasibility of kSZ² × LAE (2 Nov 2026)

Code: `src/ksz_lae_xcorr/snr/feasibility.py`, `scripts/52_feasibility.py`, `tests/test_feasibility.py`. Outputs: `data/plots/2026-11-02_feasibility/`.
Runs in about a minute on a laptop; it needs no simulation maps.

## What the forecast is

- **Noise**: La Plante+22 Eq. 13, `(S/N)² = f_sky Σ_ℓ (2ℓ+1) Cs² / (C^{T²T²}_f C^{gg} + Cs²)`. `C^{gg} = b² C^{mm}_Limber(window) + 1/n_2D`, so LAE shot noise is included. Narrow-band shells are real top-hat shells (Δz = 0.1).
- **Signal**: the toy model (ε = 8.25, amplitude fitted to La Plante Fig. 7), linear in the tracer bias, scaled to each experiment by the filtered reionisation-kSZ variance `∫ℓ dℓ/2π f² C_kSZ,reion`. This is the part to replace with the simulation's measured D_ℓ once the LAE lightcone excess is trusted.
- **Validation**: with ONE common rescaling (1.45) of the toy signal, the pipeline reproduces La Plante Table 2 (Roman HLS, f_sky 0.053, no shot noise, post-ILC): SO 2.43 vs 2.4, CMB-S4 4.82 vs 4.8, CMB-HD 22.7 vs 23. The white-noise ("instrument") column was not used in the calibration and comes out at 3.2/21.5/54 vs 2.9/18/64.
- **Experiments**: ACT DR6, SPT-3G Main (white noise only, so optimistic), and SO, CMB-S4, CMB-HD (La Plante's post-ILC curves, and also white noise for a like-for-like comparison).
- **LAE samples**: observed Lyα LFs (`data/reference/luminosity_functions`) integrated above each survey's L_min give n; bias from Sheth–Tormen abundance matching (upper value) and 0.45 × that (the simulation's measured LAE-to-halo bias ratio at z = 6).

## Result

LAE shells (fiducial history, x_HII at the shell in the last column):

| survey | z, Δz | L_min [erg/s] | LAEs/deg² | N in survey | b (ST-matched) | x_HII |
|---|---|---|---|---|---|---|
| HSC SILVERRUSH NB816 | 5.7, 0.1 | 6.3e42 | 174 | 2,441 (14 deg²) | 4.8 | 0.94 |
| HSC SILVERRUSH NB921 | 6.6, 0.1 | 7.9e42 | 28 | 584 (21 deg²) | 6.5 | 0.82 |
| Subaru PFS (repo area, assumed L_min) | 5.7 / 6.6, 0.2 | 3e42 | 1256 / 493 | 18,840 / 7,388 (15 deg²) | 4.1 / 5.1 | 0.94 / 0.82 |
| Roman grism deep, no evolution from z = 7 | 8.0, 1.5 | 1e43 | 187 | 187 (1 deg²) | 7.8 | 0.44 |
| Roman grism deep, LF as observed at z ≥ 7.3 | 8.0, 1.5 | 1e43 | 17 | 17 (1 deg²) | 9.6 | 0.44 |

S/N with shot noise (fiducial history, ST-matched bias; full overlap with the CMB map assumed):

| survey | ACT DR6 | SPT-3G Main | SO | CMB-S4 | CMB-HD | CMB-HD, white noise |
|---|---|---|---|---|---|---|
| HSC SILVERRUSH | 0.007 | 0.051 | 0.005 | 0.009 | 0.046 | 0.125 |
| Subaru PFS | 0.011 | 0.083 | 0.008 | 0.015 | 0.074 | 0.202 |
| Roman deep, no evolution | 0.013 | 0.102 | 0.009 | 0.019 | 0.091 | 0.245 |
| Roman deep, evolved LF | 0.006 | 0.044 | 0.004 | 0.008 | 0.039 | 0.106 |

(ACT, SPT-3G and the last column are white noise only. SO, S4 and CMB-HD use the post-ILC curves.) For a late reionisation history (50% at z ≈ 6.7) the z = 5.7–6.6 shells gain a factor 3–4: PFS × CMB-HD white noise reaches 0.76, HSC × SPT-3G 0.20. The 0.45 × bias case lowers every number by about 25–55%.

- No current or planned LAE survey in this list gives a detection. The best case, 0.8σ, needs both a late history and CMB-HD with no foregrounds.
- The limit is the LAE survey area (1–21 deg²), not the CMB noise. S/N² scales with the overlapping area at fixed density; area for 3σ is in `feasibility_snr.csv`: for example ~1,000 deg² for CMB-HD (post-ILC) × a Roman-deep-like density, and more than the whole sky for SO, ACT and CMB-S4 with the surveys above. 67 of the 128 combinations in the CSV need more than the whole sky.
- At z = 5.7–6.6 reionisation is nearly over (x_HII 0.8–0.94 in the fiducial history), so the signal is already on the falling side. The kSZ² × galaxy signal peaks at x_HII ≈ 0.1–0.25 (z ≈ 9–10).
- `feasibility_map.png`: for CMB-HD post-ILC, a Δz = 1 window at z = 8.5 (bias 5) reaches 3σ with about 100–300 deg² at ≳ 300–3000 LAEs/deg², or about 2,000 deg² at ~25/deg². CMB-S4 needs more than ~2,000 deg² at ≳ 10⁴/deg². SO has no 3σ region in the plotted range.

Reading: a LAE-only measurement needs a wide (100+ deg²) z ≈ 8–10 sample of ≳ 10³ per deg², which is a Roman HLS-like galaxy sample (La Plante's case) and not a LAE sample with today's densities. For LAEs the density at z ≥ 7.3 is already 10× lower than at z ≈ 7.

## Halo-mass cut vs LAE luminosity cut (simulation, 10-seed means, `halo_luminosity_mapping.csv/png`)

| z | LAE cut | n [Mpc⁻³] | b (LAE) | b of host halos (ST) | halo cut with same n (log M, b) | halo cut with same b (log M) |
|---|---|---|---|---|---|---|
| 6.0 | L ≥ 10^42.0 | 8.0e-4 | 3.97 ± 0.09 | 3.28 | 10.99, 4.4 | 10.77 |
| 6.0 | L ≥ 10^42.5 | 2.2e-4 | 2.33 ± 0.10 | 3.53 | 11.28, 5.4 | 9.38 |
| 6.0 | L ≥ 10^43.0 | 1.1e-4 | −0.49 ± 0.10 | 3.38 | 11.45, 6.0 | outside the halo range |
| 7.1 | L ≥ 10^42.0 | 1.1e-5 | 5.82 ± 0.18 | 4.13 | 11.57, 8.7 | 10.97 |
| 7.1 | L ≥ 10^42.5 | 6.5e-6 | 4.43 ± 0.27 | 4.20 | 11.65, 9.1 | 10.50 |
| 7.1 | L ≥ 10^43.0 | 5.9e-6 | 3.91 ± 0.33 | 4.13 | 11.66, 9.2 | 10.05 |

No single halo-mass cut reproduces both the number density and the bias of a LAE luminosity cut. Matched in number, the halo sample is 2–4× more biased than the LAE sample at z = 6 (the 10^42.5 cut is the 10^11.3 Msun halos by number but the 10^9.4 halos by bias). The host halos of the LAEs are 10^10–10^10.5 Msun, and the measured LAE bias differs from the host-halo bias by up to −4 (L ≥ 10^43 at z = 6). So the luminosity selection is shaped by the IGM state on top of halo mass. Consequence for the S/N: at fixed number, the signal and the shot-noise-limited S/N both scale with bias, so in this model a halo sample matched in number is better than the LAE sample by about the bias ratio (at z = 6, L ≥ 10^42.5: 5.4/2.3). A LAE advantage would have to come from a stronger response per unit bias (selection tied to the ionisation field), which is the open control run (matched-density halo sample on the same lightcone).

## Caveats

- The signal is the toy model (shape error 0.3–0.4 in ln D over x_HII 0.1–0.9, ε not measured) rescaled by one factor to La Plante Table 2. At x_HII > 0.8 the toy keeps about 6% of its peak against La Plante's ~2%, so the z = 5.7–6.6 numbers are not safe to a factor ~2.
- Foregrounds: only what is inside La Plante's post-ILC curves. ACT DR6 and SPT-3G use white noise (combined-band depth), which is optimistic; lensing leakage, masks and mode coupling are ignored. Redshift-space distortions and σ_z damping of the tracer map are not included, and the Limber approximation is used for Δz = 0.1 shells at ℓ ≲ 1000, which is marginal.
- Overlap of each LAE field with each CMB footprint is assumed to be complete. ACT DR6 (~19,000 deg², equatorial) should cover the equatorial HSC fields, but I did not check either that or SPT-3G Main against the HSC fields.
- LAE counts use the transcribed Lyα LF fits, which the SOURCES.md note says are not yet checked by eye. L_min for PFS (3e42) is an assumption; the repo has no flux cut for it. Roman: Wold+23 predict ~395/deg² assuming no evolution, and the z = 7.0 fit used here gives ~215/deg².
- Windows are treated as independent, and the abundance-matched bias is an upper value (no scatter or duty cycle).

## Sources

- ACT DR6 maps: Naess et al., arXiv:2503.14451 (depth 14 µK-arcmin at 150 GHz, 9.6 combined, ~19,000 deg²).
- SPT-3G: arXiv:2403.17925 Table 1 (Main field, 1500 deg², coadd 1.9 µK-arcmin for 2019–2023 data). The 1.2′ beam is not from that paper.
- SO: arXiv:1808.07445 (LAT baseline 10 µK-arcmin, 1.4′ at 145 GHz, f_sky 0.4).
- CMB-S4 and CMB-HD noise and beams: La Plante+22 Table 1 (arXiv:2111.13717). Curves: their Fig. 1, digitised in `data/reference/la_plante_2022`.
- SILVERRUSH: Konno+18, arXiv:1705.01222 (14 deg² at z = 5.7, 21 deg² at z = 6.6, 1,266 LAEs). L_min from Weinberger+19 Table 1 via `configs/lae_surveys.yaml`.
- Roman grism: Wold+23, arXiv:2305.01562 (1 deg², z = 7.25–8.75, flux limit ~8e-18 erg/s/cm²).
- Noise treatment inspired by Weinberger, Kulkarni & Haehnelt 2020, arXiv:1911.11783 (LAE shot noise, volume matching).

## Not done

- Replace the toy signal by the simulation's measured D_ℓ per tracer (scripts 28/44) and the matched-number halo control.
- A forecast for a reionisation history other than the shifted fiducial, and a real foreground model for ACT and SPT-3G.
- Footprint overlap of each CMB survey with each LAE field.
- LAEs at z ≥ 8 from the big boxes (Jahaan's LAEfication, see the pinned note in the Project): the small box has essentially no LAEs there (catalogue/observed n(>10^42.5) = 0.004 at z = 7, 0 at z ≥ 8).
