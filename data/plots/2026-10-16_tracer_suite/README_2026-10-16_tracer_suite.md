# Tracer suite, 16 Oct 2026 (21cmFAST 300 Mpc boxes)

Question: how do our halos, LBGs and LAEs compare with what La Plante+22 assume (delta_g = b_g(z) delta_m, Waters+16 bias), and do we need bigger boxes?

## `bias/`  (10 seeds, seed mean; scripts 35 + 36)
- `tracer_bias_vs_z`: linear bias of every cut against z. Dashed = analytic Sheth-Mo-Tormen bias of the same sample, black = La Plante / Waters+16 bias.
- `tracer_number_density`: n(>cut)(z), right axis = objects in the 300 Mpc box. Halos: dashed = Sheth-Tormen. LBG / LAE: open diamonds = published luminosity-function fits.
- `tracer_bias_vs_number`: objects in the box against bias at z = 6, 7, 8, 10; black line = La Plante bias. `tracer_bias_summary.csv` has the numbers (incl. the box side needed for 10^4 objects).

Findings: our halo bias matches Sheth-Mo-Tormen to ~5-10%. Typical halos and LBGs have 0.3-0.5 of the La Plante bias. Reaching it needs halos of ~1e12 Msun (b/b_W ~ 1 at z=6 with only ~135 in the box; 16 at z=7; at most 2 at z>=8); a box of ~1.2 Gpc (z=6) to ~2.5 Gpc (z=7) would hold 10^4 of them.

## `lf/`  (seed 1 only so far; scripts 39 + 40)
- `halo_mass_function_vs_ST`: halo mass function against Sheth-Tormen. The catalogue sits below it: n(>1e10 Msun) / ST = 0.69, 0.53, 0.40, 0.18 at z = 6, 7, 8, 10 (cause not yet checked, e.g. which mass function 21cmFAST was run with).
- `uv_luminosity_function_vs_lit`: UV LF of the LBG catalogue against Bouwens+21, Donnan+24, Finkelstein+24, McLeod+24. Agrees with observations to ~10-30% at z = 6-10 (n(M_UV<-18) catalogue/observed = 1.00, 0.88, 1.01, 1.21 at z = 6, 7, 8, 10).
- `lya_luminosity_function_vs_lit`: Lya LF of the LAE catalogue (observed luminosity) against SILVERRUSH (Konno+18, Umeda+25), Santos+16, Kageura+25. Catalogue / observed n(>1e42.5) = 0.37 at z=6, ~0.004 at z=7, 0 at z>=8: the LAE sample is strongly suppressed once the box is still mostly neutral (21cmFAST reionises late), so LAE cross-correlations at z>=7 are shot-noise dominated.
- `lf_summary.csv`: the numbers behind these statements.

Observed luminosity functions: `data/reference/luminosity_functions/` (paper, arXiv id and table of every row in `SOURCES.md`). They were read from the papers by an automated reader and still need a by-eye check.

## Not yet
kSZ^2 x tracer cross-power: tracer maps exist for seed 1 only; the cross step needs >= 2 seeds (maps for seeds 2-10 are being made).
