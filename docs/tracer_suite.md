# Tracer suite: halo mass cuts, LAE luminosity cuts, LBG MUV cuts

Purpose: compare the 21cmFAST tracers with the La Plante+22 assumption (delta_g = b_g(z) delta_m, Waters+16 bias) and
show how many objects a 300 Mpc box has when a sample is as biased as theirs -> do we need bigger boxes?

| step | script | where | output |
|---|---|---|---|
| A. bias, n(z), N_obj per cut | `35_tracer_bias_suite.py` (`pbs/submit_tracer_bias.sh`) | swarm | `data/products_tracers/bias/tracer_bias_seed*.csv` |
| A2. plots + volume table | `36_tracer_bias_plot.py` | desktop | `tracer_bias_vs_z`, `tracer_number_density`, `tracer_bias_vs_number`, `tracer_bias_summary.csv` |
| B. tracer maps on the kSZ lightcone | `37_tracer_lightcone_maps.py` (`pbs/submit_tracer_cross_chain.sh`) | swarm | `data/products_tracers/maps/<name>/seed*_wo0_lpmaps.pkl` |
| B2. D_l^cross per tracer | `28_lp_cross_vs_z0.py` (looped by `pbs/pbs_tracer_cross.sh`) | swarm | `<OUT_ROOT>/<name>/lp_cross_vs_z0_SO_wo0.csv` |
| C. halo mass function, UV LF, Lya LF of the catalogues | `39_tracer_luminosity_functions.py` (`pbs/submit_tracer_lf.sh`) | swarm | `data/products_tracers/lf/lf_seed*.csv` |
| C2. vs Sheth-Tormen and observed LFs | `40_tracer_lf_plots.py` | desktop | `halo_mass_function_vs_ST`, `uv_luminosity_function_vs_lit`, `lya_luminosity_function_vs_lit`, `lf_summary.csv` |
| B3. comparison | `38_tracer_cross_compare.py` | desktop | `tracer_cross_vs_xhii`, `tracer_cross_summary.csv` |

Names: `halo_M10.0` (M >= 1e10 Msun), `lae_L42.5` (L_Lya >= 10^42.5 erg/s), `lbg_MUV-19.0` (M_UV <= -19).  Choose cuts with
`--specs "halo:10;lae:42.5;lbg:-19"` or `default`.

Bias: b_lin = sum P_tm / sum P_mm for k < 0.15 Mpc^-1 (cross-spectrum, immune to shot noise).  Dashed in the plots = analytic
Sheth-Mo-Tormen bias of the same haloes.  Tracer maps carry their own bias and shot noise (no b_g multiplication).

Observed luminosity functions used for the comparison live in `data/reference/luminosity_functions/` (UV: Bouwens+21, Donnan+24,
Finkelstein+24, McLeod+24; Lya: Umeda+25, Kageura+25, Konno+18, Santos+16, Itoh+18, Ouchi+10, Konno+14).  `SOURCES.md` there gives the
paper, arXiv id and table of every number; they were read out of the papers by an automated reader and still need a by-eye check.
