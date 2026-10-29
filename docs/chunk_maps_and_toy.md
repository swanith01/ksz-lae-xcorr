# Looking at the maps, and the analytic toy

## 1. Maps (scripts 48 -> 49)
`scripts/48_chunk_maps.py` (swarm, one seed) saves, for z0 = 12..7 (dz = 1): the filtered whole-lightcone kSZ^2 map (what the estimator
really crosses), the kSZ^2 maps built from each window's own pixels (full / x_e only / v only, orderings A and B) and the tracer maps of the
same windows (matter x Waters, halo, LAE, LBG).  `scripts/49_chunk_map_figures.py` (usha) puts them side by side, smoothed at the soft-mode
scale chi/ell, with the Pearson r in each panel corner and `maps_correlations.csv`.

    swarm:  git pull; qsub -v SEED=1 pbs/pbs_chunk_maps.sh      # -> data/products_tracers/chunks/seed1_wo0_chunks.pkl
    usha:   python scripts/49_chunk_map_figures.py --chunks seed1_wo0_chunks.pkl --out-dir data/plots/2026-10-23_maps

## 2. Toy (correlation/toy_model.py, scripts/50)
Squeezed-limit, slice-by-slice (diagonal) picture:
    C_ell = (1/dz) int dz  s'(z) R(z) b_g(z) P_mm(ell/chi, z) / chi^2
    s' = K^2 v^2 [ xbar^2 + eps xbar (1-xbar) ]          (homogeneous velocity term + patchy term)
    s' R = K^2 v^2 { [xbar^2 + eps xbar(1-xbar)] (2 + r_v) + [2 xbar + eps (1-2 xbar)] dxbar/ddelta }
    dxbar/ddelta = (1+z_mean) b_zm (D(z_mean)/D(z)) (-dxbar/dz)       (zreion)
Free: eps (shape) and one amplitude per ell.  Against La Plante Fig. 7 (fiducial history):
* eps <~ 3 (homogeneous velocity term competitive): peak at x_HII ~ 0.3-0.7 and 30-70% of the peak left at x_HII = 0.9 (La Plante: 2%);
* eps ~ 8: peak at x_HII ~ 0.24 (LP 0.12-0.19), falls to ~0 by x_HII ~ 0.8, misfit 0.3 in log -- looks like Fig. 7.
Full write-up: docs/toy_model_report.tex (PDF goes next to the toy plots).
Reading: the early rise is the patchy-ionisation term (power grows as x(1-x)), the late fall is the vanishing patchy power plus the
vanishing ionisation response (dxbar/dz -> 0).  The density/velocity response of the homogeneous term alone peaks at x_HII ~ 0.4-0.5 and
would not fall fast enough.
Caveats: eps ~ 8 is the best of a scan and sharp (the ionisation response changes sign at x = eps/(2(eps-1)) ~ 0.57, which is what makes the late fall);
x_HII < 0.1 is controlled by the digitised history's first points; the toy is a diagonal-slice model (no cross-slice pairs); b_g(z) is
the Waters/BlueTides shape for every tracer (amplitude fitted).
    python scripts/50_toy_model.py --out-dir data/plots/2026-10-23_toy [--ours-root data/plots/2026-10-16_tracer_cross]

## 3. Teaching GIF (scripts/51, 2026-10-28)
Three panels: filtered whole-lightcone kSZ^2 map | tracer map of one dz = 1 window, stepping from high z to low z, with z, x_HII, D_ell and the
Pearson r | D_ell(x_HII) of the seed with a marker synced to the window shown (--no-curve for two panels, --paper adds La Plante Fig. 7).  Arrow (both panels, same pixel) = where (kSZ^2 fluctuation) x (tracer overdensity), both smoothed at chi/ell, is largest in that window; the
product map is normalised so its mean is r.  The window where this seed's D_ell(x_HII) peaks gets a big flashing arrow + banner (cf. Guo et al.
2026 Fig. 9, which shows the 21-cm slice at the redshift of the peak).  One seed: the flashing window is that seed's own peak (--peak-z to force one).
    python scripts/51_pedagogy_gif.py --seed 1 [--cltt-file ell,Cl_uK2.csv] --out-dir data/plots/2026-10-28_gif       # needs lpmaps of >= 2 seeds + the tracer maps
    python scripts/51_pedagogy_gif.py --chunks data/products_tracers/chunks/seed1_wo0_chunks.pkl                       # alternative: 6 windows from scripts/48
