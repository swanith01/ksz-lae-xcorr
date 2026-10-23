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
    dxbar/ddelta = (1+z) b_zm (D(z_mean)/D(z)) (-dxbar/dz)       (zreion)
Free: eps (shape) and one amplitude per ell.  Against La Plante Fig. 7 (fiducial history):
* eps <~ 3 (homogeneous velocity term matters): peak at x_HII ~ 0.5-0.7, falls too slowly -- does not look like Fig. 7;
* eps ~ 10: peak at x_HII ~ 0.2, falls to ~0 by x_HII ~ 0.8, within ~30% in log -- looks like Fig. 7.
Reading: the early rise is the patchy-ionisation term (power grows as x(1-x)), the late fall is the vanishing patchy power plus the
vanishing ionisation response (dxbar/dz -> 0).  The density/velocity response of the homogeneous term alone peaks at x_HII ~ 0.4-0.5 and
would not fall fast enough.
Caveats: eps ~ 10 is the best of a coarse scan and sharp (the ionisation response changes sign at x = eps/(2(eps-1)) ~ 0.56, which is what makes the late fall);
x_HII < 0.1 is controlled by the digitised history's first points; the toy is a diagonal-slice model (no cross-slice pairs); b_g(z) is
the Waters/BlueTides shape for every tracer (amplitude fitted).
    python scripts/50_toy_model.py --out-dir data/plots/2026-10-23_toy [--ours-root data/plots/2026-10-16_tracer_cross]
