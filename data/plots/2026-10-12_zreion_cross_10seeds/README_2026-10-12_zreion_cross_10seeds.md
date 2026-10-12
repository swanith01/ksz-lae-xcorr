# 2026-10-12: zreion ionisation field, 10 seeds (plot guide)

Same pipeline as the 21cmFAST runs (300 Mpc boxes, 10 seeds), but with the **ionisation field replaced by zreion**
(zmean=8, alpha=0.2, k0=0.9 h/Mpc, rsmooth=1, Zel'dovich, on the 21cmFAST IC density). Density and velocity are unchanged.
Made with `pbs/submit_zreion_chain.sh` (scripts 34, 26, 27, 28), overlay with `scripts/31_lp_cross_vs_xhii.py --compare-csv`.

| File | What it shows |
|---|---|
| `lp_cross_vs_xhii_SO_wo0_compare.png` | **Main plot.** D_l^cross vs x_HII at l=500/1000/3000: La Plante+22 Fig. 7 (orange), zreion (blue), 21cmFAST (green); seed mean +- s.e.m. |
| `lp_cross_vs_z0_SO_wo0.png` (+ `.csv`) | The same zreion cross-power vs galaxy-window redshift z0 (La Plante Fig. 5 style). |
| `ksz_auto_power_wrapcycle_wo0.png` | kSZ auto-power, 10 seeds. Ignore the red D_diag curve (per-pixel baseline, not validated). |
| `zreion_vs_21cmfast_seed{1..10}_*.png` | Per seed: history (21cmFAST vs zreion vs paper), x_HI slice at z=7.9, and correlation r(k) of the two ionisation fields. |

**Take-away:** at l=3000, x_HII~0.16: paper 0.0115, zreion 0.0119 +- 0.0033, 21cmFAST 0.0029 +- 0.0038 (zreion removes the low-x_HII deficit).
At l=1000 zreion improves but stays low (0.0121 +- 0.0070 vs paper 0.0193). l=500 is noise (9 modes).
Likely cause of the remaining low-l gap: 300 Mpc box (few and missing large-scale modes). Errors are large; this is evidence, not proof.
