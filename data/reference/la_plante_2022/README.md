# La Plante et al. 2022 -- digitized reference data

Digitized from the paper's published figures (ApJ 928, 162; arXiv:2111.13717),
SO ILC filter panels specifically. Two different digitization passes went
into this, noted per file below -- worth knowing which is which if a
number here ever looks off.

## Files

- **dell_vs_z0_bands_digitized.csv** -- their Figure 5 (D_ell vs z0,
  uncertainty bands only, central lines NOT digitized -- band envelope
  is what's comparable to our own seed-to-seed scatter). Three ell
  values: 500, 1000, 3000. Manually digitized by Swanith
  (2026-09-09) -- upper/lower bound clicked directly off the figure.
  Columns: ell, bound (lo/hi), z0, D_ell_uK2.

- **dell_vs_ell_xhii043_band_digitized.csv** -- their Figure 4 analog
  (D_ell vs ell) at x_HII~0.43, band only. All the individual z-window
  curves shown in the original overlap within one band at this x_HII,
  so this is a single band, not per-curve. Manually digitized by
  Swanith (2026-09-09). Columns: bound (lo/hi), ell, D_ell_uK2.

- **reionization_histories_digitized.csv** -- their x_HII(z) figure,
  three scenarios (Fiducial, Early, Short). Digitized by Claude via
  automated pixel-color extraction (deviation-from-white projected onto
  each line's color direction, axis-calibrated from detected tick
  pixel positions), cross-checked by re-plotting against the original
  figure by eye. NOT manually verified point-by-point the way the other
  two files were -- treat as good for shape/qualitative comparison, less
  authoritative than the manually-digitized files above if a discrepancy
  ever comes up between this and something else.

## Known digitization uncertainty

Low-resolution source images (~400x350px screenshots) -- axis
calibration here is good to maybe 1-2% on well-separated ticks, worse
right at plot edges where lines get thin/anti-aliased. Fine for
shape/order-of-magnitude comparison, not for anything requiring
sub-percent precision. If better precision is ever needed, re-digitize
from a vector PDF of the paper rather than a raster screenshot.

- **fig1_filter_components_digitized.csv** -- their Figure 1 (D_ell = l(l+1)C_l/2pi in uK^2 vs l):
  post-ILC noise for SO / CMB-S4 / CMB-HD, the lensed primary CMB, the 30-sim mean reionization kSZ
  and the late-time kSZ power law. Columns: component, ell, D_ell_uK2. Digitised 2026-10-07 by EXTRACTING THE
  VECTOR PATHS of the figure from the paper PDF (PyMuPDF `get_drawings`, axes calibrated from the tick marks),
  not by clicking on pixels, so accuracy is ~path precision (<<1%), not 1-2%. Check built in: the extracted
  late-kSZ line reproduces their Eq. 9 (1.38 (l/3000)^0.21) at both ends. The CMB and CMB-HD curves are drawn
  as filled outlines / dots in the PDF; those two are binned (80 log-l bins, geometric mean) from the outlines
  and are good to a few per cent. Range l ~ 96-10050 (the kSZ-reion curve starts at l = 200).

- **fig7_dell_vs_xhii_fiducial_digitized.csv** -- right panel of their Figure 7: D_ell^cross against the
  volume-averaged ionised fraction x_HII (the paper's axis runs 1 -> 0 left to right), FIDUCIAL history
  only (solid lines), ell = 500, 1000, 3000. Columns: ell, x_HII, D_ell_uK2 (100 bins of 0.01 in x_HII, bin
  medians, then a width-7 running median to remove tracking glitches where curves cross). Digitised 2026-10-10 from the paper PDF by rendering the panel at 8x and tracking each solid
  curve by colour, with long-connected-component filtering to drop the dashed (Early) and dotted (Short)
  curves of the same colour; axes calibrated from the vector tick marks (x_HII = 1 at x=365.9 pt, 0 at 556.7 pt;
  D = 0 at y=192.6 pt, 0.030 at 61.7 pt). Checked by overlay on the figure. Peaks: ell=500 0.0157 at x_HII~0.21;
  ell=1000 0.0196 at ~0.12-0.2; ell=3000 0.0118 at ~0.12. Accuracy ~1-2% of the y range; least reliable at
  x_HII > ~0.8 (|D| < 0.002, all curves overlap there). Early/Short scenarios are NOT included.
