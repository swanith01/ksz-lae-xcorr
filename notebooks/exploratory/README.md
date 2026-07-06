# Exploratory notebooks

These are kept for reference / provenance only -- they are the original,
pre-refactor scripts this repo's `src/ksz_lae_xcorr/` modules were derived
from, and describe the 400 Mpc / 64^3 / 5-seed exploratory run (which
produced under-resolved, unreliable LAE catalogues and is NOT the fiducial
run for the paper -- see top-level README "Fiducial simulation").

Per the project convention, notebooks only load pre-computed products and
are not part of the reproducible pipeline -- use `scripts/01`-`06` for that.
Do not add new analysis logic here; add it to `src/` and call it from a
script instead.

- `kSZ_LAE_lightcone_plotter_and_SNR_notebook_6Jul2026.py` -- source for
  `src/ksz_lae_xcorr/correlation/`, `src/ksz_lae_xcorr/snr/`, and
  `src/ksz_lae_xcorr/plotting/spectra_plots.py`.
- `replot_lightcone.py` -- source for `src/ksz_lae_xcorr/plotting/lightcone_panels.py`.
