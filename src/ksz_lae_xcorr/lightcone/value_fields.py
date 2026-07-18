"""
lightcone/value_fields.py
===========================
Full 3D (NGRID, NGRID, N_LC_PIX) value-weighted lightcones: occupancy PLUS
per-cell mean Lya luminosity or rest-frame equivalent width (REW), for the
LAE tracer. This is a different product from tracers/type_b_grids.py's 2D
diagnostic slices -- the realistic survey-selection extension
(snr/survey_selection.py) needs a full 3D field so a flux/REW cut can be
applied at every (x, y, z) position and then cross-correlated properly,
not just visualized.

Produces:
    <lightcone_root>/seed_{N}/lc_lae_lum_3d.npz   (lc_occ, lc_val -- luminosity)
    <lightcone_root>/seed_{N}/lc_lae_rew_3d.npz   (lc_occ, lc_val -- REW)

Both have lc_occ.shape == lc_val.shape == (NGRID, NGRID, N_LC_PIX), matching
lc_lae.npz's 'lc' shape exactly -- Cell 11a's original validation step
(comparing occupancy here against tracer_data['lae_count_lc']) should be
run once real data exists; see scripts/07_stitch_lae_value_fields.py.

External input: this requires Jahaan's pipeline to also export a per-LAE
REW value (lya_rew_obs), alongside the luminosity (lya_lum_obs) it already
provides -- see data/README.md, this is a NEW external dependency beyond
what the core pipeline (scripts 01-06) needs.
"""

from __future__ import annotations

import os
import traceback

import numpy as np

from ksz_lae_xcorr.lightcone.stitch import Stitcher


class ValueFieldStitcher(Stitcher):
    """Adds full-3D occupancy+value stitching for LAE luminosity/REW."""

    VALUE_SPECS = {
        "lae_lum": {"id_subdir": "halo_ids_obs", "id_prefix": "halo_ids_obs",
                    "val_subdir": "lya_lum_obs", "val_prefix": "lya_lum_obs"},
        "lae_rew": {"id_subdir": "halo_ids_obs", "id_prefix": "halo_ids_obs",
                    "val_subdir": "lya_rew_obs", "val_prefix": "lya_rew_obs"},
    }

    def load_lae_value_grid(self, seed: int, z: float, value_field: str, logger):
        """
        Returns (occ_grid, val_grid), both full 3D (ngrid, ngrid, ngrid).
        occ_grid is identical in construction to load_lae_grid (same mass-cut
        + id convention -- see that docstring's IMPORTANT note, same caveat
        applies here). val_grid holds the per-cell MEAN of value_field
        (luminosity or REW) for cells with occ_grid > 0, else 0.
        """
        spec = self.VALUE_SPECS[value_field]
        idpath = os.path.join(self.root_lae, spec["id_subdir"], f"{spec['id_prefix']}_z{z:.4f}_s{seed}.npy")
        valpath = os.path.join(self.root_lae, spec["val_subdir"], f"{spec['val_prefix']}_z{z:.4f}_s{seed}.npy")
        empty = np.zeros((self.ngrid,) * 3, dtype=np.float32)
        if not os.path.exists(idpath) or not os.path.exists(valpath):
            logger.warning(f"  {value_field} inputs missing at z={z:.4f} seed={seed}, using empty grid")
            return empty, empty.astype(np.float64)

        ids = np.load(idpath, mmap_mode="r")
        vals = np.load(valpath, mmap_mode="r")
        coords, masses = self._halo_coords_masses(seed, z)
        mass_cut_coords = coords[masses > self.lae_lbg_mass_cut]
        lae_coords = mass_cut_coords[ids]

        occ = np.zeros((self.ngrid,) * 3, dtype=np.float32)
        vgrid = np.zeros((self.ngrid,) * 3, dtype=np.float64)
        ix = (lae_coords[:, 0] / self.cell).astype(int) % self.ngrid
        iy = (lae_coords[:, 1] / self.cell).astype(int) % self.ngrid
        iz = (lae_coords[:, 2] / self.cell).astype(int) % self.ngrid
        for a, b, c, v in zip(ix, iy, iz, vals):
            occ[a, b, c] += 1.0
            vgrid[a, b, c] += v
        mask = occ > 0
        vgrid[mask] /= occ[mask]
        return occ, vgrid

    def stitch_value_field(self, seed: int, value_field: str, snap_z, z_arr, logger):
        """Full 3D stitch of (occ, val) -- same nearest-snapshot approach as stitch_discrete."""
        lc_occ = np.zeros((self.ngrid, self.ngrid, self.n_lc_pix), dtype=np.float32)
        lc_val = np.zeros((self.ngrid, self.ngrid, self.n_lc_pix), dtype=np.float64)

        logger.info(f"  Loading {len(snap_z)} grids for {value_field}...")
        occ_grids, val_grids = {}, {}
        for z in snap_z:
            try:
                occ_grids[z], val_grids[z] = self.load_lae_value_grid(seed, z, value_field, logger)
            except Exception as e:  # noqa: BLE001
                logger.error(f"  ERROR at z={z:.4f}: {e}\n{traceback.format_exc()}")
        loaded_z = np.array(sorted(occ_grids.keys()))
        logger.info(f"  Loaded {len(loaded_z)} / {len(snap_z)} grids")

        for n, z in enumerate(z_arr):
            y_cell = self.comoving_pixel(z)
            nearest = loaded_z[self.find_nearest_snapshot(z, loaded_z)]
            lc_occ[:, :, n] = self.get_slab(occ_grids[nearest], y_cell)
            lc_val[:, :, n] = self.get_slab(val_grids[nearest], y_cell)
        return lc_occ, lc_val

    def process_seed_value_fields(self, seed: int, logger) -> None:
        snap_z = self.get_halo_redshifts(seed)  # LAE ids only exist where halo catalogs do
        z_arr = np.linspace(self.z_min, self.z_max, self.n_lc_pix)
        seed_dir = os.path.join(self.out_root, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for value_field in ("lae_lum", "lae_rew"):
            lc_occ, lc_val = self.stitch_value_field(seed, value_field, snap_z, z_arr, logger)
            out = os.path.join(seed_dir, f"lc_{value_field}_3d.npz")
            np.savez_compressed(out, lc_occ=lc_occ, lc_val=lc_val, field=value_field, seed=seed,
                                 zmin=self.z_min, zmax=self.z_max, ngrid=self.ngrid,
                                 box_len=self.box_len, z_arr=z_arr)
            logger.info(f"Saved: {out}  shape={lc_occ.shape}")


def stitch_value_fields(cfg, seeds: list[int]) -> None:
    from ksz_lae_xcorr.lightcone.stitch import setup_logger

    os.makedirs(cfg.paths.lightcone_root, exist_ok=True)
    stitcher = ValueFieldStitcher(cfg)
    for seed in seeds:
        logger = setup_logger(seed, cfg.paths.lightcone_root)
        stitcher.process_seed_value_fields(seed, logger)
