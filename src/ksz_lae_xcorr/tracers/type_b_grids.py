"""
tracers/type_b_grids.py
========================
Builds "Type B" lightcones -- occupation count PLUS the physical value per
cell (halo mass, Lya luminosity, MUV magnitude). These feed the proportional
-marker diagnostic plots only (plotting.lightcone_panels); they are NOT used
anywhere in the cross-power or SNR pipeline (correlation/, snr/), which use
the plain occupation-count lightcones from lightcone.stitch instead.

Direct refactor of type_b_grids.py. Note this stays a 2D (single-slice)
product as in the original -- unlike lightcone.stitch's full 3D transverse
plane -- since it is only ever used for a 2D lightcone slice plot. Reuses
Stitcher's geometry helpers (comoving_pixel, get_slab, rotate_index) rather
than re-implementing them, so a single cosmology/config is the source of
truth for both modules.
"""

from __future__ import annotations

import os

import numpy as np

from ksz_lae_xcorr.lightcone.stitch import Stitcher


class TypeBBuilder(Stitcher):
    """Adds physical-value (mass/luminosity/MUV) grid builders on top of Stitcher."""

    def __init__(self, cfg, x_slice: int | None = None):
        super().__init__(cfg)
        # Which transverse column to keep for the 2D slice product.
        self.x_slice = x_slice if x_slice is not None else self.ngrid // 2

    def build_halo_mass_grid(self, seed: int, z: float):
        coords, masses = self._halo_coords_masses(seed, z)
        sel = masses > self.halo_mass_cut
        coords, masses = coords[sel], masses[sel]

        occ = np.zeros((self.ngrid,) * 3, dtype=np.float32)
        mval = np.zeros((self.ngrid,) * 3, dtype=np.float64)
        ix = (coords[:, 0] / self.cell).astype(int) % self.ngrid
        iy = (coords[:, 1] / self.cell).astype(int) % self.ngrid
        iz = (coords[:, 2] / self.cell).astype(int) % self.ngrid
        for a, b, c, m in zip(ix, iy, iz, masses):
            occ[a, b, c] += 1.0
            mval[a, b, c] += m
        mask = occ > 0
        mval[mask] /= occ[mask]
        return occ, mval

    def build_lae_lum_grid(self, seed: int, z: float, logger):
        idpath = os.path.join(self.root_lae, "halo_ids_obs", f"halo_ids_obs_z{z:.4f}_s{seed}.npy")
        lumpath = os.path.join(self.root_lae, "lya_lum_obs", f"lya_lum_obs_z{z:.4f}_s{seed}.npy")
        if not os.path.exists(idpath):
            logger.warning(f"  LAE ids missing z={z:.4f} s{seed} -> empty grid")
            return (np.zeros((self.ngrid,) * 3, dtype=np.float32),
                    np.zeros((self.ngrid,) * 3, dtype=np.float64))
        ids = np.load(idpath, mmap_mode="r")
        lum = np.load(lumpath, mmap_mode="r")
        coords, _ = self._halo_coords_masses(seed, z)
        lae_coords = coords[ids]

        occ = np.zeros((self.ngrid,) * 3, dtype=np.float32)
        lgrid = np.zeros((self.ngrid,) * 3, dtype=np.float64)
        ix = (lae_coords[:, 0] / self.cell).astype(int) % self.ngrid
        iy = (lae_coords[:, 1] / self.cell).astype(int) % self.ngrid
        iz = (lae_coords[:, 2] / self.cell).astype(int) % self.ngrid
        for a, b, c, l in zip(ix, iy, iz, lum):
            occ[a, b, c] += 1.0
            lgrid[a, b, c] += l
        mask = occ > 0
        lgrid[mask] /= occ[mask]
        return occ, lgrid

    def build_lbg_muv_grid(self, seed: int, z: float, logger):
        idpath = os.path.join(self.root_lbg, "halo_ids_lbg", f"halo_ids_lbg_z{z:.4f}_s{seed}.npy")
        muvpath = os.path.join(self.root_lbg, "MUV_lbg", f"MUV_lbg_z{z:.4f}_s{seed}.npy")
        if not os.path.exists(idpath) or not os.path.exists(muvpath):
            logger.warning(f"  LBG files missing z={z:.4f} s{seed} -> empty grid")
            return (np.zeros((self.ngrid,) * 3, dtype=np.float32),
                    np.zeros((self.ngrid,) * 3, dtype=np.float64))
        ids = np.load(idpath, mmap_mode="r")
        muv = np.load(muvpath, mmap_mode="r")
        coords, _ = self._halo_coords_masses(seed, z)
        bright = muv < self.muv_cut
        lbg_coords = coords[ids[bright]]
        muv_sel = muv[bright]

        occ = np.zeros((self.ngrid,) * 3, dtype=np.float32)
        mgrid = np.zeros((self.ngrid,) * 3, dtype=np.float64)
        ix = (lbg_coords[:, 0] / self.cell).astype(int) % self.ngrid
        iy = (lbg_coords[:, 1] / self.cell).astype(int) % self.ngrid
        iz = (lbg_coords[:, 2] / self.cell).astype(int) % self.ngrid
        for a, b, c, m in zip(ix, iy, iz, muv_sel):
            occ[a, b, c] += 1.0
            mgrid[a, b, c] += m
        mask = occ > 0
        mgrid[mask] /= occ[mask]
        return occ, mgrid

    def _stitch_2d_slice(self, snap_z, grids_occ, grids_val, z_arr, logger):
        lc_occ = np.zeros((self.ngrid, self.n_lc_pix), dtype=np.float32)
        lc_val = np.zeros((self.ngrid, self.n_lc_pix), dtype=np.float64)
        for n, z in enumerate(z_arr):
            y_cell = self.comoving_pixel(z)
            nearest = self.find_nearest_snapshot(z, snap_z)
            slab_occ = self.get_slab(grids_occ[nearest], y_cell)
            slab_val = self.get_slab(grids_val[nearest], y_cell)
            lc_occ[:, n] = slab_occ[:, self.x_slice]
            lc_val[:, n] = slab_val[:, self.x_slice]
        return lc_occ, lc_val

    def process_seed(self, seed: int, logger) -> None:  # type: ignore[override]
        snap_z = self.get_snapshot_redshifts(seed, logger)
        z_arr = np.linspace(self.z_min, self.z_max, self.n_lc_pix)
        seed_dir = os.path.join(self.out_root, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        specs = [
            ("halos_mass", lambda z: self.build_halo_mass_grid(seed, z)),
            ("lae_lum", lambda z: self.build_lae_lum_grid(seed, z, logger)),
            ("lbg_muv", lambda z: self.build_lbg_muv_grid(seed, z, logger)),
        ]
        for field, builder in specs:
            occ, val = {}, {}
            for i, z in enumerate(snap_z):
                try:
                    occ[i], val[i] = builder(z)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"  {field} z={z:.4f}: {e}")
                    occ[i] = np.zeros((self.ngrid,) * 3, dtype=np.float32)
                    val[i] = np.zeros((self.ngrid,) * 3, dtype=np.float64)

            lc_occ, lc_val = self._stitch_2d_slice(snap_z, occ, val, z_arr, logger)
            out = os.path.join(seed_dir, f"lc_{field}.npz")
            np.savez_compressed(out, lc_occ=lc_occ, lc_val=lc_val, field=field, seed=seed,
                                 zmin=self.z_min, zmax=self.z_max, ngrid=self.ngrid,
                                 box_len=self.box_len, z_arr=z_arr)
            logger.info(f"Saved: {out}")


def build_type_b_grids(cfg, seeds: list[int]) -> None:
    from ksz_lae_xcorr.lightcone.stitch import setup_logger

    builder = TypeBBuilder(cfg)
    for seed in seeds:
        logger = setup_logger(seed, cfg.paths.lightcone_root)
        builder.process_seed(seed, logger)
