"""
lightcone/stitch.py
====================
Stitches per-redshift coeval boxes (from ksz_lae_xcorr.halos.coeval_pipeline)
plus external LAE/LBG catalogues (from Jahaan's pipeline, see
data/README.md) into full 3D lightcones: shape (NGRID, NGRID, N_LC_PIX).

Direct refactor of stitch_lightcones_3D.py. All box/grid/path parameters
now come from the Config rather than being hardcoded to 400 Mpc / 64^3 / 5
seeds -- this repo's fiducial run is 300 Mpc, HII_DIM=300.

Note on resolution: the coeval fields saved by coeval_pipeline live on two
different grids -- hires_density is DIM=600, but velocity_z/neutral_fraction
are HII_DIM=300 (see the FIELD ACCESS NOTES in coeval_pipeline.py). The
lightcone grid resolution (NGRID below) should match whichever field you're
stitching; the fiducial config stitches everything on the HII_DIM grid for
consistency with the halo/tracer catalogues, which are continuous-coordinate
(not grid-snapped) until binned here.

Usage: see scripts/02_stitch_lightcones.py for the CLI wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from ksz_lae_xcorr.utils.cosmology import get_cosmology


def setup_logger(seed: int, out_root: str) -> logging.Logger:
    log_dir = os.path.join(out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"seed_{seed}_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger = logging.getLogger(f"stitch_seed_{seed}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Stitcher:
    """Holds config-derived constants and does the actual stitching for one seed."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cosmo = get_cosmology(cfg)
        self.box_len = cfg.box.box_len_mpc
        self.ngrid = cfg.box.hii_dim
        self.cell = self.box_len / self.ngrid
        self.z_min = cfg.box.z_min
        self.z_max = cfg.box.z_max
        self.n_lc_pix = cfg.lightcone.n_lc_pix
        self.angle_deg = cfg.lightcone.angle_deg
        self.halo_mass_cut = cfg.tracers.halo_mass_cut_msun
        self.muv_cut = cfg.tracers.lbg_muv_cut

        self.root_coeval = cfg.paths.coeval_root
        self.root_halos = cfg.paths.halo_root
        self.root_lae = cfg.paths.lae_catalogue_root
        self.root_lbg = cfg.paths.lbg_catalogue_root
        self.out_root = cfg.paths.lightcone_root

    # -- geometry -------------------------------------------------------

    def comoving_distance_mpc(self, z: float) -> float:
        return self.cosmo.comoving_distance(z).to(u.Mpc).value

    def comoving_pixel(self, z: float, z0: float | None = None) -> int:
        z0 = self.z_min if z0 is None else z0
        d = self.comoving_distance_mpc(z) - self.comoving_distance_mpc(z0)
        return int(d / self.cell) % self.ngrid

    def periodic(self, n) -> int:
        return int(n) % self.ngrid

    def rotate_index(self, i, j, k):
        a = np.deg2rad(self.angle_deg)
        ir = np.cos(a) * i - np.sin(a) * j
        jr = np.sin(a) * i + np.cos(a) * j
        return self.periodic(ir), self.periodic(jr), self.periodic(k)

    def get_slab(self, box: np.ndarray, y_cell: int) -> np.ndarray:
        """Rotated (NGRID, NGRID) transverse slab at LoS position y_cell."""
        n = self.ngrid
        slab = np.zeros((n, n), dtype=np.float64)
        for ix in range(n):
            for iy in range(n):
                rx, ry, rz = self.rotate_index(ix, iy, y_cell)
                slab[ix, iy] = box[rx, ry, rz]
        return slab

    # -- snapshot discovery ----------------------------------------------

    def get_snapshot_redshifts(self, seed: int, logger) -> np.ndarray:
        coeval_dir = os.path.join(self.root_coeval, f"seed_{seed}")
        dirs = [d for d in os.listdir(coeval_dir) if d.startswith("coeval_z")]
        zvals = []
        for d in dirs:
            try:
                z = float(d.replace("coeval_z", ""))
                if self.z_min - 0.5 <= z <= self.z_max + 0.5:
                    zvals.append(z)
            except ValueError:
                logger.warning(f"Could not parse redshift from dir: {d}")
        zvals = np.array(sorted(zvals))
        logger.info(f"Found {len(zvals)} snapshots: z={zvals[0]:.4f} -> z={zvals[-1]:.4f}")
        return zvals

    def get_halo_redshifts(self, seed: int) -> np.ndarray:
        halo_dir = os.path.join(self.root_halos, f"seed_{seed}", "halo_catalogs")
        if not os.path.isdir(halo_dir):
            return np.array([])
        files = [f for f in os.listdir(halo_dir) if f.startswith("halo_coords_z")]
        zvals = []
        for f in files:
            try:
                z = float(f.replace("halo_coords_z", "").replace(".npy", ""))
                if self.z_min - 0.5 <= z <= self.z_max + 0.5:
                    zvals.append(z)
            except ValueError:
                pass
        return np.array(sorted(zvals))

    @staticmethod
    def find_nearest_snapshot(z: float, snap_z: np.ndarray) -> int:
        return int(np.argmin(np.abs(snap_z - z)))

    # -- box / catalogue loaders ------------------------------------------

    def load_field_box(self, seed: int, z: float, field_name: str) -> np.ndarray:
        field_map = {
            "xH": "neutral_fraction.npy",
            "density": "hires_density.npy",
            "vz": "velocity_z.npy",
        }
        fname = field_map[field_name]
        path = os.path.join(self.root_coeval, f"seed_{seed}", f"coeval_z{z:.6f}", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        box = np.load(path, mmap_mode="r")
        if field_name == "vz":
            box = np.array(box) / (1 + z) * 3.086e19
        return box

    def _halo_coords_masses(self, seed: int, z: float):
        halo_dir = os.path.join(self.root_halos, f"seed_{seed}", "halo_catalogs")
        cpath = os.path.join(halo_dir, f"halo_coords_z{z:.6f}.npy")
        mpath = os.path.join(halo_dir, f"halo_masses_z{z:.6f}.npy")
        if not os.path.exists(cpath) or not os.path.exists(mpath):
            raise FileNotFoundError(f"Missing halo files at z={z:.6f}")
        return np.load(cpath, mmap_mode="r"), np.load(mpath, mmap_mode="r")

    def _bin_to_grid(self, coords: np.ndarray) -> np.ndarray:
        grid = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype=np.float32)
        ix = (coords[:, 0] / self.cell).astype(int) % self.ngrid
        iy = (coords[:, 1] / self.cell).astype(int) % self.ngrid
        iz = (coords[:, 2] / self.cell).astype(int) % self.ngrid
        for a, b, c in zip(ix, iy, iz):
            grid[a, b, c] += 1.0
        return grid

    def load_halo_grid(self, seed: int, z: float, logger) -> np.ndarray:
        coords, masses = self._halo_coords_masses(seed, z)
        sel = masses > self.halo_mass_cut
        return self._bin_to_grid(coords[sel])

    def load_lae_grid(self, seed: int, z: float, logger) -> np.ndarray:
        """
        Load LAE catalogue (external, Jahaan's pipeline -- see data/README.md).
        `ids` index into the mass-cut halo coordinate subset (see
        tracers.lae_lbg_mass_cut_msun in the config); format/path TBD until
        the catalogues are handed over -- this raises FileNotFoundError
        gracefully and returns an empty grid until then.
        """
        idpath = os.path.join(self.root_lae, "halo_ids_obs", f"halo_ids_obs_z{z:.4f}_s{seed}.npy")
        if not os.path.exists(idpath):
            logger.warning(f"  LAE ids missing at z={z:.4f} seed={seed}, using empty grid")
            return np.zeros((self.ngrid,) * 3, dtype=np.float32)
        ids = np.load(idpath, mmap_mode="r")
        coords, _ = self._halo_coords_masses(seed, z)
        return self._bin_to_grid(coords[ids])

    def load_lbg_grid(self, seed: int, z: float, logger) -> np.ndarray:
        """Load LBG catalogue (external, same source as LAE -- see data/README.md)."""
        idpath = os.path.join(self.root_lbg, "halo_ids_lbg", f"halo_ids_lbg_z{z:.4f}_s{seed}.npy")
        muvpath = os.path.join(self.root_lbg, "MUV_lbg", f"MUV_lbg_z{z:.4f}_s{seed}.npy")
        if not os.path.exists(idpath) or not os.path.exists(muvpath):
            logger.warning(f"  LBG files missing at z={z:.4f} seed={seed}, using empty grid")
            return np.zeros((self.ngrid,) * 3, dtype=np.float32)
        ids = np.load(idpath, mmap_mode="r")
        muv = np.load(muvpath, mmap_mode="r")
        coords, _ = self._halo_coords_masses(seed, z)
        bright = muv < self.muv_cut
        return self._bin_to_grid(coords[ids[bright]])

    # -- stitching ----------------------------------------------------------

    def stitch_continuous(self, seed, field_name, snap_z, z_arr, logger) -> np.ndarray:
        lc = np.zeros((self.ngrid, self.ngrid, self.n_lc_pix), dtype=np.float32)
        logger.info(f"  Loading {len(snap_z)} boxes for {field_name}...")
        boxes = {}
        for z in snap_z:
            try:
                boxes[z] = self.load_field_box(seed, z, field_name)
            except FileNotFoundError as e:
                logger.error(f"  SKIP snapshot z={z:.4f}: {e}")
        loaded_z = np.array(sorted(boxes.keys()))
        logger.info(f"  Loaded {len(loaded_z)} / {len(snap_z)} boxes")

        for n, z in enumerate(z_arr):
            y_cell = self.comoving_pixel(z)
            slabs = np.stack([self.get_slab(boxes[sz], y_cell) for sz in loaded_z], axis=-1)
            interp = interp1d(loaded_z, slabs, axis=-1, bounds_error=False, fill_value="extrapolate")
            lc[:, :, n] = interp(z)
            if n % 50 == 0:
                logger.debug(f"  LoS pixel {n}/{self.n_lc_pix}  z={z:.3f}")
        return lc

    def stitch_discrete(self, seed, field_name, snap_z, z_arr, logger) -> np.ndarray:
        lc = np.zeros((self.ngrid, self.ngrid, self.n_lc_pix), dtype=np.float32)
        loader_map = {"halos": self.load_halo_grid, "lae": self.load_lae_grid, "lbg": self.load_lbg_grid}
        loader = loader_map[field_name]

        logger.info(f"  Loading {len(snap_z)} grids for {field_name}...")
        grids = {}
        for z in snap_z:
            try:
                grids[z] = loader(seed, z, logger)
            except FileNotFoundError as e:
                logger.error(f"  SKIP snapshot z={z:.4f}: {e}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"  ERROR at z={z:.4f}: {e}\n{traceback.format_exc()}")
        loaded_z = np.array(sorted(grids.keys()))
        logger.info(f"  Loaded {len(loaded_z)} / {len(snap_z)} grids")

        for n, z in enumerate(z_arr):
            y_cell = self.comoving_pixel(z)
            nearest = loaded_z[self.find_nearest_snapshot(z, loaded_z)]
            lc[:, :, n] = self.get_slab(grids[nearest], y_cell)
            if n % 50 == 0:
                logger.debug(f"  LoS pixel {n}/{self.n_lc_pix}  z={z:.3f}  nearest={nearest:.4f}")
        return lc

    # -- per-seed pipeline ----------------------------------------------------

    def output_path(self, seed: int, field: str) -> str:
        seed_dir = os.path.join(self.out_root, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        return os.path.join(seed_dir, f"lc_{field}.npz")

    def checkpoint_path(self, seed: int, field: str) -> str:
        return os.path.join(self.out_root, "checkpoints", f"seed_{seed}_{field}.done")

    def is_done(self, seed: int, field: str) -> bool:
        return os.path.exists(self.checkpoint_path(seed, field))

    def mark_done(self, seed: int, field: str, meta: dict | None = None) -> None:
        os.makedirs(os.path.join(self.out_root, "checkpoints"), exist_ok=True)
        with open(self.checkpoint_path(seed, field), "w") as f:
            json.dump({"seed": seed, "field": field, "time": datetime.now().isoformat(),
                       "meta": meta or {}}, f, indent=2)

    def process_seed(self, seed: int) -> None:
        logger = setup_logger(seed, self.out_root)
        logger.info(f"SEED {seed} started at {datetime.now().isoformat()}")

        snap_z = self.get_snapshot_redshifts(seed, logger)
        halo_snap_z = self.get_halo_redshifts(seed)
        z_arr = np.linspace(self.z_min, self.z_max, self.n_lc_pix)

        fields = [(f, "continuous") for f in self.cfg.lightcone.fields.continuous]
        fields += [(f, "discrete") for f in self.cfg.lightcone.fields.discrete]

        for field_name, field_type in fields:
            logger.info(f"-- Field: {field_name} ({field_type}) --")
            if self.is_done(seed, field_name):
                logger.info(f"  SKIP: checkpoint exists")
                continue

            use_snap_z = halo_snap_z if field_name in ("halos", "lae", "lbg") else snap_z
            try:
                if field_type == "continuous":
                    lc = self.stitch_continuous(seed, field_name, use_snap_z, z_arr, logger)
                else:
                    lc = self.stitch_discrete(seed, field_name, use_snap_z, z_arr, logger)

                out_file = self.output_path(seed, field_name)
                np.savez_compressed(out_file, lc=lc, z_arr=z_arr, seed=seed, field=field_name,
                                     zmin=self.z_min, zmax=self.z_max, ngrid=self.ngrid,
                                     box_len=self.box_len)
                logger.info(f"  Saved: {out_file}  shape={lc.shape}  min={lc.min():.4f}  max={lc.max():.4f}")
                self.mark_done(seed, field_name, meta={"shape": list(lc.shape)})
            except Exception as e:  # noqa: BLE001
                msg = f"FAILED seed={seed} field={field_name}: {e}\n{traceback.format_exc()}"
                logger.error(msg)
                err_dir = os.path.join(self.out_root, "logs")
                os.makedirs(err_dir, exist_ok=True)
                with open(os.path.join(err_dir, f"error_seed{seed}_{field_name}.txt"), "w") as f:
                    f.write(msg)
                continue


def stitch_seeds(cfg, seeds: list[int]) -> None:
    os.makedirs(cfg.paths.lightcone_root, exist_ok=True)
    stitcher = Stitcher(cfg)
    for seed in seeds:
        stitcher.process_seed(seed)
