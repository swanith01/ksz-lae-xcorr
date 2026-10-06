"""
correlation/ksz_auto_wrapcycle.py
===================================
Per-seed kSZ auto-power from a WRAP-CYCLE stitched lightcone (see
lightcone/wrap_cycle.py for the method and its provenance in ksz-pipeline).
Intended to supersede scripts/09 + 22's kSZ panel once validated.

Pipeline for ONE realisation (seed):
  1. build a uniform-comoving-spacing LOS grid (spacing == box cell size);
  2. stitch xH / density(raw delta) / v_z from that seed's coeval snapshots
     with per-cycle rotations (streaming; velocity passed through UNSCALED --
     our py21cmfast-v4 velocity_z is already Mpc/s);
  3. assemble the same field_data_seed dict the rest of the repo uses
     (density_lc = 1 + delta exactly once), plus the optional normalisation
     keys 'ne_scale' (helium-inclusive n_e) and 'tau0' (optical depth below
     z_min) that compute_ksz_slices understands;
  4. P_total / P_diag / P_off -> D_ell over (a) the seed's patchy window and
     (b) the full z_min-z_max range, via the existing, tested
     decompose_p_total_diag_off.

With the wrap-cycle lightcone D_total (the coherent LOS sum) is the kSZ
auto-power; D_diag and D_off are kept as consistency diagnostics -- D_off /
D_total should now be small, unlike the fixed-angle lightcone where P_off is
the periodicity artifact itself.
"""

from __future__ import annotations

import time

import numpy as np

from ksz_lae_xcorr.correlation.coherence_decomposition import (
    compute_ksz_slices,
    compute_patchy_window_diag_power,
    decompose_p_total_diag_off,
)
from ksz_lae_xcorr.lightcone.wrap_cycle import build_los_z_grid, stitch_wrapcycle
from ksz_lae_xcorr.snr.roman_hls_benchmark import chi_eff_power_weighted
from ksz_lae_xcorr.utils.cosmology import get_cosmology
from ksz_lae_xcorr.utils.field_units import delta_at_z_from_ic, peculiar_velocity_from_raw
from ksz_lae_xcorr.utils.optical_depth import analytic_tau_below, ne_scale_helium

FIELD_CONVENTIONS = ("physical", "legacy")
NE_CONVENTIONS = ("helium", "legacy")
TAU0_MODES = ("analytic", "none")


def normalisation_keys(cfg, ne_convention: str = "helium", tau0_mode: str = "analytic",
                        Y_He: float = 0.24) -> dict:
    """The optional compute_ksz_slices keys for the chosen conventions.
    'legacy' / 'none' reproduce the pre-2026-10-06 normalisation exactly."""
    if ne_convention not in NE_CONVENTIONS:
        raise ValueError(f"ne_convention must be one of {NE_CONVENTIONS}, got {ne_convention!r}")
    if tau0_mode not in TAU0_MODES:
        raise ValueError(f"tau0_mode must be one of {TAU0_MODES}, got {tau0_mode!r}")
    ne_scale = ne_scale_helium(Y_He) if ne_convention == "helium" else 1.0
    tau0 = analytic_tau_below(cfg, cfg.box.z_min, Y_He=Y_He) if tau0_mode == "analytic" else 0.0
    return {"ne_scale": float(ne_scale), "tau0": float(tau0)}


def build_wrapcycle_field_data(cfg, seed: int, wrap_cycle_seed: int, stitcher, logger,
                                mode: str = "grid-wrap", norm: dict | None = None,
                                field_convention: str = "physical") -> dict:
    """Stitch one seed's wrap-cycle lightcone into a field_data_seed dict.

    field_convention (see utils/field_units.py -- both fixes found 2026-10-06):
      'physical' (default): velocity_z / (1+z) (raw is the comoving rate, not v_pec) and
                  delta(z) = D(z)/D(0) * hires_density (raw is the z=0-normalised IC field);
                  1+delta floored at 0 (linear delta can dip below -1 in the tails).
      'legacy'  : the raw fields exactly as before (known to be ~(1+z)^2 * (rms ratio)^2 too high)."""
    if field_convention not in FIELD_CONVENTIONS:
        raise ValueError(f"field_convention must be one of {FIELD_CONVENTIONS}, got {field_convention!r}")
    Om0 = float(cfg.cosmology.Om0)
    cosmo = get_cosmology(cfg)
    ngrid = int(cfg.box.hii_dim)
    cell = float(cfg.box.box_len_mpc) / ngrid

    snap_z = stitcher.get_snapshot_redshifts(seed, logger)
    z_arr, chi_mpc = build_los_z_grid(cosmo, cfg.box.z_min, cfg.box.z_max, cell)
    n_cycles = int(np.ceil(len(z_arr) / ngrid))
    logger.info(f"wrap-cycle lightcone: {len(z_arr)} LOS pixels at {cell:.3f} Mpc spacing, "
                f"z={z_arr[0]:.3f}-{z_arr[-1]:.3f}, {n_cycles} wrap cycles, "
                f"wrap_cycle_seed={wrap_cycle_seed}, mode={mode}")

    def load_box(z, field):
        box = np.ascontiguousarray(stitcher.load_field_box(seed, z, field), dtype=np.float32)
        if field_convention == "physical":
            if field == "vz":
                box = peculiar_velocity_from_raw(box, z).astype(np.float32)
            elif field == "density":
                box = delta_at_z_from_ic(box, Om0, z).astype(np.float32)
        return box

    t0 = time.time()
    lc = stitch_wrapcycle(
        load_box, snap_z, z_arr, chi_mpc, ngrid=ngrid, cell_size=cell,
        wrap_cycle_seed=wrap_cycle_seed, mode=mode,
        progress=lambda k, n: logger.info(f"  stitched {k + 1}/{n} LOS pixels ({time.time() - t0:.0f}s)"),
    )
    fd = {
        "z_lc": z_arr,
        "xHI_lc": lc["xH"].astype(np.float64),
        "density_lc": (np.maximum(1.0 + lc["density"].astype(np.float64), 0.0)
                       if field_convention == "physical"
                       else 1.0 + lc["density"].astype(np.float64)),   # delta -> 1+delta, ONCE
        "velocity_lc": lc["vz"].astype(np.float64),             # Mpc/s, v4: NO conversion
    }
    fd.update(norm if norm is not None else normalisation_keys(cfg))
    fd["n_cycles"] = n_cycles
    fd["field_convention"] = field_convention
    return fd


def compute_wrapcycle_auto_power(cfg, fd: dict) -> dict:
    """Decompose one seed's wrap-cycle lightcone over (a) the patchy window
    and (b) the full z range. Returns plain arrays/floats (pickle-friendly)."""
    patchy = compute_patchy_window_diag_power(cfg, fd)

    theta, chi_mpc, z = compute_ksz_slices(cfg, fd)
    chi_eff_full = chi_eff_power_weighted(cfg, fd, float(z[0]), float(z[-1]) + 1e-9)
    ell, D_total, D_diag, D_off = decompose_p_total_diag_off(cfg, theta, chi_eff_full)
    full = {"ell": ell, "D_total": D_total, "D_diag": D_diag, "D_off": D_off,
            "z_lo": float(z[0]), "z_hi": float(z[-1]), "chi_eff": float(chi_eff_full)}

    w_z = np.sqrt(np.mean(theta ** 2, axis=(0, 1)))
    return {
        "patchy": patchy, "full": full,
        "z_lc": z, "chi_mpc": chi_mpc,
        "xHI_mean": fd["xHI_lc"][:, :, : len(z)].mean(axis=(0, 1)),
        "theta_rms_z": w_z,
        "ne_scale": fd["ne_scale"], "tau0": fd["tau0"], "n_cycles": fd["n_cycles"],
        "field_convention": fd.get("field_convention", "legacy"),
    }


def implied_velocity_kms(cfg, res: dict, z_targets=(6, 7, 8, 9, 10, 12)) -> list:
    """Rough sanity diagnostic: effective rms[(1+delta) v] (km/s) implied by the per-slab kSZ rms,
    theta_rms = tau_pref * rms(x_e) (1+z)^2 ds e^{-tau} rms[(1+delta) v/c] (e^{-tau} ~ 1 ignored).
    rms(x_e) is approximated by sqrt(mean x_e) (the binary-pixel limit: pixels are mostly 0 or 1);
    dividing by the MEAN x_e instead (as the 2026-10-06 first look did) inflates the result by
    1/sqrt(x_e) -- ~2x at z=9, ~6x at z=12.  Still approximate: expect ~100-300 km/s at z=7-10
    once the fields are physical; ~(1+z) x higher flags a missing 1/(1+z), more flags the density."""
    from ksz_lae_xcorr.utils import constants
    z, chi, th = res["z_lc"], res["chi_mpc"], res["theta_rms_z"]
    xe = 1.0 - res["xHI_mean"]
    pref = constants.tau_prefactor(cfg) * float(res.get("ne_scale", 1.0))
    ds = np.abs(np.gradient(chi))
    v = th / (pref * np.sqrt(np.maximum(xe, 1e-9)) * (1 + z) ** 2 * ds) * 299792.458
    out = []
    for zt in z_targets:
        i = int(np.argmin(np.abs(z - zt)))
        out.append((float(z[i]), float(xe[i]), float(v[i])))
    return out


def summarise_at_ell(res: dict, ell_target: float = 3000.0) -> dict:
    """Headline numbers of one 'patchy' or 'full' block at ell ~ ell_target."""
    i = int(np.argmin(np.abs(res["ell"] - ell_target)))
    tot = res["D_total"][i]
    return {"ell": float(res["ell"][i]), "D_total": float(tot), "D_diag": float(res["D_diag"][i]),
            "D_off": float(res["D_off"][i]),
            "D_off_over_total": float(res["D_off"][i] / tot) if tot != 0 else float("nan")}
