"""
utils/cosmology.py
===================
Single source of the astropy cosmology object used EVERYWHERE in this repo:
lightcone stitching (comoving-distance-to-pixel mapping), the Limber
approximation (ell <-> k at a given z), and the CAMB C_ell^TT call.

Historical note: earlier scratch code split this into two different
cosmologies (a 21cmFAST-default one used for stitching, a Planck18-like one
used for the analysis notebook), which silently biased the z<->ell
correspondence. This project standardizes on the 21cmFAST-default cosmology
(matching ksz2-21cm and the underlying py21cmfast simulation) everywhere.
Do not construct a second FlatLambdaCDM anywhere else in this repo — import
get_cosmology() instead.
"""

from __future__ import annotations

from astropy.cosmology import FlatLambdaCDM


def get_cosmology(cfg) -> FlatLambdaCDM:
    """Build the FlatLambdaCDM object from a loaded Config's `cosmology` block."""
    c = cfg.cosmology
    return FlatLambdaCDM(H0=c.H0, Om0=c.Om0, Ob0=c.Ob0)


def little_h(cfg) -> float:
    return cfg.cosmology.H0 / 100.0
