"""
correlation/seed_stats.py
===========================
Aggregates the per-seed output of correlation.cross_correlation.compute_cross_spectra
into median +/- 1 sigma across seeds, at each redshift bin -- the summary form
used for reporting and plotting (matches Figure 2/3 style in the paper draft:
"D_ell vs ell (N seeds, median +/-1 sigma)").

compute_cross_spectra's raw output is per-seed: results[tracer][seed][z_c][signal].
This module collapses the seed axis.
"""

from __future__ import annotations

import numpy as np


def aggregate_over_seeds(cross_results: dict, tracer: str, signal: str, seeds: list[int],
                          ddof: int = 1) -> dict:
    """
    Collapse cross_results[tracer][seed][z_c][signal] over the seed axis.

    ddof=1 (default): sample standard deviation (divides by N_seeds - 1),
    the standard choice when treating seeds as a finite sample estimating
    the true seed-to-seed scatter. Set ddof=0 for population std instead.

    Returns: {z_c: {'ell': array, 'median': array, 'sigma': array,
                     'lower': median-sigma, 'upper': median+sigma,
                     'n_seeds': int}}
    for every z_c that has at least 2 seeds with data for this signal
    (sigma is undefined/NaN-prone with fewer than 2).
    """
    if tracer not in cross_results:
        raise KeyError(f"tracer '{tracer}' not in cross_results -- available: {list(cross_results.keys())}")

    # collect the z-bins that appear for at least one seed
    all_z = set()
    for seed in seeds:
        if seed in cross_results[tracer]:
            all_z.update(cross_results[tracer][seed].keys())

    out = {}
    for z_c in sorted(all_z):
        ell_ref = None
        D_stack = []
        for seed in seeds:
            entry = cross_results[tracer].get(seed, {}).get(z_c, {}).get(signal)
            if entry is None:
                continue
            if ell_ref is None:
                ell_ref = entry["ell"]
            D_stack.append(entry["D_ell"])

        if len(D_stack) < 2:
            continue  # can't report a sigma with fewer than 2 seeds

        D_stack = np.array(D_stack)  # shape (n_seeds_available, n_ell)
        median = np.nanmedian(D_stack, axis=0)
        sigma = np.nanstd(D_stack, axis=0, ddof=ddof)

        out[z_c] = {
            "ell": ell_ref,
            "median": median,
            "sigma": sigma,
            "lower": median - sigma,
            "upper": median + sigma,
            "n_seeds": len(D_stack),
        }
    return out


def aggregate_all_z(cross_results: dict, tracer: str, signal: str, seeds: list[int],
                     ddof: int = 1) -> dict:
    """Convenience wrapper: same as aggregate_over_seeds, just named for clarity
    when iterating over every z-bin (e.g. for a rainbow-over-z plot)."""
    return aggregate_over_seeds(cross_results, tracer, signal, seeds, ddof=ddof)


def aggregate_coherence_over_seeds(coherence_results: dict, seeds: list[int],
                                    field: str = "D_diag", ddof: int = 1) -> dict:
    """
    Median +/- 1 sigma across seeds of one field ('D_total', 'D_diag', or
    'D_off') from correlation.coherence_decomposition's per-seed output
    (as saved by scripts/09_coherence_decomposition.py). Simpler than
    aggregate_over_seeds -- no z_c dimension, since the coherence
    decomposition is evaluated at a single reference redshift per seed,
    same as auto_power.compute_auto_spectra.

    Requires every seed's 'ell' grid to match (same KGrid/cfg -- true by
    construction if every seed came from the same config) -- raises rather
    than silently averaging mismatched ell grids.

    Returns: {'ell', 'median', 'sigma', 'lower', 'upper', 'n_seeds'}
    """
    available = [s for s in seeds if s in coherence_results]
    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 seeds with coherence_decomposition results to "
            f"report a sigma -- got {len(available)} ({available})."
        )

    ell_ref = coherence_results[available[0]]["ell"]
    stack = []
    for seed in available:
        ell_this = coherence_results[seed]["ell"]
        if len(ell_this) != len(ell_ref) or not np.allclose(ell_this, ell_ref):
            raise ValueError(
                f"Seed {seed}'s ell grid doesn't match seed {available[0]}'s -- "
                f"were these run with different configs? Cannot median-combine."
            )
        stack.append(coherence_results[seed][field])

    stack = np.array(stack)  # (n_seeds, n_ell)
    median = np.nanmedian(stack, axis=0)
    sigma = np.nanstd(stack, axis=0, ddof=ddof)

    return {
        "ell": ell_ref, "median": median, "sigma": sigma,
        "lower": median - sigma, "upper": median + sigma, "n_seeds": len(available),
    }
