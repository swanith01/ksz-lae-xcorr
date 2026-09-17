"""
Tests for utils/figio.py -- particularly the regression test for a real
bug caught 2026-09-17: a base_path stem containing an internal decimal
point but no real file extension (e.g. 'foo_z9.5_symlog', produced by
scripts/20 for a z=9.5 run) was mis-split by a naive os.path.splitext,
which silently truncated the filename to 'foo_z9.pdf' instead of the
intended 'foo_z9.5_symlog.pdf'.

Run with:
    pytest tests/test_figio.py -v
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz_lae_xcorr.utils.figio import save_fig


def _tiny_fig():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig


def test_save_fig_with_no_extension_and_no_internal_dot(tmp_path):
    stem = str(tmp_path / "plot")
    pdf_path, png_path = save_fig(_tiny_fig(), stem)
    assert pdf_path == stem + ".pdf"
    assert png_path == stem + ".png"
    assert os.path.exists(pdf_path) and os.path.exists(png_path)


def test_save_fig_with_explicit_pdf_extension(tmp_path):
    stem = str(tmp_path / "plot.pdf")
    pdf_path, png_path = save_fig(_tiny_fig(), stem)
    assert pdf_path == str(tmp_path / "plot.pdf")
    assert png_path == str(tmp_path / "plot.png")


def test_save_fig_regression_decimal_in_stem_no_real_extension(tmp_path):
    """THE regression test: a stem with an internal decimal point and NO
    real extension must NOT be truncated at that decimal point."""
    stem = str(tmp_path / "direct_bispectrum_seed1_z9.5_symlog")
    pdf_path, png_path = save_fig(_tiny_fig(), stem)
    assert pdf_path == stem + ".pdf", (
        f"Got {pdf_path} -- the decimal in 'z9.5' was incorrectly treated "
        f"as an extension separator, truncating the filename."
    )
    assert png_path == stem + ".png"
    assert "z9.5_symlog.pdf" in pdf_path
    assert os.path.exists(pdf_path) and os.path.exists(png_path)


def test_save_fig_regression_decimal_with_real_extension_still_works(tmp_path):
    """The companion case: a decimal in the stem is fine to strip
    correctly WHEN a real extension follows it -- this must keep working,
    not just the no-extension case."""
    stem = str(tmp_path / "direct_vs_stitched_seed1_z9.5.pdf")
    pdf_path, png_path = save_fig(_tiny_fig(), stem)
    assert pdf_path == str(tmp_path / "direct_vs_stitched_seed1_z9.5.pdf")
    assert png_path == str(tmp_path / "direct_vs_stitched_seed1_z9.5.png")
