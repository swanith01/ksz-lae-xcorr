"""
utils/figio.py
=================
save_fig(fig, base_path, ...) writes BOTH a .pdf and a .png from one
call -- Girish reads PNGs, PDFs are for the repo/paper -- so every
plot-producing script should go through this rather than calling
fig.savefig() directly with a hardcoded extension.

base_path may be given with or without an extension -- either
'output/plot' or 'output/plot.pdf' produces the same pair
('output/plot.pdf' and 'output/plot.png').
"""

import os

# Only strip the suffix if it's actually one of these -- otherwise a
# stem with an internal decimal point and no real extension (e.g.
# 'output/direct_bispectrum_z9.5_symlog', no '.pdf'/'.png' yet) gets
# mis-split by a naive os.path.splitext, which treats the LAST dot
# anywhere in the string as an extension separator regardless of what
# follows it -- caught 2026-09-17 when this produced
# 'direct_bispectrum_seed1_z9.pdf' instead of
# 'direct_bispectrum_seed1_z9.5_symlog.pdf'.
_KNOWN_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}


def save_fig(fig, base_path: str, dpi: int = 200, bbox_inches="tight") -> tuple[str, str]:
    """
    Save fig as both PDF and PNG next to each other. Returns (pdf_path, png_path).
    """
    root, ext = os.path.splitext(base_path)
    if ext.lower() not in _KNOWN_EXTENSIONS:
        root = base_path  # not a real extension -- treat the whole thing as the stem
    pdf_path = root + ".pdf"
    png_path = root + ".png"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    return pdf_path, png_path


def compute_symlog_linthresh(*arrays):
    """
    A sensible linthresh for matplotlib's symlog scale, from one or more
    arrays of values to be plotted on that axis (e.g. central values,
    band edges, error bars). Filters out NaN/inf before computing --
    np.isfinite excludes both; a naive "!= 0" filter does NOT exclude
    NaN (NaN is never equal to anything, including itself), so a single
    NaN anywhere in the input silently poisons np.percentile into NaN,
    which can make the ENTIRE symlog axis render blank rather than just
    skipping the bad point. Caught 2026-09-17 when a stitched-pathway
    CSV (known to contain NaN from that pathway's own instability) did
    exactly this to a whole plot.
    """
    import numpy as np

    vals = np.concatenate([np.asarray(a, dtype=float) for a in arrays if len(a)])
    finite = vals[np.isfinite(vals)]
    pos = np.abs(finite[finite != 0])
    return max(np.percentile(pos, 5), 1e-30) if len(pos) else 1e-6
