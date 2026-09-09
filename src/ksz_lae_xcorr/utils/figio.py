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


def save_fig(fig, base_path: str, dpi: int = 200, bbox_inches="tight") -> tuple[str, str]:
    """
    Save fig as both PDF and PNG next to each other. Returns (pdf_path, png_path).
    """
    root, _ext = os.path.splitext(base_path)
    pdf_path = root + ".pdf"
    png_path = root + ".png"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    return pdf_path, png_path
