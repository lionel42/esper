"""Plot symbols and colors shared by the views."""

from __future__ import annotations

from PySide6.QtGui import QPainterPath


def _cross() -> QPainterPath:
    """A '+' made of two lines (pyqtgraph's '+' is a filled outline)."""
    path = QPainterPath()
    path.moveTo(-0.5, 0)
    path.lineTo(0.5, 0)
    path.moveTo(0, -0.5)
    path.lineTo(0, 0.5)
    return path


CROSS = _cross()

COLORS = ["#1f9e1f", "#ff9900", "#9b30b0", "#202020", "#1030e0", "#a00000", "#e070d0", "#008080"]

# Overlaid chromatograms (no black, used by the main mass)
OVERLAY_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

# One symbol per run type in the search view
SYMBOLS = [CROSS, "o", "s", "t", "d", "x", "star", "h"]

# Table cell backgrounds: good -> bad
QUALITY_COLORS = ["#c8e6c9", "#e6ee9c", "#ffcc80", "#ef9a9a"]  # green, yellow-green, orange, red
# Lower limits of a spectral score (0-1) for green, yellow-green, orange; below: red
SCORE_THRESHOLDS = (0.95, 0.85, 0.70)
# Upper limits of an absolute retention index difference for green, yellow-green, orange; above: red
RI_DIFF_THRESHOLDS = (5.0, 10.0, 20.0)
