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

# One symbol per run type in the search view
SYMBOLS = [CROSS, "o", "s", "t", "d", "x", "star", "h"]
