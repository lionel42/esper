"""Plot symbols and colors shared by the views."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor, QPainterPath


def _cross() -> QPainterPath:
    """A '+' made of two lines (pyqtgraph's '+' is a filled outline)."""
    path = QPainterPath()
    path.moveTo(-0.5, 0)
    path.lineTo(0.5, 0)
    path.moveTo(0, -0.5)
    path.lineTo(0, 0.5)
    return path


CROSS = _cross()

# Unrelated series (masses in the search view, run types not in RUNTYPE_COLORS): readable
# on white and unlike the run type colors
COLORS = [
    "#009e9e",  # teal
    "#e8505b",  # coral
    "#8a7d00",  # olive
    "#3d9be0",  # sky blue
    "#8b4513",  # brown
    "#7f7f7f",  # grey
    "#5a3e9b",  # indigo
    "#2f4f4f",  # dark slate
]

# Run types (sample_type) by default, by prefix (e.g. 'test' for 'testT1'); the others get COLORS
RUNTYPE_COLORS = {
    "air": "#0000e6",
    "std": "#990000",
    "tank": "#ff9900",
    "cal": "#008000",
    "blank": "#9919b3",
    "tert": "#cc66cc",
    "labair": "#000000",
}


def runtype_colors(runtypes: list[str]) -> dict[str, str]:
    """Color of each run type: from RUNTYPE_COLORS (longest matching prefix), else from COLORS."""
    colors = {}
    others = 0
    for runtype in runtypes:
        prefixes = [k for k in RUNTYPE_COLORS if runtype.startswith(k)]
        if prefixes:
            colors[runtype] = RUNTYPE_COLORS[max(prefixes, key=len)]
        else:
            colors[runtype] = COLORS[others % len(COLORS)]
            others += 1
    return colors


# Overlaid chromatograms, the main mass being black: a few (the qualifiers) get these colors,
# more get colors sampled from magma, without its ends (near black: the main mass; near
# white: the background)
QUALIFIER_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
MAGMA_RANGE = (0.25, 0.8)


def overlay_colors(n: int) -> list[QColor | str]:
    """Colors of n overlaid curves."""
    if n <= len(QUALIFIER_COLORS):
        return QUALIFIER_COLORS[:n]
    magma = pg.colormap.get("magma")
    return [magma.map(x, mode="qcolor") for x in np.linspace(*MAGMA_RANGE, n)]


# One symbol per run type in the search view
SYMBOLS = [CROSS, "o", "s", "t", "d", "x", "star", "h"]

# Table cell backgrounds: good -> bad
QUALITY_COLORS = [
    "#c8e6c9",
    "#e6ee9c",
    "#ffcc80",
    "#ef9a9a",
]  # green, yellow-green, orange, red
# Lower limits of a spectral score (0-1) for green, yellow-green, orange; below: red
SCORE_THRESHOLDS = (0.95, 0.85, 0.70)
# Upper limits of an absolute retention index difference for green, yellow-green, orange; above: red
RI_DIFF_THRESHOLDS = (5.0, 10.0, 20.0)
