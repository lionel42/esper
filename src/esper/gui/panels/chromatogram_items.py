"""Plot items of the peaks in a chromatogram: integrations and fitted peak shapes.

The functions return the items; the panel adds them to its plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt

from esper.fitting import peak_shape
from esper.integration import integration_bounds, peak_area

# Fitted peaks: drawn where above this fraction of their height, filled with this
# opacity (0-255; peaks of the selected compounds: FIT_ALPHA_SELECTED)
FIT_VISIBLE = 1e-3
FIT_ALPHA = 40
FIT_ALPHA_SELECTED = 90
# Integration view: bases of a fitted peak where it falls to this fraction of its height
FIT_BASE_FRACTION = 0.01
INTEGRATION_COLOR = "#c80000"
# Integration of the extraction (ips)
IPS_COLOR = "r"
# Peak labels without a baseline: below the lowest signal within this of the apex [s]
LABEL_HALF_WINDOW = 5.0


def pairs(a0, a1) -> np.ndarray:
    """Points of segments (a0[i], a1[i]), for ``connect="pairs"``."""
    return np.column_stack([a0, a1]).ravel()


def fitted_shapes(t: np.ndarray, peaks, fraction: float):
    """(peak, slice of t, shape above the baseline) of the fitted peaks of one mass,
    where the shape is above ``fraction`` of its height."""
    fitted = peaks[peaks["fit_rt_s"].notna() & (peaks["fit_height"] > 0)]
    for p in fitted.itertuples():
        shape = peak_shape(t, p)
        inside = np.flatnonzero(shape > fraction * p.fit_height)
        if len(inside) < 2:
            continue
        part = slice(inside[0], inside[-1] + 1)
        yield p, part, shape[part]


def lowest_near(t: np.ndarray, y: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """Lowest signal within LABEL_HALF_WINDOW of each rt (the signal at rt if no sample is)."""
    out = np.interp(rt, t, y)
    for i, r in enumerate(rt):
        near = y[(t >= r - LABEL_HALF_WINDOW) & (t <= r + LABEL_HALF_WINDOW)]
        if len(near):
            out[i] = near.min()
    return out


def integration_marks(start, stop, y_start, y_stop, rt, y_base, y_apex, color) -> list:
    """Baseline of each peak (start to stop), its bounds ("|") and a line from the
    baseline to the apex."""
    return [
        pg.PlotDataItem(
            pairs(start, stop),
            pairs(y_start, y_stop),
            connect="pairs",
            pen=pg.mkPen(color, width=1),
        ),
        pg.PlotDataItem(
            pairs(rt, rt),
            pairs(y_base, y_apex),
            connect="pairs",
            pen=pg.mkPen("k", width=0.5),
        ),
        pg.ScatterPlotItem(
            np.concatenate([start, stop]),
            np.concatenate([y_start, y_stop]),
            symbol="|",
            size=20,
            pen=pg.mkPen(color, width=1.5),
        ),
    ]


def fit_integration_items(t: np.ndarray, peaks) -> list:
    """Outline, bases, constant baseline and apex line of the fitted peaks of one mass."""
    items = []
    starts, stops, bases, rts, tops = [], [], [], [], []
    for p, part, shape in fitted_shapes(t, peaks, FIT_BASE_FRACTION):
        starts.append(t[part][0])
        stops.append(t[part][-1])
        bases.append(p.fit_baseline)
        rts.append(p.fit_rt_s)
        tops.append(p.fit_baseline + p.fit_height)
        outline = pg.PlotCurveItem(
            t[part],
            p.fit_baseline + shape,
            pen=pg.mkPen(INTEGRATION_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        outline.setZValue(-1)
        items.append(outline)
    if starts:
        items += integration_marks(
            starts, stops, bases, bases, rts, bases, tops, INTEGRATION_COLOR
        )
    return items


def ips_items(
    t: np.ndarray, y: np.ndarray, peaks
) -> tuple[list, np.ndarray, np.ndarray]:
    """Integration of the peaks with ips (straight baseline between their bounds).

    Returns the items, the areas and the baselines below the apex (NaN without ips).
    """
    rt = peaks["rt_s"].to_numpy(dtype=float)
    start, stop = integration_bounds(peaks, t)
    valid = np.isfinite(start) & np.isfinite(stop)
    areas = np.full(len(peaks), np.nan)
    y_base = np.full(len(peaks), np.nan)
    if not valid.any():
        return [], areas, y_base
    rt, start, stop = rt[valid], start[valid], stop[valid]
    y_start, y_stop, y_apex = (
        np.interp(start, t, y),
        np.interp(stop, t, y),
        np.interp(rt, t, y),
    )
    y_base[valid] = y_start + (y_stop - y_start) * (rt - start) / np.where(
        stop > start, stop - start, 1
    )
    areas[valid] = [peak_area(t, y, a, b) for a, b in zip(start, stop)]
    items = integration_marks(
        start, stop, y_start, y_stop, rt, y_base[valid], y_apex, IPS_COLOR
    )
    return items, areas, y_base


def fit_fill_items(t: np.ndarray, peaks, color, selected=()) -> list:
    """Fitted peaks of one mass, filled between the baseline and the peak, in the
    color of the mass (more opaque for the ``selected`` compounds)."""
    items = []
    for p, part, shape in fitted_shapes(t, peaks, FIT_VISIBLE):
        tt = t[part]
        base = np.full(len(tt), p.fit_baseline)
        alpha = (
            FIT_ALPHA_SELECTED
            if not pd.isna(p.compound_idx) and p.compound_idx in selected
            else FIT_ALPHA
        )
        fill_color = pg.mkColor(color)
        fill_color.setAlpha(alpha)
        lower = pg.PlotCurveItem(tt, base)
        upper = pg.PlotCurveItem(tt, base + shape, pen=pg.mkPen(color, width=1))
        fill = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(fill_color))
        for item in (fill, upper):
            item.setZValue(-2)
            items.append(item)
    return items
