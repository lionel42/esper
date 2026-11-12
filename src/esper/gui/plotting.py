"""Helpers to build the plots of the panels consistently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg

from esper.gui.settings import CurveStyle
from esper.gui.style import CROSS

# Fixed width so that stacked plots stay aligned
LEFT_AXIS_WIDTH = 75


def make_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
    for axis in ("bottom", "left"):
        plot.getAxis(axis).enableAutoSIPrefix(False)
    return plot


def make_highlight() -> pg.ScatterPlotItem:
    """Large cyan cross marking the selected point."""
    item = pg.ScatterPlotItem(
        symbol=CROSS, size=50, pen=pg.mkPen("c", width=2), brush=None
    )
    item.setZValue(10)
    return item


# Size of the symbols of a curve drawn as points or crosses [px]
POINT_SIZE = 3
CROSS_SIZE = 6


def curve_style(color, width: float = 1, style=CurveStyle.LINES) -> dict:
    """The options of a PlotDataItem drawn as ``style``: lines, or each value as a point
    or a cross."""
    if style == CurveStyle.POINTS:
        return dict(
            pen=None,
            symbol="o",
            symbolSize=POINT_SIZE,
            symbolPen=None,
            symbolBrush=pg.mkBrush(color),
        )
    if style == CurveStyle.CROSSES:
        return dict(
            pen=None,
            symbol="x",
            symbolSize=CROSS_SIZE,
            symbolPen=pg.mkPen(color, width=width),
            symbolBrush=None,
        )
    return dict(pen=pg.mkPen(color, width=width), symbol=None)


def add_curve(
    plot: pg.PlotWidget,
    t,
    y,
    color,
    z: float = -1,
    width: float = 1,
    style: CurveStyle = CurveStyle.LINES,
) -> pg.PlotDataItem:
    """A curve (``style``: see ``curve_style``)."""
    # connect "finite": NaN breaks the curve (gaps of chromatogram_store.with_breaks)
    item = plot.plot(t, y, connect="finite", **curve_style(color, width, style))
    # Only once in the plot: pyqtgraph fails if set before
    item.setClipToView(True)
    item.setDownsampling(auto=True, method="peak")
    item.setZValue(z)
    return item


def set_x_axis(plot: pg.PlotWidget, time: bool) -> None:
    """Date x axis for acquisition times, plain axis otherwise."""
    set_time_axis(plot, "bottom", time)


def set_time_axis(plot: pg.PlotWidget, side: str, time: bool) -> None:
    """Date axis (``side``: bottom, left) for times, plain axis otherwise.

    Replaced only when the kind changes: a new axis shows 0-1 until the view range changes.
    """
    if isinstance(plot.getAxis(side), pg.DateAxisItem) == time:
        return
    axis = pg.DateAxisItem(orientation=side, utcOffset=0) if time else pg.AxisItem(side)
    plot.setAxisItems({side: axis})


def time_values(df: pd.DataFrame, column: str = "acq_datetime") -> np.ndarray:
    """A time column (default: the acquisition time) as seconds since epoch, for a
    DateAxisItem."""
    return ((df[column] - pd.Timestamp(0)) / pd.Timedelta(seconds=1)).to_numpy(
        dtype=float
    )
