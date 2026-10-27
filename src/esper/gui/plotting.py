"""Helpers to build the plots of the panels consistently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg

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


def add_curve(plot: pg.PlotWidget, t, y, color, z: float = -1) -> pg.PlotDataItem:
    item = plot.plot(t, y, pen=pg.mkPen(color, width=1))
    # Only once in the plot: pyqtgraph fails if set before
    item.setClipToView(True)
    item.setDownsampling(auto=True, method="peak")
    item.setZValue(z)
    return item


def set_x_axis(plot: pg.PlotWidget, time: bool) -> None:
    """Date axis for acquisition times, plain axis otherwise."""
    axis = pg.DateAxisItem(utcOffset=0) if time else pg.AxisItem("bottom")
    plot.setAxisItems({"bottom": axis})


def time_values(df: pd.DataFrame) -> np.ndarray:
    """Acquisition time as seconds since epoch, for a DateAxisItem."""
    return ((df["acq_datetime"] - pd.Timestamp(0)) / pd.Timedelta(seconds=1)).to_numpy(
        dtype=float
    )
