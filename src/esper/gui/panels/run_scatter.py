"""Base of the plots of points across all the measurements (results, search)."""

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.plotting import (
    make_highlight,
    make_plot,
    set_time_axis,
    time_values,
)
from esper.gui.selection import Selection
from esper.gui.widgets import ALL


class RunScatterPanel(BasePanel):
    """Points of all the measurements (compounds or peaks), one plot item per group.

    The x axis is the acquisition time, the position of the measurement or a column
    (``X_COLUMNS``), the y axis a column (``Y_OPTIONS``), filtered by run type; a
    subclass can choose its axes otherwise (``axis_values``, ``is_time``,
    ``axis_label``).
    Subclasses load the data (``load``), title the plot (``plot_title``), draw the points (``draw_points``, with
    ``add_points``) and select a clicked point (``select_point``); a double click
    asks for its chromatogram.

    The point of the selection is highlighted: the row whose ``HIGHLIGHT_FIELDS``
    columns equal the selection fields of the same name. A change of ``watches``
    redraws the plot, of the ``HIGHLIGHT_FIELDS`` only moves the highlight.
    """

    # x option -> column, after "Time" and "Measurement"
    X_COLUMNS: dict[str, str] = {}
    # y option -> column
    Y_OPTIONS: dict[str, str] = {}
    HIGHLIGHT_FIELDS: tuple[str, ...] = ("measurement_id", "compound_idx")
    watches: set[str] = set()

    def __init__(
        self, data: EsperData, selection: Selection, parent=None, **options
    ) -> None:
        super().__init__(data, selection, parent)
        self.options = {
            "runtype": ALL,
            "x": self.x_options()[0],
            "y": next(iter(self.Y_OPTIONS), None),
            **options,
        }
        self.plotted: pd.DataFrame | None = None
        # The points with an x value, whatever their y (e.g. for other y axes)
        self.with_x: pd.DataFrame | None = None

        self.plot = make_plot()
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.highlight = make_highlight()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    @classmethod
    def x_options(cls) -> list[str]:
        return ["Time", "Measurement", *cls.X_COLUMNS]

    def set_options(self, **options) -> None:
        self.options.update(options)
        self.invalidate()

    def on_selection_changed(self, fields: set) -> None:
        if fields & self.watches:
            self.invalidate()
        elif fields & set(self.HIGHLIGHT_FIELDS):
            self._update_highlight()

    # --- to implement ---------------------------------------------------------------

    def load(self) -> pd.DataFrame | None:
        """The points (None: nothing to plot), with the ``Y_OPTIONS`` and ``X_COLUMNS``
        columns and those of ``EsperData.measurements`` (sample_type, ...)."""
        raise NotImplementedError

    def plot_title(self) -> str | None:
        """Title of the plot (None: left as it is)."""
        return None

    def filtered(self, df: pd.DataFrame) -> pd.DataFrame:
        """The points shown: those of the run type."""
        if self.options["runtype"] != ALL:
            df = df[df["sample_type"] == self.options["runtype"]]
        return df

    def draw_points(self, df: pd.DataFrame) -> None:
        """Draw the points (``x``, ``y`` columns), e.g. by group with ``add_points``."""
        raise NotImplementedError

    def axis_values(self, df: pd.DataFrame, which: str) -> np.ndarray:
        """The values of the points on an axis (``which``: x, y; a time: seconds since
        epoch). The x option: "Time", "Measurement" or of ``X_COLUMNS``; y: of
        ``Y_OPTIONS``."""
        if which == "y":
            return df[self.Y_OPTIONS[self.options["y"]]].to_numpy(dtype=float)
        match self.options["x"]:
            case "Time":
                return time_values(df)
            case "Measurement":
                return df["measurement_pos"].to_numpy(dtype=float)
            case x:
                return df[self.X_COLUMNS[x]].to_numpy(dtype=float)

    def is_time(self, which: str) -> bool:
        """Whether an axis shows times (a date axis)."""
        return which == "x" and self.options["x"] == "Time"

    def axis_label(self, which: str) -> str | None:
        """The label of an axis (None: none)."""
        return None

    def select_point(self, data: tuple) -> None:
        """Select the point clicked (its ``data`` in ``add_points``)."""
        raise NotImplementedError

    # --- drawing --------------------------------------------------------------------

    def redraw(self) -> None:
        df = self.load()
        self.plot.clear()
        self.legend.clear()
        for which, side in (("x", "bottom"), ("y", "left")):
            set_time_axis(self.plot, side, self.is_time(which))
            label = self.axis_label(which)
            if label is None:
                self.plot.showLabel(side, False)
            else:
                self.plot.setLabel(side, label)
        title = self.plot_title()
        if title is not None:
            self.plot.setTitle(title, color="k")
        self.plot.addItem(self.highlight)
        self.plotted = self.with_x = None
        if df is None:
            self.highlight.clear()
            return
        df = self.filtered(df)
        df = df.assign(x=self.axis_values(df, "x"), y=self.axis_values(df, "y"))
        self.with_x = df[np.isfinite(df["x"])]
        df = self.with_x[np.isfinite(self.with_x["y"])]
        self.plotted = df
        self.draw_points(df)
        self._update_highlight()
        self.plot.autoRange()

    def add_points(
        self,
        df: pd.DataFrame,
        data_columns: list[str],
        tip,
        view: pg.ViewBox | None = None,
        **options,
    ) -> pg.ScatterPlotItem:
        """A clickable scatter item of the points of ``df``, with the values of
        ``data_columns`` as the data of each point (given to ``tip`` and ``select_point``),
        in ``view`` (default: the plot's)."""
        item = pg.ScatterPlotItem(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            data=list(zip(*(df[c] for c in data_columns))),
            brush=None,
            hoverable=True,
            hoverPen=pg.mkPen("c", width=3),
            tip=tip,
            **options,
        )
        item.sigClicked.connect(self._on_clicked)
        (view or self.plot).addItem(item)
        return item

    def _update_highlight(self) -> None:
        df = self.plotted
        if df is None or df.empty:
            self.highlight.clear()
            return
        sel = self.selection
        row = df[
            reduce(
                lambda a, b: a & b,
                (df[f] == getattr(sel, f) for f in self.HIGHLIGHT_FIELDS),
            )
        ]
        if row.empty:
            self.highlight.clear()
        else:
            self.highlight.setData(row["x"].to_numpy(), row["y"].to_numpy())

    def _on_clicked(self, item, points, ev=None) -> None:
        if not len(points):
            return
        self.select_point(points[0].data())
        if ev is not None and ev.double():
            self.open_chromatogram.emit()
