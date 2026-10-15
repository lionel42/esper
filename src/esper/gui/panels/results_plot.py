"""One compound across all measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.plotting import make_highlight, make_plot, set_x_axis, time_values
from esper.gui.selection import Selection
from esper.gui.style import COLORS, CROSS

ALL = "all"

# Label -> column of EsperData.results
Y_OPTIONS = {
    "prominence sum": "prominence_sum",
    "main ion height": "max_height",
    "main ion prominence": "max_prominence",
    "rt": "rt_mean_s",
    "retention index": "retention_index",
    "score": "best_score",
    "score combined": "best_score_combined",
    "n peaks": "n_peaks",
}
X_OPTIONS = ["Time", "Measurement", "rt"]


class ResultsPlotPanel(BasePanel):
    """The selected compound (by name) in every measurement, one point per occurrence.

    Clicking a point selects that measurement and compound, a double click
    asks for its chromatogram.
    """

    title = "Results"

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.options = {"runtype": ALL, "x": X_OPTIONS[0], "y": next(iter(Y_OPTIONS)), "valid_only": True}
        self.df: pd.DataFrame | None = None
        self.plotted: pd.DataFrame | None = None
        self._loaded_key = None

        self.plot = make_plot()
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.highlight = make_highlight()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def set_options(self, **options) -> None:
        """runtype, x, y (labels of X_OPTIONS / Y_OPTIONS), valid_only."""
        self.options.update(options)
        self.invalidate()

    def on_selection_changed(self, fields: set) -> None:
        if "cmp_name" in fields:
            self.invalidate()
        elif fields & {"measurement_id", "compound_idx"}:
            self._update_highlight()

    def redraw(self) -> None:
        # Query the database only when the compound or the validity filter changed
        name = self.selection.cmp_name
        key = (name, self.options["valid_only"])
        if key != self._loaded_key:
            self._loaded_key = key
            self.df = None if name is None else self.data.results(name, self.options["valid_only"])
        self._plot()

    def _x_values(self, df: pd.DataFrame) -> np.ndarray:
        match self.options["x"]:
            case "Time":
                return time_values(df)
            case "Measurement":
                return df["measurement_pos"].to_numpy(dtype=float)
            case _:
                return df["rt_mean_s"].to_numpy(dtype=float)

    def _plot(self) -> None:
        self.plot.clear()
        self.legend.clear()
        x_option, y_option = self.options["x"], self.options["y"]
        set_x_axis(self.plot, x_option, time=x_option == "Time")
        self.plot.getAxis("left").setLabel(y_option)
        self.plot.setTitle(f"{self.selection.cmp_name or ''} {y_option}", color="k")
        self.plot.addItem(self.highlight)
        self.plotted = None

        df = self.df
        if df is None or df.empty:
            self.highlight.clear()
            return
        if self.options["runtype"] != ALL:
            df = df[df["sample_type"] == self.options["runtype"]]
        df = df.assign(x=self._x_values(df), y=df[Y_OPTIONS[y_option]].astype(float))
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
        self.plotted = df

        for i, stype in enumerate(self.data.sample_types()):
            sub = df[df["sample_type"] == stype]
            if sub.empty:
                continue
            color = COLORS[i % len(COLORS)]
            item = pg.ScatterPlotItem(
                sub["x"].to_numpy(), sub["y"].to_numpy(),
                data=list(zip(sub["measurement_id"], sub["compound_idx"], sub["name"], sub["rt_mean_s"])),
                symbol=CROSS, size=14, pen=pg.mkPen(color, width=2), brush=None,
                hoverable=True, hoverPen=pg.mkPen("c", width=3), hoverSize=20,
                tip=lambda x, y, data: f"{data[2]}\nrt {data[3]:.1f} s\n{y:.4g}",
                name=stype,
            )
            item.sigClicked.connect(self._on_clicked)
            self.plot.addItem(item)
        self._update_highlight()
        self.plot.autoRange()

    def _update_highlight(self) -> None:
        df = self.plotted
        sel = self.selection
        if df is None or df.empty:
            self.highlight.clear()
            return
        row = df[(df["measurement_id"] == sel.measurement_id) & (df["compound_idx"] == sel.compound_idx)]
        if row.empty:
            self.highlight.clear()
        else:
            self.highlight.setData(row["x"].to_numpy(), row["y"].to_numpy())

    def _on_clicked(self, item, points, ev=None) -> None:
        if not len(points):
            return
        measurement_id, compound_idx, *_ = points[0].data()
        self.selection.select_compound(int(measurement_id), int(compound_idx))
        if ev is not None and ev.double():
            self.open_chromatogram.emit()
