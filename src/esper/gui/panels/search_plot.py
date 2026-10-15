"""Peaks of given masses in an rt or retention index window, across all measurements.

Meant for unknowns: there is no compound to follow, only fragment masses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.plotting import make_highlight, make_plot, set_x_axis, time_values
from esper.gui.selection import Selection
from esper.gui.style import COLORS, SYMBOLS

ALL = "all"

# Label -> (sql window, column of EsperData.search)
WINDOWS = {"rt [s]": ("rt", "rt_s"), "RI": ("ri", "retention_index")}

Y_OPTIONS = {
    "prominence": "prominence",
    "height": "height",
    "rt": "rt_s",
    "retention index": "retention_index",
    "width [s]": "width_s",
}
X_OPTIONS = ["Time", "Measurement", "rt", "retention index"]


@dataclass(frozen=True)
class SearchQuery:
    masses: tuple[int, ...]
    window: str  # key of WINDOWS
    lo: float
    hi: float


class SearchPlotPanel(BasePanel):
    """Peaks found for a SearchQuery. Color = mass, symbol = run type.

    Clicking a point selects that peak, a double click asks for its chromatogram.
    """

    title = "Search"
    # Not redrawn on selection changes, only the highlight moves
    watches: set[str] = set()

    # Description of what is plotted, e.g. '36 peaks in 12 runs'
    summary_changed = Signal(str)

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.query: SearchQuery | None = None
        self.options = {
            "runtype": ALL,
            "x": X_OPTIONS[0],
            "y": next(iter(Y_OPTIONS)),
            "all_masses": False,
            "valid_only": False,
        }
        self.df: pd.DataFrame | None = None
        self.plotted: pd.DataFrame | None = None
        self._loaded_query: SearchQuery | None = None

        self.plot = make_plot()
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.highlight = make_highlight()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def set_query(self, query: SearchQuery) -> None:
        self.query = query
        self.invalidate()

    def set_options(self, **options) -> None:
        """runtype, x, y (labels of X_OPTIONS / Y_OPTIONS), all_masses, valid_only."""
        self.options.update(options)
        self.invalidate()

    def on_selection_changed(self, fields: set) -> None:
        if fields & {"measurement_id", "compound_idx", "mass"}:
            self._update_highlight()

    def redraw(self) -> None:
        if self.query is not None and self.query != self._loaded_query:
            self._loaded_query = self.query
            q = self.query
            self.df = self.data.search(list(q.masses), WINDOWS[q.window][0], q.lo, q.hi)
        self._plot()

    def _x_values(self, df: pd.DataFrame) -> np.ndarray:
        match self.options["x"]:
            case "Time":
                return time_values(df)
            case "Measurement":
                return df["measurement_pos"].to_numpy(dtype=float)
            case "rt":
                return df["rt_s"].to_numpy(dtype=float)
            case _:
                return df["retention_index"].to_numpy(dtype=float)

    def _filtered(self) -> pd.DataFrame:
        df = self.df
        if self.options["runtype"] != ALL:
            df = df[df["sample_type"] == self.options["runtype"]]
        if self.options["valid_only"]:
            df = df[df["valid_candidate"] == 1]
        if self.options["all_masses"]:
            n_masses = df.groupby("measurement_id")["mass"].nunique()
            complete = n_masses.index[n_masses == len(self.query.masses)]
            df = df[df["measurement_id"].isin(complete)]
        return df

    def _plot(self) -> None:
        self.plot.clear()
        self.legend.clear()
        x_option, y_option = self.options["x"], self.options["y"]
        set_x_axis(self.plot, x_option, time=x_option == "Time")
        self.plot.getAxis("left").setLabel(y_option)
        self.plot.addItem(self.highlight)
        self.plotted = None

        if self.df is None:
            self.highlight.clear()
            return
        q = self.query
        self.plot.setTitle(
            f"m/z {', '.join(map(str, q.masses))}  in {q.window} {q.lo:g} – {q.hi:g}", color="k"
        )
        df = self._filtered()
        df = df.assign(x=self._x_values(df), y=df[Y_OPTIONS[y_option]].astype(float))
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
        self.plotted = df
        self.summary_changed.emit(f"{len(df)} peaks in {df['measurement_id'].nunique()} runs")

        sample_types = self.data.sample_types()
        symbols = {t: SYMBOLS[i % len(SYMBOLS)] for i, t in enumerate(sample_types)}
        colors = {m: COLORS[i % len(COLORS)] for i, m in enumerate(q.masses)}

        for (mass, stype), sub in df.groupby(["mass", "sample_type"]):
            item = pg.ScatterPlotItem(
                sub["x"].to_numpy(), sub["y"].to_numpy(),
                data=list(zip(
                    sub["measurement_id"], sub["compound_idx"], sub["mass"], sub["rt_s"],
                    sub["name"], sub["compound_name"],
                )),
                symbol=symbols[stype], size=12, pen=pg.mkPen(colors[mass], width=1.5), brush=None,
                hoverable=True, hoverPen=pg.mkPen("c", width=3), hoverSize=18,
                tip=lambda x, y, data: f"{data[4]}\nm/z {data[2]}, rt {data[3]:.1f} s\n{data[5]}\n{y:.4g}",
            )
            item.sigClicked.connect(self._on_clicked)
            self.plot.addItem(item)

        # Legend: one entry per mass (color) and per run type (symbol)
        for mass in q.masses:
            if mass in set(df["mass"]):
                entry = pg.ScatterPlotItem(symbol="s", size=10, pen=None, brush=pg.mkBrush(colors[mass]))
                self.legend.addItem(entry, f"m/z {mass}")
        for stype in sample_types:
            if stype in set(df["sample_type"]):
                entry = pg.ScatterPlotItem(symbol=symbols[stype], size=12, pen=pg.mkPen("k", width=1.5), brush=None)
                self.legend.addItem(entry, stype)

        self._update_highlight()
        self.plot.autoRange()

    def _update_highlight(self) -> None:
        df = self.plotted
        sel = self.selection
        if df is None or df.empty:
            self.highlight.clear()
            return
        row = df[
            (df["measurement_id"] == sel.measurement_id)
            & (df["compound_idx"] == sel.compound_idx)
            & (df["mass"] == sel.mass)
        ]
        if row.empty:
            self.highlight.clear()
        else:
            self.highlight.setData(row["x"].to_numpy(), row["y"].to_numpy())

    def _on_clicked(self, item, points, ev=None) -> None:
        if not len(points):
            return
        measurement_id, compound_idx, mass, rt, *_ = points[0].data()
        self.selection.select_peak(int(measurement_id), int(compound_idx), int(mass), float(rt))
        if ev is not None and ev.double():
            self.open_chromatogram.emit()
