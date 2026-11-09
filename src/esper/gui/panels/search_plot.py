"""Peaks of given masses in an rt or retention index window, across all measurements.

Meant for unknowns: there is no compound to follow, only fragment masses.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal

from esper.gui.data import EsperData
from esper.gui.panels.run_scatter import RunScatterPanel
from esper.gui.selection import Selection
from esper.gui.style import COLORS, SYMBOLS

# Label -> (sql window, column of EsperData.search)
WINDOWS = {"rt [s]": ("rt", "rt_s"), "RI": ("ri", "retention_index")}

Y_OPTIONS = {
    "prominence": "prominence",
    "height": "height",
    "rt": "rt_s",
    "retention index": "retention_index",
    "width [s]": "width_s",
}
X_COLUMNS = {"rt": "rt_s", "retention index": "retention_index"}


@dataclass(frozen=True)
class SearchQuery:
    masses: tuple[int, ...]
    window: str  # key of WINDOWS
    lo: float
    hi: float


class SearchPlotPanel(RunScatterPanel):
    """Peaks found for a SearchQuery. Color = mass, symbol = run type.

    Clicking a point selects that peak, a double click asks for its chromatogram.
    """

    title = "Search"
    X_COLUMNS = X_COLUMNS
    Y_OPTIONS = Y_OPTIONS
    HIGHLIGHT_FIELDS = ("measurement_id", "compound_idx", "mass")

    # Description of what is plotted, e.g. '36 peaks in 12 runs'
    summary_changed = Signal(str)

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent, all_masses=False, valid_only=False)
        self.query: SearchQuery | None = None
        self._loaded_query: SearchQuery | None = None
        self._loaded: pd.DataFrame | None = None

    def set_query(self, query: SearchQuery) -> None:
        self.query = query
        self.invalidate()

    def load(self) -> pd.DataFrame | None:
        if self.query is not None and self.query != self._loaded_query:
            self._loaded_query = self.query
            q = self.query
            self._loaded = self.data.search(
                list(q.masses), WINDOWS[q.window][0], q.lo, q.hi
            )
        return self._loaded

    def plot_title(self) -> str | None:
        q = self._loaded_query
        if q is None:
            return None
        return f"m/z {', '.join(map(str, q.masses))}  in {q.window} {q.lo:g} – {q.hi:g}"

    def filtered(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().filtered(df)
        if self.options["valid_only"]:
            df = df[df["valid_candidate"] == 1]
        if self.options["all_masses"]:
            n_masses = df.groupby("measurement_id")["mass"].nunique()
            complete = n_masses.index[n_masses == len(self._loaded_query.masses)]
            df = df[df["measurement_id"].isin(complete)]
        return df

    def draw_points(self, df: pd.DataFrame) -> None:
        q = self._loaded_query
        self.summary_changed.emit(
            f"{len(df)} peaks in {df['measurement_id'].nunique()} runs"
        )
        sample_types = self.data.sample_types()
        symbols = {t: SYMBOLS[i % len(SYMBOLS)] for i, t in enumerate(sample_types)}
        colors = {m: COLORS[i % len(COLORS)] for i, m in enumerate(q.masses)}

        for (mass, stype), sub in df.groupby(["mass", "sample_type"]):
            self.add_points(
                sub,
                ["measurement_id", "compound_idx", "mass", "rt_s", "name", "main_name"],
                tip=lambda x, y, data: (
                    f"{data[4]}\nm/z {data[2]}, rt {data[3]:.1f} s\n{data[5]}\n{y:.4g}"
                ),
                symbol=symbols[stype],
                size=12,
                pen=pg.mkPen(colors[mass], width=1.5),
                hoverSize=18,
            )

        # Legend: one entry per mass (color) and per run type (symbol)
        for mass in q.masses:
            if mass in set(df["mass"]):
                entry = pg.ScatterPlotItem(
                    symbol="s", size=10, pen=None, brush=pg.mkBrush(colors[mass])
                )
                self.legend.addItem(entry, f"m/z {mass}")
        for stype in sample_types:
            if stype in set(df["sample_type"]):
                entry = pg.ScatterPlotItem(
                    symbol=symbols[stype],
                    size=12,
                    pen=pg.mkPen("k", width=1.5),
                    brush=None,
                )
                self.legend.addItem(entry, stype)

    def select_point(self, data: tuple) -> None:
        measurement_id, compound_idx, mass, rt, *_ = data
        self.selection.select_peak(
            int(measurement_id),
            None if pd.isna(compound_idx) else int(compound_idx),
            int(mass),
            float(rt),
        )
