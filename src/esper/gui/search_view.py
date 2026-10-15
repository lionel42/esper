"""Peaks of given masses in an rt or retention index window, across all measurements.

Meant for unknowns: there is no compound to follow, only fragment masses.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from esper.gui.data import EsperData
from esper.gui.selection import Selection
from esper.gui.style import COLORS, CROSS, SYMBOLS
from esper.gui.widgets import StepCombo

ALL = "all"

# Label -> (sql window, column of EsperData.search)
WINDOWS = {"rt [s]": ("rt", "rt_s"), "RI": ("ri", "retention_index")}

# Half width of the window filled from the selection
SELECTION_HALF_WINDOW = {"rt": 5.0, "ri": 5.0}
# Number of masses of the selected compound used to fill the masses
SELECTION_N_MASSES = 3

Y_OPTIONS = {
    "prominence": "prominence",
    "height": "height",
    "rt": "rt_s",
    "retention index": "retention_index",
    "width [s]": "width_s",
}
X_OPTIONS = ["Time", "Measurement", "rt", "retention index"]


def parse_masses(text: str) -> list[int]:
    """'85, 87 101-103' -> [85, 87, 101, 102, 103]"""
    masses = set()
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            masses.update(range(int(lo), int(hi) + 1))
        else:
            masses.add(int(token))
    return sorted(masses)


class SearchView(QWidget):
    # Double click on a point
    open_chromatogram = Signal()

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        self.df: pd.DataFrame | None = None
        self.plotted: pd.DataFrame | None = None
        self.masses: list[int] = []

        # Search row
        self.window_combo = QComboBox()
        self.window_combo.addItems(list(WINDOWS))
        self.lo_edit = QLineEdit()
        self.hi_edit = QLineEdit()
        for edit in (self.lo_edit, self.hi_edit):
            edit.setValidator(QDoubleValidator())
            edit.setMaximumWidth(90)
        self.masses_edit = QLineEdit()
        self.masses_edit.setPlaceholderText("e.g. 85, 87 101-103")
        search_button = QPushButton("Search")
        from_selection = QPushButton("From selection")
        from_selection.setToolTip(
            f"Window around the selected compound and its {SELECTION_N_MASSES} most prominent masses"
        )
        self.status = QLabel()

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Window:"))
        search_row.addWidget(self.window_combo)
        search_row.addWidget(QLabel("from"))
        search_row.addWidget(self.lo_edit)
        search_row.addWidget(QLabel("to"))
        search_row.addWidget(self.hi_edit)
        search_row.addSpacing(12)
        search_row.addWidget(QLabel("Masses:"))
        search_row.addWidget(self.masses_edit, 2)
        search_row.addWidget(search_button)
        search_row.addWidget(from_selection)
        search_row.addSpacing(12)
        search_row.addWidget(self.status, 1)

        # Display row
        self.runtype_box = StepCombo("Runtype:", editable=False, min_width=90)
        self.x_combo = QComboBox()
        self.x_combo.addItems(X_OPTIONS)
        self.y_combo = QComboBox()
        self.y_combo.addItems(list(Y_OPTIONS))
        self.all_masses = QCheckBox("only runs with all masses")
        self.all_masses.setToolTip("Hide the runs where one of the masses has no peak in the window")
        self.valid_only = QCheckBox("valid compounds only")

        display_row = QHBoxLayout()
        display_row.addWidget(self.runtype_box)
        display_row.addSpacing(12)
        display_row.addWidget(QLabel("Xaxis:"))
        display_row.addWidget(self.x_combo)
        display_row.addSpacing(12)
        display_row.addWidget(QLabel("Yaxis:"))
        display_row.addWidget(self.y_combo)
        display_row.addSpacing(12)
        display_row.addWidget(self.all_masses)
        display_row.addWidget(self.valid_only)
        display_row.addStretch()

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.getAxis("left").setWidth(75)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.highlight = pg.ScatterPlotItem(symbol=CROSS, size=50, pen=pg.mkPen("c", width=2), brush=None)
        self.highlight.setZValue(10)

        layout = QVBoxLayout(self)
        layout.addLayout(search_row)
        layout.addLayout(display_row)
        layout.addWidget(self.plot, 1)

        for edit in (self.lo_edit, self.hi_edit, self.masses_edit):
            edit.returnPressed.connect(self.search)
        search_button.clicked.connect(self.search)
        from_selection.clicked.connect(self.fill_from_selection)
        self.runtype_box.selected.connect(lambda _: self.redraw())
        self.x_combo.currentIndexChanged.connect(lambda _: self.redraw())
        self.y_combo.currentIndexChanged.connect(lambda _: self.redraw())
        self.all_masses.toggled.connect(lambda _: self.redraw())
        self.valid_only.toggled.connect(lambda _: self.redraw())
        self.selection.changed.connect(self._on_selection_changed)

    def panel_actions(self) -> list:
        return []

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.runtype_box.combo.count() == 0:
            self.runtype_box.set_items([(ALL, ALL)] + [(t, t) for t in self.data.sample_types()], ALL)

    def _on_selection_changed(self, fields: set) -> None:
        if fields & {"measurement_id", "compound_idx", "mass"}:
            self._update_highlight()

    # --- search ---------------------------------------------------------------------

    def fill_from_selection(self) -> None:
        sel = self.selection
        if sel.measurement_id is None or sel.compound_idx is None:
            self.status.setText("No compound selected")
            return
        window, _ = WINDOWS[self.window_combo.currentText()]
        compound = self.data.compounds(sel.measurement_id).loc[sel.compound_idx]
        centre = compound["rt_mean_s"] if window == "rt" else compound["retention_index"]
        if not np.isfinite(centre):
            self.status.setText("The selected compound has no retention index")
            return
        half = SELECTION_HALF_WINDOW[window]
        self.lo_edit.setText(f"{centre - half:.1f}")
        self.hi_edit.setText(f"{centre + half:.1f}")

        peaks = self.data.compound_peaks(sel.measurement_id, sel.compound_idx)
        masses = peaks.sort_values("prominence", ascending=False)["mass"].drop_duplicates()
        self.masses_edit.setText(", ".join(str(m) for m in masses.head(SELECTION_N_MASSES)))
        self.search()

    def search(self) -> None:
        try:
            masses = parse_masses(self.masses_edit.text())
            lo, hi = sorted([float(self.lo_edit.text()), float(self.hi_edit.text())])
        except ValueError:
            self.status.setText("Give a window and masses")
            return
        if not masses:
            self.status.setText("Give at least one mass")
            return
        window, _ = WINDOWS[self.window_combo.currentText()]
        self.masses = masses
        self.df = self.data.search(masses, window, lo, hi)
        self.redraw()

    # --- drawing --------------------------------------------------------------------

    def _x_values(self, df: pd.DataFrame) -> np.ndarray:
        match self.x_combo.currentText():
            case "Time":
                return (df["acq_datetime"] - pd.Timestamp(0)) / pd.Timedelta(seconds=1)
            case "Measurement":
                return df["measurement_pos"].to_numpy(dtype=float)
            case "rt":
                return df["rt_s"].to_numpy(dtype=float)
            case _:
                return df["retention_index"].to_numpy(dtype=float)

    def _filtered(self) -> pd.DataFrame:
        df = self.df
        runtype = self.runtype_box.combo.currentData()
        if runtype not in (None, ALL):
            df = df[df["sample_type"] == runtype]
        if self.valid_only.isChecked():
            df = df[df["valid_candidate"] == 1]
        if self.all_masses.isChecked():
            n_masses = df.groupby("measurement_id")["mass"].nunique()
            complete = n_masses.index[n_masses == len(self.masses)]
            df = df[df["measurement_id"].isin(complete)]
        return df

    def redraw(self) -> None:
        self.plot.clear()
        self.legend.clear()
        x_option, y_option = self.x_combo.currentText(), self.y_combo.currentText()
        if x_option == "Time":
            self.plot.setAxisItems({"bottom": pg.DateAxisItem(utcOffset=0)})
        else:
            self.plot.setAxisItems({"bottom": pg.AxisItem("bottom")})
        self.plot.getAxis("bottom").setLabel(x_option)
        self.plot.getAxis("left").setLabel(y_option)
        self.plot.addItem(self.highlight)
        self.plotted = None

        if self.df is None:
            self.highlight.clear()
            return
        self.plot.setTitle(
            f"m/z {', '.join(map(str, self.masses))}  in {self.window_combo.currentText()} "
            f"{self.lo_edit.text()} – {self.hi_edit.text()}",
            color="k",
        )
        df = self._filtered()
        df = df.assign(x=self._x_values(df), y=df[Y_OPTIONS[y_option]].astype(float))
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
        self.plotted = df
        self.status.setText(f"{len(df)} peaks in {df['measurement_id'].nunique()} runs")

        sample_types = self.data.sample_types()
        symbols = {t: SYMBOLS[i % len(SYMBOLS)] for i, t in enumerate(sample_types)}
        colors = {m: COLORS[i % len(COLORS)] for i, m in enumerate(self.masses)}

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
        for mass in self.masses:
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
