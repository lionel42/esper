"""One compound across all measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from esper.gui.chromatogram_view import ChromatogramPanel
from esper.gui.data import EsperData
from esper.gui.selection import Selection
from esper.gui.style import COLORS, CROSS
from esper.gui.widgets import StepCombo

ALL = "all"
# Initial height of the results plot and of the chromatogram below it
SPLIT_RATIO = (2, 1)

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




class ResultsView(QWidget):
    # Double click on a point
    open_chromatogram = Signal()

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        self.df: pd.DataFrame | None = None
        self._names_loaded = False
        self._dirty = True

        # Toolbar
        self.cmp_box = StepCombo("Cmp:", min_width=220)
        self.runtype_box = StepCombo("Runtype:", editable=False, min_width=90)
        self.x_combo = QComboBox()
        self.x_combo.addItems(X_OPTIONS)
        self.y_combo = QComboBox()
        self.y_combo.addItems(list(Y_OPTIONS))
        self.valid_only = QCheckBox("valid only")
        self.valid_only.setChecked(True)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.cmp_box, 3)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.runtype_box, 1)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Xaxis:"))
        toolbar.addWidget(self.x_combo, 1)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Yaxis:"))
        toolbar.addWidget(self.y_combo, 1)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.valid_only)
        toolbar.addStretch()

        # Plot
        self.plot = pg.PlotWidget()
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plot.getAxis("left").setWidth(75)
        self.highlight = pg.ScatterPlotItem(symbol=CROSS, size=50, pen=pg.mkPen("c", width=2), brush=None)
        self.highlight.setZValue(10)

        # Chromatogram of the selected point
        self.chromatogram = ChromatogramPanel(data, selection, show_info=False, relative=False)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.plot)
        self.splitter.addWidget(self.chromatogram)
        for i, stretch in enumerate(SPLIT_RATIO):
            self.splitter.setStretchFactor(i, stretch)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.splitter, 1)

        # Optional panels, listed in the View menu
        self.chromatogram_action = QAction("Chromatogram", self, checkable=True, checked=True)
        self.chromatogram_action.toggled.connect(self.chromatogram.setVisible)

        self.cmp_box.selected.connect(self.selection.select_cmp_name)
        self.cmp_box.text_entered.connect(self._on_cmp_text)
        self.runtype_box.selected.connect(lambda _: self.redraw())
        self.x_combo.currentIndexChanged.connect(lambda _: self.redraw())
        self.y_combo.currentIndexChanged.connect(lambda _: self.redraw())
        self.valid_only.toggled.connect(self._on_valid_only)
        self.selection.changed.connect(self._on_selection_changed)

    def panel_actions(self) -> list[QAction]:
        return [self.chromatogram_action]

    # --- selection ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._names_loaded:
            self._load_names()
        if self._dirty:
            self.reload()

    def _on_selection_changed(self, fields: set) -> None:
        if "cmp_name" in fields:
            self._dirty = True
            if self.isVisible():
                self.reload()
        elif fields & {"measurement_id", "compound_idx"}:
            self._update_highlight()

    def _load_names(self) -> None:
        self._names_loaded = True
        self.runtype_box.set_items([(ALL, ALL)] + [(t, t) for t in self.data.sample_types()], ALL)
        counts = self.data.compound_names(self.valid_only.isChecked())
        self.cmp_box.set_items([(name, name) for name in counts.index])
        if self.selection.cmp_name is None and not counts.empty:
            self.selection.select_cmp_name(counts.idxmax())
        self.cmp_box.set_current(self.selection.cmp_name, self.selection.cmp_name)

    def _on_valid_only(self) -> None:
        self._load_names()
        self.reload()

    def _on_cmp_text(self, text: str) -> None:
        for i in range(self.cmp_box.combo.count()):
            name = self.cmp_box.combo.itemData(i)
            if text.lower() in name.lower():
                self.selection.select_cmp_name(name)
                return

    # --- drawing --------------------------------------------------------------------

    def reload(self) -> None:
        """Query the database for the selected compound."""
        self._dirty = False
        name = self.selection.cmp_name
        self.cmp_box.set_current(name, name)
        self.df = None if name is None else self.data.results(name, self.valid_only.isChecked())
        self.redraw()

    def _x_values(self, df: pd.DataFrame) -> np.ndarray:
        match self.x_combo.currentText():
            case "Time":
                return (df["acq_datetime"] - pd.Timestamp(0)) / pd.Timedelta(seconds=1)
            case "Measurement":
                return df["measurement_pos"].to_numpy(dtype=float)
            case _:
                return df["rt_mean_s"].to_numpy(dtype=float)

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
        self.plot.setTitle(f"{self.selection.cmp_name or ''} {y_option}", color="k")
        self.plot.addItem(self.highlight)

        df = self.df
        if df is None or df.empty:
            self.highlight.clear()
            return
        runtype = self.runtype_box.combo.currentData()
        if runtype not in (None, ALL):
            df = df[df["sample_type"] == runtype]
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
        df = getattr(self, "plotted", None)
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
