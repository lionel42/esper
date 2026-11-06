"""Mass calibration of all the measurements over time, and the calibrants of the selected run."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from esper.gui.data import EsperData
from esper.gui.panels.mass_cal_plot import (
    ALL,
    UNITS,
    X_OPTIONS,
    MassCalPlotPanel,
    MassCalResidualsPanel,
    MassCalRunPanel,
)
from esper.gui.selection import Selection
from esper.gui.views.base import BaseView
from esper.gui.widgets import StepCombo


class MassCalView(BaseView):
    title = "Mass calibration"

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self._loaded = False

        self.runtype_box = StepCombo("Runtype:", editable=False, min_width=90)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(list(UNITS))
        self.unit_combo.setToolTip(
            "Drift and residuals in ppm (m/z error) or in tof index (shift of the peaks)"
        )
        self.x_combo = QComboBox()
        self.x_combo.addItems(X_OPTIONS)
        self.status = QLabel()

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.runtype_box, 1)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Drift in:"))
        toolbar.addWidget(self.unit_combo)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Xaxis:"))
        toolbar.addWidget(self.x_combo)
        toolbar.addStretch()
        toolbar.addWidget(self.status)
        self.add_toolbar(toolbar)

        self.over_time = self.add_panel(MassCalPlotPanel(data, selection), stretch=2)
        self.run = self.add_panel(
            MassCalRunPanel(data, selection), stretch=1, optional=True
        )
        self.residuals = self.add_panel(
            MassCalResidualsPanel(data, selection),
            stretch=1,
            optional=True,
            visible=False,
        )
        self.panels = (self.over_time, self.run, self.residuals)

        self.runtype_box.selected.connect(lambda r: self._set_options(runtype=r))
        self.unit_combo.currentTextChanged.connect(
            lambda u: self._set_options(unit=UNITS[u])
        )
        self.x_combo.currentTextChanged.connect(lambda x: self._set_options(x=x))

    def _set_options(self, **options) -> None:
        for panel in self.panels:
            panel.set_options(**options)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            # The parameters and the drift over time get most of the height
            height = self.splitter.height()
            self.splitter.setSizes([height * 2 // 3, height // 3, height // 3])
            self.runtype_box.set_items(
                [(ALL, ALL)] + [(t, t) for t in self.data.sample_types()], ALL
            )
            points = self.data.mass_cal_points
            n_runs = points["measurement_id"].nunique()
            missing = len(self.data.measurements) - n_runs
            text = f"{n_runs} runs, {len(points)} points"
            if missing:
                text += f" ({missing} runs without calibration)"
            if points["acq_start"].isna().any():
                text += "; times to the minute for some runs: run esper upload again"
            self.status.setText(text)
