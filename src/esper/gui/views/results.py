"""One compound across all measurements."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.chromatogram import ChromatogramPanel
from esper.gui.panels.results_plot import ResultsPlotPanel
from esper.gui.panels.spectrum import SpectrumPanel
from esper.gui.plot_axes import Axis, ion_element
from esper.gui.selection import Selection
from esper.gui.views.base import BaseView
from esper.gui.widgets import RuntypeBox, StepCombo


class ResultsView(BaseView):
    title = "Results"

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self._names_loaded = False

        self.cmp_box = StepCombo("Cmp:", min_width=220)
        self.runtype_box = RuntypeBox()
        self.valid_only = QCheckBox("valid only")
        self.valid_only.setChecked(True)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.cmp_box, 3)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.runtype_box, 1)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.valid_only)
        toolbar.addStretch()
        self.add_toolbar(toolbar)

        # What each plots: chosen above it (the compound, run type and validity: here)
        self.results = self.add_panel(ResultsPlotPanel(data, selection), stretch=2)
        main_area = Axis(ion_element(1), ion_element(1).variable("fit area"))
        self.results2 = self.add_panel(
            ResultsPlotPanel(data, selection, title="Results 2", y=main_area),
            stretch=2,
            optional=True,
            visible=False,
        )
        self.chromatogram = self.add_panel(
            ChromatogramPanel(data, selection, show_info=False),
            stretch=1,
            optional=True,
        )
        self.spectrum = self.add_panel(
            SpectrumPanel(data, selection, chromatogram=self.chromatogram),
            stretch=1,
            optional=True,
        )

        self.cmp_box.selected.connect(self.selection.select_candidate)
        self.cmp_box.text_entered.connect(self._on_cmp_text)
        self.runtype_box.selected.connect(
            lambda runtype: self._set_options(runtype=runtype)
        )
        self.valid_only.toggled.connect(self._on_valid_only)
        self.selection.changed.connect(self._on_selection_changed)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._names_loaded:
            self._load_names()

    def _on_selection_changed(self, fields: set) -> None:
        if "identification" in fields:
            # A compound may have a new name
            self._names_loaded = False
            if self.isVisible():
                self._load_names()
        if fields & {"candidate", "target"}:
            self.cmp_box.set_current(self.selection.candidate, self.selection.label)

    def _load_names(self) -> None:
        """Candidate names for the Cmp box; selects the most frequent one if nothing is selected."""
        self._names_loaded = True
        # Keeping the runtype the plot is filtered by
        self.runtype_box.fill(self.data.sample_types(), self.results.options["runtype"])
        counts = self.data.candidate_names(self.valid_only.isChecked())
        self.cmp_box.set_items([(name, name) for name in counts.index])
        if self.selection.label is None and not counts.empty:
            self.selection.select_candidate(counts.idxmax())
        self.cmp_box.set_current(self.selection.candidate, self.selection.label)

    def _set_options(self, **options) -> None:
        """Options of both results plots."""
        for panel in (self.results, self.results2):
            panel.set_options(**options)

    def _on_valid_only(self, valid_only: bool) -> None:
        self._load_names()
        self._set_options(valid_only=valid_only)

    def _on_cmp_text(self, text: str) -> None:
        for i in range(self.cmp_box.combo.count()):
            name = self.cmp_box.combo.itemData(i)
            if text.lower() in name.lower():
                self.selection.select_candidate(name)
                return
