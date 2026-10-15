"""Chromatograms of one measurement."""

from __future__ import annotations

from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit

from esper.gui.data import TIC, EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.panels.chromatogram import ChromatogramPanel, RelativeChromatogramPanel, compound_label
from esper.gui.panels.compound_table import CompoundTablePanel
from esper.gui.panels.matches import BestMatchesPanel
from esper.gui.panels.spectrum import SpectrumPanel
from esper.gui.panels.target_table import TargetTablePanel
from esper.gui.selection import Selection
from esper.gui.views.base import BaseView
from esper.gui.widgets import StepCombo


class ChromatogramToolbar(BasePanel):
    """File, compound, mass and time of the selection.

    A panel so that it is only updated while visible.
    """

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self._cmp_items_mid = None
        self._mass_items_key = None

        self.file_box = StepCombo("File:", min_width=200)
        self.cmp_box = StepCombo("Cmp:", min_width=220)
        self.mass_box = StepCombo("Mass:", min_width=70)
        self.time_edit = QLineEdit()
        self.time_edit.setValidator(QDoubleValidator(0, 1e6, 1))
        self.time_edit.setMaximumWidth(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.file_box, 3)
        layout.addSpacing(12)
        layout.addWidget(self.cmp_box, 3)
        layout.addSpacing(12)
        layout.addWidget(self.mass_box, 1)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Time:"))
        layout.addWidget(self.time_edit)
        layout.addStretch()

        self.file_box.selected.connect(self._on_file_selected)
        self.file_box.text_entered.connect(self._on_file_text)
        self.cmp_box.selected.connect(self._on_cmp_selected)
        self.cmp_box.text_entered.connect(self._on_cmp_text)
        self.mass_box.selected.connect(self.selection.select_mass)
        self.mass_box.text_entered.connect(self._on_mass_text)
        self.time_edit.editingFinished.connect(self._on_time)

    def _on_file_selected(self, measurement_id) -> None:
        if measurement_id is not None:
            self.selection.select_measurement(int(measurement_id))

    def _on_file_text(self, text: str) -> None:
        names = self.data.measurements["name"]
        matching = names[names.str.contains(text, regex=False)]
        if not matching.empty:
            self.selection.select_measurement(int(matching.index[0]))

    def _on_cmp_selected(self, compound_idx) -> None:
        if compound_idx is not None and self.selection.measurement_id is not None:
            self.selection.select_compound(self.selection.measurement_id, int(compound_idx))

    def _on_cmp_text(self, text: str) -> None:
        mid = self.selection.measurement_id
        if mid is None:
            return
        compounds = self.data.compounds(mid)
        names = compounds["compound_name"].dropna()
        matching = names[names.str.lower() == text.lower()]
        if matching.empty:
            matching = names[names.str.contains(text, case=False, regex=False)]
        if not matching.empty:
            self.selection.select_cmp_name(matching.iloc[0])

    def _on_mass_text(self, text: str) -> None:
        text = text.strip().lower()
        if text == "tic":
            self.selection.select_mass(TIC)
        elif text.isdigit():
            self.selection.select_mass(int(text))

    def _on_time(self) -> None:
        text = self.time_edit.text().strip()
        try:
            rt = float(text)
        except ValueError:
            return
        self.selection.select_rt(rt)

    def redraw(self) -> None:
        sel = self.selection
        meas = self.data.measurements

        if self.file_box.combo.count() != len(meas):
            self.file_box.set_items([(name, int(mid)) for mid, name in meas["name"].items()])
        self.file_box.set_current(sel.measurement_id)

        if self._cmp_items_mid != sel.measurement_id:
            self._cmp_items_mid = sel.measurement_id
            items = []
            if sel.measurement_id is not None:
                compounds = self.data.compounds(sel.measurement_id)
                items = [
                    (f"{compound_label(c.compound_name, idx)}  ({c.rt_mean_s:.1f})", int(idx))
                    for idx, c in compounds.iterrows()
                ]
            self.cmp_box.set_items(items)
        self.cmp_box.set_current(sel.compound_idx, sel.cmp_name)

        key = (sel.measurement_id, sel.compound_idx)
        if self._mass_items_key != key:
            self._mass_items_key = key
            if sel.measurement_id is not None and sel.compound_idx is not None:
                peaks = self.data.compound_peaks(*key).sort_values("prominence", ascending=False)
                masses = [int(m) for m in peaks["mass"].drop_duplicates()]
            else:
                masses = list(range(2, 301))
            self.mass_box.set_items([("tic", TIC)] + [(str(m), m) for m in masses])
        self.mass_box.set_current(sel.mass, "tic" if sel.mass is TIC else str(sel.mass))

        self.time_edit.setText("" if sel.rt is None else f"{sel.rt:.1f}")


class ChromatogramsView(BaseView):
    title = "Chromatograms"

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.toolbar = ChromatogramToolbar(data, selection)
        self.add_toolbar(self.toolbar)
        self.chromatogram = self.add_panel(ChromatogramPanel(data, selection), stretch=3)
        self.relative = self.add_panel(RelativeChromatogramPanel(self.chromatogram), stretch=1, optional=True)
        self.compounds = self.add_panel(CompoundTablePanel(data, selection), optional=True, left=True)
        self.targets = self.add_panel(TargetTablePanel(data, selection), optional=True, left=True)
        self.best_matches = self.add_panel(BestMatchesPanel(data, selection), optional=True, left=True)
        self.spectrum = self.add_panel(SpectrumPanel(data, selection, chromatogram=self.chromatogram), stretch=1, optional=True)
