"""Search for unknowns: peaks of given masses in an rt or retention index window."""

from __future__ import annotations

import re

import numpy as np
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton

from esper.gui.data import EsperData
from esper.gui.panels.search_plot import ALL, WINDOWS, X_OPTIONS, Y_OPTIONS, SearchPlotPanel, SearchQuery
from esper.gui.selection import Selection
from esper.gui.views.base import BaseView
from esper.gui.widgets import StepCombo

# Half width of the window filled from the selection
SELECTION_HALF_WINDOW = {"rt": 5.0, "ri": 5.0}
# Number of masses of the selected compound used to fill the masses
SELECTION_N_MASSES = 3


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


class SearchView(BaseView):
    title = "Search"

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)

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
        self.add_toolbar(search_row)

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
        self.add_toolbar(display_row)

        self.results = self.add_panel(SearchPlotPanel(data, selection))

        for edit in (self.lo_edit, self.hi_edit, self.masses_edit):
            edit.returnPressed.connect(self.search)
        search_button.clicked.connect(self.search)
        from_selection.clicked.connect(self.fill_from_selection)
        self.runtype_box.selected.connect(lambda runtype: self.results.set_options(runtype=runtype))
        self.x_combo.currentTextChanged.connect(lambda x: self.results.set_options(x=x))
        self.y_combo.currentTextChanged.connect(lambda y: self.results.set_options(y=y))
        self.all_masses.toggled.connect(lambda checked: self.results.set_options(all_masses=checked))
        self.valid_only.toggled.connect(lambda checked: self.results.set_options(valid_only=checked))
        self.results.summary_changed.connect(self.status.setText)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.runtype_box.combo.count() == 0:
            self.runtype_box.set_items([(ALL, ALL)] + [(t, t) for t in self.data.sample_types()], ALL)

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
        self.results.set_query(SearchQuery(tuple(masses), self.window_combo.currentText(), lo, hi))
