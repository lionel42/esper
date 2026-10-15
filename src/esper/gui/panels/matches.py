"""Library spectra matching the selected compound best, computed live."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt

from esper.gui.data import EsperData
from esper.gui.library_loader import library_loader
from esper.gui.panels.table import Column, TablePanel, ri_diff_color, score_color
from esper.gui.references import current_reference
from esper.gui.selection import Selection
from esper.gui.settings import settings_manager


class BestMatchesPanel(TablePanel):
    """The library spectra matching the selected compound best, computed live.

    Uses the enabled libraries and the matching settings; only computed while
    the panel is visible. The search box searches the spectra of all the
    libraries, also the disabled ones ('library/name'), scored against the compound. Choosing a row
    compares its spectrum with the compound.
    """

    title = "Best matches"
    watches = {"measurement_id", "compound_idx"}
    follows = {"reference"}
    default_sort = ("score", Qt.SortOrder.DescendingOrder)
    search_reloads = True

    columns = [
        Column("name", "name"),
        Column("library", "library"),
        Column(
            "score", "score", "{:.3f}", "Cosine score with the compound (settings: intensity, processing)", score_color
        ),
        Column("matched", "matched", "{:d}", "Number of masses in common"),
        Column("ri", "RI", "{:.1f}", "Retention index of the target with this name, if any"),
        Column("ri_diff", "RI diff", "{:+.1f}", "Retention index of the compound minus the target's", ri_diff_color),
    ]

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.settings = settings_manager()
        self.settings.changed.connect(self.invalidate)
        self.loader = library_loader()
        self.loader.loaded.connect(self.invalidate)
        self._loading = False
        self.filter_edit.setPlaceholderText("Search the spectra of all libraries (library/name)…")

    def load(self) -> pd.DataFrame:
        sel = self.selection
        self._loading = False
        empty = pd.DataFrame(columns=[c.key for c in self.columns])
        if sel.measurement_id is None or sel.compound_idx is None:
            return empty
        search = self.search_text
        self._loading = not self.loader.ensure(all_libraries=bool(search))
        if self._loading:
            return empty
        settings = self.settings.settings
        spectrum = self.data.compound_spectrum(sel.measurement_id, sel.compound_idx, settings.spectrum_intensity)
        matches = self.settings.libraries.best_matches(
            spectrum, None if search else settings.n_best_matches, search
        )
        if not matches:
            return empty
        df = pd.DataFrame([vars(m) for m in matches])
        df["spectrum"] = df["library"] + "/" + df["name"]
        df.index = df["spectrum"]
        targets = self.data.targets
        df["ri"] = [
            targets.at[t, "ri"] if (t := self.data.target_of(name)) is not None else np.nan
            for name in df["name"]
        ]
        compound_ri = self.data.compounds(sel.measurement_id).at[sel.compound_idx, "retention_index"]
        df["ri_diff"] = compound_ri - df["ri"]
        return df

    def _update_count(self) -> None:
        if self._loading:
            self.count_label.setText("loading libraries…")
        else:
            super()._update_count()

    def current_key(self):
        reference = current_reference(self.data, self.selection)
        return None if reference is None else "/".join(reference)

    def select(self, key) -> None:
        library, name = str(key).split("/", 1)
        self.selection.select_reference((library, name))
