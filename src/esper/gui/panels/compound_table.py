"""Table of the compounds extracted by alpinac in the selected measurement."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox

from esper.gui.data import EsperData
from esper.gui.panels.chromatogram import compound_label
from esper.gui.panels.table import Column, TablePanel, bool_color, score_color
from esper.gui.selection import Selection


class CompoundTablePanel(TablePanel):
    """Compounds of the selected measurement. Choosing a row selects the compound."""

    title = "Compounds"
    watches = {"measurement_id"}
    follows = {"compound_idx"}
    default_sort = ("compound_idx", Qt.SortOrder.AscendingOrder)

    columns = [
        Column("compound_idx", "#", "{:d}", "Compound index in pipeline.nc"),
        Column("rt_mean_s", "rt", "{:.1f}", "Mean retention time of the peaks [s]"),
        Column("retention_index", "RI", "{:.1f}", "Retention index"),
        Column("name", "name"),
        Column("best_score", "score", "{:.3f}", "Score of the best library match", score_color),
        Column("best_score_combined", "comb.", "{:.3f}", "Score combined with the RI difference", score_color),
        Column("valid_candidate", "valid", lambda v: "✓" if v else "", "Valid candidate", bool_color),
        Column("n_peaks", "peaks", "{:d}", "Number of peaks (masses) of the compound"),
        Column("main_ion", "ion", "{:d}", "Most prominent mass"),
        Column("best_spectrum", "spectrum", tooltip="Library spectrum of the best match"),
    ]

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.valid_only = QCheckBox("valid only")
        self.valid_only.toggled.connect(lambda _: self.refilter())
        self.toolbar.insertWidget(1, self.valid_only)

    def load(self) -> pd.DataFrame:
        mid = self.selection.measurement_id
        if mid is None:
            return pd.DataFrame(columns=[c.key for c in self.columns])
        compounds = self.data.compounds(mid)
        peaks = self.data.peaks(mid)
        grouped = peaks.groupby("compound_idx")
        main_ion = peaks.loc[grouped["prominence"].idxmax()].set_index("compound_idx")["mass"]

        df = compounds.copy()
        df["compound_idx"] = df.index
        df["name"] = [compound_label(n, i) for i, n in df["main_name"].items()]
        df["n_peaks"] = grouped.size().reindex(df.index).fillna(0).astype(int)
        df["main_ion"] = main_ion.reindex(df.index).astype("Int64")
        return df

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.valid_only.isChecked() and len(df):
            df = df[df["valid_candidate"] == 1]
        return df

    def current_key(self):
        return self.selection.compound_idx

    def select(self, key) -> None:
        if self.selection.measurement_id is not None:
            self.selection.select_compound(self.selection.measurement_id, int(key))
