"""Table of the target compounds, with where they are expected in the selected measurement."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox

from esper.gui.data import EsperData
from esper.gui.panels.table import Column, TablePanel
from esper.gui.selection import Selection


class TargetTablePanel(TablePanel):
    """Target compounds (imported with ``esper targets``).

    Choosing a row goes to the compound of the target if it is found in the
    measurement, otherwise to the rt where it is expected, with its expected ions.
    """

    title = "Targets"
    watches = {"measurement_id"}
    follows = {"target"}
    default_sort = ("name", Qt.SortOrder.AscendingOrder)

    columns = [
        Column("name", "name"),
        Column("ri", "RI", "{:.1f}", "Expected retention index"),
        Column("expected_rt", "exp. rt", "{:.1f}", "rt expected in this measurement, from the RI [s]"),
        Column("found", "found", lambda v: "✓" if v else "", "A compound of this measurement is identified as the target"),
        Column("ions", "ions", tooltip="Expected unit masses, main ion first"),
        Column("formula", "formula"),
        Column("cas", "CAS"),
        Column("cmp_class", "class"),
        Column("chemical_name", "chemical name"),
    ]

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.with_ri = QCheckBox("with RI")
        self.with_ri.setToolTip("Only the targets with a retention index")
        self.found_only = QCheckBox("found")
        self.found_only.setToolTip("Only the targets found in this measurement")
        for check in (self.with_ri, self.found_only):
            check.toggled.connect(lambda _: self.refilter())
        self.toolbar.insertWidget(1, self.with_ri)
        self.toolbar.insertWidget(2, self.found_only)
        if self.data.targets.empty:
            self.filter_edit.setPlaceholderText("No targets, import them with: esper targets")

    def load(self) -> pd.DataFrame:
        df = self.data.targets.copy()
        df["name"] = df.index
        df["ions"] = df["masses"].apply(lambda m: ", ".join(map(str, m)))
        mid = self.selection.measurement_id
        if mid is None or df.empty:
            df["expected_rt"] = float("nan")
            df["found"] = False
            return df
        df["expected_rt"] = self.data.expected_rts(mid, df["ri"].to_numpy(dtype=float))
        df["found"] = df.index.isin(self.data.targets_found(mid))
        return df

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.with_ri.isChecked() and len(df):
            df = df[df["ri"].notna()]
        if self.found_only.isChecked() and len(df):
            df = df[df["found"]]
        return df

    def current_key(self):
        return self.selection.target

    def select(self, key) -> None:
        self.selection.select_target(str(key))
