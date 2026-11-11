"""Table of the target compounds, with where they are expected in the selected measurement."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox

from esper.gui.data import EsperData
from esper.gui.panels.table import Column, TablePanel, bool_color
from esper.gui.selection import Selection


class TargetTablePanel(TablePanel):
    """Target compounds (imported with ``esper targets``).

    Choosing a row goes to the compound of the target if it is found in the
    measurement, otherwise to the rt where it is expected, with its expected ions.
    """

    title = "Targets"
    watches = {"measurement_id", "identification"}
    follows = {"target"}
    default_sort = ("name", Qt.SortOrder.AscendingOrder)

    columns = [
        Column("name", "name"),
        Column("ri", "RI", "{:.1f}", "Expected retention index"),
        Column(
            "expected_rt",
            "exp. rt",
            "{:.1f}",
            "rt expected in this measurement [s]: from the identification of the "
            "time of the run (GCwerks peakid), else from the RI",
        ),
        Column(
            "found",
            "found",
            lambda v: "✓" if v else "",
            "A compound of this measurement is identified as the target",
            bool_color,
        ),
        Column("ions", "ions", tooltip="Expected unit masses, main ion first"),
        Column("formula", "formula"),
        Column("cas", "CAS"),
        Column("cmp_class", "class"),
        Column("chemical_name", "chemical name"),
    ]

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.targets_only = QCheckBox("targets only")
        self.targets_only.setToolTip(
            "Only the targets with an expected position (retention index or expected "
            "rt), not the suspects known only by their spectrum"
        )
        self.found_only = QCheckBox("found")
        self.found_only.setToolTip("Only the targets found in this measurement")
        for check in (self.targets_only, self.found_only):
            check.toggled.connect(lambda _: self.refilter())
        self.toolbar.insertWidget(1, self.targets_only)
        self.toolbar.insertWidget(2, self.found_only)
        if self.data.targets.empty:
            self.filter_edit.setPlaceholderText(
                "No targets, import them with: esper targets (GCwerks: esper upload)"
            )

    def load(self) -> pd.DataFrame:
        df = self.data.targets.copy()
        df["name"] = df.index
        df["ions"] = df["masses"].apply(lambda m: ", ".join(map(str, m)))
        mid = self.selection.measurement_id
        if mid is None or df.empty:
            df["expected_rt"] = float("nan")
            df["found"] = False
            # no run to place them in: any retention reference makes a target
            has_rts = [self.data.target_index.has_rts(t) for t in df.index]
            df["is_target"] = df["ri"].notna() | pd.Series(has_rts, index=df.index, dtype=bool)
            return df
        df["expected_rt"] = self.data.expected_target_rts(mid)
        # a target is expected somewhere: an rt in this run (GCwerks peakid, or from the
        # RI with the RI references of the run), or at least an RI (run without references)
        df["is_target"] = df["expected_rt"].notna() | df["ri"].notna()
        # the ions of the version of the measurement (GCwerks peakid)
        df["ions"] = [", ".join(map(str, self.data.target_masses(mid, t))) for t in df.index]
        df["found"] = df.index.isin(self.data.targets_found(mid))
        return df

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.targets_only.isChecked() and len(df):
            df = df[df["is_target"]]
        if self.found_only.isChecked() and len(df):
            df = df[df["found"]]
        return df

    def current_key(self):
        return self.selection.target

    def select(self, key) -> None:
        self.selection.select_target(str(key))
