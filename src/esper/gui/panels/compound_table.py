"""Table of the compounds extracted by alpinac in the selected measurement."""

from __future__ import annotations

import sqlite3

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QMenu, QMessageBox

from esper import db
from esper.gui.data import EsperData
from esper.gui.identification_dialog import IdentificationDialog
from esper.gui.panels.chromatogram import compound_label
from esper.gui.panels.table import Column, TablePanel, bool_color, score_color
from esper.gui.references import identification_candidate
from esper.gui.selection import Selection


def valid_color(text: str) -> str | None:
    """Background of the valid column, whose text may carry the pencil of an edited identification."""
    return bool_color("✓" in text)


class CompoundTablePanel(TablePanel):
    """Compounds of the selected measurement. Choosing a row selects the compound.

    The identification of a compound can be edited from the context menu of its row.
    Several compounds can be selected (ctrl/shift click) to see them combined,
    and merged from the context menu into the one with the lowest index.
    """

    title = "Compounds"
    watches = {"measurement_id", "identification"}
    follows = {"compound_idx", "compound_idxs"}
    multi_select = True
    default_sort = ("compound_idx", Qt.SortOrder.AscendingOrder)

    columns = [
        Column("compound_idx", "#", "{:d}", "Compound index in pipeline.nc"),
        Column("rt_mean_s", "rt", "{:.1f}", "Mean retention time of the peaks [s]"),
        Column("retention_index", "RI", "{:.1f}", "Retention index"),
        Column("name", "name"),
        Column(
            "best_score",
            "score",
            "{:.3f}",
            "Score of the best library match",
            score_color,
        ),
        Column(
            "best_score_combined",
            "comb.",
            "{:.3f}",
            "Score combined with the RI difference",
            score_color,
        ),
        Column(
            "valid",
            "valid",
            tooltip="Valid candidate (✎: identification edited by hand)",
            color=valid_color,
        ),
        Column("n_peaks", "peaks", "{:d}", "Number of peaks (masses) of the compound"),
        Column("main_ion", "ion", "{:d}", "Most prominent mass"),
        Column(
            "best_spectrum", "spectrum", tooltip="Library spectrum of the best match"
        ),
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
        main_ion = peaks.loc[grouped["prominence"].idxmax()].set_index("compound_idx")[
            "mass"
        ]

        df = compounds.copy()
        df["compound_idx"] = df.index
        df["name"] = [compound_label(n, i) for i, n in df["main_name"].items()]
        df["n_peaks"] = grouped.size().reindex(df.index).fillna(0).astype(int)
        df["main_ion"] = main_ion.reindex(df.index).astype("Int64")
        # Pencil: identification edited by hand (identified_at is None if by alpinac)
        df["valid"] = [
            " ".join(
                mark
                for mark, on in (("✓", valid), ("✎", isinstance(edited, str)))
                if on
            )
            for valid, edited in zip(df["valid_candidate"], df["identified_at"])
        ]
        return df

    def add_context_actions(self, menu: QMenu, key) -> None:
        mid = self.selection.measurement_id
        if mid is None:
            return
        compound_idx = int(key)
        compound = self._frame.loc[compound_idx]
        menu.addSeparator()
        idxs = self.selection.compound_idxs
        if len(idxs) > 1 and compound_idx in idxs:
            others = ", ".join(f"#{i}" for i in idxs[1:])
            menu.addAction(
                f"Merge {others} into #{idxs[0]}…", lambda: self._merge(mid, idxs)
            )
        menu.addAction(
            "Edit identification…", lambda: self._edit_identification(mid, compound_idx)
        )
        alpinac = (
            compound["alpinac_name"]
            if isinstance(compound["alpinac_name"], str)
            else "none"
        )
        # '&&': a single '&' would be a shortcut mnemonic
        reset = menu.addAction(
            f"Reset to alpinac ({alpinac.replace('&', '&&')})",
            lambda: self._write(self.data.reset_identification, mid, compound_idx),
        )
        reset.setEnabled(isinstance(compound["identified_at"], str))

    def _edit_identification(self, measurement_id: int, compound_idx: int) -> None:
        dialog = IdentificationDialog(self.data, measurement_id, compound_idx, self)
        if not dialog.exec():
            return
        if dialog.reset:
            self._write(self.data.reset_identification, measurement_id, compound_idx)
        else:
            candidate = (None, None, None)
            if dialog.name is not None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    candidate = identification_candidate(
                        self.data, measurement_id, compound_idx, dialog.name
                    )
                finally:
                    QApplication.restoreOverrideCursor()
            self._write(
                self.data.set_identification,
                measurement_id,
                compound_idx,
                dialog.name,
                *candidate,
            )

    def _merge(self, measurement_id: int, compound_idxs: tuple[int, ...]) -> None:
        main, *others = compound_idxs
        answer = QMessageBox.question(
            self,
            "Merge compounds",
            f"Merge {', '.join(f'#{i}' for i in others)} into #{main}?\n\n"
            f"Their peaks are given to #{main} and they are deleted from the database"
            " (until the measurement is imported again). This cannot be undone.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._write(self.data.merge_compounds, measurement_id, compound_idxs):
            self.selection.select_compound(measurement_id, main)

    def _write(self, edit, *args) -> bool:
        try:
            edit(*args)
        except (sqlite3.Error, OSError, db.SchemaVersionError) as e:
            QMessageBox.warning(self, "esper", f"Could not save the edit: {e}")
            return False
        self.selection.identification_edited()
        return True

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.valid_only.isChecked() and len(df):
            df = df[df["valid_candidate"] == 1]
        return df

    def current_key(self):
        return self.selection.compound_idx

    def current_keys(self) -> list:
        return list(self.selection.compound_idxs)

    def select(self, key) -> None:
        if self.selection.measurement_id is not None:
            self.selection.select_compound(self.selection.measurement_id, int(key))

    def select_many(self, keys: list) -> None:
        if self.selection.measurement_id is not None:
            self.selection.select_compounds(self.selection.measurement_id, keys)
