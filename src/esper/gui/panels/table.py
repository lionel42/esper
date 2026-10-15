"""Base of the table panels: a sortable, filterable table whose current row follows the selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QTableView, QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.selection import Selection

SORT_ROLE = Qt.ItemDataRole.UserRole
ROW_HEIGHT = 20
# Initial width limit of a column (long names), the user can widen it
MAX_COLUMN_WIDTH = 220


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


@dataclass
class Column:
    """A column of a table panel.

    ``key`` is the column of the loaded DataFrame, ``fmt`` a format string
    (e.g. ``"{:.1f}"``) or a function giving the displayed text.
    """

    key: str
    header: str
    fmt: str | Callable[[Any], str] | None = None
    tooltip: str = ""

    def text(self, value) -> str:
        if _missing(value):
            return ""
        if callable(self.fmt):
            return self.fmt(value)
        if self.fmt:
            return self.fmt.format(value)
        return str(value)


class DataFrameModel(QAbstractTableModel):
    """Read-only Qt model of a DataFrame; rows are identified by its index."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._columns: list[Column] = []
        self._values: list[list] = []
        self._numeric: list[bool] = []

    def set_frame(self, df: pd.DataFrame, columns: list[Column]) -> None:
        self.beginResetModel()
        self._df = df
        self._columns = columns
        # Plain Python values per row: much faster than DataFrame lookups in data()
        self._values = df[[c.key for c in columns]].astype(object).values.tolist() if len(df) else []
        self._numeric = [pd.api.types.is_numeric_dtype(df[c.key]) for c in columns]
        self.endResetModel()

    def key(self, row: int):
        return self._df.index[row]

    def row_of(self, key) -> int:
        try:
            return int(self._df.index.get_loc(key))
        except KeyError:
            return -1

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._values)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._values[index.row()][index.column()]
        column = self._columns[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.text(value)
        if role == SORT_ROLE:
            # Missing values sort first
            if _missing(value):
                return float("-inf") if self._numeric[index.column()] else ""
            return value if isinstance(value, (int, float, str)) else str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        column = self._columns[section]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.header
        if role == Qt.ItemDataRole.ToolTipRole:
            return column.tooltip or None
        return None


class TablePanel(BasePanel):
    """A table of items (rows of a DataFrame, identified by its index).

    Subclasses define ``columns`` and implement:

    - ``load()``: the rows to show, indexed by item key
    - ``current_key()``: key of the item matching the selection (or None)
    - ``select(key)``: update the selection when a row is chosen

    and may override ``filter_rows`` and add widgets to ``self.toolbar``.
    ``watches`` triggers a reload, ``follows`` only moves the current row.
    """

    columns: list[Column] = []
    follows: set[str] = set()
    # Initial sorting: (key of the column, order)
    default_sort: tuple[str, Qt.SortOrder] | None = None

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self._frame = pd.DataFrame()
        self._updating = False

        self.model = DataFrameModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(SORT_ROLE)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().currentRowChanged.connect(self._on_current_row_changed)
        if self.default_sort is not None:
            key, order = self.default_sort
            column = [c.key for c in self.columns].index(key)
            self.table.horizontalHeader().setSortIndicator(column, order)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._on_filter_text)
        self.count_label = QLabel(self)

        self.toolbar = QHBoxLayout()
        self.toolbar.addWidget(self.filter_edit, 1)
        self.toolbar.addWidget(self.count_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.toolbar)
        layout.addWidget(self.table, 1)

    # --- to implement ---------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def current_key(self):
        raise NotImplementedError

    def select(self, key) -> None:
        raise NotImplementedError

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extra filtering of the loaded rows (e.g. by a checkbox in the toolbar)."""
        return df

    # --- behaviour ------------------------------------------------------------------

    def on_selection_changed(self, fields: set) -> None:
        if self.watches is None or fields & self.watches:
            self.invalidate()
        elif fields & self.follows:
            self.show_current()

    def redraw(self) -> None:
        self._frame = self.load()
        self.refilter()

    def refilter(self) -> None:
        """Apply ``filter_rows`` again to the loaded rows, keeping the sort order."""
        df = self.filter_rows(self._frame)
        first_fill = self.model.rowCount() == 0
        self._updating = True
        self.model.set_frame(df, self.columns)
        self._updating = False
        if first_fill and len(df):
            self.table.resizeColumnsToContents()
            for i in range(len(self.columns)):
                self.table.setColumnWidth(i, min(self.table.columnWidth(i), MAX_COLUMN_WIDTH))
        self._update_count()
        self.show_current()

    def _update_count(self) -> None:
        self.count_label.setText(f"{self.proxy.rowCount()} / {len(self._frame)}")

    def _on_filter_text(self, text: str) -> None:
        self.proxy.setFilterFixedString(text)
        self._update_count()
        self.show_current()

    def show_current(self) -> None:
        """Make the row of the selected item current, without selecting it again."""
        row = self.model.row_of(self.current_key()) if self.model.rowCount() else -1
        index = self.proxy.mapFromSource(self.model.index(row, 0)) if row >= 0 else QModelIndex()
        self._updating = True
        selection = self.table.selectionModel()
        if index.isValid():
            selection.setCurrentIndex(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
            )
            self.table.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)
        else:
            selection.clearSelection()
        self._updating = False

    def _on_current_row_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        """Row chosen by the user (click or arrow keys)."""
        if self._updating or not current.isValid():
            return
        self.select(self.model.key(self.proxy.mapToSource(current).row()))
