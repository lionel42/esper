"""Base of the table panels: a sortable, filterable table whose current row follows the selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence
from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QTableView,
    QVBoxLayout,
)

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.selection import Selection
from esper.gui.style import QUALITY_COLORS, RI_DIFF_THRESHOLDS, SCORE_THRESHOLDS

SORT_ROLE = Qt.ItemDataRole.UserRole
ROW_HEIGHT = 20
# Delay before reloading a table searched in its source (``search_reloads``) [ms]
SEARCH_DELAY_MS = 250
# Initial width limit of a column (long names), the user can widen it
MAX_COLUMN_WIDTH = 220


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def score_color(value) -> str | None:
    """Background of a spectral score (0-1): high is good."""
    for threshold, color in zip(SCORE_THRESHOLDS, QUALITY_COLORS):
        if value >= threshold:
            return color
    return QUALITY_COLORS[-1]


def ri_diff_color(value) -> str | None:
    """Background of a retention index difference: close to 0 is good."""
    for threshold, color in zip(RI_DIFF_THRESHOLDS, QUALITY_COLORS):
        if abs(value) <= threshold:
            return color
    return QUALITY_COLORS[-1]


def bool_color(value) -> str | None:
    """Background of a yes/no column: green or red."""
    return QUALITY_COLORS[0] if value else QUALITY_COLORS[-1]


@dataclass
class Column:
    """A column of a table panel.

    ``key`` is the column of the loaded DataFrame, ``fmt`` a format string
    (e.g. ``"{:.1f}"``) or a function giving the displayed text, ``color`` a
    function giving the background color of a (non missing) value.
    """

    key: str
    header: str
    fmt: str | Callable[[Any], str] | None = None
    tooltip: str = ""
    color: Callable[[Any], str | None] | None = None

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
        if role == Qt.ItemDataRole.BackgroundRole:
            if column.color is None or _missing(value):
                return None
            color = column.color(value)
            return QColor(color) if color else None
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

    and may override ``filter_rows`` and ``add_context_actions``, and add
    widgets to ``self.toolbar``. With ``multi_select``, several rows can be
    selected (ctrl/shift click): they implement ``current_keys()`` and
    ``select_many(keys)`` too.
    ``watches`` triggers a reload, ``follows`` only moves the current row.

    The text box filters the loaded rows; with ``search_reloads`` it is a
    search passed to ``load()`` instead (``self.search_text``).
    """

    columns: list[Column] = []
    follows: set[str] = set()
    # Initial sorting: (key of the column, order)
    default_sort: tuple[str, Qt.SortOrder] | None = None
    # The text box searches in the source (reload) instead of filtering the rows
    search_reloads = False
    # Several rows can be selected together
    multi_select = False

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
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
            if self.multi_select
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._on_rows_selected)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        if self.default_sort is not None:
            key, order = self.default_sort
            column = [c.key for c in self.columns].index(key)
            self.table.horizontalHeader().setSortIndicator(column, order)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._on_filter_text)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(SEARCH_DELAY_MS)
        self._search_timer.timeout.connect(self.invalidate)
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

    def current_keys(self) -> list:
        """Keys of the items matching the selection, the current one first (with ``multi_select``)."""
        key = self.current_key()
        return [] if key is None else [key]

    def select_many(self, keys: list) -> None:
        """Update the selection when several rows are chosen (with ``multi_select``)."""
        raise NotImplementedError

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extra filtering of the loaded rows (e.g. by a checkbox in the toolbar)."""
        return df

    def add_context_actions(self, menu: QMenu, key) -> None:
        """Add actions on the item ``key`` to the context menu of its row."""

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
        if self.search_reloads:
            self.count_label.setText(f"{self.proxy.rowCount()}")
        else:
            self.count_label.setText(f"{self.proxy.rowCount()} / {len(self._frame)}")

    @property
    def search_text(self) -> str:
        return self.filter_edit.text().strip()

    def _on_filter_text(self, text: str) -> None:
        if self.search_reloads:
            self._search_timer.start()
            return
        # Rows hidden by the filter are deselected, without changing the selection
        self._updating = True
        self.proxy.setFilterFixedString(text)
        self._updating = False
        self._update_count()
        self.show_current()

    def show_current(self) -> None:
        """Make the rows of the selected items selected (the first one current), without selecting them again."""
        keys = self.current_keys() if self.model.rowCount() else []
        rows = [row for row in map(self.model.row_of, keys) if row >= 0]
        indexes = [i for i in (self.proxy.mapFromSource(self.model.index(row, 0)) for row in rows) if i.isValid()]
        self._updating = True
        selection = self.table.selectionModel()
        if indexes:
            items = QItemSelection()
            for index in indexes:
                items.select(index, index)
            flags = QItemSelectionModel.SelectionFlag
            selection.select(items, flags.ClearAndSelect | flags.Rows)
            selection.setCurrentIndex(indexes[0], flags.NoUpdate)
            self.table.scrollTo(indexes[0], QAbstractItemView.ScrollHint.EnsureVisible)
        else:
            selection.clearSelection()
        self._updating = False

    def _on_rows_selected(self, *_) -> None:
        """Rows chosen by the user (click, ctrl/shift click or arrow keys)."""
        if self._updating:
            return
        rows = self.table.selectionModel().selectedRows()
        keys = [self.model.key(self.proxy.mapToSource(index).row()) for index in rows]
        if len(keys) == 1:
            self.select(keys[0])
        elif len(keys) > 1:
            self.select_many(keys)

    # --- context menu ---------------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row = self.proxy.mapToSource(index).row()
        column = self.proxy.mapToSource(index).column()
        menu = QMenu(self)
        copy = menu.addAction("Copy", lambda: self._copy(self.model.index(row, column).data()))
        copy.setShortcut(QKeySequence.StandardKey.Copy)
        menu.addAction("Copy row", lambda: self._copy(self._row_text(row)))
        self.add_context_actions(menu, self.model.key(row))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _row_text(self, row: int) -> str:
        """The displayed values of a row, separated by tabs (pasted in cells of a spreadsheet)."""
        return "\t".join(self.model.index(row, i).data() or "" for i in range(self.model.columnCount()))

    @staticmethod
    def _copy(text: str | None) -> None:
        QGuiApplication.clipboard().setText(text or "")
