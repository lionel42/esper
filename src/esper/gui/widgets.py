"""Small reusable widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)


class StepCombo(QWidget):
    """Label, combobox and up/down buttons to step through the items.

    ``selected`` is emitted with the data of the chosen item, ``text_entered``
    when typed text does not match any item.
    """

    selected = Signal(object)
    text_entered = Signal(str)

    def __init__(self, label: str, editable: bool = True, min_width: int = 120, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.combo = QComboBox(self)
        self.combo.setEditable(editable)
        self.combo.setMinimumWidth(min_width)
        self.combo.setMaxVisibleItems(30)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        if editable:
            self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            completer = self.combo.completer()
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            self.combo.lineEdit().returnPressed.connect(self._on_return)
        self.combo.activated.connect(self._on_activated)

        up = QToolButton(self)
        up.setArrowType(Qt.ArrowType.UpArrow)
        up.clicked.connect(lambda: self.step(-1))
        down = QToolButton(self)
        down.setArrowType(Qt.ArrowType.DownArrow)
        down.clicked.connect(lambda: self.step(1))

        layout.addWidget(QLabel(label, self))
        layout.addWidget(self.combo, 1)
        layout.addWidget(up)
        layout.addWidget(down)

    def set_items(self, items: list[tuple[str, object]], current=None) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        for text, data in items:
            self.combo.addItem(text, data)
        self.combo.blockSignals(False)
        self.set_current(current)

    def set_current(self, data, text: str | None = None) -> None:
        """Select the item with that data, or show ``text`` if there is none."""
        self.combo.blockSignals(True)
        index = self.find(data)
        self.combo.setCurrentIndex(index)
        if index < 0 and self.combo.isEditable():
            self.combo.setEditText(text or "")
        self.combo.blockSignals(False)

    def find(self, data) -> int:
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == data:
                return i
        return -1

    def step(self, delta: int) -> None:
        if self.combo.count() == 0:
            return
        index = self.combo.currentIndex()
        index = 0 if index < 0 else min(max(index + delta, 0), self.combo.count() - 1)
        self.combo.setCurrentIndex(index)
        self.selected.emit(self.combo.itemData(index))

    def _on_activated(self, index: int) -> None:
        self.selected.emit(self.combo.itemData(index))

    def _on_return(self) -> None:
        text = self.combo.currentText().strip()
        if self.combo.findText(text) < 0:
            self.text_entered.emit(text)
