"""Base class of the panels."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

from esper.gui.data import EsperData
from esper.gui.selection import Selection


class BasePanel(QWidget):
    """A widget that shows something for the current selection.

    The panel is redrawn when the selection changes, but only while it is
    visible: a hidden panel is marked as outdated and redrawn when shown.
    Subclasses implement ``redraw`` and call ``invalidate`` when their own
    options change.
    """

    # Name in the View menu
    title = ""
    # Selection fields that make the panel outdated (None: all of them)
    watches: set[str] | None = None

    # Ask the main window to show the chromatogram of the selection
    open_chromatogram = Signal()
    # Shown/hidden with the View menu (not when the whole view changes)
    toggled = Signal(bool)

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        self._outdated = True
        self.selection.changed.connect(self.on_selection_changed)

    def on_selection_changed(self, fields: set) -> None:
        if self.watches is None or fields & self.watches:
            self.invalidate()

    def invalidate(self) -> None:
        """Redraw now if visible, otherwise when shown."""
        self._outdated = True
        if self.isVisible():
            self._redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._outdated:
            self._redraw()

    def _redraw(self) -> None:
        self._outdated = False
        self.redraw()

    def redraw(self) -> None:
        raise NotImplementedError

    def toggle_action(self) -> QAction:
        """Checkable action showing/hiding the panel, for the View menu."""
        action = QAction(self.title, self, checkable=True, checked=not self.isHidden())
        action.toggled.connect(self.setVisible)
        action.toggled.connect(self.toggled)
        return action
