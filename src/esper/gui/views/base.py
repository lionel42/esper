"""Base class of the views."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLayout, QSplitter, QVBoxLayout, QWidget

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.selection import Selection


class BaseView(QWidget):
    """A tab of the main window: toolbar rows above a vertical stack of panels.

    The first panel is the main one. Optional panels can be shown and hidden
    from the View menu; all heights are adjustable with the splitter.
    """

    # Name of the tab
    title = ""

    # Ask the main window to show the chromatogram of the selection
    open_chromatogram = Signal()

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        self._panel_actions: list[QAction] = []

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(True)
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self.splitter, 1)

    def add_toolbar(self, toolbar: QWidget | QLayout) -> None:
        """Add a row above the panels."""
        index = self._layout.count() - 1
        if isinstance(toolbar, QLayout):
            self._layout.insertLayout(index, toolbar)
        else:
            self._layout.insertWidget(index, toolbar)

    def add_panel(
        self, panel: BasePanel, stretch: int = 1, optional: bool = False, visible: bool = True
    ) -> BasePanel:
        """Add a panel below the others, ``stretch`` sets its initial height."""
        self.splitter.addWidget(panel)
        self.splitter.setStretchFactor(self.splitter.count() - 1, stretch)
        panel.setVisible(visible)
        panel.open_chromatogram.connect(self.open_chromatogram)
        if optional:
            self._panel_actions.append(panel.toggle_action())
        return panel

    def panel_actions(self) -> list[QAction]:
        """Actions showing/hiding the optional panels, for the View menu."""
        return list(self._panel_actions)
