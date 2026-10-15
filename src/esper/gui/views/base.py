"""Base class of the views."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLayout, QSplitter, QVBoxLayout, QWidget

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.selection import Selection


# Initial width of the left column and of the main stack
COLUMN_RATIO = (1, 2)


class BaseView(QWidget):
    """A tab of the main window: toolbar rows above a vertical stack of panels.

    The first panel is the main one. Panels can also be put in a column on the
    left (e.g. tables). Optional panels can be shown and hidden from the View
    menu; all sizes are adjustable with the splitters.
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

        # Main stack of panels, and the left column
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(True)
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setChildrenCollapsible(True)
        self.left_splitter.hide()
        self._columns = QSplitter(Qt.Orientation.Horizontal)
        self._columns.addWidget(self.left_splitter)
        self._columns.addWidget(self.splitter)
        for i, stretch in enumerate(COLUMN_RATIO):
            self._columns.setStretchFactor(i, stretch)
        self._sized = False

        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self._columns, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Initial widths of the columns; afterwards the user's splitter positions are kept
        if not self._sized and self.width() > 0:
            self._sized = True
            total = sum(COLUMN_RATIO)
            self._columns.setSizes([self.width() * r // total for r in COLUMN_RATIO])

    def add_toolbar(self, toolbar: QWidget | QLayout) -> None:
        """Add a row above the panels."""
        index = self._layout.count() - 1
        if isinstance(toolbar, QLayout):
            self._layout.insertLayout(index, toolbar)
        else:
            self._layout.insertWidget(index, toolbar)

    def add_panel(
        self,
        panel: BasePanel,
        stretch: int = 1,
        optional: bool = False,
        visible: bool = True,
        left: bool = False,
    ) -> BasePanel:
        """Add a panel below the others, in the main stack or in the left column.

        ``stretch`` sets its initial height relative to the other panels of the column.
        """
        splitter = self.left_splitter if left else self.splitter
        splitter.addWidget(panel)
        splitter.setStretchFactor(splitter.count() - 1, stretch)
        panel.setVisible(visible)
        panel.open_chromatogram.connect(self.open_chromatogram)
        if optional:
            self._panel_actions.append(panel.toggle_action())
        if left:
            panel.toggled.connect(lambda _: self._update_left_column())
            self._update_left_column()
        return panel

    def _update_left_column(self) -> None:
        """Hide the left column when none of its panels is shown."""
        panels = [self.left_splitter.widget(i) for i in range(self.left_splitter.count())]
        self.left_splitter.setVisible(any(not p.isHidden() for p in panels))

    def panel_actions(self) -> list[QAction]:
        """Actions showing/hiding the optional panels, for the View menu."""
        return list(self._panel_actions)
