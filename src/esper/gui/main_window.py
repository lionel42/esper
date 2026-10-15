"""Main window of the esper GUI."""

from __future__ import annotations

import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
)

from esper import __version__
from esper.gui.chromatogram_view import ChromatogramView
from esper.gui.data import EsperData
from esper.gui.results_view import ResultsView
from esper.gui.search_view import SearchView
from esper.gui.selection import Selection

VIEWS = ["Chromatograms", "Results", "Search"]


class MainWindow(QMainWindow):
    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__()
        self.data: EsperData | None = None
        self.selection: Selection | None = None
        self.setWindowTitle("esper")
        self.resize(1350, 900)

        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)
        self._build_menus()
        self._build_statusbar()

        if data_dir is not None:
            self.open_data(data_dir)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open data directory…", self, shortcut=QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._ask_data_dir)
        file_menu.addAction(open_action)
        quit_action = QAction("&Quit", self, shortcut=QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Panels of the current view, filled by show_view
        self.view_menu = self.menuBar().addMenu("&View")
        self._fill_view_menu()

        # Switch views with Ctrl+1, Ctrl+2, ...
        for i in range(len(VIEWS)):
            action = QAction(self, shortcut=QKeySequence(f"Ctrl+{i + 1}"))
            action.triggered.connect(lambda _=False, i=i: self.show_view(i))
            self.addAction(action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(
            lambda: QMessageBox.about(self, "esper", f"esper {__version__}\nPeaks extracted by alpinac.")
        )
        help_menu.addAction(about)

    def _fill_view_menu(self) -> None:
        self.view_menu.clear()
        view = self.stack.currentWidget()
        actions = view.panel_actions() if view is not None else []
        for action in actions:
            self.view_menu.addAction(action)
        if not actions:
            empty = self.view_menu.addAction("No panels for this view")
            empty.setEnabled(False)

    def _build_statusbar(self) -> None:
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.view_buttons = QButtonGroup(self)
        for i, name in enumerate(VIEWS):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setFlat(True)
            self.view_buttons.addButton(button, i)
            self.statusBar().addPermanentWidget(button)
        self.view_buttons.idClicked.connect(self.show_view)

    def _ask_data_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open esper data directory")
        if path:
            self.open_data(Path(path))

    def open_data(self, data_dir: Path) -> None:
        try:
            data = EsperData(data_dir)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "esper", str(e))
            return

        while self.stack.count():
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        if self.data is not None:
            self.data.close()

        self.data = data
        self.selection = Selection(data)
        self.selection.changed.connect(self._update_status)
        self.chromatogram_view = ChromatogramView(data, self.selection)
        self.results_view = ResultsView(data, self.selection)
        self.results_view.open_chromatogram.connect(lambda: self.show_view(0))
        self.stack.addWidget(self.chromatogram_view)
        self.stack.addWidget(self.results_view)
        self.search_view = SearchView(data, self.selection)
        self.search_view.open_chromatogram.connect(lambda: self.show_view(0))
        self.stack.addWidget(self.search_view)
        self.setWindowTitle(f"esper - {data_dir}")

        if not data.measurements.empty:
            self.selection.select_measurement(int(data.measurements.index[-1]))
        self.show_view(0)

    def show_view(self, index: int) -> None:
        if self.stack.count() == 0:
            return
        self.stack.setCurrentIndex(index)
        self.view_buttons.button(index).setChecked(True)
        self._fill_view_menu()

    def _update_status(self, fields=None) -> None:
        sel = self.selection
        if sel.measurement_id is None:
            self.status_label.setText("")
            return
        name = self.data.measurements.loc[sel.measurement_id, "name"]
        cmp = sel.cmp_name or ""
        self.status_label.setText(f"{name} ({cmp})" if cmp else name)


def run(data_dir: Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(background="w", foreground="k", antialias=True)
    window = MainWindow(data_dir)
    window.show()
    return app.exec()
