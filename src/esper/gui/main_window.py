"""Main window of the esper GUI."""

from __future__ import annotations

import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
)

from esper import __version__
from esper.gui.data import EsperData
from esper.gui.library_loader import library_loader
from esper.gui.open_dialog import OpenDataDialog, ask_data_dir, is_data_dir
from esper.gui.settings import settings_manager
from esper.gui.selection import Selection
from esper.gui.views.base import BaseView
from esper.gui.views.chromatograms import ChromatogramsView
from esper.gui.views.results import ResultsView
from esper.gui.views.search import SearchView

# Tabs, in order; the first one shows the chromatogram of the selection
VIEWS: list[type[BaseView]] = [ChromatogramsView, ResultsView, SearchView]


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
        self.recent_menu = file_menu.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        settings_action = QAction("&Settings…", self, shortcut=QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)
        quit_action = QAction("&Quit", self, shortcut=QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Panels of the current view, filled by show_view, then the display options
        self.view_menu = self.menuBar().addMenu("&View")
        self.uncertainties_action = QAction("Show &uncertainties", self, checkable=True)
        self.uncertainties_action.setChecked(settings_manager().settings.show_uncertainties)
        self.uncertainties_action.toggled.connect(self._set_show_uncertainties)
        self._fill_view_menu()

        calculate_menu = self.menuBar().addMenu("&Calculate")
        self.integrate_action = QAction("&Integrate…", self)
        self.integrate_action.setEnabled(False)
        self.integrate_action.triggered.connect(self._open_integrate)
        calculate_menu.addAction(self.integrate_action)
        self.integrate_dialog = None

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
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.uncertainties_action)

    def _set_show_uncertainties(self, show: bool) -> None:
        manager = settings_manager()
        if manager.settings.show_uncertainties != show:
            manager.update(replace(manager.settings, show_uncertainties=show))

    def _build_statusbar(self) -> None:
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        # Background reading of the spectral libraries
        self.library_progress = QProgressBar()
        self.library_progress.setMaximumWidth(260)
        self.library_progress.setFormat("%v / %m")
        self.library_progress.hide()
        self.library_label = QLabel()
        self.library_label.hide()
        self.statusBar().addPermanentWidget(self.library_label)
        self.statusBar().addPermanentWidget(self.library_progress)
        loader = library_loader()
        loader.progress.connect(self._on_library_progress)
        loader.busy.connect(self._on_library_busy)

        self.view_buttons = QButtonGroup(self)
        for i, view_class in enumerate(VIEWS):
            button = QPushButton(view_class.title)
            button.setCheckable(True)
            button.setFlat(True)
            self.view_buttons.addButton(button, i)
            self.statusBar().addPermanentWidget(button)
        self.view_buttons.idClicked.connect(self.show_view)

    def _on_library_progress(self, library: str, done: int, total: int) -> None:
        self.library_label.setText(f"Loading library {library}")
        self.library_progress.setMaximum(total)
        self.library_progress.setValue(done)

    def _on_library_busy(self, busy: bool) -> None:
        self.library_label.setVisible(busy)
        self.library_progress.setVisible(busy)
        if busy:
            self.library_label.setText("Loading libraries")
            self.library_progress.setValue(0)

    def _open_settings(self) -> None:
        from esper.gui.settings_dialog import SettingsDialog

        SettingsDialog(self).exec()

    def _open_integrate(self) -> None:
        from esper.gui.integrate_dialog import IntegrateDialog

        if self.integrate_dialog is None:
            # Not modal: the data can be browsed during a long integration
            self.integrate_dialog = IntegrateDialog(self.data, self.selection, self)
            self.integrate_dialog.integrated.connect(self._on_integrated)
        self.integrate_dialog.show()
        self.integrate_dialog.raise_()

    def _close_integrate_dialog(self) -> None:
        """Close the integration dialog, stopping a running integration first."""
        if self.integrate_dialog is not None:
            self.integrate_dialog.shutdown()
            self.integrate_dialog.close()
            self.integrate_dialog.deleteLater()
            self.integrate_dialog = None

    def closeEvent(self, event) -> None:
        # Its thread is owned by the dialog: stop it before the window and the dialog go
        self._close_integrate_dialog()
        super().closeEvent(event)

    def _on_integrated(self) -> None:
        self.data.reload()
        self.selection.refresh()

    def _ask_data_dir(self) -> None:
        path = ask_data_dir(self, "Open esper data directory")
        if path is not None:
            self.open_data(path)

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = settings_manager().settings.recent_data_dirs
        for path in recent:
            action = self.recent_menu.addAction(path)
            action.setEnabled(is_data_dir(Path(path)))
            action.triggered.connect(lambda _=False, p=path: self.open_data(Path(p)))
        if not recent:
            self.recent_menu.addAction("No recent data directory").setEnabled(False)

    def choose_data_dir(self) -> bool:
        """Ask which data directory to open (recent ones or browse); False if cancelled."""
        dialog = OpenDataDialog(self)
        if dialog.exec() and dialog.data_dir is not None:
            self.open_data(dialog.data_dir)
            return self.data is not None
        return False

    def open_data(self, data_dir: Path) -> None:
        try:
            data = EsperData(data_dir)
        except (FileNotFoundError, ValueError) as e:
            QMessageBox.critical(self, "esper", str(e))
            return

        while self.stack.count():
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        self._close_integrate_dialog()
        if self.data is not None:
            self.data.close()
        settings_manager().remember_data_dir(data_dir)

        self.data = data
        self.selection = Selection(data)
        self.selection.changed.connect(self._update_status)
        self.views: list[BaseView] = []
        for view_class in VIEWS:
            view = view_class(data, self.selection)
            view.open_chromatogram.connect(lambda: self.show_view(0))
            self.stack.addWidget(view)
            self.views.append(view)
        self.setWindowTitle(f"esper - {data_dir}")
        self.integrate_action.setEnabled(True)

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
    app.setApplicationName("esper")
    # Matches the desktop entry (window grouping, Wayland icon)
    app.setDesktopFileName("esper")
    icon = resources.files("esper") / "resources" / "esper.svg"
    app.setWindowIcon(QIcon(str(icon)))
    pg.setConfigOptions(background="w", foreground="k", antialias=True)
    window = MainWindow(data_dir)
    window.show()
    if data_dir is None and not window.choose_data_dir() and window.data is None:
        # Cancelled: nothing to show
        return 0
    return app.exec()
