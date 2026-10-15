"""Choose the data directory to open, among the ones opened before."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from esper import db
from esper.gui.settings import settings_manager

PATH_ROLE = Qt.ItemDataRole.UserRole


def is_data_dir(path: Path) -> bool:
    return db.db_path(path).is_file()


def ask_data_dir(parent=None, title: str = "Choose a folder with an esper database") -> Path | None:
    """File dialog for a data directory, checked to contain an esper database."""
    folder = QFileDialog.getExistingDirectory(parent, title)
    if not folder:
        return None
    if not is_data_dir(Path(folder)):
        QMessageBox.warning(parent, "esper", f"No esper database ({db.DB_FILENAME}) in {folder}")
        return None
    return Path(folder)


class OpenDataDialog(QDialog):
    """List of the recent data directories; ``data_dir`` is the chosen one after ``exec()``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open esper data")
        self.resize(560, 320)
        self.data_dir: Path | None = None
        self.settings = settings_manager()

        self.list = QListWidget(self)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        browse = self.buttons.addButton("Browse…", QDialogButtonBox.ButtonRole.ActionRole)
        self.remove = self.buttons.addButton("Remove from list", QDialogButtonBox.ButtonRole.ActionRole)
        browse.clicked.connect(self._browse)
        self.remove.clicked.connect(self._remove)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self._fill()
        self.list.currentItemChanged.connect(lambda *_: self._update_buttons())

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Data directories opened before:"))
        layout.addWidget(self.list, 1)
        layout.addWidget(self.buttons)
        self._update_buttons()

    def _fill(self) -> None:
        self.list.clear()
        for path in self.settings.settings.recent_data_dirs:
            item = QListWidgetItem(path)
            item.setData(PATH_ROLE, path)
            if not is_data_dir(Path(path)):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip("No esper database there (moved or not accessible)")
            self.list.addItem(item)
        for row in range(self.list.count()):
            if self.list.item(row).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.list.setCurrentRow(row)
                break

    def _update_buttons(self) -> None:
        item = self.list.currentItem()
        usable = item is not None and bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(usable)
        self.remove.setEnabled(item is not None)

    def _browse(self) -> None:
        path = ask_data_dir(self)
        if path is not None:
            self.data_dir = path
            super().accept()

    def _remove(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self.settings.forget_data_dir(item.data(PATH_ROLE))
            self._fill()
            self._update_buttons()

    def accept(self) -> None:
        item = self.list.currentItem()
        if item is None or not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        self.data_dir = Path(item.data(PATH_ROLE))
        super().accept()
