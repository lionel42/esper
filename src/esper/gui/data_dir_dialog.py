"""File > Data directory: information about the open data directory, copy its path, show
it in the file manager."""

from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from esper.data_dir_info import Section, as_text, describe


def copy_path(data_dir: Path) -> str:
    """Put the full path of ``data_dir`` in the clipboard; returns it."""
    path = str(Path(data_dir).resolve())
    QGuiApplication.clipboard().setText(path)
    return path


def show_in_file_manager(data_dir: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(data_dir).resolve())))


def as_html(sections: list[Section]) -> str:
    rows = []
    for section in sections:
        rows.append(
            f"<tr><td colspan=2 style='padding-top:10px'><b>{escape(section.title)}"
            "</b></td></tr>"
        )
        rows += [
            f"<tr><td style='padding-right:16px'>{escape(label)}</td>"
            f"<td>{escape(value)}</td></tr>"
            for label, value in section.rows
        ]
    return f"<table>{''.join(rows)}</table>"


class DataDirDialog(QDialog):
    """The information of :func:`esper.data_dir_info.describe`, selectable, with buttons
    to copy the path or all of it."""

    def __init__(self, data_dir: Path, con, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data directory")
        self.resize(720, 520)
        self.data_dir = Path(data_dir)
        self.sections = describe(self.data_dir, con)

        self.label = QLabel(as_html(self.sections))
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        content = QWidget()
        QVBoxLayout(content).addWidget(self.label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        for text, slot in (
            ("Copy path", lambda: copy_path(self.data_dir)),
            ("Copy all", self.copy_all),
            ("Show in file manager", lambda: show_in_file_manager(self.data_dir)),
        ):
            button = QPushButton(text)
            button.setAutoDefault(False)
            button.clicked.connect(slot)
            buttons.addButton(button, QDialogButtonBox.ButtonRole.ActionRole)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(buttons)

    def copy_all(self) -> None:
        QGuiApplication.clipboard().setText(as_text(self.sections))
