"""Dialog to edit the settings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from esper.gui.settings import INTENSITIES, Settings, settings_manager
from esper.libraries import Library

LIBRARY_ROLE = Qt.ItemDataRole.UserRole


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("esper settings")
        self.resize(640, 420)
        settings = settings_manager().settings

        tabs = QTabWidget(self)
        tabs.addTab(self._spectrum_tab(settings), "Spectrum")
        tabs.addTab(self._libraries_tab(settings), "Libraries")
        tabs.addTab(self._matching_tab(settings), "Matching")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: self._fill(Settings())
        )

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # --- tabs -----------------------------------------------------------------------

    def _spectrum_tab(self, settings: Settings) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.intensity = QComboBox()
        self.intensity.addItems(INTENSITIES)
        self.intensity.setCurrentText(settings.spectrum_intensity)
        form.addRow("Intensity of the measured peaks:", self.intensity)
        note = QLabel(
            "Used for the spectrum and the best matches. The area is integrated in the "
            "chromatograms (slower). alpinac's stored scores were computed with heights."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return tab

    def _libraries_tab(self, settings: Settings) -> QWidget:
        tab = QWidget()
        self.library_list = QListWidget()
        self._fill_libraries(settings.libraries)

        up = QPushButton("Up")
        down = QPushButton("Down")
        add = QPushButton("Add…")
        remove = QPushButton("Remove")
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        add.clicked.connect(self._add_library)
        remove.clicked.connect(self._remove_library)

        buttons = QVBoxLayout()
        for button in (up, down, add, remove):
            buttons.addWidget(button)
        buttons.addStretch()

        row = QHBoxLayout()
        row.addWidget(self.library_list, 1)
        row.addLayout(buttons)

        layout = QVBoxLayout(tab)
        note = QLabel(
            "Checked libraries are used. When a spectrum is in several libraries, "
            "the first one in this list is shown."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addLayout(row)
        return tab

    def _matching_tab(self, settings: Settings) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0, 1000)
        self.cutoff.setSuffix(" Da")
        self.cutoff.setValue(settings.cutoff_mass)
        self.ignored = QLineEdit(", ".join(f"{m:g}" for m in settings.ignored_masses))
        self.ignored.setPlaceholderText("e.g. 28, 32, 40")
        self.normalize = QCheckBox()
        self.normalize.setChecked(settings.normalize)
        self.n_best = QSpinBox()
        self.n_best.setRange(1, 500)
        self.n_best.setValue(settings.n_best_matches)
        form.addRow("Cutoff mass (masses below are removed):", self.cutoff)
        form.addRow("Ignored masses:", self.ignored)
        form.addRow("Normalize intensities:", self.normalize)
        form.addRow("Number of best matches:", self.n_best)
        return tab

    # --- libraries ------------------------------------------------------------------

    def _fill_libraries(self, libraries: list[Library]) -> None:
        self.library_list.clear()
        for library in libraries:
            self._add_item(library)

    def _add_item(self, library: Library) -> None:
        item = QListWidgetItem(f"{library.name}    {library.path}")
        item.setData(LIBRARY_ROLE, library)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if library.enabled else Qt.CheckState.Unchecked
        )
        if not Path(library.path).is_dir():
            item.setToolTip("Folder not found")
            item.setForeground(Qt.GlobalColor.red)
        self.library_list.addItem(item)

    def _move(self, delta: int) -> None:
        row = self.library_list.currentRow()
        new = row + delta
        if row < 0 or not 0 <= new < self.library_list.count():
            return
        item = self.library_list.takeItem(row)
        self.library_list.insertItem(new, item)
        self.library_list.setCurrentRow(new)

    def _add_library(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Library folder (.jdx or alpinac .txt spectra)"
        )
        if not folder:
            return
        name, ok = QInputDialog.getText(
            self,
            "Library name",
            "Name (prefix before '_' = family, e.g. nist_xyz):",
            text=Path(folder).name.lower(),
        )
        if ok and name:
            self._add_item(Library(name, Path(folder), True))

    def _remove_library(self) -> None:
        row = self.library_list.currentRow()
        if row >= 0:
            self.library_list.takeItem(row)

    def _libraries(self) -> list[Library]:
        libraries = []
        for i in range(self.library_list.count()):
            item = self.library_list.item(i)
            library = item.data(LIBRARY_ROLE)
            enabled = item.checkState() == Qt.CheckState.Checked
            libraries.append(Library(library.name, Path(library.path), enabled))
        return libraries

    # --- result ---------------------------------------------------------------------

    def _fill(self, settings: Settings) -> None:
        self.intensity.setCurrentText(settings.spectrum_intensity)
        self._fill_libraries(settings.libraries)
        self.cutoff.setValue(settings.cutoff_mass)
        self.ignored.setText(", ".join(f"{m:g}" for m in settings.ignored_masses))
        self.normalize.setChecked(settings.normalize)
        self.n_best.setValue(settings.n_best_matches)

    def settings(self) -> Settings:
        """The current settings with the values of the dialog (other fields kept)."""
        ignored = [
            float(t)
            for t in self.ignored.text().replace(";", ",").replace(" ", ",").split(",")
            if t.strip()
        ]
        return replace(
            settings_manager().settings,
            spectrum_intensity=self.intensity.currentText(),
            libraries=self._libraries(),
            cutoff_mass=self.cutoff.value(),
            ignored_masses=ignored,
            normalize=self.normalize.isChecked(),
            n_best_matches=self.n_best.value(),
        )

    def accept(self) -> None:
        try:
            settings = self.settings()
        except ValueError:
            self.ignored.setFocus()
            self.ignored.selectAll()
            return
        settings_manager().update(settings)
        super().accept()
