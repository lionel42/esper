"""Dialog to edit the identification of a compound: as a target, another name, or not identified."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from esper.gui.data import EsperData


def _format(value, fmt: str) -> str:
    return fmt.format(value) if value == value and value is not None else "-"


class IdentificationDialog(QDialog):
    """Choose what a compound is: a target (or any name typed), or not identified.

    After ``exec``, ``name`` is the chosen name (None: not identified), or
    ``reset`` is True to go back to alpinac's identification.
    """

    def __init__(
        self, data: EsperData, measurement_id: int, compound_idx: int, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Identification of compound #{compound_idx}")
        self.setMinimumWidth(460)
        self.name: str | None = None
        self.reset = False

        compound = data.compounds(measurement_id).loc[compound_idx]
        alpinac = (
            compound["alpinac_name"]
            if isinstance(compound["alpinac_name"], str)
            else None
        )
        alpinac_target = data.target_of(alpinac)
        edited = isinstance(compound["identified_at"], str)
        current = (
            compound["main_name"] if isinstance(compound["main_name"], str) else None
        )

        self.combo = QComboBox(self)
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo.setMaxVisibleItems(30)
        completer = self.combo.completer()
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # alpinac's library candidates first, then all the targets with their expected rt
        for _, candidate in data.candidates(measurement_id, compound_idx).iterrows():
            name = data.main_name(candidate["candidate_name"])
            if name is not None:
                score = _format(candidate["score"], "{:.3f}")
                self.combo.addItem(
                    f"{name}  (candidate {candidate['rank']}, score {score})", name
                )
        if self.combo.count():
            self.combo.insertSeparator(self.combo.count())
        targets = data.targets.index.tolist()
        expected = (
            data.expected_rts(measurement_id, data.targets["ri"].to_numpy(dtype=float))
            if targets
            else []
        )
        for target, rt in sorted(zip(targets, expected), key=lambda t: t[0].lower()):
            self.combo.addItem(
                f"{target}  (expected rt {_format(rt, '{:.1f}')} s)", target
            )
        index = self.combo.findData(current) if current is not None else -1
        self.combo.setCurrentIndex(index)
        if index < 0:
            self.combo.setEditText(current or "")

        self.identified = QRadioButton("Identified as", self)
        self.not_identified = QRadioButton("Not identified", self)
        group = QButtonGroup(self)
        group.addButton(self.identified)
        group.addButton(self.not_identified)
        (self.identified if current is not None else self.not_identified).setChecked(
            True
        )
        self.identified.toggled.connect(self.combo.setEnabled)
        self.combo.setEnabled(self.identified.isChecked())
        self.combo.lineEdit().textEdited.connect(
            lambda _: self.identified.setChecked(True)
        )

        info = QFormLayout()
        info.addRow(
            "Retention time:",
            QLabel(f"{_format(compound['rt_mean_s'], '{:.1f}')} s", self),
        )
        info.addRow(
            "Retention index:",
            QLabel(_format(compound["retention_index"], "{:.1f}"), self),
        )
        alpinac_text = alpinac or "none"
        if alpinac_target is not None and alpinac_target != alpinac:
            alpinac_text += f" (target {alpinac_target})"
        if not compound["alpinac_valid"]:
            alpinac_text += ", not a valid candidate"
        info.addRow("alpinac:", QLabel(alpinac_text, self))
        if edited:
            info.addRow(
                "Edited:", QLabel(compound["identified_at"].replace("T", " "), self)
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        reset = buttons.addButton(
            "Reset to alpinac", QDialogButtonBox.ButtonRole.ResetRole
        )
        reset.setToolTip("Forget the identification edited by hand")
        reset.setEnabled(edited)
        reset.clicked.connect(self._on_reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(info)
        layout.addWidget(self.identified)
        layout.addWidget(self.combo)
        layout.addWidget(self.not_identified)
        layout.addWidget(buttons)

    def _chosen_name(self) -> str | None:
        """The item chosen in the list, or the name typed."""
        text = self.combo.currentText().strip()
        index = self.combo.currentIndex()
        if index >= 0 and text == self.combo.itemText(index):
            return self.combo.itemData(index)
        return text or None

    def _on_reset(self) -> None:
        self.reset = True
        super().accept()

    def accept(self) -> None:
        if self.identified.isChecked():
            self.name = self._chosen_name()
            if self.name is None:
                self.combo.setFocus()
                return
        super().accept()
