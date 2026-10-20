"""Calculate > Integrate: compute the peak areas of chosen measurements, in the background."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from PySide6.QtCore import QDateTime, QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from esper import db
from esper.gui.data import EsperData
from esper.gui.selection import Selection
from esper.integration import INTEGRATION_VERSION, integrate_measurements

CUSTOM = "Custom"
# Time ranges, counted back from now
RANGES = {
    "All": None,
    "Last year": pd.DateOffset(years=1),
    "Last month": pd.DateOffset(months=1),
    "Last week": pd.DateOffset(weeks=1),
    "Last day": pd.DateOffset(days=1),
    CUSTOM: None,
}
# Half width of the default rt range, around the selected rt [s]
RT_HALF_RANGE = 30.0


class IntegrationWorker(QObject):
    """Runs integrate_measurements in its own thread, with its own (writable) connection."""

    progress = Signal(int, int, str)
    finished = Signal(object)  # IntegrationSummary, or the exception

    def __init__(self, data_dir, **options) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.options = options
        self.cancel_requested = False

    def run(self) -> None:
        try:
            con = db.connect(self.data_dir, create=True)
            try:
                result = integrate_measurements(
                    con, self.data_dir, **self.options,
                    progress=self.progress.emit, cancelled=lambda: self.cancel_requested,
                )
            finally:
                con.close()
        except Exception as e:
            result = e
        self.finished.emit(result)


class IntegrateDialog(QDialog):
    """Choose the measurements (time range) and compounds (name, rt range) to integrate."""

    # Emitted when an integration wrote to the database
    integrated = Signal()

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        self._thread: QThread | None = None
        self._worker: IntegrationWorker | None = None
        self.setWindowTitle("Integrate peaks")
        self.setMinimumWidth(460)

        # Measurements
        self.range_combo = QComboBox()
        self.range_combo.addItems(list(RANGES))
        self.from_edit = QDateTimeEdit(calendarPopup=True)
        self.to_edit = QDateTimeEdit(calendarPopup=True)
        for edit in (self.from_edit, self.to_edit):
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        now = QDateTime.currentDateTime()
        self.from_edit.setDateTime(now.addDays(-7))
        self.to_edit.setDateTime(now)
        dates = QHBoxLayout()
        dates.addWidget(QLabel("from"))
        dates.addWidget(self.from_edit, 1)
        dates.addWidget(QLabel("to"))
        dates.addWidget(self.to_edit, 1)
        self.count_label = QLabel()
        self.force_check = QCheckBox("Also redo the measurements already integrated")

        measurements = QGroupBox("Measurements")
        form = QFormLayout(measurements)
        form.addRow("Acquired:", self.range_combo)
        form.addRow("", dates)
        form.addRow("", self.count_label)
        form.addRow("", self.force_check)

        # Compounds
        # Follows the selected compound name (see _update_name)
        self.name_check = QCheckBox()
        self.rt_check = QCheckBox("Only rt from")
        self.rt_from = QDoubleSpinBox()
        self.rt_to = QDoubleSpinBox()
        for spin in (self.rt_from, self.rt_to):
            spin.setRange(0, 100_000)
            spin.setDecimals(1)
            spin.setSuffix(" s")
        rt = selection.rt
        self.rt_from.setValue(max(rt - RT_HALF_RANGE, 0) if rt is not None else 0)
        self.rt_to.setValue(rt + RT_HALF_RANGE if rt is not None else 600)
        rt_row = QHBoxLayout()
        rt_row.addWidget(self.rt_check)
        rt_row.addWidget(self.rt_from, 1)
        rt_row.addWidget(QLabel("to"))
        rt_row.addWidget(self.rt_to, 1)
        compounds = QGroupBox("Compounds (all if nothing is checked)")
        compounds_layout = QVBoxLayout(compounds)
        compounds_layout.addWidget(self.name_check)
        compounds_layout.addLayout(rt_row)
        partial = QLabel(
            "Integrating only some compounds does not mark the measurement as integrated: "
            "a later integration of all compounds still does it."
        )
        partial.setWordWrap(True)
        partial.setStyleSheet("color: gray")
        compounds_layout.addWidget(partial)

        # Progress and buttons
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.progress.hide()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.buttons = QDialogButtonBox()
        self.run_button = QPushButton("Integrate")
        self.run_button.setDefault(True)
        self.buttons.addButton(self.run_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.close_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.run_button.clicked.connect(self._run)
        self.close_button.clicked.connect(self._close_or_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(measurements)
        layout.addWidget(compounds)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

        self.range_combo.currentTextChanged.connect(self._on_range_changed)
        for widget in (self.from_edit, self.to_edit):
            widget.dateTimeChanged.connect(self._update_count)
        for check in (self.force_check, self.name_check, self.rt_check):
            check.toggled.connect(self._update_count)
        self.range_combo.setCurrentText("All")
        self._on_range_changed("All")
        self._update_name()
        self.selection.changed.connect(self._on_selection_changed)

    # --- choice -------------------------------------------------------------------------

    def _on_range_changed(self, label: str) -> None:
        custom = label == CUSTOM
        self.from_edit.setEnabled(custom)
        self.to_edit.setEnabled(custom)
        if not custom:
            # Show the dates the range stands for
            start, end = self._range()
            acq = self.data.measurements["acq_datetime"].dropna()
            if start is None:
                start = acq.min() if not acq.empty else pd.Timestamp.now()
            for edit, value in ((self.from_edit, start), (self.to_edit, end)):
                edit.blockSignals(True)
                edit.setDateTime(QDateTime(value.to_pydatetime()))
                edit.blockSignals(False)
        self._update_count()

    def _on_selection_changed(self, fields: set) -> None:
        if "cmp_name" in fields:
            self._update_name()

    def _update_name(self) -> None:
        """Keep the 'Only <name>' filter on the selected compound name.

        Unchecked when no compound name is selected: it never turns into "all
        compounds" silently. A running integration keeps the name it started with.
        """
        name = self.selection.cmp_name
        self.name_check.setText(f"Only {name}" if name else "Only the selected substance (none selected)")
        if name is None:
            self.name_check.setChecked(False)
        self.name_check.setEnabled(name is not None and self._worker is None)

    def _range(self) -> tuple[pd.Timestamp | None, pd.Timestamp]:
        """Acquisition time range (start None: no lower bound)."""
        label = self.range_combo.currentText()
        if label == CUSTOM:
            return (
                pd.Timestamp(self.from_edit.dateTime().toPython()),
                pd.Timestamp(self.to_edit.dateTime().toPython()),
            )
        now = pd.Timestamp(datetime.now()).floor("min")
        offset = RANGES[label]
        return (None if offset is None else now - offset), now

    def _measurement_ids(self) -> list[int]:
        m = self.data.measurements
        start, end = self._range()
        if self.range_combo.currentText() == "All":
            return m.index.tolist()
        acq = m["acq_datetime"]
        keep = acq <= end
        if start is not None:
            keep &= acq >= start
        return m.index[keep].tolist()

    def _options(self) -> dict:
        return {
            "measurement_ids": self._measurement_ids(),
            "compound_name": self.selection.cmp_name if self.name_check.isChecked() else None,
            "rt_range": (self.rt_from.value(), self.rt_to.value()) if self.rt_check.isChecked() else None,
            "force": self.force_check.isChecked(),
        }

    def _update_count(self) -> None:
        ids = self._measurement_ids()
        m = self.data.measurements.loc[ids]
        if "integration_version" in m:
            done = int((m["integration_version"] >= INTEGRATION_VERSION).sum())
        else:
            done = 0
        text = f"{len(ids)} measurements, {done} already integrated"
        if not self.force_check.isChecked() and done:
            text += f" (skipped): {len(ids) - done} to integrate"
        self.count_label.setText(text)
        self.run_button.setEnabled(bool(ids) and (self.force_check.isChecked() or done < len(ids)))

    # --- run ----------------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        for widget in (self.range_combo, self.force_check, self.name_check, self.rt_check, self.rt_from, self.rt_to, self.run_button):
            widget.setEnabled(not running)
        if running:
            self.from_edit.setEnabled(False)
            self.to_edit.setEnabled(False)
        else:
            self._on_range_changed(self.range_combo.currentText())
            self._update_name()
        self.close_button.setText("Cancel" if running else "Close")
        self.progress.setVisible(running)

    def _run(self) -> None:
        self._worker = IntegrationWorker(self.data.data_dir, **self._options())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        # Direct: quit from the worker thread, even while the main thread waits for it
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._worker.deleteLater)
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self._set_running(True)
        self._thread.start()

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"Integrating {name}" if name else "")

    def _on_finished(self, result) -> None:
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._set_running(False)
        if isinstance(result, Exception):
            self.status.setText("")
            QMessageBox.critical(self, "esper", f"Integration failed: {result}")
            return
        parts = [f"{len(result.integrated)} integrated"]
        if result.skipped_integrated:
            parts.append(f"{len(result.skipped_integrated)} already integrated")
        if result.no_chromatogram:
            parts.append(f"{len(result.no_chromatogram)} without chromatogram")
        if result.failed:
            parts.append(f"{len(result.failed)} failed: " + ", ".join(name for name, _ in result.failed[:5]))
        self.status.setText(("Cancelled: " if result.cancelled else "Done: ") + ", ".join(parts))
        if result.integrated:
            self.integrated.emit()
        self._update_count()

    def _close_or_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel_requested = True
            self.status.setText("Cancelling after the current measurement…")
        else:
            self.close()

    def shutdown(self) -> None:
        """Stop a running integration: it finishes the current measurement (each is
        committed), then the thread ends. Blocks until then."""
        if self._worker is not None:
            self._worker.cancel_requested = True
            self._thread.wait()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
