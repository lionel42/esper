"""Dialog to refit the peaks of a target with other parameters: a fit override
(:mod:`esper.fit_overrides`), previewed on the selected run before it is saved and the
runs it applies to are fitted again.

Opened from the right click menu of the compound table. The parameters start as the
override of the target on the selected mass that applies to the run, else as the fit of
the data directory. The preview draws the chromatogram as the chromatogram panel does
(``chromatogram_curve``, ``draw_curve``), the current fit (``fit_integration_items``) and
the fit of the override (``fit_fill_items``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyqtgraph as pg
from peaxart.backgrounds import BackgroundName
from peaxart.shapes import ShapeName
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from esper import db
from esper.fit_overrides import (
    DEFAULT_SIGMA_RANGE_S,
    FitOverride,
    OverrideParams,
    delete_override,
    override_fit,
    read_overrides,
    reference_rt,
    refit_runs,
    run_overrides,
    save_override,
)
from esper.fitting import FITTED_PEAK_SOURCES, extraction
from esper.gui.data import EsperData
from esper.gui.labels import compound_label, mass_label
from esper.gui.panels.chromatogram import chromatogram_curve, draw_curve
from esper.gui.panels.chromatogram_items import (
    fit_fill_items,
    fit_integration_items,
    fit_sum_items,
)
from esper.gui.plotting import make_plot
from esper.gui.style import TARGET_COLOR
from esper.gui.widgets import StepCombo

MASS_ROLE = Qt.ItemDataRole.UserRole
# Window fitted by default: the bounds of the compound's peaks, widened by this [s]
RT_MARGIN_S = 10.0
# Without bounds: this around the reference rt [s]
RT_HALF_WINDOW_S = 15.0
# The plot shows the window and this around it [s]
PLOT_MARGIN_S = 10.0
PREVIEW_COLOR = "#0050c8"
PREVIEW_SUM_COLOR = "#e6007e"
POSITION_COLOR = "#e07000"


@dataclass(frozen=True)
class TableColumn:
    header: str
    tooltip: str
    value: Callable  # (kind, peak row) -> text


def _get(name: str, digits: int = 0, preview_only: bool = False) -> Callable:
    def value(kind, p) -> str:
        if preview_only and kind != "preview":
            return ""
        return _number(p.get(name), digits)

    return value


def _background(kind, p) -> str:
    """Kept by the fit (preview); stored: only its slope (constant or not)."""
    if kind == "preview" and isinstance(p.get("background"), str):
        return p["background"]
    slope = p.get("fit_baseline_slope")
    if slope is None or pd.isna(slope):
        return "–" if pd.isna(p.get("fit_baseline")) else "constant"
    return "constant" if slope == 0 else "linear/exp."


def _flag(name: str, yes: str) -> Callable:
    def value(kind, p) -> str:
        v = p.get(name)
        return "" if v is None or pd.isna(v) else (yes if v else "")

    return value


TABLE_COLUMNS = [
    TableColumn("fit", "Current fit (stored) or preview", lambda kind, p: kind),
    TableColumn(
        "function",
        "Peak function: EMG (exponentially modified Gaussian) or Pap-Papai",
        lambda kind, p: str(p.get("fit_shape") or "–"),
    ),
    TableColumn(
        "rt [s]", "Apex", lambda kind, p: _number(p.get("fit_rt_s", p.get("rt_s")), 2)
    ),
    TableColumn("± rt", "Standard error of the location [s]", _get("fit_rt_u_s", 2)),
    TableColumn("height", "Apex above the background", _get("fit_height")),
    TableColumn("± h", "Uncertainty of the height", _get("fit_height_u")),
    TableColumn("area", "Area of the peak function [counts s]", _get("fit_area")),
    TableColumn("± area", "Uncertainty of the area", _get("fit_area_u")),
    TableColumn("σ [s]", "Width (standard deviation of its Gaussian)", _get("fit_sigma_s", 2)),
    TableColumn(
        "tail",
        "Tail parameter: EMG τ/σ; Pap-Papai a (0: Gaussian)",
        _get("fit_tailing", 2),
    ),
    TableColumn("S/N", "Signal to noise (height over the limit of detection)", _get("fit_sn", 1)),
    TableColumn("baseline", "Background at the apex", _get("fit_baseline")),
    TableColumn("slope /s", "Slope of the background at the apex", _get("fit_baseline_slope", 3)),
    TableColumn(
        "background",
        "Background of the window: kept by the fit among those tried (preview);"
        " stored fits: only constant or not, from the slope",
        _background,
    ),
    TableColumn(
        "iter.",
        "Iterations of the least-squares fit kept (preview only, not stored)",
        _get("iterations", preview_only=True),
    ),
    TableColumn(
        "limit",
        "Stopped at the iteration limit: the parameters may not be converged",
        _flag("aborted", "stopped"),
    ),
    TableColumn(
        "line ratio",
        "Residual with the peak / with a straight line around it (small: a real peak;"
        " above 0.9 the default fit removes it)",
        _get("linear_ratio", 2, preview_only=True),
    ),
    TableColumn(
        "coverage",
        "Fraction of the scans within 1 σ of the apex that are measured",
        _get("coverage", 2, preview_only=True),
    ),
    TableColumn("points", "Points of the window fitted", _get("n_points", preview_only=True)),
    TableColumn(
        "override",
        "Fit override the stored peak was fitted with (none: the data directory's fit)",
        lambda kind, p: ""
        if kind == "preview"
        else ("" if pd.isna(p.get("fit_override_id")) else f"#{int(p['fit_override_id'])}"),
    ),
]


def _spin(lo, hi, value, decimals=2, step=0.1, suffix="") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _row(*widgets) -> QWidget:
    widget = QWidget()
    row = QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        row.addWidget(QLabel(w) if isinstance(w, str) else w)
    row.addStretch()
    return widget


def _number(value, digits=0) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.{digits}f}"


class RefitDialog(QDialog):
    """The fit override of a target on some masses: its window, peaks, fit parameters
    and the runs it applies to; previewed on this run, saved and applied."""

    def __init__(
        self,
        data: EsperData,
        measurement_id: int,
        compound_idx: int | None = None,
        mass=None,
        parent=None,
        target: str | None = None,
    ) -> None:
        """From a compound of the run (its target, if it is one), or from a ``target``
        (its compound in the run if found; else its expected masses and rt)."""
        super().__init__(parent)
        self.data = data
        # the run shown (the preview run), at first the one the dialog is opened on
        self.measurement_id = measurement_id
        self.opened_id = measurement_id
        # window and positions relative to the reference rt, kept while the preview run
        # has none
        self._relative: tuple | None = None
        self.reference: float | None = None
        if compound_idx is None and target is not None:
            compound_idx = data.find_target(measurement_id, target)
        self.compound_idx = compound_idx
        self.base = data.processing.fit
        self.saved = False  # runs fitted again: the caller reloads
        self._positions: list[float] = []  # absolute, in this run
        self._position_lines: list[pg.InfiniteLine] = []
        self._extraction = None  # (peaks, frame) of the run, read once
        run = data.measurements.loc[measurement_id]
        self.run_time = run["acq_datetime"]
        if compound_idx is not None:
            compound = data.compounds(measurement_id).loc[compound_idx]
            self.peaks = data.compound_peaks(measurement_id, compound_idx).sort_values(
                "height", ascending=False
            )
            if target is None:
                target = data.target_index.target_of(compound["main_name"])
            name = compound_label(compound["main_name"], compound_idx)
            title = f"{name} (#{compound_idx})"
        else:  # a target not found in the run
            self.peaks = data.peaks(measurement_id).iloc[:0]
            title = f"{target} (not found)"
        self.setWindowTitle(f"Refit {title} — {run['name']}")
        self.resize(1250, 780)

        left = QWidget()
        form = QVBoxLayout(left)
        form.addWidget(self._target_box(target))
        form.addWidget(self._masses_box(mass, target))
        form.addWidget(self._window_box())
        form.addWidget(self._peaks_box())
        form.addWidget(self._parameters_box())
        form.addWidget(self._scope_box(run))
        form.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(470)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self._preview_widget())
        splitter.setStretchFactor(1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.preview_button = buttons.addButton(
            "Preview", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.preview_button.clicked.connect(self.preview)
        self.delete_button = buttons.addButton(
            "Delete override…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.delete_button.clicked.connect(self._delete)
        self.save_button = buttons.addButton(
            "Save and refit…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.save_button.clicked.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(buttons)
        self._on_target_changed()

    # --- parts ----------------------------------------------------------------------

    def _target_box(self, target: str | None) -> QGroupBox:
        box = QGroupBox("Target")
        self.target_box = QComboBox()
        targets = sorted(self.data.targets.index)
        self.target_box.addItems(targets)
        if target in self.data.targets.index:
            self.target_box.setCurrentText(target)
        else:
            self.target_box.setCurrentIndex(-1)
        self.target_box.currentTextChanged.connect(lambda _: self._on_target_changed())
        self.reference_label = QLabel()
        self.reference_label.setWordWrap(True)
        self.existing_label = QLabel()
        self.existing_label.setWordWrap(True)
        layout = QFormLayout(box)
        layout.addRow("Assigned to:", self.target_box)
        layout.addRow(self.reference_label)
        layout.addRow(self.existing_label)
        return box

    def _masses_box(self, mass, target: str | None) -> QGroupBox:
        box = QGroupBox("Masses (one override each)")
        self.mass_list = QListWidget()
        self.mass_list.setMaximumHeight(110)
        # the masses of the compound, then the expected ions of the target
        ions = (
            []
            if target is None
            else self.data.target_masses(self.measurement_id, target)
        )
        masses = list(dict.fromkeys([*self.peaks["mass"], *ions]))
        if mass is not None and mass not in masses:
            masses.insert(0, mass)
        checked = mass if mass in masses else (masses[0] if masses else None)
        for m in masses:
            heights = self.peaks.loc[self.peaks["mass"] == m, "height"]
            text = mass_label(m) + (
                f"    H {heights.max():.0f}" if len(heights) else "    no peak"
            )
            item = QListWidgetItem(text)
            item.setData(MASS_ROLE, m)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if m == checked else Qt.CheckState.Unchecked
            )
            self.mass_list.addItem(item)
        self.mass_list.itemChanged.connect(lambda _: self._on_masses_changed())
        self.preview_mass = QComboBox()
        self.preview_mass.currentIndexChanged.connect(lambda _: self._on_preview_mass())
        layout = QFormLayout(box)
        layout.addRow(self.mass_list)
        layout.addRow("Preview:", self.preview_mass)
        return box

    def _window_box(self) -> QGroupBox:
        box = QGroupBox("Window fitted (in this run; follows the target in the others)")
        self.rt_from = _spin(0, 1e6, 0, 1, 0.5, " s")
        self.rt_to = _spin(0, 1e6, 0, 1, 0.5, " s")
        for spin in (self.rt_from, self.rt_to):
            spin.valueChanged.connect(self._on_window_spin)
        self.window_label = QLabel()
        layout = QVBoxLayout(box)
        layout.addWidget(_row("from", self.rt_from, "to", self.rt_to))
        layout.addWidget(self.window_label)
        return box

    def _peaks_box(self) -> QGroupBox:
        box = QGroupBox("Peaks")
        self.position_list = QListWidget()
        self.position_list.setMaximumHeight(90)
        self.position_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        from_fit = QPushButton("From the current fit")
        from_fit.setToolTip("The apexes of the current fit of this mass in the window")
        from_fit.clicked.connect(self._positions_from_fit)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_positions)
        clear = QPushButton("Clear")
        clear.setToolTip("No positions: the fit starts at the peaks of the extraction")
        clear.clicked.connect(lambda: self._set_positions([]))
        note = QLabel(
            "Approximate positions where the fit starts, one per peak (all kept):"
            " double click in the preview to add one, drag to move. None: the fit"
            " starts at the extracted peaks."
        )
        note.setWordWrap(True)
        self.tolerance = _spin(0, 30, 2.0, 2, 0.1, " s")
        self.tolerance.setSpecialValueText("fixed")
        self.default_tolerance = QCheckBox("default (3 scans)")
        self.default_tolerance.toggled.connect(self.tolerance.setDisabled)
        self.tolerance.setToolTip(
            "How far the fit can move each peak from its position; fixed: 0 (the"
            " location of the peak function: the apex of a tailing peak can still be"
            " after it, by its tail)"
        )
        self.min_peaks = QSpinBox()
        self.min_peaks.setRange(0, 20)
        self.min_peaks.setSpecialValueText("–")
        self.min_peaks.setToolTip(
            "At least this many peaks: added where the residual is largest, none removed"
        )
        self.max_peaks = QSpinBox()
        self.max_peaks.setRange(0, 20)
        self.max_peaks.setSpecialValueText("–")
        self.max_peaks.setToolTip(
            "At most this many peaks. –: with positions, exactly them; else up to 2"
            " more where the data show a coelution"
        )
        layout = QFormLayout(box)
        layout.addRow(self.position_list)
        layout.addRow(_row(from_fit, remove, clear))
        layout.addRow(note)
        layout.addRow("Peaks move by up to ±", _row(self.tolerance, self.default_tolerance))
        layout.addRow("Number of peaks:", _row("min", self.min_peaks, "max", self.max_peaks))
        return box

    def _parameters_box(self) -> QGroupBox:
        box = QGroupBox("Fit parameters")
        form = QFormLayout(box)
        self.shape = QComboBox()
        self.shape.addItems([str(s) for s in ShapeName])
        form.addRow("Peak function:", self.shape)
        self.background_checks = {}
        checks = []
        for name in BackgroundName:
            check = QCheckBox(str(name))
            self.background_checks[str(name)] = check
            checks.append(check)
        backgrounds = _row(*checks)
        backgrounds.setToolTip(
            "Several: fitted with the first, another kept where clearly better"
        )
        form.addRow("Backgrounds:", backgrounds)
        self.sigma_min = _spin(0.05, 100, DEFAULT_SIGMA_RANGE_S[0], 2, 0.1, " s")
        self.sigma_max = _spin(0.05, 100, DEFAULT_SIGMA_RANGE_S[1], 2, 0.5, " s")
        width = _row(self.sigma_min, "–", self.sigma_max)
        width.setToolTip(
            "Bounds of the width (σ of the peak function); the fit starts at the half"
            " width of the highest peak of the window"
        )
        form.addRow("Width σ:", width)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(10, 10000)
        form.addRow("Max iterations:", self.max_iterations)
        defaults = QPushButton("Data directory's fit")
        defaults.clicked.connect(
            lambda: self._fill_params(OverrideParams.from_config(self.base))
        )
        form.addRow(defaults)
        return box

    def _scope_box(self, run) -> QGroupBox:
        box = QGroupBox("Applies to the runs")
        # the runs with a date, in acquisition order, as the File box of the view
        runs = self.data.measurements
        runs = runs[runs["acq_datetime"].notna()]
        self._run_times = {
            int(mid): f"{time:%Y-%m-%dT%H:%M:%S}"
            for mid, time in runs["acq_datetime"].items()
        }
        items = [(name, int(mid)) for mid, name in runs["name"].items()]
        self.from_all = QCheckBox("from the first run")
        self.from_run = StepCombo("", min_width=200)
        self.to_open = QCheckBox("no end (also the future runs)")
        self.to_run = StepCombo("", min_width=200)
        for box_ in (self.from_run, self.to_run):
            box_.set_items(items, self.measurement_id)
            box_.selected.connect(lambda _: self._update_count())
            box_.text_entered.connect(lambda text, b=box_: self._on_run_text(b, text))
        self.from_all.toggled.connect(self.from_run.setDisabled)
        self.to_open.toggled.connect(self.to_run.setDisabled)
        for check in (self.from_all, self.to_open):
            check.toggled.connect(lambda _: self._update_count())
            check.setChecked(True)
        self.type_list = QListWidget()
        self.type_list.setMaximumHeight(90)
        types = sorted(self.data.measurements["sample_type"].dropna().astype(str).unique())
        for sample_type in types:
            item = QListWidgetItem(sample_type)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.type_list.addItem(item)
        self.type_list.itemChanged.connect(lambda _: self._update_count())
        self.count_label = QLabel()
        layout = QFormLayout(box)
        layout.addRow("From run:", self.from_run)
        layout.addRow("", self.from_all)
        layout.addRow("To run (included):", self.to_run)
        layout.addRow("", self.to_open)
        layout.addRow("Sample types:", self.type_list)
        layout.addRow(self.count_label)
        return box

    def _on_run_text(self, box: StepCombo, text: str) -> None:
        """A run typed in a run box: the first whose name contains it."""
        for i in range(box.combo.count()):
            if text in box.combo.itemText(i):
                box.set_current(box.combo.itemData(i))
                self._update_count()
                return

    def _run_at(self, iso: str, after: bool) -> int | None:
        """The run acquired at ``iso``, else the first after it (``after``) or the last
        before it."""
        time = pd.Timestamp(iso)
        times = pd.Series({mid: pd.Timestamp(t) for mid, t in self._run_times.items()})
        if times.empty:
            return None
        candidates = times[times >= time] if after else times[times <= time]
        if candidates.empty:  # none: the nearest end
            return int(times.idxmax() if after else times.idxmin())
        return int(candidates.idxmin() if after else candidates.idxmax())

    def _preview_widget(self) -> QWidget:
        widget = QWidget()
        self.plot = make_plot()
        self.plot.setMinimumHeight(380)
        self.region = pg.LinearRegionItem(
            brush=pg.mkBrush(0, 80, 200, 25), pen=pg.mkPen(PREVIEW_COLOR)
        )
        self.region.setZValue(-20)
        self.region.sigRegionChangeFinished.connect(self._on_region_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([c.header for c in TABLE_COLUMNS])
        for j, column in enumerate(TABLE_COLUMNS):
            self.table.horizontalHeaderItem(j).setToolTip(column.tooltip)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.preview_run = StepCombo("Preview run:", min_width=240)
        self.preview_run.setToolTip(
            "The runs the override applies to (and the one the dialog was opened on):"
            " the window and positions follow the target in each"
        )
        self.preview_run.selected.connect(self._set_run)
        self.preview_run.text_entered.connect(self._on_preview_run_text)
        layout = QVBoxLayout(widget)
        layout.addWidget(self.preview_run)
        layout.addWidget(self.plot, 3)
        layout.addWidget(self.result_label)
        layout.addWidget(self.table, 1)
        return widget

    # --- state ----------------------------------------------------------------------

    @property
    def target(self) -> str | None:
        text = self.target_box.currentText()
        return text if text in self.data.targets.index else None

    def masses(self) -> list:
        """The checked masses."""
        return [
            self.mass_list.item(i).data(MASS_ROLE)
            for i in range(self.mass_list.count())
            if self.mass_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _reference(self) -> float | None:
        target = self.target
        if target is None:
            return None
        return reference_rt(
            self.data.con,
            self.measurement_id,
            target,
            self.data.target_index,
            self.run_time,
        )

    def _existing(self, mass) -> FitOverride | None:
        """The override of the target on ``mass`` that applies to this run."""
        if self.target is None:
            return None
        sample_type = self.data.measurements.at[self.measurement_id, "sample_type"]
        for override in run_overrides(
            read_overrides(self.data.con, self.target), self.run_time, sample_type
        ):
            if float(override.mass) == float(mass):
                return override
        return None

    def relative(self) -> tuple[float, float, list[float]]:
        """The window and positions relative to the reference rt (of the run shown;
        without one, as they were in the last run that had one)."""
        if self.reference is None:
            return self._relative or (0.0, 0.0, [])
        reference = self.reference
        return (
            self.rt_from.value() - reference,
            self.rt_to.value() - reference,
            [p - reference for p in sorted(self._positions)],
        )

    def params(self) -> OverrideParams:
        """The fit parameters of the dialog, positions relative to the reference rt."""
        return OverrideParams(
            peak_shape=self.shape.currentText(),
            backgrounds=tuple(
                name for name, check in self.background_checks.items() if check.isChecked()
            )
            or ("constant",),
            sigma_range_s=(self.sigma_min.value(), self.sigma_max.value()),
            max_iterations=self.max_iterations.value(),
            positions=tuple(round(p, 3) for p in self.relative()[2]),
            position_tolerance=None
            if self.default_tolerance.isChecked()
            else self.tolerance.value(),
            min_peaks=self.min_peaks.value(),
            max_peaks=self.max_peaks.value(),
        )

    def _scope(self) -> dict:
        def iso(box: StepCombo) -> str | None:
            return self._run_times.get(box.combo.currentData())

        checked = [
            self.type_list.item(i).text()
            for i in range(self.type_list.count())
            if self.type_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        return {
            "valid_from": None if self.from_all.isChecked() else iso(self.from_run),
            "valid_to": None if self.to_open.isChecked() else iso(self.to_run),
            "sample_types": None
            if len(checked) == self.type_list.count()
            else tuple(checked),
        }

    def override(self, mass, override_id=None) -> FitOverride:
        """The override of the dialog on ``mass`` (window relative to the reference)."""
        lo, hi, _ = self.relative()
        return FitOverride(
            target=self.target or "",
            mass=float(mass),
            rt_from=round(lo, 3),
            rt_to=round(hi, 3),
            params=self.params(),
            override_id=override_id,
            **self._scope(),
        )

    def _runs(self, overrides: list[FitOverride]) -> list[tuple[int, str]]:
        """(measurement_id, name) of the fitted runs that one of ``overrides`` applies to."""
        runs = self.data.measurements
        if "peak_source" in runs:
            runs = runs[runs["peak_source"].isin(FITTED_PEAK_SOURCES)]
        return [
            (int(mid), row["name"])
            for mid, row in runs.iterrows()
            if any(o.applies(row["acq_datetime"], row["sample_type"]) for o in overrides)
        ]

    # --- filling --------------------------------------------------------------------

    def _fill_params(self, params: OverrideParams) -> None:
        self.shape.setCurrentText(str(params.peak_shape))
        for name, check in self.background_checks.items():
            check.setChecked(name in params.backgrounds)
        sigma = params.sigma_range_s or DEFAULT_SIGMA_RANGE_S
        self.sigma_min.setValue(sigma[0])
        self.sigma_max.setValue(sigma[1])
        self.max_iterations.setValue(params.max_iterations)
        self.default_tolerance.setChecked(params.position_tolerance is None)
        if params.position_tolerance is not None:
            self.tolerance.setValue(params.position_tolerance)
        self.min_peaks.setValue(params.min_peaks)
        self.max_peaks.setValue(params.max_peaks)

    def _fill_scope(self, override: FitOverride) -> None:
        self.from_all.setChecked(override.valid_from is None)
        if override.valid_from is not None:
            self.from_run.set_current(self._run_at(override.valid_from, after=True))
        self.to_open.setChecked(override.valid_to is None)
        if override.valid_to is not None:
            self.to_run.set_current(self._run_at(override.valid_to, after=False))
        for i in range(self.type_list.count()):
            item = self.type_list.item(i)
            shown = override.sample_types is None or item.text() in override.sample_types
            item.setCheckState(Qt.CheckState.Checked if shown else Qt.CheckState.Unchecked)
        self._update_count()

    def _default_window(self) -> tuple[float, float]:
        reference = self.reference
        starts = self.peaks["start_s"].to_numpy(dtype=float)
        stops = self.peaks["stop_s"].to_numpy(dtype=float)
        if np.isfinite(starts).any() and np.isfinite(stops).any():
            return np.nanmin(starts) - RT_MARGIN_S, np.nanmax(stops) + RT_MARGIN_S
        if self.target is not None:  # its identification window (GCwerks peakid)
            window = self.data.target_window(self.measurement_id, self.target)
            if window is not None:
                return window[0] - RT_MARGIN_S / 2, window[1] + RT_MARGIN_S / 2
        if len(self.peaks):
            center = float(self.peaks["rt_s"].mean())
        else:
            center = reference if reference is not None else 0.0
        return center - RT_HALF_WINDOW_S, center + RT_HALF_WINDOW_S

    def _on_target_changed(self) -> None:
        self.reference = self._reference()
        usable = self._update_reference_label()
        for button in (self.preview_button, self.save_button):
            button.setEnabled(usable)
        self._on_masses_changed(load=True)

    def _update_reference_label(self) -> bool:
        """The reference rt of the target in the run shown; whether it has one."""
        target = self.target
        run = self.data.measurements.at[self.measurement_id, "name"]
        if target is None:
            self.reference_label.setText(
                "<i>Choose a target: an override is assigned to a target.</i>"
            )
        elif self.reference is None:
            self.reference_label.setText(
                f"<i>{target} has no rt in {run} (not expected, not identified):"
                " choose another preview run.</i>"
            )
        else:
            self.reference_label.setText(
                f"Reference rt of {target} in {run}: {self.reference:.2f} s"
                " (window and positions are saved relative to it)"
            )
        return target is not None and self.reference is not None

    def _set_run(self, measurement_id) -> None:
        """Show another run: the window and positions moved to its reference rt."""
        if measurement_id is None or int(measurement_id) == self.measurement_id:
            return
        if self.reference is not None:
            self._relative = self.relative()
        self.measurement_id = int(measurement_id)
        self.run_time = self.data.measurements.at[self.measurement_id, "acq_datetime"]
        self._extraction = None
        self.reference = self._reference()
        usable = self._update_reference_label()
        for button in (self.preview_button, self.save_button):
            button.setEnabled(usable)
        if usable and self._relative is not None:
            lo, hi, positions = self._relative
            self._set_window(self.reference + lo, self.reference + hi)
            self._set_positions([self.reference + p for p in positions])
        self._on_masses_changed()

    def _on_preview_run_text(self, text: str) -> None:
        self._on_run_text(self.preview_run, text)
        self._set_run(self.preview_run.combo.currentData())

    def _fill_preview_runs(self, runs: list[tuple[int, str]]) -> None:
        """The runs of the scope in the preview box, with the run opened and the one
        shown (marked when outside the scope)."""
        if not hasattr(self, "preview_run"):
            return
        inside = {mid for mid, _ in runs}
        shown = inside | {self.opened_id, self.measurement_id}
        names = self.data.measurements["name"]
        items = [
            (name if mid in inside else f"{name}   (outside the scope)", int(mid))
            for mid, name in names.items()
            if mid in shown
        ]
        self.preview_run.set_items(items, self.measurement_id)

    def _on_masses_changed(self, load: bool = False) -> None:
        masses = self.masses()
        current = self.preview_mass.currentData()
        self.preview_mass.blockSignals(True)
        self.preview_mass.clear()
        for m in masses:
            self.preview_mass.addItem(mass_label(m), m)
        index = self.preview_mass.findData(current)
        self.preview_mass.setCurrentIndex(max(index, 0))
        self.preview_mass.blockSignals(False)
        existing = [o for m in masses if (o := self._existing(m)) is not None]
        self.delete_button.setEnabled(bool(existing))
        self.existing_label.setText(
            "Overrides applying to this run, replaced when saved: "
            + ", ".join(f"#{o.override_id} ({mass_label(o.mass)})" for o in existing)
            if existing
            else "No override of these masses applies to this run yet."
        )
        if load:
            self._load(existing[0] if existing else None)
        self._update_count()
        self._on_preview_mass()

    def _load(self, override: FitOverride | None) -> None:
        """The dialog from an override, else from the fit of the data directory."""
        if override is None or self.reference is None:
            self._fill_params(OverrideParams.from_config(self.base))
            lo, hi = self._default_window()
            self._set_window(lo, hi)
            self._set_positions([])
            return
        self._fill_params(override.params)
        self._fill_scope(override)
        self._set_window(*override.window(self.reference))
        self._set_positions(list(override.starts(self.reference)))

    def _update_count(self) -> None:
        if not hasattr(self, "count_label") or self.target is None:
            return
        if self.reference is None and self._relative is None:
            self.count_label.setText("")
            return
        runs = self._runs([self.override(m) for m in self.masses()])
        self.count_label.setText(f"<b>{len(runs)}</b> fitted runs")
        self._fill_preview_runs(runs)

    # --- window and positions -------------------------------------------------------

    def _set_window(self, lo: float, hi: float) -> None:
        for spin, value in ((self.rt_from, lo), (self.rt_to, hi)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)
        self._update_window_label()

    def _update_window_label(self) -> None:
        if self.reference is None:
            self.window_label.setText("")
            return
        lo, hi = self.rt_from.value(), self.rt_to.value()
        self.window_label.setText(
            f"relative to the target: {lo - self.reference:+.1f} to"
            f" {hi - self.reference:+.1f} s"
        )

    def _on_window_spin(self) -> None:
        self._set_window(self.rt_from.value(), self.rt_to.value())

    def _on_region_moved(self) -> None:
        self._set_window(*self.region.getRegion())

    def _set_positions(self, positions: list[float]) -> None:
        self._positions = sorted(float(p) for p in positions)
        self.position_list.clear()
        for p in self._positions:
            self.position_list.addItem(f"{p:.2f} s")
        for line in self._position_lines:
            self.plot.removeItem(line)
        self._position_lines = []
        for p in self._positions:
            line = pg.InfiniteLine(
                pos=p,
                angle=90,
                movable=True,
                pen=pg.mkPen(POSITION_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(POSITION_COLOR, width=3),
            )
            line.sigPositionChangeFinished.connect(self._on_line_moved)
            self.plot.addItem(line)
            self._position_lines.append(line)

    def _on_line_moved(self) -> None:
        self._set_positions([line.value() for line in self._position_lines])

    def _positions_from_fit(self) -> None:
        current = self._current_peaks()
        self._set_positions(list(current["fit_rt_s"].dropna()))

    def _remove_positions(self) -> None:
        rows = {index.row() for index in self.position_list.selectedIndexes()}
        self._set_positions([p for i, p in enumerate(self._positions) if i not in rows])

    def _on_plot_clicked(self, event) -> None:
        if not event.double():
            return
        point = self.plot.getViewBox().mapSceneToView(event.scenePos())
        self._set_positions([*self._positions, point.x()])

    # --- preview --------------------------------------------------------------------

    def _run_data(self):
        if self._extraction is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                peaks, _ = extraction(self.data.con, self.measurement_id)
                frame = self.data.chromatogram_frame(self.measurement_id)
            finally:
                QApplication.restoreOverrideCursor()
            self._extraction = (peaks, frame)
        return self._extraction

    def _current_peaks(self) -> pd.DataFrame:
        """The peaks of the preview mass in the window, as fitted now."""
        mass = self.preview_mass.currentData()
        peaks = self.data.peaks(self.measurement_id)
        lo, hi = self.rt_from.value(), self.rt_to.value()
        on_mass = peaks[peaks["mass"] == mass]
        rt = on_mass["fit_rt_s"].fillna(on_mass["rt_s"])
        return on_mass[rt.between(lo, hi)]

    def _on_preview_mass(self) -> None:
        """Draw the chromatogram of the preview mass and its current fit."""
        self.plot.clear()
        self.legend.clear()
        self.table.setRowCount(0)
        self.result_label.setText("")
        mass = self.preview_mass.currentData()
        self.plot.addItem(self.region)
        for line in self._position_lines:
            self.plot.addItem(line)
        if mass is None:
            return
        series = self.data.chromatogram(self.measurement_id, mass)
        drawn = self.data.chromatogram_drawn(self.measurement_id, mass)
        if series is None or series.empty:
            self.result_label.setText(f"No chromatogram of {mass_label(mass)}.")
            return
        curve = chromatogram_curve(series, drawn, mass)
        item = draw_curve(self.plot, curve)
        self.legend.addItem(item, curve.label)
        current = self._current_peaks()
        for item in fit_integration_items(curve.t, current) + fit_sum_items(
            curve.t, current
        ):
            self.plot.addItem(item)
        if self.reference is not None:
            self.plot.addItem(
                pg.InfiniteLine(
                    pos=self.reference,
                    angle=90,
                    pen=pg.mkPen(TARGET_COLOR, width=1, style=Qt.PenStyle.DotLine),
                    label=self.target,
                    labelOpts={"position": 0.95, "color": TARGET_COLOR},
                )
            )
        lo, hi = self.rt_from.value(), self.rt_to.value()
        inside = (curve.t >= lo - PLOT_MARGIN_S) & (curve.t <= hi + PLOT_MARGIN_S)
        self.plot.setXRange(lo - PLOT_MARGIN_S, hi + PLOT_MARGIN_S, padding=0)
        if inside.any():
            bottom, top = curve.y[inside].min(), curve.y[inside].max()
            span = max(top - bottom, 1.0)
            self.plot.setYRange(bottom - 0.05 * span, top + 0.1 * span, padding=0)
        self._fill_table(current, None)

    def preview(self) -> None:
        """Fit the window of the preview mass with the parameters, and draw it."""
        mass = self.preview_mass.currentData()
        if mass is None or self.reference is None:
            return
        self._on_preview_mass()
        peaks, frame = self._run_data()
        if frame is None:
            self.result_label.setText("No chromatograms for this run.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fitted = override_fit(
                frame,
                peaks,
                self.override(mass),
                self.reference,
                self.base,
                diagnostics=True,
            )
        except Exception as e:
            self.result_label.setText(f"<b>The fit failed:</b> {e!r}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if peaks[peaks["mass"] == mass].empty:
            self.result_label.setText(
                f"No extracted peak on {mass_label(mass)}: nothing to refit."
            )
            return
        summary = self._fit_summary(fitted)
        fitted = fitted.assign(compound_idx=pd.NA)
        t = frame.index.to_numpy(dtype=float)
        items = fit_fill_items(t, fitted, PREVIEW_COLOR)
        sums = fit_sum_items(t, fitted, PREVIEW_SUM_COLOR)
        for item in items + sums:
            self.plot.addItem(item)
        if items:
            self.legend.addItem(items[-1], "preview")
        if sums:
            self.legend.addItem(sums[0], "preview: sum")
        self._fill_table(self._current_peaks(), fitted)
        self.result_label.setText(
            (
                f"Preview: {len(fitted)} peak(s) (filled, blue; their sum: magenta);"
                " current fit: red dashed (sum: purple)."
                if len(fitted)
                else "<b>Preview: no peak fitted</b> (the fit was rejected)."
            )
            + (f"<br>{summary}" if summary else "")
        )

    def _fill_table(self, current: pd.DataFrame, preview: pd.DataFrame | None) -> None:
        rows = [("current", p) for _, p in current.sort_values("rt_s").iterrows()]
        if preview is not None:
            rows += [("preview", p) for _, p in preview.sort_values("fit_rt_s").iterrows()]
        self.table.setRowCount(len(rows))
        for i, (kind, p) in enumerate(rows):
            for j, column in enumerate(TABLE_COLUMNS):
                item = QTableWidgetItem(column.value(kind, p))
                if kind == "preview":
                    item.setForeground(pg.mkColor(PREVIEW_COLOR))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

    def _fit_summary(self, fitted: pd.DataFrame) -> str:
        """The least-squares fits of the preview and its start positions."""
        attrs = fitted.attrs
        if "n_least_squares" not in attrs:
            return ""
        starts = ", ".join(f"{s:.2f}" for s in attrs.get("starts", []))
        if attrs.get("start_sigma") is not None:
            starts += f" (σ from {attrs['start_sigma']:.2f})"
        text = (
            f"{attrs['n_least_squares']} least-squares fits (with the other backgrounds"
            f" and extra peaks tried), {attrs['iterations_total']} iterations in all"
        )
        if attrs.get("n_aborted"):
            text += (
                f", <b>{attrs['n_aborted']} stopped at the limit"
                f" ({self.max_iterations.value()} iterations)</b>"
            )
        return f"{text}. Starts: {starts} s."

    # --- saving ---------------------------------------------------------------------

    def _save(self) -> None:
        masses = self.masses()
        if not masses or self.target is None or self.reference is None:
            return
        if self.rt_to.value() <= self.rt_from.value():
            QMessageBox.warning(self, "esper", "The window is empty.")
            return
        scope = self._scope()
        if scope["valid_from"] and scope["valid_to"] and scope["valid_to"] < scope["valid_from"]:
            QMessageBox.warning(self, "esper", "The last run is before the first one.")
            return
        old = {m: self._existing(m) for m in masses}
        new = [
            self.override(m, None if old[m] is None else old[m].override_id)
            for m in masses
        ]
        runs = self._runs(new + [o for o in old.values() if o is not None])
        answer = QMessageBox.question(
            self,
            "Save the fit override",
            f"Save the override of {self.target} on"
            f" {', '.join(mass_label(m) for m in masses)} and fit again {len(runs)}"
            " run(s)?\n\nTheir peaks in the window are replaced by the fit of the"
            " override (the extraction is kept: deleting the override and fitting again"
            " brings the default fit back). The future runs matching it are fitted with"
            " it too.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        con = db.connect(self.data.data_dir, create=True)  # schema with fit_override
        try:
            for override in new:
                save_override(con, override)
        finally:
            con.close()
        self._refit(runs)

    def _delete(self) -> None:
        existing = [o for m in self.masses() if (o := self._existing(m)) is not None]
        if not existing:
            return
        runs = self._runs(existing)
        answer = QMessageBox.question(
            self,
            "Delete the fit override",
            "Delete "
            + ", ".join(f"#{o.override_id} ({mass_label(o.mass)})" for o in existing)
            + f" and fit again the {len(runs)} run(s) it applied to (default fit)?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        con = db.connect(self.data.data_dir, create=True)
        try:
            for override in existing:
                delete_override(con, override.override_id)
        finally:
            con.close()
        self._refit(runs)

    def _refit(self, runs: list[tuple[int, str]]) -> None:
        progress = QProgressDialog("Fitting the runs…", "Stop", 0, len(runs), self)
        progress.setWindowTitle("esper")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        def step(i: int, name: str) -> bool:
            progress.setValue(i)
            progress.setLabelText(f"Fitting {name} ({i + 1} of {len(runs)})…")
            QApplication.processEvents()
            return not progress.wasCanceled()

        done, failed = refit_runs(self.data.data_dir, runs, step)
        progress.setValue(len(runs))
        self.saved = True
        self.data.reload()
        message = f"{len(done)} of {len(runs)} run(s) fitted again."
        if len(done) + len(failed) < len(runs):
            message += (
                "\nStopped: the others keep their fit until fitted again"
                " (esper process --overrides)."
            )
        if failed:
            message += "\nFailed: " + ", ".join(name for name, _ in failed)
        QMessageBox.information(self, "esper", message)
        self._extraction = None
        self._on_masses_changed()

