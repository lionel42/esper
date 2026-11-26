"""What an axis of the results plot shows: a variable of an element of the compounds.

An element is what the value is of: the measurement, the compound, or one of its ions
(main, 2nd, ...: those of the compound plotted, ``results_elements``); each has its own
variables (e.g. the area of a peak is of an ion, the score of the compound). Both axes are
chosen the same way, with an ``AxisSelector``.

To compare compounds, an axis also names its compound (``Axis.compound``): its values
are then in the columns of that compound in ``EsperData.results_by_run``.

An axis can be a ratio: divided by another one (``Axis.over``, the "÷" of its selector),
e.g. an ion over another, the compound over its main ion, or (comparing compounds) a
compound over another.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QWidget

from esper.gui.data import COMPOUND_SEP
from esper.gui.plotting import time_values
from esper.target_index import unit_mass


@dataclass(frozen=True)
class Variable:
    label: str
    column: str  # of the element (an ion: after its prefix)
    time: bool = False  # a datetime column, on a date axis
    tip: str = ""  # in the list of the variables


@dataclass(frozen=True)
class Element:
    key: str
    label: str
    variables: tuple[Variable, ...]
    prefix: str = ""  # of the columns of its variables
    tip: str = ""

    def variable(self, label: str) -> Variable | None:
        return next((v for v in self.variables if v.label == label), None)


@dataclass(frozen=True)
class Axis:
    element: Element
    variable: Variable
    # Comparing compounds: the compound of the axis (None: the compound plotted; the
    # run has none)
    compound: str | None = None
    # A ratio: divided by this axis (not of the run)
    over: Axis | None = None

    @property
    def column(self) -> str:
        """The column of the values (of a ratio: of its numerator)."""
        column = self.element.prefix + self.variable.column
        if self.compound is None:
            return column
        return f"{self.compound}{COMPOUND_SEP}{column}"

    @property
    def time(self) -> bool:
        return self.variable.time and self.over is None

    @property
    def label(self) -> str:
        """'time' (of the run), 'Compound area', 'Ion 1 (119) area'; comparing:
        'SF6 Ion 1 (89) area'; a ratio: 'Ion 2 (87) area / Ion 1 (85) area'."""
        if self.element.key == "measurement":
            label = self.variable.label
        else:
            label = f"{self.element.label} {self.variable.label}"
            if self.compound is not None:
                label = f"{self.compound} {label}"
        return label if self.over is None else f"{label} / {self.over.label}"

    def values(self, df: pd.DataFrame) -> np.ndarray:
        """The values of the rows (a time: seconds since epoch, for a date axis; a
        ratio: NaN where divided by 0)."""
        if self.variable.time:
            values = time_values(df, self.column)
        else:
            values = pd.to_numeric(df[self.column], errors="coerce").to_numpy(float)
        if self.over is None:
            return values
        over = self.over.values(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(over != 0, values / over, np.nan)

    def format(self, value: float) -> str:
        """A value of the axis, for the tooltips."""
        if self.time:
            return f"{pd.Timestamp(value, unit='s'):%Y-%m-%d %H:%M}"
        return f"{value:.4g}"


# The variables have the same label on every element where they mean the same (the
# compound: of all its peaks; an ion: of its peak), so that changing the element keeps
# the variable. The values of the fit have the plain names, those of the peak detection
# (before the fit) end with 'detected'.
MEASUREMENT = Element(
    "measurement",
    "Run",
    (
        Variable("time", "acq_datetime", time=True, tip="Start of the acquisition"),
        Variable("position", "measurement_pos", tip="Position in the runs, by time"),
        Variable("id", "measurement_id", tip="measurement_id in the database"),
        Variable("port", "port", tip="Inlet port"),
    ),
    tip="The run of the compound",
)
COMPOUND = Element(
    "compound",
    "Compound",
    (
        Variable("area", "fit_area_sum", tip="Sum of the fitted areas of its peaks"),
        Variable(
            "area ±",
            "fit_area_u_sum",
            tip="Uncertainty of the area: of its peaks, added in quadrature",
        ),
        Variable("area ± %", "fit_area_rel_u", tip="Relative uncertainty of the area"),
        Variable(
            "rt [s]",
            "fit_rt_mean_s",
            tip="Mean of the fitted rts of its peaks (detected where not fitted)",
        ),
        Variable("RI", "retention_index", tip="Retention index"),
        Variable("score", "best_score", tip="Score of the best library match"),
        Variable(
            "score + RI",
            "best_score_combined",
            tip="Score combined with the retention index difference",
        ),
        Variable("peaks", "n_peaks", tip="Number of peaks (masses)"),
        Variable("fitted peaks", "n_fitted", tip="Number of fitted peaks"),
        Variable(
            "prominence detected",
            "prominence_sum",
            tip="Sum of the prominences of its peaks (peak detection)",
        ),
    ),
    tip="The compound: all its peaks",
)
# Variables of an ion: of its most prominent peak
ION_VARIABLES = (
    Variable("area", "fit_area", tip="Fitted area"),
    Variable("area ±", "fit_area_u", tip="Uncertainty of the fitted area"),
    Variable("area ± %", "fit_area_rel_u", tip="Relative uncertainty of the area"),
    Variable("height", "fit_height", tip="Fitted height, above the baseline"),
    Variable("S/N", "fit_sn", tip="Signal to noise of the fit"),
    Variable("rt [s]", "fit_rt_s", tip="Fitted apex"),
    Variable("σ [s]", "fit_sigma_s", tip="Fitted width (sigma of its Gaussian)"),
    Variable("tailing", "fit_tailing", tip="Tail parameter (EMG: τ/σ)"),
    Variable("baseline", "fit_baseline", tip="Fitted background at the apex"),
    Variable("m/z", "mass", tip="Mass of the peak"),
    Variable("prominence detected", "prominence", tip="Prominence (peak detection)"),
    Variable("height detected", "height", tip="Height (peak detection)"),
    Variable("rt detected [s]", "rt_s", tip="Apex (peak detection)"),
    Variable(
        "FWHM detected [s]", "width_s", tip="Width at half maximum (peak detection)"
    ),
)


def _ordinal(k: int) -> str:
    suffix = (
        "th" if 10 <= k % 100 < 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    )
    return f"{k}{suffix}"


def ion_element(k: int, mass: int | None = None) -> Element:
    """Ion ``k`` (1: the main ion) of the compounds; ``mass``: its unit mass when the
    same in all the runs (in its label)."""
    return Element(
        f"ion{k}",
        f"Ion {k}" if mass is None else f"Ion {k} ({mass})",
        ION_VARIABLES,
        prefix=f"ion{k}_",
        tip=f"Ion {k} of the target at the time of the run (by rank of its ions); "
        f"of a compound that is no target, its {_ordinal(k)} mass by prominence",
    )


def results_elements(df: pd.DataFrame | None) -> tuple[Element, ...]:
    """The elements of results (``EsperData.results``): the measurement, the compound and
    its ions with a peak in one of the runs at least (the main ion always)."""
    ions = []
    k = 1
    while f"ion{k}_mass" in (df.columns if df is not None else ()) or k == 1:
        masses = (
            df[f"ion{k}_mass"].dropna().map(unit_mass).unique()
            if df is not None and f"ion{k}_mass" in df
            else []
        )
        if k == 1 or len(masses):
            ions.append(ion_element(k, int(masses[0]) if len(masses) == 1 else None))
        k += 1
    return (MEASUREMENT, COMPOUND, *ions)


RESULTS_ELEMENTS = results_elements(None)
RESULTS_X = Axis(MEASUREMENT, MEASUREMENT.variables[0])
RESULTS_Y = Axis(COMPOUND, COMPOUND.variable("area"))
# What an axis is divided by when made a ratio, at first
RATIO_OVER = Axis(ion_element(2), ion_element(2).variable("area"))


def _ratio_elements(elements: tuple[Element, ...]) -> tuple[Element, ...]:
    """The elements an axis can be divided by: those of the compound."""
    return tuple(e for e in elements if e.key != MEASUREMENT.key)


class AxisSelector(QWidget):
    """Label, element and variable of an axis; the variables are those of the element
    (the same variable kept when the new element has it).

    ``optional``: the label is a checkbox, the axis is shown only when checked (``value``:
    None otherwise); ``collapsed``: its boxes hidden while not checked. Comparing
    compounds (``set_compare``): a compound too, before the element (none for the
    measurement); the elements are then those of that compound (``elements_of``).
    ``ratio``: a "÷" after it, to divide the axis by another (a selector of its own,
    ``over``; not an axis of the run). ``changed``: the new ``value``.
    """

    changed = Signal(object)

    def __init__(
        self,
        label: str,
        elements: tuple[Element, ...],
        axis: Axis,
        optional: bool = False,
        elements_of: Callable[[str], tuple[Element, ...]] | None = None,
        parent=None,
        ratio: bool = False,
        collapsed: bool = False,
    ) -> None:
        super().__init__(parent)
        self.elements_of = elements_of
        self.compare = False
        self.collapsed = collapsed
        self._quiet = False  # the changes of ``over`` not emitted (emitted once after)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.check: QCheckBox | None = None
        if optional:
            self.check = QCheckBox(label)
            self.check.setToolTip(f"Show {label.rstrip(':')} on an axis of its own")
            layout.addWidget(self.check)
        else:
            layout.addWidget(QLabel(label))
        self.compound_combo = QComboBox()
        self.compound_combo.setEditable(True)
        self.compound_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.compound_combo.setMinimumWidth(140)
        self.compound_combo.setMaxVisibleItems(30)
        self.compound_combo.setToolTip("The compound of this axis")
        self.compound_combo.hide()
        self.element_combo = QComboBox()
        self.element_combo.setMaxVisibleItems(25)
        # the labels of the ions in full (their m/z)
        self.element_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._fill_elements(elements)
        self.variable_combo = QComboBox()
        self.variable_combo.setMaxVisibleItems(20)
        layout.addWidget(self.compound_combo)
        layout.addWidget(self.element_combo)
        layout.addWidget(self.variable_combo)
        self.over: AxisSelector | None = None
        if ratio:
            self.over = AxisSelector(
                "÷",
                _ratio_elements(elements),
                RATIO_OVER,
                optional=True,
                elements_of=elements_of
                and (lambda name: _ratio_elements(elements_of(name))),
                collapsed=True,
            )
            self.over.check.setToolTip(
                "Divide by another variable: a ratio, e.g. of two ions, of the compound"
                " to an ion, or (comparing) of two compounds"
            )
            layout.addSpacing(4)
            layout.addWidget(self.over)
            self.over.changed.connect(self._on_over_changed)
        self._emitted: Axis | None = None
        self.set_axis(axis)
        self.compound_combo.activated.connect(self._on_compound_changed)
        self.element_combo.currentIndexChanged.connect(self._on_element_changed)
        self.variable_combo.currentIndexChanged.connect(self._emit)
        if self.check is not None:
            self.check.toggled.connect(self._on_checked)
            self._on_checked(False)

    @property
    def value(self) -> Axis | None:
        """The axis, None when not checked (optional)."""
        if self.check is not None and not self.check.isChecked():
            return None
        return self.axis

    def set_checked(self, checked: bool) -> None:
        if self.check is not None:
            self.check.setChecked(checked)

    @property
    def parts(self) -> list[AxisSelector]:
        """This selector, and the one of what it is divided by."""
        return [self] if self.over is None else [self, self.over]

    def _update_enabled(self) -> None:
        shown = self.check is None or self.check.isChecked()
        of_run = self.element.key == MEASUREMENT.key
        if self.collapsed:
            self.element_combo.setVisible(shown)
            self.variable_combo.setVisible(shown)
            self.compound_combo.setVisible(shown and self.compare)
        self.element_combo.setEnabled(shown)
        self.variable_combo.setEnabled(shown)
        self.compound_combo.setEnabled(shown and not of_run)
        if self.over is not None:
            self.over.setVisible(not of_run)
            self.over.setEnabled(shown)

    def _on_checked(self, checked: bool) -> None:
        self._update_enabled()
        self._emit()

    @property
    def element(self) -> Element:
        return self.elements[self.element_combo.currentData()]

    @property
    def compound(self) -> str | None:
        """The compound chosen when comparing compounds (None otherwise, or none)."""
        if not self.compare:
            return None
        return self.compound_combo.currentText() or None

    @property
    def axis(self) -> Axis:
        element = self.element
        variable = element.variable(self.variable_combo.currentText())
        of_run = element.key == MEASUREMENT.key
        compound = None if of_run else self.compound
        over = None if self.over is None or of_run else self.over.value
        return Axis(element, variable or element.variables[0], compound, over)

    def _on_over_changed(self, _value) -> None:
        if not self._quiet:
            self._emit()

    @contextmanager
    def _over_quiet(self):
        """The changes of ``over`` not emitted (``changed`` emitted once after)."""
        self._quiet = True
        try:
            yield
        finally:
            self._quiet = False

    # --- comparing compounds ----------------------------------------------------------

    def set_compounds(self, names: list[str]) -> None:
        """The compounds to choose from (the one shown kept)."""
        if self.over is not None:
            self.over.set_compounds(names)
        combo = self.compound_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        combo.setCurrentIndex(combo.findText(current))
        combo.blockSignals(False)

    def set_compare(self, compare: bool, compound: str | None = None) -> None:
        """Compare compounds or not; ``compound``: shown if none is (e.g. the one
        selected)."""
        self.compare = compare
        if self.over is not None:
            with self._over_quiet():
                self.over.set_compare(compare, compound)
        self.compound_combo.setVisible(
            compare and (not self.collapsed or self.value is not None)
        )
        if compare and not self.compound_combo.currentText() and compound:
            self._show_compound(compound)
        if compare and self.compound is not None:
            self._load_elements(self.compound)
        self._update_enabled()
        self._emit()

    def set_compound(self, compound: str) -> None:
        """Show a compound (with its elements)."""
        self._show_compound(compound)
        self._on_compound_changed()

    def _show_compound(self, compound: str) -> None:
        combo = self.compound_combo
        combo.blockSignals(True)
        index = combo.findText(compound)
        if index < 0:
            combo.addItem(compound)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _on_compound_changed(self, *_) -> None:
        if self.compound is not None:
            self._load_elements(self.compound)
        self._emit()

    def _load_elements(self, compound: str) -> None:
        """The elements of a compound (its ions), the axis kept as far as it can."""
        if self.elements_of is not None:
            current = self.axis
            self._fill_elements(self.elements_of(compound))
            self._set_axis(current)

    # --- elements and variables -------------------------------------------------------

    def set_elements(self, elements: tuple[Element, ...]) -> None:
        """Change the elements (e.g. the ions of another compound), keeping the element
        shown when there is one of the same key, else the main ion (if an ion was shown)
        or the first one; ``changed`` emitted if the axis changes."""
        current = self.axis
        if self.over is not None:
            with self._over_quiet():
                self.over.set_elements(_ratio_elements(elements))
        self._fill_elements(elements)
        self.set_axis(current)

    def _fill_elements(self, elements: tuple[Element, ...]) -> None:
        self.elements = {e.key: e for e in elements}
        combo = self.element_combo
        combo.blockSignals(True)
        combo.clear()
        for i, element in enumerate(elements):
            combo.addItem(element.label, element.key)
            combo.setItemData(i, element.tip, Qt.ItemDataRole.ToolTipRole)
        combo.blockSignals(False)

    def set_axis(self, axis: Axis) -> None:
        """Show an axis (its element: of the same key among the elements; its variable,
        else the first of the element; its compound if it has one); ``changed`` emitted
        if it changes. An element not among them: the main ion if it is an ion, else the
        first element."""
        if axis.compound is not None and axis.compound != self.compound:
            self._show_compound(axis.compound)
            if self.compare:
                self._load_elements(axis.compound)
        self._set_axis(axis)
        if self.over is not None:
            with self._over_quiet():
                if axis.over is not None:
                    self.over.set_axis(axis.over)
                self.over.set_checked(axis.over is not None)
        self._emit()

    def _set_axis(self, axis: Axis) -> None:
        key = axis.element.key
        if key not in self.elements:
            fallback = "ion1" if key.startswith("ion") else None
            key = fallback if fallback in self.elements else next(iter(self.elements))
        self.element_combo.blockSignals(True)
        self.element_combo.setCurrentIndex(self.element_combo.findData(key))
        self.element_combo.blockSignals(False)
        self._fill_variables(axis.variable.label)
        self._update_enabled()

    def _fill_variables(self, keep: str) -> None:
        combo = self.variable_combo
        combo.blockSignals(True)
        combo.clear()
        for i, variable in enumerate(self.element.variables):
            combo.addItem(variable.label)
            combo.setItemData(i, variable.tip, Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(max(combo.findText(keep), 0))
        combo.blockSignals(False)

    def _on_element_changed(self) -> None:
        self._fill_variables(self.variable_combo.currentText())
        self._update_enabled()
        self._emit()

    def _emit(self) -> None:
        value = self.value
        if value != self._emitted:
            self._emitted = value
            self.changed.emit(value)
