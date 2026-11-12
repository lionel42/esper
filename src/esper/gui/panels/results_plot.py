"""One compound across all measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QToolButton

from esper.gui.data import COMPOUND_SEP, EsperData
from esper.gui.panels.run_scatter import RunScatterPanel
from esper.gui.plot_axes import (
    RESULTS_ELEMENTS,
    RESULTS_X,
    RESULTS_Y,
    Axis,
    AxisSelector,
    ion_element,
    results_elements,
)
from esper.gui.plotting import make_highlight
from esper.gui.selection import Selection
from esper.gui.settings import settings_manager
from esper.gui.style import CROSS, runtype_colors

# The other y axes, on the right (the first one: next to the plot)
EXTRA_YS = ("y2", "y3")
# Symbol of the points of each y, and its sign in the labels
Y_SYMBOLS = {"y": CROSS, "y2": "o", "y3": "t"}
Y_SIGNS = {"y": "✕", "y2": "○", "y3": "△"}
# Shown on the other y axes when they are checked first
EXTRA_DEFAULTS = {
    "y2": Axis(ion_element(1), ion_element(1).variable("fit area")),
    "y3": Axis(ion_element(1), ion_element(1).variable("fit S/N")),
}
POINT_COLUMNS = ["measurement_id", "compound_idx", "name", "rt_mean_s"]


class _RightAxis:
    """A y axis on the right of a plot, with a view box of its own (x linked to the
    plot's; its y range fits the points visible). The mouse pans and zooms the plot."""

    def __init__(self, plot_item: pg.PlotItem, column: int) -> None:
        self.plot_item = plot_item
        self.column = column  # in the layout of the plot item (2: its own right axis)
        self.view = pg.ViewBox(enableMenu=False)
        self.view.setMouseEnabled(x=False, y=False)
        self.view.setXLink(plot_item.vb)
        self.view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.view.setAutoVisible(y=True)
        plot_item.scene().addItem(self.view)
        self.highlight = make_highlight()
        self.view.addItem(self.highlight)
        self.axis: pg.AxisItem | None = None
        self.items: list = []
        self.plotted: pd.DataFrame | None = None
        plot_item.vb.sigResized.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        self.view.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.view.linkedViewChanged(self.plot_item.vb, self.view.XAxis)

    def show(self, label: str | None, time: bool = False) -> None:
        """The axis with its label (a time: a date axis), None: no axis."""
        kind = None if label is None else pg.DateAxisItem if time else pg.AxisItem
        if self.axis is not None and type(self.axis) is not kind:
            self.plot_item.layout.removeItem(self.axis)
            self.axis.scene().removeItem(self.axis)
            self.axis = None
        if kind is not None and self.axis is None:
            options = {"utcOffset": 0} if time else {}
            self.axis = kind(orientation="right", **options)
            self.axis.enableAutoSIPrefix(False)  # e.g. ratios: as they are
            self.plot_item.layout.addItem(self.axis, 2, self.column)
            self.axis.linkToView(self.view)
        if self.axis is not None:
            self.axis.setLabel(label)
        self.view.setVisible(label is not None)

    def clear(self) -> None:
        for item in self.items:
            self.view.removeItem(item)
        self.items = []
        self.plotted = None
        self.highlight.clear()


class ResultsPlotPanel(RunScatterPanel):
    """The selected compound (by name) in every measurement, one point per occurrence.

    Its axes, chosen in the row above the plot, are each an ``Axis``: a variable of the
    measurement, of the compound or of one of its ions (``plot_axes``), so that
    variables can be plotted against each other; up to two more y (options y2, y3, None:
    not shown) have their own axis on the right and their own symbol. Clicking a point
    selects that measurement and compound, a double click asks for its chromatogram.

    Comparing compounds (its checkbox, remembered): each axis has a compound of its own
    and a point is a measurement (``EsperData.results_by_run``), e.g. the main ion area
    of one compound against another's; the compound selected elsewhere is then only the
    one of the axes without one.
    """

    title = "Results"
    # The elements of the compound loaded (its ions: ``plot_axes.results_elements``)
    elements_changed = Signal(object)
    watches = {"candidate", "target", "identification"}

    def __init__(
        self,
        data: EsperData,
        selection: Selection,
        parent=None,
        title: str | None = None,
        y: Axis = RESULTS_Y,
    ) -> None:
        super().__init__(
            data,
            selection,
            parent,
            valid_only=True,
            x=RESULTS_X,
            y=y,
            y2=None,
            y3=None,
            compare=False,
        )
        if title is not None:
            self.title = title
        self._loaded_key = None
        self._loaded: pd.DataFrame | None = None
        plot_item = self.plot.getPlotItem()
        self.right_axes = {
            name: _RightAxis(plot_item, column)
            for column, name in enumerate(EXTRA_YS, start=3)
        }
        self._add_axis_row(y)
        if settings_manager().compare_compounds(self.title):
            self.compare_check.setChecked(True)

    def _add_axis_row(self, y: Axis) -> None:
        """The axes of the plot: X, Y, and the optional Y2, Y3."""
        self.selectors = {
            "x": AxisSelector(
                "X:", RESULTS_ELEMENTS, RESULTS_X, elements_of=self._elements_of
            ),
            "y": AxisSelector("Y:", RESULTS_ELEMENTS, y, elements_of=self._elements_of),
            **{
                name: AxisSelector(
                    f"{name.upper()}:",
                    RESULTS_ELEMENTS,
                    EXTRA_DEFAULTS[name],
                    optional=True,
                    elements_of=self._elements_of,
                )
                for name in EXTRA_YS
            },
        }
        self.compare_check = QCheckBox("compare")
        self.compare_check.setToolTip(
            "Compare compounds: a compound for each axis, one point per measurement"
        )
        self.compare_check.toggled.connect(self.set_compare)
        swap = QToolButton()
        swap.setText("⇄")
        swap.setToolTip("Swap the X and Y axes")
        swap.clicked.connect(self.swap_axes)
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 0)
        row.addWidget(self.compare_check)
        row.addSpacing(8)
        row.addWidget(self.selectors["x"])
        row.addWidget(swap)
        for name in ("y", *EXTRA_YS):
            row.addSpacing(8)
            row.addWidget(self.selectors[name])
        row.addStretch()
        self.layout().insertLayout(0, row)
        for name, selector in self.selectors.items():
            selector.changed.connect(
                lambda value, name=name: self.set_options(**{name: value})
            )
        self.elements_changed.connect(self._on_elements_changed)

    def _on_elements_changed(self, elements: tuple) -> None:
        """The ions of the compound shown (its target's, or its masses)."""
        for selector in self.selectors.values():
            selector.set_elements(elements)

    def _elements_of(self, name: str) -> tuple:
        """The elements of a compound (its ions), for an axis of its own."""
        return results_elements(self.data.results(name, self.options["valid_only"]))

    # --- comparing compounds ----------------------------------------------------------

    def set_compare(self, compare: bool) -> None:
        """Compare compounds (an axis: a compound of its own) or not (remembered)."""
        if compare == self.options["compare"]:
            return
        self.options["compare"] = compare
        if self.compare_check.isChecked() != compare:
            self.compare_check.setChecked(compare)
        settings_manager().remember_compare_compounds(self.title, compare)
        if compare:
            self._fill_compound_lists()
        for selector in self.selectors.values():
            selector.set_compare(compare, self.selection.label)
        self._loaded_key = None  # loaded again; not comparing: its ions sent again
        self.invalidate()

    def _fill_compound_lists(self) -> None:
        names = list(self.data.candidate_names(self.options["valid_only"]).index)
        for selector in self.selectors.values():
            selector.set_compounds(names)

    def set_options(self, **options) -> None:
        super().set_options(**options)
        if "valid_only" in options and self.options["compare"]:
            self._fill_compound_lists()

    def _compounds(self) -> tuple[str, ...]:
        """The compounds of the axes shown (comparing compounds)."""
        axes = (self.options[name] for name in ("x", "y", *EXTRA_YS))
        return tuple(sorted({a.compound for a in axes if a and a.compound}))

    def swap_axes(self) -> None:
        x, y = self.selectors["x"].axis, self.selectors["y"].axis
        self.selectors["x"].set_axis(y)
        self.selectors["y"].set_axis(x)

    def on_selection_changed(self, fields: set) -> None:
        if "identification" in fields:
            # Names edited or compounds merged: the results are queried again
            self._loaded_key = None
        super().on_selection_changed(fields)

    def load(self) -> pd.DataFrame | None:
        if self.options["compare"]:
            # an axis without its compound yet (e.g. at start): the one selected
            label = self.selection.label
            for selector in self.selectors.values():
                value = selector.value
                if label and value and value.element.key != "measurement":
                    if value.compound is None:
                        selector.set_compound(label)
            names = self._compounds()
            key = ("compare", names, self.options["valid_only"])
            if key != self._loaded_key:
                self._loaded_key = key
                self._loaded = (
                    self.data.results_by_run(names, self.options["valid_only"])
                    if names
                    else None
                )
            df = self._loaded
            return None if df is None or df.empty else df
        # Query the database only when the compound or the validity filter changed
        name = self.selection.label
        key = (name, self.options["valid_only"])
        if key != self._loaded_key:
            self._loaded_key = key
            self._loaded = (
                None
                if name is None
                else self.data.results(name, self.options["valid_only"])
            )
            self.elements_changed.emit(results_elements(self._loaded))
        df = self._loaded
        return None if df is None or df.empty else df

    def _ys(self) -> dict[str, Axis]:
        """The y axes shown: y, and y2, y3 if checked."""
        return {
            name: self.options[name]
            for name in ("y", *EXTRA_YS)
            if self.options[name] is not None
        }

    def plot_title(self) -> str:
        ys = ", ".join(y.label for y in self._ys().values())
        if self.options["compare"]:
            return f"{ys} vs {self.options['x'].label}"
        return f"{self.selection.label or ''}: {ys} vs {self.options['x'].label}"

    def axis_values(self, df: pd.DataFrame, which: str) -> np.ndarray:
        return self.options[which].values(df)

    def is_time(self, which: str) -> bool:
        return self.options[which].time

    def axis_label(self, which: str) -> str:
        label = self.options[which].label
        if which == "y" and len(self._ys()) > 1:
            return f"{Y_SIGNS['y']} {label}"
        return label

    def redraw(self) -> None:
        for right in self.right_axes.values():
            right.clear()
        super().redraw()
        if self.plotted is None:  # nothing to plot: no other y axis
            for right in self.right_axes.values():
                right.show(None)

    def draw_points(self, df: pd.DataFrame) -> None:
        colors = runtype_colors(self.data.sample_types())
        self._draw_y(df, "y", colors)
        for name, right in self.right_axes.items():
            axis = self.options[name]
            if axis is None or self.with_x is None:
                right.show(None)
                continue
            right.show(f"{Y_SIGNS[name]} {axis.label}", axis.time)
            points = self.with_x.assign(y=axis.values(self.with_x))
            right.plotted = points[np.isfinite(points["y"])]
            right.items = self._draw_y(right.plotted, name, colors, right.view)
            # its symbol in the legend
            sample = pg.ScatterPlotItem(
                symbol=Y_SYMBOLS[name], size=10, pen=pg.mkPen("k"), brush=None
            )
            self.legend.addItem(sample, axis.label)
            right.view.autoRange()

    def _draw_y(
        self,
        df: pd.DataFrame,
        which: str,
        colors: dict,
        view: pg.ViewBox | None = None,
    ) -> list:
        """The points of a y axis (its ``y`` column), colored by run type; the run types
        in the legend for the first y."""
        x_axis: Axis = self.options["x"]
        y_axis: Axis = self.options[which]
        if self.options["compare"]:
            df = self._of_compound(df, y_axis.compound or x_axis.compound)

        def tip(x, y, data) -> str:
            _, _, name, rt = data
            rt = "–" if pd.isna(rt) else f"{rt:.1f}"
            return (
                f"{name}\nrt {rt} s\n{y_axis.label}: {y_axis.format(y)}"
                f"\n{x_axis.label}: {x_axis.format(x)}"
            )

        items = []
        for stype, color in colors.items():
            sub = df[df["sample_type"] == stype]
            if sub.empty:
                continue
            items.append(
                self.add_points(
                    sub,
                    POINT_COLUMNS,
                    tip=tip,
                    view=view,
                    symbol=Y_SYMBOLS[which],
                    size=14 if which == "y" else 10,
                    pen=pg.mkPen(color, width=2),
                    hoverSize=20,
                    name=stype if which == "y" else None,
                )
            )
        return items

    @staticmethod
    def _of_compound(df: pd.DataFrame, compound: str | None) -> pd.DataFrame:
        """Comparing compounds: the points with the compound and rt of one of them
        (None: of none, e.g. a measurement against another)."""
        columns = {}
        for column in ("compound_idx", "rt_mean_s"):
            name = f"{compound}{COMPOUND_SEP}{column}"
            columns[column] = df[name] if compound and name in df else np.nan
        return df.assign(**columns)

    def _update_highlight(self) -> None:
        """The point of the measurement and compound selected; comparing compounds:
        the points of the measurement."""
        sel = self.selection
        highlights = [(self.highlight, self.plotted)] + [
            (right.highlight, right.plotted) for right in self.right_axes.values()
        ]
        for highlight, df in highlights:
            if df is None or df.empty:
                highlight.clear()
                continue
            match = df["measurement_id"] == sel.measurement_id
            if not self.options["compare"]:
                match &= df["compound_idx"] == sel.compound_idx
            row = df[match]
            highlight.setData(row["x"].to_numpy(), row["y"].to_numpy())

    def select_point(self, data: tuple) -> None:
        measurement_id, compound_idx, *_ = data
        if pd.isna(compound_idx):  # comparing measurements alone
            self.selection.select_measurement(int(measurement_id))
        else:
            self.selection.select_compound(int(measurement_id), int(compound_idx))
