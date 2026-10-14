"""Chromatogram of one mass of one measurement, with the peaks found by alpinac."""

from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from esper.gui.data import TIC, EsperData
from esper.gui.selection import Selection
from esper.gui.widgets import StepCombo

# Half width of the rt window shown around the selected compound [s]
RT_HALF_WINDOW = 30.0
# Headroom above the selected peak
Y_ZOOM = 1.5
# Integration bounds, in half widths at half maximum from the apex (~3.5 sigma for a gaussian)
N_HWHM = 3.0


def _compound_label(name, compound_idx) -> str:
    return name if isinstance(name, str) else f"#{compound_idx}"


def integration_bounds(peaks, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of each peak [s].

    ``start_s``/``stop_s`` are the half maximum crossings, the peak is extended
    to ``N_HWHM`` half widths on each side but not beyond its prominence bases.
    """
    rt, start_hm, stop_hm = (peaks[c].to_numpy(dtype=float) for c in ("rt_s", "start_s", "stop_s"))
    last = len(t) - 1
    left_base = t[np.clip(peaks["left_base_scan"].to_numpy(), 0, last)]
    right_base = t[np.clip(peaks["right_base_scan"].to_numpy(), 0, last)]
    start = np.maximum(rt - N_HWHM * (rt - start_hm), left_base)
    stop = np.minimum(rt + N_HWHM * (stop_hm - rt), right_base)
    return start, stop


def peak_area(t: np.ndarray, y: np.ndarray, start: float, stop: float) -> float:
    """Area above the straight baseline between start and stop [counts * s]."""
    mask = (t >= start) & (t <= stop)
    if mask.sum() < 2:
        return np.nan
    y_start, y_stop = np.interp([start, stop], t, y)
    baseline = np.interp(t[mask], [start, stop], [y_start, y_stop])
    return float(np.trapezoid(y[mask] - baseline, t[mask]))


class ChromatogramView(QWidget):
    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(parent)
        self.data = data
        self.selection = selection
        self._dirty = True
        self._cmp_items_mid = None
        self._mass_items_key = None
        self._annotations: list = []

        # Toolbar
        self.file_box = StepCombo("File:", min_width=200)
        self.cmp_box = StepCombo("Cmp:", min_width=220)
        self.mass_box = StepCombo("Mass:", min_width=70)
        self.time_edit = QLineEdit()
        self.time_edit.setValidator(QDoubleValidator(0, 1e6, 1))
        self.time_edit.setMaximumWidth(80)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.file_box, 3)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.cmp_box, 3)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.mass_box, 1)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Time:"))
        toolbar.addWidget(self.time_edit)
        toolbar.addStretch()

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=False, y=False)
        self.plot.getAxis("bottom").setLabel("rt [s]")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").setWidth(75)
        self.curve = self.plot.plot(pen=pg.mkPen("k", width=1))
        self.curve.setClipToView(True)

        self.info_label = pg.LabelItem(justify="left", color="k", size="9pt")
        self.info_label.setParentItem(self.plot.getViewBox())
        self.info_label.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 5))
        self.mass_label = pg.LabelItem(color="k", size="10pt")
        self.mass_label.setParentItem(self.plot.getViewBox())
        self.mass_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 5))

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.plot, 1)

        self.file_box.selected.connect(self._on_file_selected)
        self.file_box.text_entered.connect(self._on_file_text)
        self.cmp_box.selected.connect(self._on_cmp_selected)
        self.cmp_box.text_entered.connect(self._on_cmp_text)
        self.mass_box.selected.connect(self.selection.select_mass)
        self.mass_box.text_entered.connect(self._on_mass_text)
        self.time_edit.editingFinished.connect(self._on_time)
        self.selection.changed.connect(self._on_selection_changed)

    # --- selection ------------------------------------------------------------------

    def _on_selection_changed(self, fields: set) -> None:
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def _on_file_selected(self, measurement_id) -> None:
        if measurement_id is not None:
            self.selection.select_measurement(int(measurement_id))

    def _on_file_text(self, text: str) -> None:
        names = self.data.measurements["name"]
        matching = names[names.str.contains(text, regex=False)]
        if not matching.empty:
            self.selection.select_measurement(int(matching.index[0]))

    def _on_cmp_selected(self, compound_idx) -> None:
        if compound_idx is not None and self.selection.measurement_id is not None:
            self.selection.select_compound(self.selection.measurement_id, int(compound_idx))

    def _on_cmp_text(self, text: str) -> None:
        mid = self.selection.measurement_id
        if mid is None:
            return
        compounds = self.data.compounds(mid)
        names = compounds["compound_name"].dropna()
        matching = names[names.str.lower() == text.lower()]
        if matching.empty:
            matching = names[names.str.contains(text, case=False, regex=False)]
        if not matching.empty:
            self.selection.select_cmp_name(matching.iloc[0])

    def _on_mass_text(self, text: str) -> None:
        text = text.strip().lower()
        if text == "tic":
            self.selection.select_mass(TIC)
        elif text.isdigit():
            self.selection.select_mass(int(text))

    def _on_time(self) -> None:
        text = self.time_edit.text().strip()
        try:
            rt = float(text)
        except ValueError:
            return
        self.selection.select_rt(rt)

    # --- drawing --------------------------------------------------------------------

    def refresh(self) -> None:
        self._dirty = False
        sel = self.selection
        self._update_toolbar()
        self._clear_annotations()
        if sel.measurement_id is None:
            self.curve.clear()
            self.info_label.setText("")
            self.mass_label.setText("")
            return

        series = self.data.chromatogram(sel.measurement_id, sel.mass)
        self.mass_label.setText("TIC" if sel.mass is TIC else f"m/z {sel.mass}")
        self._update_info()
        if series is None:
            self.curve.clear()
            self.mass_label.setText(("TIC" if sel.mass is TIC else f"m/z {sel.mass}") + " (no chromatogram)")
            return

        t = series.index.to_numpy(dtype=float)
        y = series.to_numpy(dtype=float)
        self.curve.setData(t, y)
        self._annotate(t, y)
        self._set_range(t, y)

    def _update_toolbar(self) -> None:
        sel = self.selection
        meas = self.data.measurements

        if self.file_box.combo.count() != len(meas):
            self.file_box.set_items([(name, int(mid)) for mid, name in meas["name"].items()])
        self.file_box.set_current(sel.measurement_id)

        if self._cmp_items_mid != sel.measurement_id:
            self._cmp_items_mid = sel.measurement_id
            items = []
            if sel.measurement_id is not None:
                compounds = self.data.compounds(sel.measurement_id)
                items = [
                    (f"{_compound_label(c.compound_name, idx)}  ({c.rt_mean_s:.1f})", int(idx))
                    for idx, c in compounds.iterrows()
                ]
            self.cmp_box.set_items(items)
        self.cmp_box.set_current(sel.compound_idx, sel.cmp_name)

        key = (sel.measurement_id, sel.compound_idx)
        if self._mass_items_key != key:
            self._mass_items_key = key
            if sel.measurement_id is not None and sel.compound_idx is not None:
                peaks = self.data.compound_peaks(*key).sort_values("prominence", ascending=False)
                masses = [int(m) for m in peaks["mass"].drop_duplicates()]
            else:
                masses = list(range(2, 301))
            self.mass_box.set_items([("tic", TIC)] + [(str(m), m) for m in masses])
        self.mass_box.set_current(sel.mass, "tic" if sel.mass is TIC else str(sel.mass))

        self.time_edit.setText("" if sel.rt is None else f"{sel.rt:.1f}")

    def _update_info(self) -> None:
        sel = self.selection
        m = self.data.measurements.loc[sel.measurement_id]
        lines = [
            ("sample", f"{m.sample_type} {m.sample_number}"),
            ("acquired", f"{m.acq_datetime:%Y-%m-%d %H:%M}"),
            ("peaks", f"{m.n_peaks}"),
            ("compounds", f"{m.n_compounds}"),
            ("mass cal", f"mode {m.mass_cal_mode}, {m.s_per_bin:.4f} s/bin"),
        ]
        if sel.compound_idx is not None:
            c = self.data.compounds(sel.measurement_id).loc[sel.compound_idx]
            lines += [
                ("cmp", f"{_compound_label(c.compound_name, sel.compound_idx)} (#{sel.compound_idx})"),
                ("rt", f"{c.rt_mean_s:.1f} ± {c.rt_std_s:.2f} s"),
                ("RI", f"{c.retention_index:.1f}"),
                ("spectrum", f"{c.best_spectrum}"),
                ("score", f"{c.best_score:.3f} (combined {c.best_score_combined:.3f})"),
                ("valid", "yes" if c.valid_candidate else "no"),
            ]
        self.info_label.setText(
            "<br>".join(f"{k}&nbsp;&nbsp;{html.escape(str(v))}" for k, v in lines)
        )

    def _clear_annotations(self) -> None:
        for item in self._annotations:
            self.plot.removeItem(item)
        self._annotations = []

    def _add(self, item) -> None:
        self.plot.addItem(item)
        self._annotations.append(item)

    def _annotate(self, t: np.ndarray, y: np.ndarray) -> None:
        """Baseline, apex line and label of each peak found at this mass."""
        sel = self.selection
        if sel.mass is TIC:
            return
        peaks = self.data.peaks(sel.measurement_id)
        peaks = peaks[peaks["mass"] == sel.mass]
        if peaks.empty:
            return
        names = self.data.compounds(sel.measurement_id)["compound_name"]

        rt = peaks["rt_s"].to_numpy(dtype=float)
        start, stop = integration_bounds(peaks, t)
        y_start, y_stop, y_apex = np.interp(start, t, y), np.interp(stop, t, y), np.interp(rt, t, y)
        y_base = y_start + (y_stop - y_start) * (rt - start) / np.where(stop > start, stop - start, 1)

        def pairs(a0, a1):
            return np.column_stack([a0, a1]).ravel()

        self._add(pg.PlotDataItem(pairs(start, stop), pairs(y_start, y_stop), connect="pairs", pen=pg.mkPen("r", width=1)))
        self._add(pg.PlotDataItem(pairs(rt, rt), pairs(y_base, y_apex), connect="pairs", pen=pg.mkPen("k", width=0.5)))
        self._add(
            pg.ScatterPlotItem(
                np.concatenate([start, stop]),
                np.concatenate([y_start, y_stop]),
                symbol="|", size=20, pen=pg.mkPen("r", width=1.5),
            )
        )

        # Invisible markers at the apex to select a compound by clicking on its peak
        apex = pg.ScatterPlotItem(
            rt, y_apex, data=peaks["compound_idx"].to_numpy(), symbol="o", size=14,
            pen=None, brush=pg.mkBrush(0, 0, 0, 0), hoverable=True,
            hoverBrush=pg.mkBrush(0, 120, 255, 80),
            tip=lambda x, y, data: f"{_compound_label(names.get(data), data)}\nrt {x:.1f}",
        )
        apex.sigClicked.connect(self._on_apex_clicked)
        self._add(apex)

        for i, (_, p) in enumerate(peaks.iterrows()):
            selected = p.compound_idx == sel.compound_idx
            area = peak_area(t, y, start[i], stop[i])
            text = (
                f"{p.rt_s:.1f}<br>{html.escape(_compound_label(names.get(p.compound_idx), p.compound_idx))}"
                f"<br>H = {p.height:.0f}<br>A = {area:.0f}<br>W = {p.width_s:.2f}"
            )
            color = "#0050c8" if selected else "#000"
            weight = "bold" if selected else "normal"
            label = pg.TextItem(
                html=f'<div style="text-align:center; color:{color}; font-weight:{weight}; font-size:8pt">{text}</div>',
                anchor=(0.5, 0),
            )
            label.setPos(rt[i], min(y_start[i], y_stop[i]))
            self._add(label)

    def _on_apex_clicked(self, item, points, ev=None) -> None:
        if len(points) and self.selection.measurement_id is not None:
            self.selection.select_compound(self.selection.measurement_id, int(points[0].data()))

    def _set_range(self, t: np.ndarray, y: np.ndarray) -> None:
        sel = self.selection
        if sel.rt is None:
            self.plot.autoRange()
            return
        x0, x1 = sel.rt - RT_HALF_WINDOW, sel.rt + RT_HALF_WINDOW
        mask = (t >= x0) & (t <= x1)
        if not mask.any():
            self.plot.setXRange(x0, x1, padding=0)
            return
        bottom, top = float(y[mask].min()), float(y[mask].max())

        # Zoom on the selected compound's peak at this mass, if any
        if sel.compound_idx is not None and sel.mass is not TIC:
            peaks = self.data.compound_peaks(sel.measurement_id, sel.compound_idx)
            peaks = peaks[peaks["mass"] == sel.mass]
            if not peaks.empty:
                apex = float(np.interp(peaks["rt_s"].iloc[0], t, y))
                top = bottom + Y_ZOOM * (apex - bottom)

        span = max(top - bottom, 1.0)
        self.plot.setXRange(x0, x1, padding=0)
        self.plot.setYRange(bottom - 0.35 * span, top + 0.05 * span, padding=0)
