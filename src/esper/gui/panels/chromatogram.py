"""Chromatogram panels: absolute, with the peaks found by alpinac, and relative."""

from __future__ import annotations

import html
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidgetAction

from esper.fitting import SOURCE_FIT, peak_shape
from esper.gui.data import TIC, EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.plotting import add_curve, make_plot
from esper.gui.selection import Selection
from esper.gui.settings import INTEGRATIONS, settings_manager
from esper.gui.style import overlay_colors
from esper.integration import integration_bounds, peak_area

# Half width of the rt window shown around the selected compound [s]
RT_HALF_WINDOW = 30.0
# Headroom above the selected peak
Y_ZOOM = 1.5
# Overlays: key -> menu text. 'rankN' is the Nth most prominent mass of the selected compound
# (checking the 3rd checks the 2nd too, which can then be unchecked alone).
OVERLAYS = {
    "rank2": "2nd mass",
    "rank3": "3rd mass",
    "compound": "all compound masses",
    "peaking": "masses peaking here",
    "all": "all masses",
    "target": "target ions",
    "tic": "TIC",
}
# Overlays checked at start: the qualifier ions of the target
DEFAULT_OVERLAYS = {"target"}
TARGET_COLOR = "#008000"
# Half width of the window used for the y zoom around a target not found [s]
TARGET_HALF_WINDOW = 10.0
# Above this number of overlaid curves, no legend
MAX_LEGEND_ENTRIES = 12
# Above this number of overlaid curves, the whole chromatogram file is read at once
READ_ALL_ABOVE = 20
# Delay before rescaling the relative plot after a pan/zoom [ms]
RESCALE_DELAY_MS = 100
# Delay before updating the masses peaking in the view after a pan/zoom [ms]
PEAKING_DELAY_MS = 300
# Integration menu: key -> menu text
INTEGRATION_TEXTS = {
    "fit_integration": "fit: integration",
    "fit_shapes": "fit: peak shapes",
    "extraction": "extraction (ips)",
    "none": "none",
}
# Fitted peaks: drawn where above this fraction of their height, filled with this
# opacity (0-255; peaks of the selected compounds: FIT_ALPHA_SELECTED)
FIT_VISIBLE = 1e-3
FIT_ALPHA = 40
FIT_ALPHA_SELECTED = 90
# Integration view: bases of a fitted peak where it falls to this fraction of its height
FIT_BASE_FRACTION = 0.01
INTEGRATION_COLOR = "#c80000"
# Peak labels without a baseline: below the lowest signal within this of the apex [s]
LABEL_HALF_WINDOW = 5.0
# compound_idx of the peaks of the fit that match no compound
NO_COMPOUND = -1


def compound_label(name, compound_idx) -> str:
    return name if isinstance(name, str) else f"#{compound_idx}"


def combined_label(compound_idxs: tuple[int, ...]) -> str:
    """The other compounds selected with the main one, e.g. ' + #12, #15'."""
    return (
        " + " + ", ".join(f"#{i}" for i in compound_idxs[1:])
        if len(compound_idxs) > 1
        else ""
    )


def peak_compound_label(names, compound_idx: int) -> str:
    """Compound of a peak (``NO_COMPOUND``: none)."""
    if compound_idx == NO_COMPOUND:
        return "no compound"
    return compound_label(names.get(compound_idx), compound_idx)


def _pairs(a0, a1) -> np.ndarray:
    """Points of segments (a0[i], a1[i]), for ``connect="pairs"``."""
    return np.column_stack([a0, a1]).ravel()


def fitted_shapes(t: np.ndarray, peaks, fraction: float):
    """(peak, slice of t, shape above the baseline) of the fitted peaks of one mass,
    where the shape is above ``fraction`` of its height."""
    fitted = peaks[peaks["fit_rt_s"].notna() & (peaks["fit_height"] > 0)]
    for p in fitted.itertuples():
        shape = peak_shape(t, p)
        inside = np.flatnonzero(shape > fraction * p.fit_height)
        if len(inside) < 2:
            continue
        part = slice(inside[0], inside[-1] + 1)
        yield p, part, shape[part]


def lowest_near(t: np.ndarray, y: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """Lowest signal within LABEL_HALF_WINDOW of each rt (the signal at rt if no sample is)."""
    out = np.interp(rt, t, y)
    for i, r in enumerate(rt):
        near = y[(t >= r - LABEL_HALF_WINDOW) & (t <= r + LABEL_HALF_WINDOW)]
        if len(near):
            out[i] = near.min()
    return out


def mass_label(mass: int | None) -> str:
    return "TIC" if mass is TIC else f"m/z {mass}"


@dataclass
class Curve:
    mass: int | None  # None: TIC
    label: str
    t: np.ndarray
    y: np.ndarray
    color: object


class ChromatogramPanel(BasePanel):
    """Chromatogram of the selected measurement and mass, with the peaks found at that mass.

    The peaks of the selected compounds (several can be) are highlighted. Other masses can be overlaid from the right click menu (Masses).
    """

    title = "Chromatogram"

    # After each redraw: main Curve (or None) and list of overlaid Curves
    curves_changed = Signal(object, object)

    def __init__(
        self, data: EsperData, selection: Selection, show_info: bool = True, parent=None
    ) -> None:
        super().__init__(data, selection, parent)
        self.show_info = show_info
        self._keep_range = False
        self._annotations: list = []
        self._overlay_items: list = []

        self.plot = make_plot()
        self.curve = self.plot.plot(pen=pg.mkPen("k", width=1))
        self.curve.setClipToView(True)

        self.info_label = pg.LabelItem(justify="left", color="k", size="9pt")
        self.info_label.setParentItem(self.plot.getViewBox())
        self.info_label.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 5))
        self.title_label = pg.LabelItem(color="k", size="10pt")
        self.title_label.setParentItem(self.plot.getViewBox())
        self.title_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 5))
        self.legend = self.plot.addLegend(offset=(-10, 30))

        self._check_sets: list[dict[str, QCheckBox]] = []
        self.overlay_checks = self.add_masses_menu(self.plot)
        self.settings = settings_manager()
        self.integration = self.settings.settings.chromatogram_integration
        self.integration_actions = self._add_integration_menu(self.plot)
        self.settings.changed.connect(self._on_settings_changed)
        # Masses overlaid now, to redraw only when "peaking here" changes them after a pan/zoom
        self._overlaid: list = []
        self._peaking_timer = QTimer(self, singleShot=True, interval=PEAKING_DELAY_MS)
        self._peaking_timer.timeout.connect(self._update_peaking)
        self.plot.sigXRangeChanged.connect(self._on_x_range_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    # --- overlays -------------------------------------------------------------------

    def add_masses_menu(self, plot: pg.PlotWidget) -> dict[str, QCheckBox]:
        """Add the 'Masses' submenu to the right click menu of a plot.

        Checkboxes keep the menu open. Every plot gets its own checkboxes,
        kept in sync with the ones of this panel.
        """
        menu = plot.getViewBox().menu
        menu.addSeparator()
        submenu = menu.addMenu("Masses")
        checks: dict[str, QCheckBox] = {}
        current = self._check_sets[0] if self._check_sets else {}
        for key, text in OVERLAYS.items():
            check = QCheckBox(text)
            check.setChecked(
                current[key].isChecked() if current else key in DEFAULT_OVERLAYS
            )
            check.setStyleSheet("QCheckBox { padding: 3px 12px; }")
            check.toggled.connect(lambda value, key=key: self._on_check(key, value))
            action = QWidgetAction(submenu)
            action.setDefaultWidget(check)
            submenu.addAction(action)
            checks[key] = check
        self._check_sets.append(checks)
        return checks

    def _add_integration_menu(self, plot: pg.PlotWidget) -> dict[str, QAction]:
        """'Integration' submenu of the right click menu: ips, fitted peaks or none."""
        submenu = plot.getViewBox().menu.addMenu("Integration")
        group = QActionGroup(submenu)
        actions = {}
        for key in INTEGRATIONS:
            action = QAction(INTEGRATION_TEXTS[key], submenu, checkable=True)
            action.setChecked(key == self.integration)
            action.triggered.connect(
                lambda _=False, key=key: self._set_integration(key)
            )
            group.addAction(action)
            submenu.addAction(action)
            actions[key] = action
        return actions

    def _set_integration(self, key: str) -> None:
        """Saved in the settings, for all the chromatogram panels."""
        self.settings.update(
            replace(self.settings.settings, chromatogram_integration=key)
        )

    def _on_settings_changed(self) -> None:
        key = self.settings.settings.chromatogram_integration
        if key == self.integration:
            return
        self.integration = key
        self.integration_actions[key].setChecked(True)
        self._keep_range = True
        self.invalidate()

    def _on_check(self, key: str, value: bool) -> None:
        keys = [key, "rank2"] if key == "rank3" and value else [key]
        for checks in self._check_sets:
            for k in keys:
                checks[k].blockSignals(True)
                checks[k].setChecked(value)
                checks[k].blockSignals(False)
        self._keep_range = True
        self.invalidate()

    def _overlay_masses(self) -> list[int | None]:
        """Masses to overlay on the main one (None: TIC), in display order."""
        sel = self.selection
        checked = {k for k, c in self.overlay_checks.items() if c.isChecked()}
        compound_masses = []
        if sel.compound_idx is not None:
            peaks = self.data.compound_peaks(sel.measurement_id, sel.compound_idxs)
            peaks = peaks.sort_values("prominence", ascending=False)
            compound_masses = [int(m) for m in peaks["mass"].drop_duplicates()]

        masses: list = []
        if "all" in checked:
            columns = self.data.chromatogram_columns(sel.measurement_id)
            masses = sorted(int(c) for c in columns if c.isdigit())
        elif "compound" in checked:
            masses = compound_masses
        else:
            for rank in (2, 3):
                if f"rank{rank}" in checked and len(compound_masses) >= rank:
                    masses.append(compound_masses[rank - 1])
        if "peaking" in checked and "all" not in checked:
            masses.extend(self._peaking_masses())
        # Expected ions of the target; always when the target is not found in the measurement
        if sel.target is not None and ("target" in checked or sel.compound_idx is None):
            masses.extend(self.data.target_masses(sel.target))
        if "tic" in checked:
            masses.append(TIC)
        return [m for m in dict.fromkeys(masses) if m != sel.mass]

    def _peaking_masses(self) -> list[int]:
        """Masses with a peak apex in the visible rt range, most prominent first."""
        lo, hi = self.plot.viewRange()[0]
        peaks = self.data.peaks(self.selection.measurement_id)
        here = peaks[(peaks["rt_s"] >= lo) & (peaks["rt_s"] <= hi)]
        return [
            int(m)
            for m in here.groupby("mass")["prominence"]
            .max()
            .sort_values(ascending=False)
            .index
        ]

    def _on_x_range_changed(self, *_) -> None:
        if self.overlay_checks["peaking"].isChecked():
            self._peaking_timer.start()

    def _update_peaking(self) -> None:
        """After a pan/zoom: redraw (keeping the view) if other masses peak in it."""
        if not self.isVisible():
            return
        if (
            self.selection.measurement_id is not None
            and self._overlay_masses() != self._overlaid
        ):
            self._keep_range = True
            self.invalidate()

    # --- drawing --------------------------------------------------------------------

    def redraw(self) -> None:
        keep_range, self._keep_range = self._keep_range, False
        sel = self.selection
        view_range = self.plot.viewRange()
        self._clear()
        if sel.measurement_id is None:
            self.curve.clear()
            self.info_label.setText("")
            self.title_label.setText("")
            self.curves_changed.emit(None, [])
            return

        series = self.data.chromatogram(sel.measurement_id, sel.mass)
        title = mass_label(sel.mass)
        if not self.show_info:
            name = self.data.measurements.loc[sel.measurement_id, "name"]
            title = f"{name}   {html.escape(sel.label or '')}   {title}"
        if series is None:
            title = f"{title} (no chromatogram)"
        elif self.integration.startswith("fit") and not self.data.is_fitted(
            sel.measurement_id
        ):
            title = f"{title} (peaks not fitted)"
        self.title_label.setText(title)
        if self.show_info:
            self._update_info()
        else:
            self.info_label.setText("")
        if series is None:
            self.curve.clear()
            self.curves_changed.emit(None, [])
            return

        t = series.index.to_numpy(dtype=float)
        y = series.to_numpy(dtype=float)
        self.curve.setData(t, y)
        self._annotate(t, y)
        self._mark_target()
        if keep_range:
            self.plot.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        else:
            self._set_range(t, y)
        overlays = self._draw_overlays()
        self.curves_changed.emit(
            Curve(sel.mass, mass_label(sel.mass), t, y, "k"), overlays
        )

    def _clear(self) -> None:
        for item in self._annotations + self._overlay_items:
            self.plot.removeItem(item)
        self._annotations = []
        self._overlay_items = []
        self.legend.clear()

    def _draw_overlays(self) -> list[Curve]:
        """Other masses of the measurement, drawn in color behind the main mass."""
        sel = self.selection
        masses = self._overlaid = self._overlay_masses()
        if not masses:
            return []
        frame = (
            self.data.chromatogram_frame(sel.measurement_id)
            if len(masses) > READ_ALL_ABOVE
            else None
        )

        n = len(masses)
        colors = overlay_colors(n)
        peaks = self.data.peaks(sel.measurement_id)
        show_legend = n <= MAX_LEGEND_ENTRIES
        if show_legend:
            self.legend.addItem(self.curve, mass_label(sel.mass))
        curves = []
        for i, mass in enumerate(masses):
            if frame is not None:
                column = "tic" if mass is TIC else str(mass)
                series = frame[column] if column in frame else None
            else:
                series = self.data.chromatogram(sel.measurement_id, mass)
            if series is None:
                continue
            color = colors[i]
            curve = Curve(
                mass,
                mass_label(mass),
                series.index.to_numpy(dtype=float),
                series.to_numpy(dtype=float),
                color,
            )
            item = add_curve(self.plot, curve.t, curve.y, color)
            self._overlay_items.append(item)
            curves.append(curve)
            # Fitted peaks of the overlaid masses, unless there are too many
            if self.integration == "fit_shapes" and mass is not TIC and show_legend:
                self._draw_fits(curve.t, peaks[peaks["mass"] == mass], color)
            if show_legend:
                self.legend.addItem(item, curve.label)
        return curves

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
                (
                    "cmp",
                    f"{compound_label(c.main_name, sel.compound_idx)} (#{sel.compound_idx}){combined_label(sel.compound_idxs)}",
                ),
                ("rt", f"{c.rt_mean_s:.1f} ± {c.rt_std_s:.2f} s"),
                ("RI", f"{c.retention_index:.1f}"),
                ("spectrum", f"{c.best_spectrum}"),
                ("score", f"{c.best_score:.3f} (combined {c.best_score_combined:.3f})"),
                ("valid", "yes" if c.valid_candidate else "no"),
            ]
        if sel.target is not None:
            target = self.data.targets.loc[sel.target]
            expected_rt = self.data.expected_rt(sel.measurement_id, sel.target)
            masses = self.data.target_masses(sel.target)
            lines += [
                (
                    "target",
                    sel.target
                    + ("" if sel.compound_idx is not None else " (not found)"),
                ),
                (
                    "expected RI",
                    "unknown"
                    if not np.isfinite(target["ri"])
                    else f"{target['ri']:.1f}",
                ),
                (
                    "expected rt",
                    "unknown" if expected_rt is None else f"{expected_rt:.1f} s",
                ),
                ("expected ions", ", ".join(map(str, masses)) or "unknown"),
            ]
        self.info_label.setText(
            "<br>".join(f"{k}&nbsp;&nbsp;{html.escape(str(v))}" for k, v in lines)
        )

    def _mark_target(self) -> None:
        """Dashed line where the selected target is expected."""
        sel = self.selection
        if sel.target is None:
            return
        rt = self.data.expected_rt(sel.measurement_id, sel.target)
        if rt is None:
            return
        line = pg.InfiniteLine(
            pos=rt,
            angle=90,
            pen=pg.mkPen(TARGET_COLOR, width=1, style=Qt.PenStyle.DashLine),
            label=f"{sel.target} expected",
            labelOpts={"position": 0.92, "color": TARGET_COLOR},
        )
        self._add(line)

    def _add(self, item) -> None:
        self.plot.addItem(item)
        self._annotations.append(item)

    def _annotate(self, t: np.ndarray, y: np.ndarray) -> None:
        """Integration (``self.integration``), apex marker and label of each peak at this mass."""
        sel = self.selection
        if sel.mass is TIC:
            return
        peaks = self.data.peaks(sel.measurement_id)
        peaks = peaks[peaks["mass"] == sel.mass]
        if peaks.empty:
            return
        names = self.data.compounds(sel.measurement_id)["main_name"]
        rt = peaks["rt_s"].to_numpy(dtype=float)
        y_apex = np.interp(rt, t, y)
        # Labels below the baseline, under the lowest signal near the peak if none is drawn
        y_label = lowest_near(t, y, rt)
        areas = np.full(len(peaks), np.nan)
        integration = self.integration
        # Not fitted: no integration of the fit (the title says so)
        if integration.startswith("fit") and not self.data.is_fitted(
            sel.measurement_id
        ):
            integration = "none"
        fitted = integration == "fit_integration"
        if fitted:
            self._draw_fit_integration(t, peaks)
            areas = peaks["fit_area"].to_numpy(dtype=float)
            base = peaks["fit_baseline"].to_numpy(dtype=float)
            y_label = np.where(np.isfinite(base), base, y_label)
        elif integration == "extraction":
            areas, y_base = self._draw_ips(t, y, peaks)
            y_label = np.where(np.isfinite(y_base), y_base, y_label)
        elif integration == "fit_shapes":
            self._draw_fits(t, peaks, "k", selected=sel.compound_idxs)
            areas = peaks["fit_area"].to_numpy(dtype=float)
            base = peaks["fit_baseline"].to_numpy(dtype=float)
            y_label = np.where(np.isfinite(base), base, y_label)

        # Invisible markers at the apex to select a compound by clicking on its peak
        compound_idxs = peaks["compound_idx"].fillna(NO_COMPOUND).to_numpy(dtype=int)
        apex = pg.ScatterPlotItem(
            rt,
            y_apex,
            data=compound_idxs,
            symbol="o",
            size=14,
            pen=None,
            brush=pg.mkBrush(0, 0, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 120, 255, 80),
            tip=lambda x, y, data: f"{peak_compound_label(names, data)}\nrt {x:.1f}",
        )
        apex.sigClicked.connect(self._on_apex_clicked)
        self._add(apex)

        for i, p in enumerate(peaks.itertuples()):
            selected = compound_idxs[i] in sel.compound_idxs
            if fitted:
                lines = self._fit_label_lines(p, names, compound_idxs[i])
            else:
                lines = [
                    f"{p.rt_s:.1f}",
                    html.escape(peak_compound_label(names, compound_idxs[i])),
                    f"H = {p.height:.0f}",
                ]
                if integration != "none":
                    lines.append(
                        "A = –" if np.isnan(areas[i]) else f"A = {areas[i]:.0f}"
                    )
                if np.isfinite(p.width_s):
                    lines.append(f"W = {p.width_s:.2f}")
            if p.source == SOURCE_FIT:
                lines.append("<i>from the fit</i>")
            color = "#0050c8" if selected else "#000"
            weight = "bold" if selected else "normal"
            label = pg.TextItem(
                html=f'<div style="text-align:center; color:{color}; font-weight:{weight}; font-size:8pt">{"<br>".join(lines)}</div>',
                anchor=(0.5, 0),
            )
            label.setPos(rt[i], y_label[i])
            self._add(label)

    @staticmethod
    def _fit_label_lines(p, names, compound_idx: int) -> list[str]:
        """Label of a peak in the integration view: the values of its fit."""
        compound = html.escape(peak_compound_label(names, compound_idx))
        if pd.isna(p.fit_rt_s):
            return [f"{p.rt_s:.1f}", compound, f"H = {p.height:.0f}", "not fitted"]
        area = f"A = {p.fit_area:.0f}"
        if np.isfinite(p.fit_area_u):
            area += f" ± {p.fit_area_u:.0f}"
        return [
            f"{p.fit_rt_s:.2f}",
            compound,
            f"H = {p.height:.0f}",
            area,
            f"σ = {p.fit_sigma_s:.2f}, tail = {p.fit_tailing:.2f}",
            f"S/N = {p.fit_sn:.0f}",
        ]

    def _draw_fit_integration(self, t: np.ndarray, peaks) -> None:
        """Bases, constant baseline, apex line and outline of the fitted peaks of one mass."""
        pen = pg.mkPen(INTEGRATION_COLOR, width=1)
        starts, stops, bases, rts, tops = [], [], [], [], []
        for p, part, shape in fitted_shapes(t, peaks, FIT_BASE_FRACTION):
            starts.append(t[part][0])
            stops.append(t[part][-1])
            bases.append(p.fit_baseline)
            rts.append(p.fit_rt_s)
            tops.append(p.fit_baseline + p.fit_height)
            outline = pg.PlotCurveItem(
                t[part],
                p.fit_baseline + shape,
                pen=pg.mkPen(INTEGRATION_COLOR, width=1, style=Qt.PenStyle.DashLine),
            )
            outline.setZValue(-1)
            self._add(outline)
        if not starts:
            return
        self._add(
            pg.PlotDataItem(
                _pairs(starts, stops), _pairs(bases, bases), connect="pairs", pen=pen
            )
        )
        self._add(
            pg.PlotDataItem(
                _pairs(rts, rts),
                _pairs(bases, tops),
                connect="pairs",
                pen=pg.mkPen("k", width=0.5),
            )
        )
        self._add(
            pg.ScatterPlotItem(
                np.concatenate([starts, stops]),
                np.concatenate([bases, bases]),
                symbol="|",
                size=20,
                pen=pg.mkPen(INTEGRATION_COLOR, width=1.5),
            )
        )

    def _draw_ips(
        self, t: np.ndarray, y: np.ndarray, peaks
    ) -> tuple[np.ndarray, np.ndarray]:
        """Baseline, bounds and apex line of the peaks with ips; their areas and baselines below the apex (NaN without ips)."""
        rt = peaks["rt_s"].to_numpy(dtype=float)
        start, stop = integration_bounds(peaks, t)
        valid = np.isfinite(start) & np.isfinite(stop)
        areas = np.full(len(peaks), np.nan)
        y_base = np.full(len(peaks), np.nan)
        if not valid.any():
            return areas, y_base
        rt, start, stop = rt[valid], start[valid], stop[valid]
        y_start, y_stop, y_apex = (
            np.interp(start, t, y),
            np.interp(stop, t, y),
            np.interp(rt, t, y),
        )
        y_base[valid] = y_start + (y_stop - y_start) * (rt - start) / np.where(
            stop > start, stop - start, 1
        )
        areas[valid] = [peak_area(t, y, a, b) for a, b in zip(start, stop)]

        self._add(
            pg.PlotDataItem(
                _pairs(start, stop),
                _pairs(y_start, y_stop),
                connect="pairs",
                pen=pg.mkPen("r", width=1),
            )
        )
        self._add(
            pg.PlotDataItem(
                _pairs(rt, rt),
                _pairs(y_base[valid], y_apex),
                connect="pairs",
                pen=pg.mkPen("k", width=0.5),
            )
        )
        self._add(
            pg.ScatterPlotItem(
                np.concatenate([start, stop]),
                np.concatenate([y_start, y_stop]),
                symbol="|",
                size=20,
                pen=pg.mkPen("r", width=1.5),
            )
        )
        return areas, y_base

    def _draw_fits(self, t: np.ndarray, peaks, color, selected=()) -> None:
        """Fitted peaks of one mass, filled between the baseline and the peak, in the color of the mass."""
        for p, part, shape in fitted_shapes(t, peaks, FIT_VISIBLE):
            tt = t[part]
            base = np.full(len(tt), p.fit_baseline)
            alpha = (
                FIT_ALPHA_SELECTED
                if not pd.isna(p.compound_idx) and p.compound_idx in selected
                else FIT_ALPHA
            )
            fill_color = pg.mkColor(color)
            fill_color.setAlpha(alpha)
            lower = pg.PlotCurveItem(tt, base)
            upper = pg.PlotCurveItem(tt, base + shape, pen=pg.mkPen(color, width=1))
            fill = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(fill_color))
            for item in (fill, upper):
                item.setZValue(-2)
                self._add(item)

    def _on_apex_clicked(self, item, points, ev=None) -> None:
        if not len(points) or self.selection.measurement_id is None:
            return
        compound_idx = int(points[0].data())
        if compound_idx != NO_COMPOUND:
            self.selection.select_compound(self.selection.measurement_id, compound_idx)

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

        # Target not found: zoom on the signal close to where it is expected
        if sel.compound_idx is None and sel.target is not None:
            near = mask & (np.abs(t - sel.rt) <= TARGET_HALF_WINDOW)
            if near.any():
                top = float(y[near].max())

        # Zoom on the selected compounds' peaks at this mass, if any
        if sel.compound_idx is not None and sel.mass is not TIC:
            peaks = self.data.compound_peaks(sel.measurement_id, sel.compound_idxs)
            peaks = peaks[peaks["mass"] == sel.mass]
            if not peaks.empty:
                apex = float(np.interp(peaks["rt_s"], t, y).max())
                top = bottom + Y_ZOOM * (apex - bottom)

        span = max(top - bottom, 1.0)
        self.plot.setXRange(x0, x1, padding=0)
        self.plot.setYRange(bottom - 0.35 * span, top + 0.05 * span, padding=0)


class RelativeChromatogramPanel(BasePanel):
    """The curves of a chromatogram panel, each scaled to the main mass in the visible window.

    Follows its source panel (x axis linked, same overlays), not the selection.
    """

    title = "Relative chromatograms"
    watches: set[str] = set()

    def __init__(self, source: ChromatogramPanel, parent=None) -> None:
        super().__init__(source.data, source.selection, parent)
        self.source = source
        self._main: Curve | None = None
        self._overlays: list[Curve] = []
        self._items: list = []

        self.plot = make_plot()
        self.plot.setXLink(source.plot)
        self.curve = self.plot.plot(pen=pg.mkPen("k", width=1))
        self.curve.setClipToView(True)
        self.curve.setZValue(1)
        source.add_masses_menu(self.plot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        self._rescale_timer = QTimer(self)
        self._rescale_timer.setSingleShot(True)
        self._rescale_timer.setInterval(RESCALE_DELAY_MS)
        self._rescale_timer.timeout.connect(self.rescale)
        source.plot.sigXRangeChanged.connect(lambda *_: self._rescale_timer.start())
        source.curves_changed.connect(self._on_curves_changed)

    def _on_curves_changed(self, main: Curve | None, overlays: list[Curve]) -> None:
        self._main, self._overlays = main, overlays
        self.invalidate()

    def redraw(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items = []
        if self._main is None:
            self.curve.clear()
            return
        self.curve.setData(self._main.t, self._main.y)
        self._items = [add_curve(self.plot, c.t, c.y, c.color) for c in self._overlays]
        self.rescale()

    def rescale(self) -> None:
        """Scale each overlay to the range of the main mass in the visible window."""
        if self._main is None or not self.isVisible():
            return
        x0, x1 = self.plot.viewRange()[0]
        t, y = self._main.t, self._main.y
        window = (t >= x0) & (t <= x1)
        if not window.any():
            return
        main_min, main_max = float(y[window].min()), float(y[window].max())
        for item, curve in zip(self._items, self._overlays):
            w = (curve.t >= x0) & (curve.t <= x1)
            if not w.any():
                continue
            lo, hi = curve.y[w].min(), curve.y[w].max()
            factor = (main_max - main_min) / (hi - lo) if hi > lo else 0.0
            item.setData(curve.t, main_min + (curve.y - lo) * factor)
        span = max(main_max - main_min, 1.0)
        self.plot.setYRange(main_min - 0.05 * span, main_max + 0.05 * span, padding=0)
