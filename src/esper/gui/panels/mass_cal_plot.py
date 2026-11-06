"""Mass calibration: its parameters and mass drift over time, and the selected run's calibrants.

The drift is in ppm (error on the m/z if a reference calibration were used) or in tof index
(shift of the tof of the m/z):

- over time, the reference is the median calibration of the plotted points;
- within a run, the calibration of its first rt window (as ``esper plot-calibration``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.plotting import make_plot, set_x_axis
from esper.gui.selection import Selection
from esper.gui.settings import settings_manager

ALL = "all"
PARAMS = [f"p{k}" for k in range(1, 6)]
UNITS = {"ppm": "ppm", "tof index": "tof"}
X_OPTIONS = ["Time", "Measurement"]
# Width of the points of one run on the Measurement axis (1: the next measurement)
RUN_WIDTH = 0.8
# m/z range of the colormap, the same in all the plots: a drift mass has the color of the
# calibrants of similar m/z
MZ_COLOR_RANGE = (15.0, 300.0)
CMAP = pg.colormap.get("viridis")
CMAP_RANGE = (0.0, 0.85)  # without the light yellow end
HIGHLIGHT_PEN = pg.mkPen("c", width=3)


def mz_color(mz: float):
    lo, hi = MZ_COLOR_RANGE
    f = min(max((mz - lo) / (hi - lo), 0.0), 1.0)
    return CMAP.map(CMAP_RANGE[0] + f * (CMAP_RANGE[1] - CMAP_RANGE[0]), mode="qcolor")


def drift_masses() -> list[float]:
    return list(settings_manager().settings.mass_drift_masses)


# --- drift of the calibration points --------------------------------------------------


def _params(df: pd.DataFrame) -> dict:
    return {p: df[p].to_numpy(dtype=float) for p in PARAMS}


def _per_mode(df: pd.DataFrame, func) -> np.ndarray:
    """``func(rows, mode)`` for the rows of each calibration mode (TOFWERK number, as stored;
    NaN: mode not supported)."""
    out = np.full(len(df), np.nan)
    for mode, rows in df.groupby("mode").indices.items():
        try:
            with np.errstate(invalid="ignore"):
                out[rows] = func(df.iloc[rows], int(mode))
        except (NotImplementedError, KeyError, ValueError):
            pass
    return out


def reference_params(df: pd.DataFrame) -> dict:
    """Median parameters of the points, per mode."""
    return {
        mode: {
            p: float(np.nanmedian(g[p])) if g[p].notna().any() else np.nan
            for p in PARAMS
        }
        for mode, g in df.groupby("mode")
    }


def drift(df: pd.DataFrame, mz: float, unit: str) -> np.ndarray:
    """Drift of ``mz`` at each calibration point, relative to the median calibration of ``df``.

    ``ppm``: error on ``mz`` if the median calibration were used instead of the point's
    (positive when the ions arrive later); ``tof``: tof of ``mz`` minus its median tof.
    """
    from masscal import Mode, mass_to_tof, tof_to_mass

    if df.empty:
        return np.array([], dtype=float)
    f = Mode.from_tofwerk  # the modes are stored as TOFWERK numbers
    tof = _per_mode(df, lambda d, mode: mass_to_tof(mz, _params(d), f(mode)))
    reference = reference_params(df)
    if unit == "tof":
        tof_ref = _per_mode(
            df, lambda d, mode: mass_to_tof(mz, reference[mode], f(mode))
        )
        return tof - tof_ref
    mz_ref = _per_mode(
        df.assign(_tof=tof),
        lambda d, mode: tof_to_mass(d["_tof"].to_numpy(), reference[mode], f(mode)),
    )
    return (mz_ref - mz) / mz * 1e6


# --- drift within a run ---------------------------------------------------------------


def run_drift(cal, mz, rt, unit: str) -> np.ndarray:
    """Drift of the calibration at ``rt`` since its first window, for ``mz`` (broadcast)."""
    rt0 = float(cal.rt[0])
    if unit == "tof":
        return cal.tof(rt, mz) - cal.tof(rt0, mz)
    return cal.drift_ppm(mz, rt, rt0)


def calibrant_drift(cal, peaks: pd.DataFrame, unit: str) -> np.ndarray:
    """Measured drift of each calibrant peak since the first window: its m/z with the
    calibration of the first window (ppm), or its tof minus the tof of its m/z then."""
    rt0 = float(cal.rt[0])
    tof, mz = peaks["tof"].to_numpy(float), peaks["mz_exact"].to_numpy(float)
    if unit == "tof":
        return tof - cal.tof(rt0, mz)
    return (cal.mz(rt0, tof) - mz) / mz * 1e6


def calibrant_residuals(cal, peaks: pd.DataFrame, unit: str) -> np.ndarray:
    """Residual of each calibrant peak: measured minus the calibration of the run at its rt."""
    if unit == "ppm":
        return peaks["ppm"].to_numpy(float)
    rt, mz = peaks["rt_s"].to_numpy(float), peaks["mz_exact"].to_numpy(float)
    return peaks["tof"].to_numpy(float) - cal.tof(rt, mz)


def set_y_range(plot: pg.PlotWidget, values, quantile: float = 0.02) -> None:
    """y range of the bulk of ``values`` (between the quantiles ``quantile`` and 1 - ``quantile``,
    padded): a few outliers (failed fits, calibrants left out) would flatten the rest. The
    View All of the context menu shows them."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    lo, hi = np.quantile(values, [quantile, 1 - quantile])
    span = hi - lo or max(abs(hi), 1e-9) * 0.1
    plot.setYRange(lo - 0.3 * span, hi + 0.3 * span, padding=0)


def _runs_connected(df: pd.DataFrame, x: str, y: str):
    """x, y with NaN between the runs (one line per run in one item), and the row of each
    point (-1 between the runs)."""
    runs = df["measurement_pos"].to_numpy()
    breaks = np.flatnonzero(np.diff(runs) != 0) + 1
    rows = np.insert(np.arange(len(df)), breaks, -1)
    return (
        np.insert(df[x].to_numpy(float), breaks, np.nan),
        np.insert(df[y].to_numpy(float), breaks, np.nan),
        rows,
    )


class _MassCalBase(BasePanel):
    """Options shared by the mass calibration panels; redrawn when the drift masses change."""

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.options = {"runtype": ALL, "unit": "ppm", "x": X_OPTIONS[0]}
        settings_manager().changed.connect(self.invalidate)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def set_options(self, **options) -> None:
        """runtype, unit (values of UNITS), x (X_OPTIONS)."""
        self.options.update(options)
        self.invalidate()

    def unit_label(self) -> str:
        return "ppm" if self.options["unit"] == "ppm" else "tof index"


class MassCalPlotPanel(_MassCalBase):
    """The parameters p1, p2, ... of the calibration points of all the measurements, and the
    drift of the masses of the settings, over time; one line per run, x axes linked.

    Clicking a point selects its measurement, a double click opens its chromatogram.
    """

    title = "Calibration over time"
    watches = {"measurement_id"}

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.plots: list[pg.PlotWidget] = []
        self.highlights: list[tuple[pg.PlotDataItem, str]] = []
        self.plotted: pd.DataFrame | None = None

    def on_selection_changed(self, fields: set) -> None:
        if "measurement_id" in fields:
            self._update_highlight()

    def _plot_widgets(self, n: int) -> list[pg.PlotWidget]:
        """n stacked plots with linked x axes (made again when n changes)."""
        if len(self.plots) != n:
            for plot in self.plots:
                self._layout.removeWidget(plot)
                plot.deleteLater()
            self.plots = [make_plot() for _ in range(n)]
            for plot in self.plots:
                self._layout.addWidget(plot)
                if plot is not self.plots[0]:
                    plot.setXLink(self.plots[0])
        return self.plots

    def _x_values(self, df: pd.DataFrame) -> np.ndarray:
        if self.options["x"] == "Time":
            return ((df["time"] - pd.Timestamp(0)) / pd.Timedelta(seconds=1)).to_numpy(
                dtype=float
            )
        # Measurement: the points of a run spread over RUN_WIDTH, by rt
        rt_max = max(float(self.data.mass_cal_points["rt_s"].max()), 1.0)
        return (
            df["measurement_pos"].to_numpy(dtype=float)
            + RUN_WIDTH * df["rt_s"].to_numpy(dtype=float) / rt_max
        )

    def redraw(self) -> None:
        df = self.data.mass_cal_points
        df = df[df["ok"]]
        if self.options["runtype"] != ALL:
            df = df[df["sample_type"] == self.options["runtype"]]
        df = df.sort_values(["measurement_pos", "rt_s"], ignore_index=True)
        df = df.assign(x=self._x_values(df))
        self.plotted = df
        params = [p for p in PARAMS if df[p].notna().any()] or PARAMS[:3]
        masses = drift_masses()
        plots = self._plot_widgets(len(params) + 1)
        self.highlights = []
        time = self.options["x"] == "Time"
        for plot in plots:
            plot.clear()
            set_x_axis(plot, time=time)
            plot.showAxis("bottom", plot is plots[-1])

        for plot, p in zip(plots, params):
            plot.setLabel("left", p)
            self._add_series(plot, df.assign(y=df[p]), pg.mkPen("k", width=1), "k")
            self.highlights.append((self._highlight_item(plot), p))
            set_y_range(plot, df[p])

        plot = plots[-1]
        plot.setLabel("left", f"drift [{self.unit_label()}]")
        legend = plot.addLegend(offset=(-10, 10), colCount=len(masses))
        drifts = []
        for mz in masses:
            y = drift(df, mz, self.options["unit"])
            color = mz_color(mz)
            item = self._add_series(
                plot, df.assign(y=y), pg.mkPen(color, width=1), color
            )
            legend.addItem(item, f"m/z {mz:g}")
            self.highlights.append((self._highlight_item(plot), y))
            drifts.append(y)
        set_y_range(plot, np.concatenate(drifts) if drifts else [])
        plots[-1].setLabel("bottom", "time" if time else "measurement")
        self._update_highlight()
        if len(df):
            plots[0].setXRange(float(df["x"].min()), float(df["x"].max()), padding=0.02)

    def _add_series(self, plot: pg.PlotWidget, df: pd.DataFrame, pen, color):
        x, y, rows = _runs_connected(df, "x", "y")
        item = pg.PlotDataItem(
            x,
            y,
            pen=pen,
            connect="finite",
            symbol="+",
            symbolSize=7,
            symbolPen=pg.mkPen(color),
            symbolBrush=None,
            data=rows,
        )
        item.sigPointsClicked.connect(self._on_clicked)
        plot.addItem(item)
        return item

    def _highlight_item(self, plot: pg.PlotWidget) -> pg.PlotDataItem:
        item = pg.PlotDataItem(pen=HIGHLIGHT_PEN)
        item.setZValue(10)
        plot.addItem(item)
        return item

    def _update_highlight(self) -> None:
        df = self.plotted
        if df is None:
            return
        rows = (df["measurement_id"] == self.selection.measurement_id).to_numpy()
        for item, y in self.highlights:
            values = df[y].to_numpy(float) if isinstance(y, str) else y
            item.setData(df["x"].to_numpy()[rows], values[rows])

    def _on_clicked(self, item, points, ev=None) -> None:
        if not len(points) or self.plotted is None:
            return
        row = points[0].data()
        if row is None or row < 0:
            return
        self.selection.select_measurement(int(self.plotted["measurement_id"].iloc[row]))
        if ev is not None and ev.double():
            self.open_chromatogram.emit()


class _RunPanel(_MassCalBase):
    """A plot of the selected run against the rt."""

    watches = {"measurement_id"}

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent)
        self.plot = make_plot()
        self._layout.addWidget(self.plot)

    def redraw(self) -> None:
        self.plot.clear()
        self.plot.setLabel("bottom", "rt [s]")
        mid = self.selection.measurement_id
        name = (
            self.data.measurements.loc[mid, "name"]
            if mid in self.data.measurements.index
            else ""
        )
        self.name = name
        self.plot.setTitle(f"{name} {self.title.lower()}", color="k")
        cal, peaks = self.data.mass_cal_run(mid) if mid is not None else (None, None)
        if cal is None:
            self.plot.setTitle(f"{name}: no mass calibration", color="k")
            return
        shown = self.draw(cal, peaks)
        self.plot.setXRange(float(cal.rt[0]), float(cal.rt[-1]), padding=0.03)
        set_y_range(self.plot, np.r_[shown, 0.0], quantile=0.0)

    def draw(self, cal, peaks: pd.DataFrame) -> np.ndarray:
        """Plot the run; returns the values to show in full (calibrants left out may be off)."""
        raise NotImplementedError

    def add_calibrants(self, peaks: pd.DataFrame, y: np.ndarray) -> None:
        """One line per calibrant, colored by m/z; left out of the fit: hollow circles."""
        peaks = peaks.assign(y=y)
        for (calibrant, mz), g in peaks.groupby(["calibrant", "mz_exact"]):
            color = mz_color(mz)
            g = g.sort_values("rt_s")
            line = self.plot.plot(
                g["rt_s"].to_numpy(),
                g["y"].to_numpy(),
                pen=pg.mkPen(color, width=0.8, style=pg.QtCore.Qt.PenStyle.DotLine),
            )
            line.setZValue(-1)
            for used, sub in g.groupby("used"):
                self.plot.addItem(
                    pg.ScatterPlotItem(
                        sub["rt_s"].to_numpy(),
                        sub["y"].to_numpy(),
                        data=[
                            (calibrant, mz, ppm) for ppm in sub["ppm"].to_numpy(float)
                        ],
                        symbol="o",
                        size=7,
                        pen=pg.mkPen(color, width=1.2),
                        brush=pg.mkBrush(color) if used else None,
                        hoverable=True,
                        hoverPen=pg.mkPen("c", width=2),
                        hoverSize=11,
                        tip=lambda x, y, data, used=used: (
                            f"{data[0]} m/z {data[1]:.4f}\nrt {x:.0f} s\n{y:.4g}"
                            f" {self.unit_label()}\nresidual {data[2]:.2f} ppm"
                            + ("" if used else "\nleft out of the fit")
                        ),
                    )
                )


class MassCalRunPanel(_RunPanel):
    """Drift within the selected run since its first rt window: of the calibration at the
    masses of the settings (lines) and of each calibrant peak (points; hollow: left out of
    the fit of its window)."""

    title = "Drift in the run"

    def draw(self, cal, peaks: pd.DataFrame) -> np.ndarray:
        unit = self.options["unit"]
        self.plot.setLabel("left", f"drift [{self.unit_label()}]")
        legend = self.plot.addLegend(offset=(10, 10), colCount=4)
        rt = np.linspace(cal.rt[0], cal.rt[-1], 200)
        shown = []
        for mz in drift_masses():
            y = run_drift(cal, mz, rt, unit)
            item = self.plot.plot(rt, y, pen=pg.mkPen(mz_color(mz), width=2))
            legend.addItem(item, f"m/z {mz:g}")
            shown.append(y)
        if not peaks.empty:
            y = calibrant_drift(cal, peaks, unit)
            self.add_calibrants(peaks, y)
            shown.append(y[peaks["used"].to_numpy(bool)])
        return np.concatenate(shown) if shown else np.array([])


class MassCalResidualsPanel(_RunPanel):
    """Residuals of the calibrant peaks of the selected run: measured minus the calibration of
    its window (points; hollow: left out of the fit)."""

    title = "Calibrant residuals"

    def draw(self, cal, peaks: pd.DataFrame) -> np.ndarray:
        self.plot.setLabel("left", f"residual [{self.unit_label()}]")
        self.plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("grey", width=1)))
        if peaks.empty:
            self.plot.setTitle(f"{self.name}: no calibrant peaks stored", color="k")
            return np.array([])
        y = calibrant_residuals(cal, peaks, self.options["unit"])
        self.add_calibrants(peaks, y)
        return y[peaks["used"].to_numpy(bool)]
