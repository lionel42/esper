"""Mass spectrum of the selected compound."""

from __future__ import annotations

import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.panels.chromatogram import ChromatogramPanel, Curve, compound_label
from esper.gui.plotting import make_plot
from esper.gui.selection import Selection

# Number of most intense masses labelled with their m/z
N_LABELS = 12
# A click selects the closest stick within this distance [m/z]
CLICK_TOLERANCE = 1.0
# Selected mass when there is no chromatogram to take the colors from
SELECTED_COLOR = "#0050c8"
# Masses not plotted in the chromatogram
OTHER_COLOR = "#aaaaaa"
# Top of the y axis [%], leaves room for the label of the 100 % stick
Y_MAX = 125


class SpectrumPanel(BasePanel):
    """Spectrum of the selected compound: one stick per mass, height = peak prominence.

    Clicking a stick selects that mass. With a visible chromatogram panel as
    ``chromatogram``, the masses plotted there get the same colors here.
    """

    title = "Spectrum"
    watches = {"measurement_id", "compound_idx", "mass"}

    def __init__(
        self,
        data: EsperData,
        selection: Selection,
        chromatogram: ChromatogramPanel | None = None,
        parent=None,
    ) -> None:
        super().__init__(data, selection, parent)
        self.chromatogram = chromatogram
        self._masses = np.array([])
        self._items: list = []
        # Mass -> color of its curve in the chromatogram panel
        self._colors: dict[int, object] = {}
        if chromatogram is not None:
            chromatogram.curves_changed.connect(self._on_curves_changed)
            chromatogram.toggled.connect(lambda _: self.invalidate())

        self.plot = make_plot(x_label="m/z", y_label="prominence [%]")
        self.plot.setMouseEnabled(y=False)
        self.title_label = pg.LabelItem(color="k", size="10pt")
        self.title_label.setParentItem(self.plot.getViewBox())
        self.title_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 5))
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def _on_curves_changed(self, main: Curve | None, overlays: list[Curve]) -> None:
        curves = ([main] if main is not None else []) + overlays
        self._colors = {c.mass: c.color for c in curves if c.mass is not None}
        self.invalidate()

    def _stick_colors(self, masses: np.ndarray) -> tuple[list, list[bool]]:
        """Color of each stick and whether it stands out.

        With a visible chromatogram panel: the color of the curve of that mass,
        grey if not plotted. Without: the selected mass in blue, the others black.
        """
        if self.chromatogram is not None and not self.chromatogram.isHidden():
            plotted = [int(m) in self._colors for m in masses]
            colors = [self._colors.get(int(m), OTHER_COLOR) for m in masses]
            return colors, plotted
        plotted = [m == self.selection.mass for m in masses]
        return [SELECTED_COLOR if p else "k" for p in plotted], plotted

    def redraw(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items = []
        self._masses = np.array([])
        sel = self.selection
        if sel.measurement_id is None or sel.compound_idx is None:
            self.title_label.setText("No compound selected")
            return

        compound = self.data.compounds(sel.measurement_id).loc[sel.compound_idx]
        self.title_label.setText(
            f"{html.escape(compound_label(compound.compound_name, sel.compound_idx))}"
            f"   rt {compound.rt_mean_s:.1f} s"
        )
        peaks = self.data.compound_peaks(sel.measurement_id, sel.compound_idx)
        if peaks.empty:
            return
        # One stick per mass: the most prominent peak if a mass appears twice
        peaks = peaks.sort_values("prominence", ascending=False).drop_duplicates("mass").sort_values("mass")
        masses = peaks["mass"].to_numpy(dtype=float)
        intensity = 100 * peaks["prominence"].to_numpy(dtype=float) / peaks["prominence"].max()
        self._masses = masses
        selected = masses == sel.mass
        colors, plotted = self._stick_colors(masses)

        for i, mass in enumerate(masses):
            item = pg.PlotDataItem(
                [mass, mass], [0, intensity[i]], pen=pg.mkPen(colors[i], width=3 if plotted[i] else 2)
            )
            item.setZValue(1 if plotted[i] else 0)
            self.plot.addItem(item)
            self._items.append(item)

        # Hover tooltips at the top of the sticks
        tips = pg.ScatterPlotItem(
            masses, intensity, data=list(zip(peaks["prominence"], peaks["height"], peaks["rt_s"])),
            symbol="o", size=10, pen=None, brush=pg.mkBrush(0, 0, 0, 0), hoverable=True,
            hoverBrush=pg.mkBrush(0, 120, 255, 80),
            tip=lambda x, y, data: (
                f"m/z {x:.0f}: {y:.1f} %\nprominence {data[0]:.0f}\nheight {data[1]:.0f}\nrt {data[2]:.1f} s"
            ),
        )
        self.plot.addItem(tips)
        self._items.append(tips)

        # m/z labels on the most intense sticks and on the selected one
        labelled = set(np.argsort(intensity)[::-1][:N_LABELS]) | set(np.flatnonzero(selected))
        for i in sorted(labelled):
            color = colors[i] if plotted[i] else "k"
            label = pg.TextItem(f"{masses[i]:.0f}", color=color, anchor=(0.5, 1))
            label.setPos(masses[i], intensity[i])
            self.plot.addItem(label)
            self._items.append(label)

        self.plot.setXRange(masses.min() - 5, masses.max() + 5, padding=0)
        self.plot.setYRange(0, Y_MAX, padding=0)

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not len(self._masses):
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        x = vb.mapSceneToView(event.scenePos()).x()
        closest = self._masses[np.argmin(np.abs(self._masses - x))]
        if abs(closest - x) <= CLICK_TOLERANCE:
            self.selection.select_mass(int(closest))
