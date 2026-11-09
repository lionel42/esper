"""The curves of a chromatogram panel, each scaled to the main mass in the visible window."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.panels.base import BasePanel
from esper.gui.panels.chromatogram import ChromatogramPanel, Curve
from esper.gui.plotting import add_curve, make_plot

# Delay before rescaling the relative plot after a pan/zoom [ms]
RESCALE_DELAY_MS = 100


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
        source.overlays.add_menu(self.plot)

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
        if self._main is None:
            return
        if not self.isVisible():
            self._outdated = True  # the window may have changed: rescaled when shown
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
