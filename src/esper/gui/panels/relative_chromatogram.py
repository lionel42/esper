"""The curves of a chromatogram panel, each scaled to the main mass in the visible window,
with their peaks."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QVBoxLayout

from esper.gui.data import TIC
from esper.gui.panels.base import BasePanel
from esper.gui.panels.chromatogram import (
    MAX_LEGEND_ENTRIES,
    ChromatogramPanel,
    Curve,
    draw_curve,
)
from esper.gui.panels.chromatogram_items import fit_fill_items, integration_items
from esper.gui.plotting import make_plot
from esper.gui.settings import Integration

# Delay before rescaling the relative plot after a pan/zoom [ms]
RESCALE_DELAY_MS = 100


class RelativeChromatogramPanel(BasePanel):
    """The curves of a chromatogram panel, each scaled to the main mass in the visible window.

    Follows its source panel (x axis linked, same overlays, integration and curve style,
    with the same right click menus), not the selection; its curves are drawn as there
    (``draw_curve``: colors, gaps where a mass was not measured), and their peaks too
    (the integration of the main mass, the fitted shapes of the overlays), without the
    labels; the peaks of an overlay are scaled with it.
    """

    title = "Relative chromatograms"
    watches: set[str] = set()

    def __init__(self, source: ChromatogramPanel, parent=None) -> None:
        super().__init__(source.data, source.selection, parent)
        self.source = source
        self._main: Curve | None = None
        self._overlays: list[Curve] = []
        self._items: list = []
        # the items of the peaks of each curve (main first), scaled with it
        self._peak_items: list[list] = []

        self.plot = make_plot()
        self.plot.setXLink(source.plot)
        source.overlays.add_menu(self.plot)
        self.draw_options = source.draw_options
        self.draw_options.add_menus(self.plot)

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
        for item in self._items + [i for items in self._peak_items for i in items]:
            self.plot.removeItem(item)
        self._items, self._peak_items = [], []
        if self._main is None:
            return
        self._items = [
            draw_curve(self.plot, c, style=self.draw_options.curve_style)
            for c in (self._main, *self._overlays)
        ]
        self._peak_items = self._draw_peaks()
        self.rescale()

    def _draw_peaks(self) -> list[list]:
        """The items of the peaks of each curve, as the source panel draws them: the
        integration of the main mass; the fitted shapes of the overlays (when not too
        many)."""
        sel = self.selection
        drawn: list[list] = [[] for _ in (self._main, *self._overlays)]
        if sel.measurement_id is None:
            return drawn
        integration = self.draw_options.shown_integration(
            self.data.is_fitted(sel.measurement_id)
        )
        if integration == Integration.NONE:
            return drawn
        peaks = self.data.peaks(sel.measurement_id)
        main = self._main
        if sel.mass is not TIC:
            drawn[0], _, _ = integration_items(
                main.t,
                main.y,
                peaks[peaks["mass"] == sel.mass],
                integration,
                selected=sel.compound_idxs,
            )
        if (
            integration == Integration.FIT_SHAPES
            and len(self._overlays) < MAX_LEGEND_ENTRIES
        ):
            for i, curve in enumerate(self._overlays, start=1):
                if curve.mass is not TIC:
                    on = peaks[peaks["mass"] == curve.mass]
                    drawn[i] = fit_fill_items(curve.t, on, curve.color)
        for item in (i for items in drawn for i in items):
            self.plot.addItem(item)
        return drawn

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
        for i, (item, curve) in enumerate(
            zip(self._items[1:], self._overlays), start=1
        ):
            w = (curve.t >= x0) & (curve.t <= x1)
            if not w.any():
                continue
            lo, hi = curve.y[w].min(), curve.y[w].max()
            factor = (main_max - main_min) / (hi - lo) if hi > lo else 0.0
            # the drawn values (NaN in the gaps) scaled
            scaled = main_min + (np.asarray(curve.drawn_y) - lo) * factor
            item.setData(curve.drawn_t, scaled, connect="finite")
            # its peaks: y -> main_min + (y - lo) * factor
            scaling = QTransform(1, 0, 0, factor, 0, main_min - lo * factor)
            for peak_item in self._peak_items[i]:
                peak_item.setTransform(scaling)
        span = max(main_max - main_min, 1.0)
        self.plot.setYRange(main_min - 0.05 * span, main_max + 0.05 * span, padding=0)
