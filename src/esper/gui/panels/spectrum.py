"""Mass spectrum of the selected compound."""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout

from esper.gui.data import EsperData
from esper.gui.panels.base import BasePanel
from esper.gui.panels.chromatogram import TARGET_COLOR, ChromatogramPanel, Curve, compound_label
from esper.gui.plotting import make_plot
from esper.gui.library_loader import library_loader
from esper.gui.references import current_reference
from esper.gui.selection import Selection
from esper.gui.settings import settings_manager
from esper.libraries import write_jdx

# Number of most intense masses labelled with their m/z
N_LABELS = 12
# A click selects the closest stick within this distance [m/z]
CLICK_TOLERANCE = 1.0
# Selected mass when there is no chromatogram to take the colors from
SELECTED_COLOR = "#0050c8"
# Masses not plotted in the chromatogram
OTHER_COLOR = "#aaaaaa"
# Top of the y axis [%], leaves room for the label of the 100 % stick
Y_MAX = 135
# Reference spectrum, mirrored below
REFERENCE_COLOR = "#7b3fa0"
# Number of most intense reference masses labelled
N_REFERENCE_LABELS = 8


class SpectrumPanel(BasePanel):
    """Spectrum of the selected compound: one stick per mass, height = peak prominence.

    Clicking a stick selects that mass. With a visible chromatogram panel as
    ``chromatogram``, the masses plotted there get the same colors here.
    The expected ions of the selected target are marked with dashed lines.
    The reference spectrum (``selection.reference``, by default alpinac's best
    candidate) is mirrored below, the library can be chosen with a right click.
    The intensity of the measured peaks is set in the settings.
    The measured spectrum can be exported with a right click.
    """

    title = "Spectrum"
    watches = {"measurement_id", "compound_idx", "mass", "target", "reference"}

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

        self.settings = settings_manager()
        self.settings.changed.connect(self.invalidate)
        self.loader = library_loader()
        self.loader.loaded.connect(self.invalidate)
        self._reference: tuple[str, str] | None = None

        self.plot = make_plot()
        self.plot.setMouseEnabled(y=False)
        self.title_label = pg.LabelItem(color="k", size="10pt")
        self.title_label.setParentItem(self.plot.getViewBox())
        self.title_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 5))
        self.reference_label = pg.LabelItem(color=REFERENCE_COLOR, size="10pt")
        self.reference_label.setParentItem(self.plot.getViewBox())
        self.reference_label.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -5))

        menu = self.plot.getViewBox().menu
        menu.addSeparator()
        self.library_menu = menu.addMenu("Reference library")
        self.library_menu.aboutToShow.connect(self._fill_library_menu)
        self.export_menu = menu.addMenu("Export spectrum")
        self.export_jdx_action = self.export_menu.addAction("JCAMP-DX (.jdx)…")
        self.export_jdx_action.triggered.connect(self._export_jdx)
        # TODO: alpinac fragment file (.txt), as read by libraries.read_alpinac_fragments
        self.export_menu.addAction("alpinac (.txt)… (not implemented yet)").setEnabled(False)
        self.export_menu.aboutToShow.connect(
            lambda: self.export_jdx_action.setEnabled(self.selection.compound_idx is not None)
        )
        # Folder of the last export
        self._export_dir = Path.home()
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
        expected = self.data.target_masses(sel.target) if sel.target is not None else []
        self._mark_expected(expected)
        if sel.measurement_id is None or sel.compound_idx is None:
            if sel.target is not None:
                self.title_label.setText(f"{html.escape(sel.target)}: not found, expected ions")
                if expected:
                    self.plot.setXRange(min(expected) - 5, max(expected) + 5, padding=0)
                    self.plot.setYRange(0, Y_MAX, padding=0)
            else:
                self.title_label.setText("No compound selected")
            return

        compound = self.data.compounds(sel.measurement_id).loc[sel.compound_idx]
        self.title_label.setText(
            f"{html.escape(compound_label(compound.main_name, sel.compound_idx))}"
            f"   rt {compound.rt_mean_s:.1f} s"
        )
        kind = self.settings.settings.spectrum_intensity
        spectrum = self.data.compound_spectrum(sel.measurement_id, sel.compound_idx, kind)
        if not len(spectrum) or spectrum.intensities.max() <= 0:
            return
        masses = spectrum.mz.astype(float)
        values = spectrum.intensities
        intensity = 100 * values / values.max()
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
            masses, intensity, data=list(values),
            symbol="o", size=10, pen=None, brush=pg.mkBrush(0, 0, 0, 0), hoverable=True,
            hoverBrush=pg.mkBrush(0, 120, 255, 80),
            tip=lambda x, y, data: f"m/z {x:.0f}: {y:.1f} %\n{kind} {data:.0f}",
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

        low, high = masses.min(), masses.max()
        if expected:
            low, high = min(low, min(expected)), max(high, max(expected))
        reference = self._draw_reference(spectrum)
        if reference is not None and len(reference):
            low, high = min(low, reference.mz.min()), max(high, reference.mz.max())
        self.plot.setXRange(low - 5, high + 5, padding=0)
        self.plot.setYRange(-Y_MAX if reference is not None else 0, Y_MAX, padding=0)

    def _draw_reference(self, spectrum):
        """Mirror the reference spectrum below the measured one; returns it (or None)."""
        self._reference = current_reference(self.data, self.selection)
        self.reference_label.setText("")
        if not self.loader.ensure():
            self.reference_label.setText("Loading libraries…")
            return None
        if self._reference is None:
            return None
        library, name = self._reference
        if not self.loader.libraries.is_loaded(library):
            # A library not enabled, chosen in a search
            self.loader.ensure(all_libraries=True)
            self.reference_label.setText("Loading libraries…")
            return None
        reference = self.settings.libraries.spectrum(library, name)
        if reference is None or not len(reference) or reference.intensities.max() <= 0:
            self.reference_label.setText(f"{html.escape(name)} not in {library}")
            return None
        score, matched = self.settings.libraries.score(spectrum, reference)
        self.reference_label.setText(
            f"{html.escape(name)} ({library})   score {score:.3f}, {matched} matched"
        )
        ref = 100 * reference.intensities / reference.intensities.max()
        x = np.repeat(reference.mz.astype(float), 2)
        y = np.column_stack([np.zeros(len(ref)), -ref]).ravel()
        item = pg.PlotDataItem(x, y, connect="pairs", pen=pg.mkPen(REFERENCE_COLOR, width=2))
        self.plot.addItem(item)
        self._items.append(item)
        axis = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("k", width=1))
        self.plot.addItem(axis)
        self._items.append(axis)
        for i in np.argsort(ref)[::-1][:N_REFERENCE_LABELS]:
            label = pg.TextItem(f"{reference.mz[i]:.0f}", color=REFERENCE_COLOR, anchor=(0.5, 0))
            label.setPos(float(reference.mz[i]), -ref[i])
            self.plot.addItem(label)
            self._items.append(label)
        return reference

    def _fill_library_menu(self) -> None:
        """Libraries containing the current reference spectrum, to show it from another one."""
        self.library_menu.clear()
        if not self.loader.ensure():
            self.library_menu.addAction("Loading libraries…").setEnabled(False)
            return
        if self._reference is None:
            self.library_menu.addAction("No reference").setEnabled(False)
            return
        current, name = self._reference
        for library in self.settings.libraries.libraries_with(name):
            action = self.library_menu.addAction(library)
            action.setCheckable(True)
            action.setChecked(library == current)
            action.triggered.connect(lambda _=False, lib=library: self.selection.select_reference((lib, name)))

    def _export_jdx(self) -> None:
        """Save the measured spectrum of the selected compound (current intensity) as .jdx."""
        sel = self.selection
        if sel.measurement_id is None or sel.compound_idx is None:
            return
        measurement = self.data.measurements.loc[sel.measurement_id, "name"]
        compound = self.data.compounds(sel.measurement_id).loc[sel.compound_idx]
        label = compound_label(compound.main_name, sel.compound_idx)
        kind = self.settings.settings.spectrum_intensity
        spectrum = self.data.compound_spectrum(sel.measurement_id, sel.compound_idx, kind)
        if not len(spectrum) or spectrum.intensities.max() <= 0:
            QMessageBox.warning(self, "Export spectrum", f"{label} has no spectrum to export.")
            return
        default = re.sub(r"[^\w.,+-]+", "_", f"{measurement}_{label}_rt{compound.rt_mean_s:.0f}s") + ".jdx"
        file, _ = QFileDialog.getSaveFileName(
            self, "Export spectrum", str(self._export_dir / default), "JCAMP-DX (*.jdx)"
        )
        if not file:
            return
        file = Path(file)
        # Not with_suffix: measurement names contain dots
        if file.suffix.lower() != ".jdx":
            file = file.with_name(file.name + ".jdx")
        comments = {
            "MEASUREMENT": measurement,
            "COMPOUND INDEX": str(sel.compound_idx),
            "RETENTION TIME [S]": f"{compound.rt_mean_s:.2f}",
            "INTENSITY": kind,
        }
        try:
            write_jdx(file, spectrum, title=label, comments=comments)
        except OSError as e:
            QMessageBox.critical(self, "Export spectrum", f"Could not write {file}:\n{e}")
            return
        self._export_dir = file.parent

    def _mark_expected(self, masses: list[int]) -> None:
        """Dashed lines at the expected ions of the target, labelled with their rank."""
        pen = pg.mkPen(TARGET_COLOR, width=1, style=Qt.PenStyle.DashLine)
        for rank, mass in enumerate(masses):
            line = pg.InfiniteLine(pos=mass, angle=90, pen=pen)
            line.setZValue(-1)
            self.plot.addItem(line)
            self._items.append(line)
            label = pg.TextItem(f"Q{rank}", color=TARGET_COLOR, anchor=(0, 0))
            label.setPos(mass, Y_MAX)
            self.plot.addItem(label)
            self._items.append(label)

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
