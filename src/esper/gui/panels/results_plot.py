"""One compound across all measurements."""

from __future__ import annotations

import pandas as pd
import pyqtgraph as pg

from esper.gui.data import EsperData
from esper.gui.panels.run_scatter import RunScatterPanel
from esper.gui.selection import Selection
from esper.gui.style import CROSS, runtype_colors

# Label -> column of EsperData.results
Y_OPTIONS = {
    "prominence sum": "prominence_sum",
    # Main ion: the target's (target_ion rank 0), else the most prominent peak
    "main ion height": "main_height",
    "main ion prominence": "main_prominence",
    "rt": "rt_mean_s",
    "retention index": "retention_index",
    "score": "best_score",
    "score combined": "best_score_combined",
    "n peaks": "n_peaks",
    # Fit (esper.fitting)
    "fit area sum": "fit_area_sum",
    "main ion fit area": "main_fit_area",
    "main ion fit area rel. u": "main_fit_area_rel_u",
    "main ion fit height": "main_fit_height",
    "main ion fit rt": "main_fit_rt_s",
    "main ion fit sigma": "main_fit_sigma_s",
    "main ion fit tailing": "main_fit_tailing",
    "main ion fit baseline": "main_fit_baseline",
    "main ion fit S/N": "main_fit_sn",
    "main ion mass": "main_mass",
    "n fitted peaks": "n_fitted",
}
X_COLUMNS = {"rt": "rt_mean_s"}


class ResultsPlotPanel(RunScatterPanel):
    """The selected compound (by name) in every measurement, one point per occurrence.

    Clicking a point selects that measurement and compound, a double click
    asks for its chromatogram.
    """

    title = "Results"
    X_COLUMNS = X_COLUMNS
    Y_OPTIONS = Y_OPTIONS
    watches = {"candidate", "target", "identification"}

    def __init__(self, data: EsperData, selection: Selection, parent=None) -> None:
        super().__init__(data, selection, parent, valid_only=True)
        self._loaded_key = None
        self._loaded: pd.DataFrame | None = None

    def on_selection_changed(self, fields: set) -> None:
        if "identification" in fields:
            # Names edited or compounds merged: the results are queried again
            self._loaded_key = None
        super().on_selection_changed(fields)

    def load(self) -> pd.DataFrame | None:
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
        df = self._loaded
        return None if df is None or df.empty else df

    def plot_title(self) -> str:
        return f"{self.selection.label or ''} {self.options['y']}"

    def draw_points(self, df: pd.DataFrame) -> None:
        sample_types = self.data.sample_types()
        colors = runtype_colors(sample_types)
        for stype in sample_types:
            sub = df[df["sample_type"] == stype]
            if sub.empty:
                continue
            self.add_points(
                sub,
                ["measurement_id", "compound_idx", "name", "rt_mean_s"],
                tip=lambda x, y, data: f"{data[2]}\nrt {data[3]:.1f} s\n{y:.4g}",
                symbol=CROSS,
                size=14,
                pen=pg.mkPen(colors[stype], width=2),
                hoverSize=20,
                name=stype,
            )

    def select_point(self, data: tuple) -> None:
        measurement_id, compound_idx, *_ = data
        self.selection.select_compound(int(measurement_id), int(compound_idx))
