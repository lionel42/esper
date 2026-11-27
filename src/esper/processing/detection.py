"""Peak detection in the chromatograms of a run, and their grouping into compounds.

The peaks of each mass are found in each window where it was measured
(:mod:`esper.windows`), after its baseline is removed (``peaxart.remove_baseline``, of
the window), with the legacy detection of alpinac (``peaxart.find_chromatogram_peaks``),
its prominence threshold relative to the noise around each peak
(``esper.processing.config.DetectionConfig``).
The peaks of all masses are then grouped into compounds of co-eluting peaks
(``peaxart.group_peaks_legacy``), whatever the method that identifies them.

The peaks and compounds have the columns of the tables of the database (as imported from
alpinac, :mod:`esper.alpinac`). The baseline is removed only to find the peaks: the fit
is done on the stored chromatograms, with its own background.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import peaxart as px
from peaxart.chromatogram import noise_std

from esper.deconvolution import chromatogram_masses
from esper.windows import cuts_of, is_continuous, measured_windows

if TYPE_CHECKING:
    from esper.processing.config import DetectionConfig

logger = logging.getLogger(__name__)

# Columns of peaxart's peak and compound tables -> columns of the database (as alpinac's)
PEAK_COLUMNS = {
    "compound_id": "compound_idx",
    "mass": "mass",
    "peak_rt": "rt_s",
    "peak_heights": "height",
    "prominences": "prominence",
    "width_heights": "width_height",
    "left_bases": "left_base_s",
    "right_bases": "right_base_s",
    "start_ip": "start_s",
    "stopp_ip": "stop_s",
    "flatness": "flatness",
    "tailing_ratio": "tailing_ratio",
    "is_max_of_cmp": "is_max_of_cmp",
}
COMPOUND_COLUMNS = {
    "peak_rt_mean": "rt_mean_s",
    "peak_rt_std": "rt_std_s",
    "peak_rt_count": "n_peaks",
    "prominences_sum": "prominence_sum",
    "start_ip_mean": "start_mean_s",
    "start_ip_std": "start_std_s",
    "stopp_ip_mean": "stop_mean_s",
    "stopp_ip_std": "stop_std_s",
    "mai_": "main_ion_mass",
}
# Columns of peaxart.find_chromatogram_peaks
FOUND_COLUMNS = [
    "peak_heights",
    "prominences",
    "left_bases",
    "right_bases",
    "widths",
    "width_heights",
    "start_ip",
    "stopp_ip",
    "peak_rt",
    "mass",
    "flatness",
    "tailing_ratio",
]
# Fewest values of a window to look for peaks in it
MIN_WINDOW_POINTS = 5


def _mass(column: str) -> int | float:
    value = float(column)
    return int(value) if value.is_integer() else value


def local_noise(t: np.ndarray, y: np.ndarray, half_window: float | None) -> np.ndarray:
    """The noise standard deviation at each value of ``y`` (``noise_std``): of the values
    within ``half_window`` s, evaluated every ``half_window / 4`` and interpolated (None, or
    a series not longer than the window: of the whole series)."""
    t = np.asarray(t, dtype=float)
    if half_window is None or len(t) < 2 or t[-1] - t[0] <= 2 * half_window:
        return np.full(len(t), noise_std(y))
    grid = np.arange(t[0], t[-1] + half_window / 4, half_window / 4)
    lo = np.searchsorted(t, grid - half_window)
    hi = np.searchsorted(t, grid + half_window, side="right")
    noise = np.array([noise_std(y[a:b]) for a, b in zip(lo, hi, strict=True)])
    return np.interp(t, grid, noise)


def find_peaks(
    frame: pd.DataFrame,
    baseline: px.BaselineConfig | None,
    detection: DetectionConfig,
) -> pd.DataFrame:
    """The peaks of all the masses of a run (``peaxart.find_chromatogram_peaks`` columns),
    found in each window of each mass; their prominence bases (``left_bases``,
    ``right_bases``) as times [s].

    ``frame``: the chromatograms (index: rt [s], one column per mass; ``tic`` left out).
    ``baseline``: removed from each window before (None: not removed).
    ``detection``: its prominence threshold relative to the noise around each peak
    (``local_noise``).
    """
    t = frame.index.to_numpy(dtype=float)
    continuous = is_continuous(frame)
    parts = []
    for column in chromatogram_masses(frame):
        y = frame[column].to_numpy(dtype=float)
        cuts = cuts_of(frame, column)
        for index in measured_windows(y, continuous, MIN_WINDOW_POINTS, cuts):
            values = y[index]
            # of the measured values: the baseline removal smooths them
            min_prominence = detection.min_prominence(
                local_noise(t[index], values, detection.noise_window_s)
            )
            if baseline is not None:
                values = px.remove_baseline(values, baseline)
            peaks = px.find_chromatogram_peaks(
                values[:, None],
                t[index],
                [_mass(column)],
                detection.width,
                detection.height,
                min_prominence,
            )
            if len(peaks):
                for bases in ("left_bases", "right_bases"):  # scans of the window
                    peaks[bases] = t[index[peaks[bases].to_numpy(dtype=int)]]
                parts.append(peaks)
    if not parts:
        return pd.DataFrame(columns=FOUND_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def detect(
    frame: pd.DataFrame,
    baseline: px.BaselineConfig | None,
    detection: DetectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The peaks (indexed by peak_idx) and compounds (indexed by compound_idx) of a run,
    with the columns of the database; the compounds not identified yet."""
    found = find_peaks(frame, baseline, detection)
    if found.empty:  # e.g. a blank: no peak (stored as such, not processed again)
        peaks = pd.DataFrame(
            columns=[*PEAK_COLUMNS.values(), "width_s"],
            index=pd.RangeIndex(0, name="peak_idx"),
        )
        compounds = pd.DataFrame(
            columns=[
                *COMPOUND_COLUMNS.values(),
                "retention_index",
                "is_ref_ri",
                "valid_candidate",
                "compound_name",
            ],
            index=pd.Index([], dtype=int, name="compound_idx"),
        )
        logger.debug("no peaks")
        return peaks, compounds
    compounds, peaks = px.group_peaks_legacy(
        found,
        detection.rt_threshold,
        detection.rt_threshold_start_ip,
        detection.rt_threshold_stopp_ip,
    )
    peaks = peaks[list(PEAK_COLUMNS)].rename(columns=PEAK_COLUMNS)
    peaks["is_max_of_cmp"] = peaks["is_max_of_cmp"].astype(bool).astype(int)
    # Width at half prominence, s: between its crossings (peaxart's ``widths`` counts the
    # values of its window, not evenly spaced in a scan run)
    peaks["width_s"] = peaks["stop_s"] - peaks["start_s"]
    peaks.index = pd.RangeIndex(len(peaks), name="peak_idx")

    compounds.columns = ["_".join(c) if isinstance(c, tuple) else c for c in compounds]
    compounds = compounds[list(COMPOUND_COLUMNS)].rename(columns=COMPOUND_COLUMNS)
    compounds.index = compounds.index.astype(int).rename("compound_idx")
    compounds = compounds.assign(
        retention_index=np.nan, is_ref_ri=0, valid_candidate=0, compound_name=None
    )
    logger.debug(f"{len(peaks)} peaks, {len(compounds)} compounds")
    return peaks, compounds
