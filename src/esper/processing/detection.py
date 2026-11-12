"""Peak detection in the chromatograms of a run, and their grouping into compounds.

The peaks of each mass are found in each window where it was measured
(:mod:`esper.windows`), after its baseline is removed (``peaxart.remove_baseline``, of
the window), with the legacy detection of alpinac (``peaxart.find_chromatogram_peaks``).
The peaks of all masses are then grouped into compounds of co-eluting peaks
(``peaxart.group_peaks_legacy``), whatever the method that identifies them.

The peaks and compounds have the columns of the tables of the database (as imported from
alpinac, :mod:`esper.alpinac`). The baseline is removed only to find the peaks: the fit
is done on the stored chromatograms, with its own background.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import peaxart as px

from esper.deconvolution import chromatogram_masses
from esper.windows import is_continuous, measured_windows

logger = logging.getLogger(__name__)

# Columns of peaxart's peak and compound tables -> columns of the database (as alpinac's)
PEAK_COLUMNS = {
    "compound_id": "compound_idx",
    "mass": "mass",
    "peak_rt": "rt_s",
    "peak_heights": "height",
    "prominences": "prominence",
    "widths": "width_scans",
    "width_heights": "width_height",
    "left_bases": "left_base_scan",
    "right_bases": "right_base_scan",
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


def find_peaks(
    frame: pd.DataFrame,
    baseline: px.BaselineConfig | None,
    detection: px.LegacyDeconvolutionConfig,
) -> pd.DataFrame:
    """The peaks of all the masses of a run (``peaxart.find_chromatogram_peaks`` columns),
    found in each window of each mass; their scans are those of the whole run.

    ``frame``: the chromatograms (index: rt [s], one column per mass; ``tic`` left out).
    ``baseline``: removed from each window before (None: not removed).
    """
    t = frame.index.to_numpy(dtype=float)
    continuous = is_continuous(frame)
    parts = []
    for column in chromatogram_masses(frame):
        y = frame[column].to_numpy(dtype=float)
        for index in measured_windows(y, continuous, MIN_WINDOW_POINTS):
            values = y[index]
            if baseline is not None:
                values = px.remove_baseline(values, baseline)
            peaks = px.find_chromatogram_peaks(
                values[:, None],
                t[index],
                [_mass(column)],
                detection.width,
                detection.height,
                detection.prominence,
            )
            if len(peaks):
                for scans in ("left_bases", "right_bases"):  # of the run
                    peaks[scans] = index[peaks[scans].to_numpy(dtype=int)]
                parts.append(peaks)
    if not parts:
        return pd.DataFrame(columns=FOUND_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def detect(
    frame: pd.DataFrame,
    baseline: px.BaselineConfig | None,
    detection: px.LegacyDeconvolutionConfig,
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
    # Width at half prominence, s: between its crossings (``width_scans`` counts the values
    # of its window, not evenly spaced in a scan run)
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
