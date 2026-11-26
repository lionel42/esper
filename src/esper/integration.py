"""Integration of the peaks found by alpinac in the unit mass chromatograms."""

from __future__ import annotations

import numpy as np

# Integration bounds, in half widths at half maximum from the apex (~3.5 sigma for a gaussian)
N_HWHM = 3.0


# The prominence bases of the peaks [s] (peak, extraction_peak)
BASE_COLUMNS = ("left_base_s", "right_base_s")


def scan_times(scans, times) -> np.ndarray:
    """The times [s] of scans (NaN: none; ``times``: of all the scans of the run, None:
    unknown, all NaN), e.g. the prominence bases of peaks given as scans (peaxart,
    alpinac)."""
    scans = np.asarray(scans, dtype=float)
    if times is None or len(times) == 0:
        return np.full(scans.shape, np.nan)
    times = np.asarray(times, dtype=float)
    index = np.clip(np.nan_to_num(scans), 0, len(times) - 1).astype(int)
    return np.where(np.isfinite(scans), times[index], np.nan)


def integration_bounds(peaks) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of each peak [s], NaN for the peaks without ips (see ``has_ips``).

    ``start_s``/``stop_s`` are the half maximum crossings, the peak is extended
    to ``N_HWHM`` half widths on each side but not beyond its prominence bases
    (``BASE_COLUMNS``).
    """
    rt, start_hm, stop_hm, left, right = (
        peaks[c].to_numpy(dtype=float)
        for c in ("rt_s", "start_s", "stop_s", *BASE_COLUMNS)
    )
    valid = has_ips(peaks)
    start = np.where(valid, np.maximum(rt - N_HWHM * (rt - start_hm), left), np.nan)
    stop = np.where(valid, np.minimum(rt + N_HWHM * (stop_hm - rt), right), np.nan)
    return start, stop


def has_ips(peaks) -> np.ndarray:
    """Peaks with the half maximum crossings and prominence bases of the extraction
    (``BASE_COLUMNS``).

    A peak split by the fit (esper.fitting) has none.
    """
    columns = ("start_s", "stop_s", *BASE_COLUMNS)
    return np.all(
        [np.isfinite(peaks[c].to_numpy(dtype=float)) for c in columns], axis=0
    )


def peak_area(t: np.ndarray, y: np.ndarray, start: float, stop: float) -> float:
    """Area above the straight baseline between start and stop [counts * s]."""
    mask = (t >= start) & (t <= stop)
    if mask.sum() < 2:
        return np.nan
    y_start, y_stop = np.interp([start, stop], t, y)
    baseline = np.interp(t[mask], [start, stop], [y_start, y_stop])
    return float(np.trapezoid(y[mask] - baseline, t[mask]))
