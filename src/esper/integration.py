"""Integration of the peaks found by alpinac in the unit mass chromatograms."""

from __future__ import annotations

import numpy as np

# Integration bounds, in half widths at half maximum from the apex (~3.5 sigma for a gaussian)
N_HWHM = 3.0


def integration_bounds(peaks, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of each peak [s], NaN for the peaks without ips (see ``has_ips``).

    ``start_s``/``stop_s`` are the half maximum crossings, the peak is extended
    to ``N_HWHM`` half widths on each side but not beyond its prominence bases.
    """
    rt, start_hm, stop_hm = (
        peaks[c].to_numpy(dtype=float) for c in ("rt_s", "start_s", "stop_s")
    )
    valid = has_ips(peaks)
    last = len(t) - 1

    def base(column):
        scans = np.nan_to_num(peaks[column].to_numpy(dtype=float)).astype(int)
        return np.where(valid, t[np.clip(scans, 0, last)], np.nan)

    start = np.maximum(rt - N_HWHM * (rt - start_hm), base("left_base_scan"))
    stop = np.minimum(rt + N_HWHM * (stop_hm - rt), base("right_base_scan"))
    return start, stop


def has_ips(peaks) -> np.ndarray:
    """Peaks with the half maximum crossings and prominence bases of the extraction.

    A peak split by the fit (esper.fitting) has none.
    """
    columns = ("start_s", "stop_s", "left_base_scan", "right_base_scan")
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
