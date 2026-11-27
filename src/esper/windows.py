"""The parts of a chromatogram measured without interruption.

A quadrupole in SIM measures each mass only in its time windows: its chromatogram is NaN
between them, and the peaks are found and fitted in each window. A scan run measures
every mass at each scan, but keeps a value only where it found a peak: its NaN are not
gaps, the values form one window (``esper.sources.CONTINUOUS``). A TOF chromatogram has
no NaN: one window, cut where the acquisition steps its background (the raw sums of
``esper chromatograms``: filament emission, resonant ejection; :data:`CUTS`), as its
baseline was.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

CUTS = "esper.cuts"
"""Key of the ``attrs`` of chromatograms: JSON {column: [rt, ...]}, the rts where the
background of a column steps (``esper.chromatograms.baseline_cuts``): a window starts
there."""


def is_continuous(frame: pd.DataFrame) -> bool:
    """Whether the NaN of the chromatograms are not gaps (scan runs)."""
    from esper.sources import CONTINUOUS

    return bool(frame.attrs.get(CONTINUOUS))


def cuts_of(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Indices (rows of ``frame``) where a window of ``column`` starts at a step of its
    background (:data:`CUTS`; none without)."""
    text = frame.attrs.get(CUTS)
    rts = json.loads(text).get(column, []) if text else []
    if not rts:
        return np.zeros(0, dtype=int)
    t = frame.index.to_numpy(dtype=float)
    return np.searchsorted(t, np.asarray(rts, dtype=float) - 1e-6)


def measured_windows(
    y: np.ndarray, continuous: bool = False, min_points: int = 1, cuts=()
) -> list[np.ndarray]:
    """Indices of the values of each window of a chromatogram, with at least
    ``min_points`` values: its runs of finite values, or all of them if ``continuous``;
    a window is also cut at the indices ``cuts`` (a window starts there, ``cuts_of``)."""
    index = np.flatnonzero(np.isfinite(np.asarray(y, dtype=float)))
    if continuous:
        windows = [index]
    else:
        windows = np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)
    cuts = np.asarray(cuts, dtype=int)
    if len(cuts):
        windows = [
            part for w in windows for part in np.split(w, np.searchsorted(w, cuts))
        ]
    return [w for w in windows if len(w) >= max(min_points, 1)]
