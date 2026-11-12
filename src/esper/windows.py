"""The parts of a chromatogram measured without interruption.

A quadrupole in SIM measures each mass only in its time windows: its chromatogram is NaN
between them, and the peaks are found and fitted in each window. A scan run measures
every mass at each scan, but keeps a value only where it found a peak: its NaN are not
gaps, the values form one window (``esper.sources.CONTINUOUS``). A TOF chromatogram has
no NaN: one window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_continuous(frame: pd.DataFrame) -> bool:
    """Whether the NaN of the chromatograms are not gaps (scan runs)."""
    from esper.sources import CONTINUOUS

    return bool(frame.attrs.get(CONTINUOUS))


def measured_windows(
    y: np.ndarray, continuous: bool = False, min_points: int = 1
) -> list[np.ndarray]:
    """Indices of the values of each window of a chromatogram, with at least
    ``min_points`` values: its runs of finite values, or all of them if ``continuous``."""
    index = np.flatnonzero(np.isfinite(np.asarray(y, dtype=float)))
    if continuous:
        windows = [index]
    else:
        windows = np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)
    return [w for w in windows if len(w) >= max(min_points, 1)]
