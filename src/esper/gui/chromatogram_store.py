"""The unit mass chromatograms of the measurements, read lazily for the GUI."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class ChromatogramStore:
    """Chromatogram files (parquet, index: rt [s], one column per unit mass and ``tic``).

    Single columns are read on demand and the last ``cache_size`` are kept; the
    whole file of the last ``n_frames`` measurements is kept too (to draw many
    masses at once).
    """

    def __init__(
        self,
        path_of: Callable[[int], Path | None],
        cache_size: int = 64,
        n_frames: int = 2,
    ) -> None:
        """``path_of``: the chromatogram file of a measurement (None: none)."""
        self._path_of = path_of
        self._cache_size = cache_size
        self._series: OrderedDict = OrderedDict()
        self._columns = lru_cache(maxsize=None)(self._read_columns)
        self._frame = lru_cache(maxsize=n_frames)(self._read_frame)

    def clear(self) -> None:
        self._series.clear()
        self._columns.cache_clear()
        self._frame.cache_clear()

    def _path(self, measurement_id: int) -> Path | None:
        path = self._path_of(measurement_id)
        return path if path is not None and path.is_file() else None

    def _read_columns(self, measurement_id: int) -> frozenset[str]:
        path = self._path(measurement_id)
        return frozenset() if path is None else frozenset(pq.read_schema(path).names)

    def _read_frame(self, measurement_id: int) -> pd.DataFrame | None:
        path = self._path(measurement_id)
        return None if path is None else pd.read_parquet(path)

    def columns(self, measurement_id: int) -> frozenset[str]:
        """Columns of the file of a measurement (empty without file)."""
        return self._columns(measurement_id)

    def frame(self, measurement_id: int) -> pd.DataFrame | None:
        """All the chromatograms of a measurement (whole file)."""
        return self._frame(measurement_id)

    def series(self, measurement_id: int, column: str) -> pd.Series | None:
        """One chromatogram; only that column of the file is read.

        Without its NaN: the masses measured only in some time windows (GCwerks
        quadrupole, esper.gcwerks) are NaN outside them.
        """
        key = (measurement_id, column)
        if key in self._series:
            self._series.move_to_end(key)
            return self._series[key]
        if column not in self.columns(measurement_id):
            return None
        series = pd.read_parquet(self._path_of(measurement_id), columns=[column])[
            column
        ].dropna()
        self._series[key] = series
        if len(self._series) > self._cache_size:
            self._series.popitem(last=False)
        return series
