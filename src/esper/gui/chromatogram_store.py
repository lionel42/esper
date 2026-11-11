"""The unit mass chromatograms of the measurements, read lazily for the GUI."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from esper.sources import CONTINUOUS


def mass_column(mass: float) -> str:
    """The column of a mass in a chromatogram file: ``"85"``; ``"84.9"`` for a mass that is
    not a unit mass (e.g. the scans of GCwerks)."""
    return f"{mass:g}"


def with_breaks(series: pd.Series) -> pd.Series:
    """A chromatogram to draw: without its NaN, but one NaN in each gap between the
    values (where the mass was not measured, between time windows), so that the
    curve, drawn with ``connect="finite"``, is not continuous over it."""
    finite = series.notna().to_numpy()
    if finite.all():
        return series
    if not finite.any():
        return series.iloc[:0]
    first, last = finite.argmax(), len(finite) - finite[::-1].argmax()
    keep = finite.copy()
    # the first NaN of each run of NaN between two values
    keep[1:] |= ~finite[1:] & finite[:-1]
    keep[:first] = keep[last:] = False
    return series[keep]


def column_mass(column: str) -> int | float | None:
    """The mass of a column of a chromatogram file (int for a unit mass), None if it is not
    a mass (``tic``)."""
    try:
        mass = float(column)
    except ValueError:
        return None
    return int(mass) if mass.is_integer() else mass


# A unit mass M stands for the masses measured in [M - 0.5, M + 0.5] (69.5: 69 and 70)
UNIT_HALF_WIDTH = 0.5
_TOLERANCE = 1e-6


def unit_masses(masses) -> list[int]:
    """The unit masses of masses as measured (``masses_of_unit``), by increasing mass."""
    units = set()
    for m in masses:
        low = math.ceil(m - UNIT_HALF_WIDTH - _TOLERANCE)
        high = math.floor(m + UNIT_HALF_WIDTH + _TOLERANCE)
        units.update(range(low, high + 1))
    return sorted(units)


def masses_of_unit(masses, unit: float) -> list[int | float]:
    """The masses as measured that a unit mass stands for: those of
    ``[unit - 0.5, unit + 0.5]``, by increasing mass (GCwerks: 69 -> 68.5 ... 69.5; a
    TOF file has the unit masses: 69 -> 69)."""
    return sorted(m for m in masses if abs(m - unit) <= UNIT_HALF_WIDTH + _TOLERANCE)


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
        self._continuous = lru_cache(maxsize=None)(self._read_continuous)
        self._frame = lru_cache(maxsize=n_frames)(self._read_frame)

    def clear(self) -> None:
        self._series.clear()
        self._columns.cache_clear()
        self._continuous.cache_clear()
        self._frame.cache_clear()

    def _path(self, measurement_id: int) -> Path | None:
        path = self._path_of(measurement_id)
        return path if path is not None and path.is_file() else None

    def _read_columns(self, measurement_id: int) -> frozenset[str]:
        path = self._path(measurement_id)
        return frozenset() if path is None else frozenset(pq.read_schema(path).names)

    def _read_continuous(self, measurement_id: int) -> bool:
        path = self._path(measurement_id)
        if path is None:
            return False
        attrs = (pq.read_schema(path).metadata or {}).get(b"PANDAS_ATTRS")
        return attrs is not None and json.loads(attrs).get(CONTINUOUS) == "1"

    def continuous(self, measurement_id: int) -> bool:
        """Whether a missing value is a mass not found in a scan, not a gap (GCwerks
        scans, ``esper.sources.CONTINUOUS``): the curves link the values around."""
        return self._continuous(measurement_id)

    def to_draw(self, measurement_id: int, series: pd.Series) -> pd.Series:
        """A chromatogram of the measurement to draw: with one NaN in each gap
        (``with_breaks``), or without NaN when ``continuous``."""
        if self.continuous(measurement_id):
            return series.dropna()
        return with_breaks(series)

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
        quadrupole, esper.sources.gcwerks) are NaN outside them.
        """
        cached = self._cached(measurement_id, column)
        return None if cached is None else cached[0]

    def drawn(self, measurement_id: int, column: str) -> pd.Series | None:
        """One chromatogram to draw (``to_draw``)."""
        cached = self._cached(measurement_id, column)
        return None if cached is None else cached[1]

    def _cached(
        self, measurement_id: int, column: str
    ) -> tuple[pd.Series, pd.Series] | None:
        """(without NaN, to draw) of one column of the file (cached)."""
        key = (measurement_id, column)
        if key in self._series:
            self._series.move_to_end(key)
            return self._series[key]
        if column not in self.columns(measurement_id):
            return None
        series = pd.read_parquet(self._path_of(measurement_id), columns=[column])[
            column
        ]
        cached = self._series[key] = (
            series.dropna(),
            self.to_draw(measurement_id, series),
        )
        if len(self._series) > self._cache_size:
            self._series.popitem(last=False)
        return cached
