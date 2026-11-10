"""Runs of a GCwerks directory: their chromatograms (instrument ``gcwerks``).

A GCwerks directory (e.g. a Medusa GC-MS) has one folder per year with the
mass-spectrometer chromatograms of each run::

    <gcdir>/
        config/gcwerks.conf
        <yy>/chromatograms/channel0/<yymmdd.hhmm.type[.number]>    # *.cm format, no suffix

The files are read with the `gcwerks <https://github.com/lionel42/gcwerks>`_ package and
written as esper chromatograms (``chromatograms/<name>.parquet``: index ``rt`` [s], one
column per unit mass and ``tic``). Only the chromatograms are imported: no peaks, no
compounds (``peak_source`` NULL: not fitted).

The quadrupole measures each mass only in its time windows (SIM): a mass column is NaN
outside them. The signal is as GCwerks stores it, baseline included.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from esper.sources import GCWERKS, MeasurementImport, Run

if TYPE_CHECKING:
    from gcwerks.io.chromatogram import Chromatogram

logger = logging.getLogger(__name__)

INSTRUMENT = GCWERKS
PEAK_SOURCE = None  # chromatograms only
# measurement.acq_start_source: the start written in the chromatogram file (to the s)
START_FILE = "gcwerks"

CHROMATOGRAMS_DIRNAME = "chromatograms"
DEFAULT_CHANNEL = "channel0"
CONFIG_FILE = Path("config") / "gcwerks.conf"
# Run files: yymmdd.hhmm.<type>[.<number>] (e.g. 260112.2114.air.7, 260112.1907.tune)
_RUN_NAME_RE = re.compile(r"^\d{6}\.\d{4}\.(?P<type>[^.]+)(\.[^.]+)*$")
# Run types not imported: the tune files are not chromatograms (a few scans of the tuning)
SKIPPED_TYPES = frozenset({"tune"})
_YEAR_RE = re.compile(r"^\d{2}$")
PROVENANCE = "esper.gcwerks"


def gcwerks_root(path: Path) -> Path | None:
    """The GCwerks directory containing ``path`` (or ``path`` itself), None if not in one.

    A GCwerks directory has ``config/gcwerks.conf``.
    """
    path = Path(path).resolve()
    for folder in (path, *path.parents):
        if (folder / CONFIG_FILE).is_file():
            return folder
    return None


def is_gcwerks_source(path: Path) -> bool:
    return gcwerks_root(path) is not None


def is_run_file(path: Path) -> bool:
    """A chromatogram file of a run to import (not of a :data:`SKIPPED_TYPES`)."""
    match = _RUN_NAME_RE.match(path.name)
    return match is not None and match["type"] not in SKIPPED_TYPES and path.is_file()


def find_chromatogram_files(source: Path, channel: str = DEFAULT_CHANNEL) -> list[Path]:
    """The chromatogram files of ``source``, sorted by name (= by time).

    ``source`` can be the GCwerks directory, a year folder (``<yy>``), its
    ``chromatograms`` or ``chromatograms/<channel>`` folder, or a single file. Only the
    files of ``channel`` are taken (not e.g. ``channel0.integration``).
    """
    source = Path(source)
    if source.is_file():
        return [source]
    if source.name == channel:
        folders = [source]
    elif source.name == CHROMATOGRAMS_DIRNAME:
        folders = [source / channel]
    elif (source / CHROMATOGRAMS_DIRNAME).is_dir():  # a year
        folders = [source / CHROMATOGRAMS_DIRNAME / channel]
    else:  # the GCwerks directory: its years
        folders = [
            year / CHROMATOGRAMS_DIRNAME / channel
            for year in source.iterdir()
            if _YEAR_RE.match(year.name)
        ]
    files = [p for folder in folders if folder.is_dir() for p in folder.iterdir()]
    return sorted((p for p in files if is_run_file(p)), key=lambda p: p.name)


def chromatogram_frame(chromatogram: Chromatogram) -> tuple[pd.DataFrame, list[str]]:
    """The chromatograms as esper stores them, and warnings.

    Index ``rt`` (s since the start of the run), one column per unit mass (``"43"``,
    NaN where the mass was not measured), then ``tic``. A mass measured in several
    windows is one column.
    """
    warnings = []
    index = pd.Index(np.asarray(chromatogram.times, dtype=float), name="rt")
    columns: dict[int, np.ndarray] = {}
    for ion in chromatogram.ions:
        mass = int(round(ion.mass))
        if abs(ion.mass - mass) > 0.05:
            warnings.append(f"mass {ion.mass:g} stored as {mass}")
        values = columns.setdefault(mass, np.full(len(index), np.nan))
        if np.isfinite(values[ion.scans]).any():
            warnings.append(f"mass {mass} measured twice in some scans, last kept")
        values[ion.scans] = ion.values
    frame = pd.DataFrame(
        {str(m): columns[m] for m in sorted(columns)}, index=index, dtype=float
    )
    frame["tic"] = np.asarray(chromatogram.tic, dtype=float)
    return frame, warnings


def find_runs(source: Path) -> list[Run]:
    """The runs of :func:`find_chromatogram_files`."""
    return [Run(path.name, path, path) for path in find_chromatogram_files(source)]


def read(run: Run) -> MeasurementImport:
    from gcwerks.io.chromatogram import read_chromatogram

    path = Path(run.path)
    chromatogram = read_chromatogram(path)
    if not len(chromatogram.times):
        raise ValueError("no scans")
    frame, warnings = chromatogram_frame(chromatogram)
    frame.attrs[PROVENANCE] = json.dumps(
        {"file": str(path.resolve()), "start": str(chromatogram.start)}
    )
    times = chromatogram.times
    return MeasurementImport(
        run=run,
        instrument=INSTRUMENT,
        peak_source=PEAK_SOURCE,
        rt_start_s=float(times[0]),
        rt_stop_s=float(times[-1]),
        chromatogram_frame=frame,
        acq_start=pd.Timestamp(chromatogram.start).to_pydatetime(),
        acq_start_source=START_FILE,
        warnings=warnings,
    )


def refresh(con: sqlite3.Connection, run: Run) -> None:
    """Nothing changes without changing the file."""


def read_targets(source: Path, config: dict):
    """The targets of the GCwerks directory of ``source``
    (:func:`esper.sources.gcwerks_targets.read_targets`, with the aliases of the
    esper.yaml ``config``)."""
    from esper.sources.gcwerks_targets import ALIASES_KEY
    from esper.sources.gcwerks_targets import read_targets as read
    from esper.targets import TargetError

    root = gcwerks_root(source)
    if root is None:
        raise TargetError(f"{source} is not in a GCwerks directory (no {CONFIG_FILE})")
    return read(root, config.get(ALIASES_KEY), DEFAULT_CHANNEL)
