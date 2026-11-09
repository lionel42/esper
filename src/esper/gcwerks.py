"""Import the chromatograms of a GCwerks directory into an esper data directory.

A GCwerks directory (e.g. a Medusa GC-MS) has one folder per year with the
mass-spectrometer chromatograms of each run::

    <gcdir>/
        config/gcwerks.conf
        <yy>/chromatograms/channel0/<yymmdd.hhmm.type[.number]>    # *.cm format, no suffix

The files are read with the `gcwerks <https://github.com/lionel42/gcwerks>`_ package and
written as esper chromatograms (``chromatograms/<name>.parquet``: index ``rt`` [s], one
column per unit mass and ``tic``). Only the chromatograms are imported: no peaks, no
compounds (the measurements have ``source_format = 'gcwerks'`` and are not fitted).

The quadrupole measures each mass only in its time windows (SIM): a mass column is NaN
outside them. The signal is as GCwerks stores it, baseline included.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from esper import db
from esper.alpinac import parse_measurement_name

if TYPE_CHECKING:
    from gcwerks.io.chromatogram import Chromatogram

    from esper.upload import UploadSummary

logger = logging.getLogger(__name__)

# measurement.source_format
SOURCE_FORMAT = "gcwerks"
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


def _store(
    con: sqlite3.Connection,
    name: str,
    path: Path,
    chromatogram: Chromatogram,
    chromatogram_file: str,
) -> int:
    """Replace the measurement in the database (single transaction), returns its id."""
    stat = path.stat()
    times = chromatogram.times
    start = pd.Timestamp(chromatogram.start)
    with con:
        con.execute("DELETE FROM measurement WHERE name = ?", (name,))
        cur = con.execute(
            """
            INSERT INTO measurement (
                name, acq_datetime, sample_type, sample_number,
                rt_start_s, rt_stop_s, cleaning, n_peaks, n_compounds,
                chromatogram_file, source_dir, source_mtime, source_size, imported_at,
                acq_start, acq_start_source, source_format
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                *parse_measurement_name(name).values(),
                float(times[0]) if len(times) else None,
                float(times[-1]) if len(times) else None,
                None,
                0,
                0,
                chromatogram_file,
                str(path.resolve()),
                stat.st_mtime,
                stat.st_size,
                datetime.now().isoformat(timespec="seconds"),
                start.isoformat(timespec="milliseconds"),
                START_FILE,
                SOURCE_FORMAT,
            ),
        )
    return cur.lastrowid


def import_file(con: sqlite3.Connection, path: Path, data_dir: Path) -> list[str]:
    """Import one chromatogram file, returns the warnings."""
    from gcwerks.io.chromatogram import read_chromatogram

    path = Path(path)
    name = path.name
    chromatogram = read_chromatogram(path)
    if not len(chromatogram.times):
        raise ValueError("no scans")
    frame, warnings = chromatogram_frame(chromatogram)
    frame.attrs[PROVENANCE] = json.dumps(
        {"file": str(path.resolve()), "start": str(chromatogram.start)}
    )
    target = db.chromatograms_dir(data_dir) / f"{name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, compression="zstd")
    _store(con, name, path, chromatogram, str(target.relative_to(data_dir)))
    return warnings


def upload(
    source: Path, data_dir: Path, summary: UploadSummary, force: bool = False
) -> UploadSummary:
    """Import the chromatograms of ``source`` (see :func:`find_chromatogram_files`)."""
    from esper.upload import _is_unchanged

    data_dir = Path(data_dir)
    files = find_chromatogram_files(source)
    logger.info(f"{len(files)} GCwerks chromatogram files in {source}")
    con = db.connect(data_dir, create=True)
    try:
        for path in files:
            name = path.name
            if not force and _is_unchanged(con, name, path):
                logger.debug(f"{name}: unchanged, skipped")
                summary.skipped_unchanged.append(name)
                continue
            try:
                warnings = import_file(con, path, data_dir)
            except Exception as e:
                logger.debug(f"{name}: import failed", exc_info=True)
                logger.warning(f"{name}: import failed: {e}")
                summary.failed.append((name, repr(e)))
                continue
            for w in warnings:
                logger.warning(f"{name}: {w}")
            logger.info(f"{name}: imported")
            summary.imported.append(name)
    finally:
        con.close()
    return summary
