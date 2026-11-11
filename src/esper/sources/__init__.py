"""Where the measurements of an esper data directory come from.

A data directory holds the runs of one instrument, declared in its ``esper.yaml``
(``instrument: tofwerk``). Each instrument is a module of this package that finds the runs
of a source path and reads them, with the peaks of its peak source if any:

=========== ===================================================== ============
instrument  runs                                                  peak_source
=========== ===================================================== ============
tofwerk     ``<run>.alpinac`` folders (made from the TOFWERK h5)  alpinac
gcwerks     ``<yy>/chromatograms/channel0/<run>`` of a GCwerks    None
            directory (chromatograms only)
=========== ===================================================== ============

A source module has:

- ``INSTRUMENT``: its name; ``PEAK_SOURCE``: where the peaks of its runs come from
  (``measurement.peak_source``), None if it has none;
- ``find_runs(path) -> list[Run]``: the runs of a path, sorted;
- ``read(run) -> MeasurementImport``: everything to store of a run;
- ``refresh(con, run)``: update what can change without importing the run again (called
  for the unchanged runs; e.g. the mass calibration);
- optionally ``read_targets(path, config) -> esper.targets.Targets``: the targets that
  the instrument defines itself (GCwerks: peak.list and peakid), imported by each upload
  (``config``: the esper.yaml of the data directory). Without it, the targets come from
  a target list (``esper targets``).

:mod:`esper.upload` stores what they read.
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from esper.mass_cal import RunCalibration

logger = logging.getLogger(__name__)

TOFWERK = "tofwerk"
GCWERKS = "gcwerks"
INSTRUMENTS = (TOFWERK, GCWERKS)
# Instrument of the data directories made before it was declared (all tofwerk)
LEGACY_INSTRUMENT = TOFWERK
# Attribute of a chromatogram frame (stored in its parquet file): "1" when a mass without
# a value in a scan was swept but not found there (GCwerks scan runs), its curve then
# links the values around; without it, a missing value is a gap (not measured: the
# time windows of SIM)
CONTINUOUS = "esper.continuous"

# e.g. 241007.0423.air.1, or without number (GCwerks: 260112.1907.tune)
_NAME_RE = re.compile(
    r"^(?P<date>\d{6})\.(?P<time>\d{4})\.(?P<type>[^.]+)(?:\.(?P<number>\d+))?$"
)


def parse_measurement_name(name: str) -> dict:
    """Split 'YYMMDD.HHMM.<type>[.<number>]' into its parts (None if not matching)."""
    m = _NAME_RE.match(name)
    if m is None:
        return {"acq_datetime": None, "sample_type": None, "sample_number": None}
    dt = datetime.strptime(m["date"] + m["time"], "%y%m%d%H%M")
    return {
        "acq_datetime": dt.isoformat(),
        "sample_type": m["type"],
        "sample_number": None if m["number"] is None else int(m["number"]),
    }


@dataclass
class Run:
    """A run found in a source path."""

    name: str
    path: Path
    """What the run was read from (``measurement.source_dir``)."""
    file: Path | None
    """The file whose size and mtime tell whether the run changed; None: the run is not
    complete (not imported)."""


@dataclass
class MeasurementImport:
    """Everything stored of a run (the optional parts are None when the source has none)."""

    run: Run
    instrument: str
    peak_source: (
        str | None
    )  # where the peaks come from (e.g. 'alpinac'), None: no peaks
    rt_start_s: float | None = None
    rt_stop_s: float | None = None
    cleaning: str | None = None
    # Chromatograms: a file to copy (with its parts, esper.chromatograms.part_path), or
    # a frame to write (index rt [s], one column per unit mass and 'tic')
    chromatogram_file: Path | None = None
    chromatogram_frame: pd.DataFrame | None = None
    # Tables of the database, without measurement_id
    peaks: pd.DataFrame | None = None
    compounds: pd.DataFrame | None = None
    candidates: pd.DataFrame | None = None
    # Mass calibration and start of the acquisition (esper.mass_cal); stored if it has
    # a calibration
    calibration: RunCalibration | None = None
    # Start of the acquisition (UTC) when there is no calibration to give it
    acq_start: datetime | None = None
    acq_start_source: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.run.name


class InstrumentError(ValueError):
    """The instrument of a data directory is unknown, not declared, or not the one asked."""


def get_source(instrument: str):
    """The module of an instrument."""
    if instrument not in INSTRUMENTS:
        raise InstrumentError(
            f"Unknown instrument {instrument!r}, expected one of {', '.join(INSTRUMENTS)}"
        )
    return importlib.import_module(f"esper.sources.{instrument}")


def declared_instrument(data_dir: Path) -> str | None:
    """The ``instrument`` of the esper.yaml of a data directory, None if not declared."""
    from esper import db

    instrument = db.read_config(data_dir).get("instrument")
    if instrument is not None and instrument not in INSTRUMENTS:
        raise InstrumentError(
            f"{Path(data_dir) / db.CONFIG_FILENAME}: unknown instrument {instrument!r},"
            f" expected one of {', '.join(INSTRUMENTS)}"
        )
    return instrument


def declare_instrument(data_dir: Path, instrument: str) -> None:
    """Add ``instrument: <instrument>`` to the esper.yaml of a data directory (created if
    needed; its other lines and comments are kept)."""
    from esper import db

    get_source(instrument)  # known
    path = Path(data_dir) / db.CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(f"{text}instrument: {instrument}\n")
    logger.info(f"{path}: instrument {instrument} declared")


def resolve_instrument(
    data_dir: Path, instrument: str | None = None, source: Path | None = None
) -> str:
    """The instrument of a data directory, declared in its esper.yaml if not yet.

    ``instrument``: as asked (e.g. ``esper upload --instrument``), must be the declared one.
    Not declared and not asked: the data directory must have a database already (made
    before the instruments were declared: :data:`LEGACY_INSTRUMENT`).
    """
    from esper import db
    from esper.sources.gcwerks import gcwerks_root

    data_dir = Path(data_dir)
    declared = declared_instrument(data_dir)
    if declared is not None:
        if instrument is not None and instrument != declared:
            raise InstrumentError(
                f"{data_dir} has the runs of {declared} (its {db.CONFIG_FILENAME}),"
                f" not of {instrument}"
            )
        return declared
    if instrument is None:
        if source is not None and gcwerks_root(source) is not None:
            raise InstrumentError(
                f"{source} is a GCwerks directory: declare the instrument of {data_dir}"
                f" (--instrument {GCWERKS})"
            )
        if not db.db_path(data_dir).is_file():
            raise InstrumentError(
                f"New data directory {data_dir}: give its instrument"
                f" (--instrument {'|'.join(INSTRUMENTS)})"
            )
        instrument = LEGACY_INSTRUMENT
    else:
        get_source(instrument)
    declare_instrument(data_dir, instrument)
    return instrument
