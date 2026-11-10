"""Runs of a TOFWERK mass spectrometer, through their ``<run>.alpinac`` folders.

The folder has the peaks of the alpinac extraction (``pipeline.nc``, :mod:`esper.alpinac`),
the unit mass chromatograms (``esper chromatograms``) and the mass calibration
(``esper calibrate``, :mod:`esper.mass_cal`, which also reads the h5 file next to the folder).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from esper import mass_cal
from esper.alpinac import ALPINAC_SUFFIX, find_pipeline, read_alpinac_folder
from esper.sources import TOFWERK, MeasurementImport, Run

logger = logging.getLogger(__name__)

INSTRUMENT = TOFWERK
PEAK_SOURCE = "alpinac"


def find_alpinac_folders(source: Path) -> list[Path]:
    """The source can be a .alpinac folder or a directory containing some."""
    source = Path(source)
    if source.name.endswith(ALPINAC_SUFFIX):
        return [source]
    return sorted(p for p in source.rglob(f"*{ALPINAC_SUFFIX}") if p.is_dir())


def find_runs(source: Path) -> list[Run]:
    """The .alpinac folders of ``source`` (without pipeline file: not complete)."""
    return [
        Run(folder.name.removesuffix(ALPINAC_SUFFIX), folder, find_pipeline(folder))
        for folder in find_alpinac_folders(source)
    ]


def read(run: Run) -> MeasurementImport:
    result = read_alpinac_folder(run.path)
    return MeasurementImport(
        run=run,
        instrument=INSTRUMENT,
        peak_source=PEAK_SOURCE,
        rt_start_s=result.attrs["rt_start_s"],
        rt_stop_s=result.attrs["rt_stop_s"],
        cleaning=result.attrs["cleaning"],
        chromatogram_file=result.chromatogram_file,
        peaks=result.peaks,
        compounds=result.compounds,
        candidates=result.candidates,
        calibration=result.run,
        warnings=result.warnings,
    )


def refresh(con: sqlite3.Connection, run: Run) -> None:
    """The calibration may have been made again (esper calibrate)."""
    mass_cal.refresh(con, run.name, run.path)
