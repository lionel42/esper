"""Runs of a TOFWERK mass spectrometer, through their ``<run>.alpinac`` folders.

The folder has the unit mass chromatograms (``esper chromatograms``) and the mass calibration
(``esper calibrate``, :mod:`esper.mass_cal`, which also reads the h5 file next to the
folder). A run is read in one of two ways, by the data directory:

- with a processing method (``processing: method:`` of its esper.yaml): its chromatograms
  and calibration only, the peaks are found and identified by esper (``esper process``,
  :mod:`esper.processing`); ``pipeline.nc`` is not read. The chromatograms stored are the
  raw sums (``<run>.unit_mass_chromatos.raw.parquet``), with the rts where their
  background steps (:data:`esper.windows.CUTS`, from the provenance of the corrected
  ones: :func:`esper.chromatograms.baseline_cuts`): the peaks are found and fitted on
  the measured counts, each piece between the steps on its own, not on the corrected
  chromatograms (smoothed, clipped at 0: their noise was underestimated ~6 x, and the
  fit kept noise). A folder is complete when it has its corrected chromatograms and their
  raw sums (``esper chromatograms``; not the legacy pipeline);
- without (legacy, until the halosearch pipeline is retired): with the peaks of the
  alpinac extraction (``pipeline.nc``, :mod:`esper.alpinac`, peak source ``alpinac``).
  A folder is complete when it has its pipeline file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from esper import mass_cal
from esper.alpinac import (
    ALPINAC_SUFFIX,
    find_chromatograms,
    find_pipeline,
    read_alpinac_folder,
)
from esper.chromatograms import PROVENANCE, baseline_cuts, part_path, provenance
from esper.sources import TOFWERK, MeasurementImport, Run
from esper.windows import CUTS

logger = logging.getLogger(__name__)

INSTRUMENT = TOFWERK
PEAK_SOURCE = "alpinac"
# The runs can also be read without their peaks (esper finds them): see the module
OPTIONAL_PEAKS = True


def find_alpinac_folders(source: Path) -> list[Path]:
    """The source can be a .alpinac folder or a directory containing some."""
    source = Path(source)
    if source.name.endswith(ALPINAC_SUFFIX):
        return [source]
    return sorted(p for p in source.rglob(f"*{ALPINAC_SUFFIX}") if p.is_dir())


def find_raw_sums(folder: Path, name: str) -> Path | None:
    """The raw sums of the chromatograms of a .alpinac folder, next to the corrected ones
    (``esper chromatograms``); None without either."""
    corrected = find_chromatograms(folder, name)
    if corrected is None:
        return None
    raw = part_path(corrected, "raw")
    return raw if raw.is_file() else None


def find_runs(source: Path, peaks: bool = True) -> list[Run]:
    """The .alpinac folders of ``source``: with their peaks (complete with a pipeline
    file), or without (complete with the raw sums of their chromatograms)."""
    runs = []
    for folder in find_alpinac_folders(source):
        name = folder.name.removesuffix(ALPINAC_SUFFIX)
        file = find_pipeline(folder) if peaks else find_raw_sums(folder, name)
        if file is None and not peaks and find_chromatograms(folder, name) is not None:
            logger.warning(
                f"{name}: chromatograms without their raw sums (legacy pipeline?):"
                " make them again with esper chromatograms"
            )
        runs.append(Run(name, folder, file, peaks=peaks))
    return runs


def read(run: Run) -> MeasurementImport:
    if not run.peaks:
        return _read_chromatograms(run)
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


def raw_chromatograms(raw: Path) -> pd.DataFrame:
    """The raw sums ``raw`` (float32), with the rts of the steps of their background
    (:data:`esper.windows.CUTS`) and the provenance of the corrected chromatograms next to
    them in their ``attrs``."""
    frame = pd.read_parquet(raw)
    corrected = Path(raw).with_name(
        Path(raw).name.removesuffix(".raw.parquet") + ".parquet"
    )
    info = provenance(pd.read_parquet(corrected, columns=[]))
    cuts = baseline_cuts(frame, info)
    frame.attrs = {
        PROVENANCE: json.dumps({**info, "part": "raw"}),
        CUTS: json.dumps(cuts),
    }
    return frame


def _read_chromatograms(run: Run) -> MeasurementImport:
    """A run without peaks: the raw sums of its chromatograms (``run.file``) and its
    calibration."""
    frame = raw_chromatograms(run.file)
    times = frame.index
    if not len(times):
        raise ValueError("no scans in the chromatograms")
    calibration = mass_cal.read_run(Path(run.path), run.name)
    return MeasurementImport(
        run=run,
        instrument=INSTRUMENT,
        peak_source=None,
        rt_start_s=float(times[0]),
        rt_stop_s=float(times[-1]),
        chromatogram_frame=frame,
        calibration=calibration,
        # also without a calibration file (the h5 file, else the run name)
        acq_start=calibration.start_time,
        acq_start_source=calibration.start_source,
        warnings=calibration.warnings,
    )


def refresh(con: sqlite3.Connection, run: Run) -> None:
    """The calibration may have been made again (esper calibrate)."""
    mass_cal.refresh(con, run.name, run.path)
