"""Mass calibration of the measurements (masscal files, found as twutils does) and its storage
in the database. The calibration is fitted by ``esper calibrate`` (:mod:`esper.calibrate`).

The calibration of a run is fitted at several retention times (rt windows), to follow its
drift. Each point is stored with its absolute time, ``time`` = start of the acquisition + rt,
to follow the calibration over the measurements (drift, failures).

The calibration file of a ``<run>.alpinac`` folder is, in order (as
``twutils.calibration_for``): ``<run>.seg0_mc.json`` of ``esper calibrate``, the
legacy ``<run>.EI_mc.txt``, then any other ``*_mc.json`` / ``*_mc.txt`` in the folder.

The start of the acquisition is read from the h5 file next to the folder (``<run>.h5``, to
the ms, ``twutils.TofFile.start_time``), else taken from the name of the run (to the minute).
All times are UTC, stored without time zone like ``measurement.acq_datetime``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from esper import db

logger = logging.getLogger(__name__)

# measurement.acq_start_source
START_H5 = "h5"
START_NAME = "name"

# masscal.MassCalibration.WINDOW_COLUMNS -> mass_cal_rt column
_WINDOW_COLUMNS = {
    "rt_start": "rt_start_s",
    "rt_stop": "rt_stop_s",
    "n_calibrants": "n_calibrants",
    "n_used": "n_used",
    "rms_ppm": "rms_ppm",
    "max_ppm": "max_ppm",
}
MAX_PARAMS = 5  # p1 ... p5 columns of mass_cal_rt


@dataclass
class RunCalibration:
    """Timing and mass calibration of a run, as found next to its .alpinac folder."""

    start_time: datetime | None  # start of the acquisition, UTC, naive
    start_source: str | None  # START_H5 / START_NAME
    s_per_bin: float | None  # scan period
    calibration: object | None = None  # masscal.MassCalibration
    ionization: str | None = None  # legacy files: 'EI', ...
    # Calibrant peaks (masscal.read_calibrant_peaks), None if not written
    peaks: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


def _naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _iso(dt) -> str | None:
    return None if dt is None or pd.isna(dt) else dt.isoformat(timespec="milliseconds")


def calibration_files(folder: Path, name: str) -> list[Path]:
    """The candidate calibration files of a .alpinac folder, by preference."""
    import twutils
    from twutils.calibration import legacy_mc_paths

    h5 = folder.parent / f"{name}.h5"
    preferred = [twutils.mc_path(h5), legacy_mc_paths(h5)[0]]
    others = sorted(folder.glob("*_mc.json")) + sorted(folder.glob("*_mc.txt"))
    files = []
    for path in preferred + others:
        if path.is_file() and path not in files:
            files.append(path)
    return files


def _h5_file(folder: Path, name: str, calibration) -> Path | None:
    """The h5 file of the run: next to the .alpinac folder, else where the calibration was made."""
    candidates = [folder.parent / f"{name}.h5"]
    source = (getattr(calibration, "metadata", None) or {}).get("source")
    if isinstance(source, dict) and source.get("file"):
        candidates.append(Path(source["file"]))
    return next((p for p in candidates if p.is_file()), None)


def read_run(folder: Path, name: str) -> RunCalibration:
    """Calibration and start of the acquisition of the run of a .alpinac folder."""
    import twutils
    from masscal import MassCalibration

    from esper.alpinac import parse_measurement_name

    folder = Path(folder)
    warnings = []
    calibration, ionization = None, None
    files = calibration_files(folder, name)
    if files:
        path = files[0]
        try:
            calibration = MassCalibration.from_file(path)
        except Exception as e:  # noqa: BLE001 - a bad file must not stop the import
            warnings.append(f"Cannot read the mass calibration {path.name}: {e!r}")
        if path.name.endswith("_mc.txt"):
            ionization = path.name.removesuffix("_mc.txt").rsplit(".", 1)[-1]
            ionization = ionization if ionization != name else None
        standard = {f"{name}.seg0_mc.json", f"{name}.EI_mc.txt"}
        if len(files) > 1 and path.name not in standard:
            warnings.append(f"Several mass calibration files, using {path.name}")
    else:
        warnings.append("No mass calibration file")

    start, source, s_per_bin = None, None, None
    h5 = _h5_file(folder, name, calibration)
    if h5 is not None:
        try:
            run = twutils.TofFile(h5)
            s_per_bin = run.s_per_bin
            start = run.start_time
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Cannot read {h5.name}: {e!r}")
    if start is not None:
        start, source = _naive_utc(start), START_H5
    else:
        acq = parse_measurement_name(name)["acq_datetime"]
        if acq is not None:
            start, source = datetime.fromisoformat(acq), START_NAME
            warnings.append(
                "Start of the acquisition from the name (to the minute): no readable h5 file"
            )
    peaks = None
    if calibration is not None:
        from masscal import read_calibrant_peaks

        try:
            peaks = read_calibrant_peaks(calibration.source)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Cannot read the calibrant peaks: {e!r}")
        if peaks is None:
            warnings.append("No calibrant peaks of the mass calibration")
    if s_per_bin is None and calibration is not None:
        s_per_bin = calibration.metadata.get("s_per_bin")
    return RunCalibration(
        start_time=start,
        start_source=source,
        s_per_bin=s_per_bin,
        calibration=calibration,
        ionization=ionization,
        peaks=peaks,
        warnings=warnings,
    )


def tables(run: RunCalibration) -> dict[str, pd.DataFrame]:
    """Rows of mass_cal, mass_cal_rt and mass_cal_mass (without measurement_id)."""
    from twutils.h5 import segments_label

    cal = run.calibration
    if cal is None:
        return {}
    n_params = cal.params.shape[1]
    if n_params > MAX_PARAMS:
        raise ValueError(
            f"{n_params} mass calibration parameters (at most {MAX_PARAMS})"
        )
    header = pd.DataFrame(
        [
            {
                "ionization": run.ionization,
                "mode": cal.mode.tofwerk,  # TOFWERK number (INTEGER column)
                "n_params": n_params,
                "sigma_tofidx_slope": cal.sigma_slope,
                "sigma_tofidx_b": cal.sigma_intercept,
                "sigma_m_slope": cal.sigma_mz_slope,
                "sigma_m_b": cal.sigma_mz_intercept,
                "alpha_pseudo_voigt": cal.alpha,
                "s_per_bin": run.s_per_bin,
                "source_file": cal.source,
                "segments": None
                if cal.segments is None
                else segments_label(cal.segments),
            }
        ]
    )

    points = cal.to_frame(start_time=run.start_time).rename(
        columns={"rt": "rt_s", **_WINDOW_COLUMNS}
    )
    if "time" in points:
        points["time"] = [_iso(t) for t in points["time"]]
    points["ok"] = points["ok"].astype(int)
    points["rt_idx"] = (
        np.round(points["rt_s"] / run.s_per_bin).astype("Int64")
        if run.s_per_bin
        else pd.NA
    )
    keep = ["rt_idx", "rt_s", "time", "ok"] + [f"p{k + 1}" for k in range(n_params)]
    keep += list(_WINDOW_COLUMNS.values())
    points = points[[c for c in keep if c in points]]

    masses = pd.DataFrame(
        {"exact_mass": cal.u_masses, "u_ppm": cal.u_ppm}
    ).drop_duplicates("exact_mass")
    result = {"mass_cal": header, "mass_cal_rt": points, "mass_cal_mass": masses}
    if run.peaks is not None:
        peaks = run.peaks.rename(columns={"rt": "rt_s"})
        result["mass_cal_peak"] = peaks.assign(used=peaks["used"].astype(int))
    return result


def store(con: sqlite3.Connection, measurement_id: int, run: RunCalibration) -> bool:
    """Replace the calibration of a measurement and its start time (no commit).

    Nothing is written without a calibration (no file found, or not readable): False.
    """
    if run.calibration is None:
        return False
    con.execute(
        "UPDATE measurement SET acq_start = ?, acq_start_source = ? WHERE measurement_id = ?",
        (_iso(run.start_time), run.start_source, measurement_id),
    )
    for table in ("mass_cal", "mass_cal_rt", "mass_cal_mass", "mass_cal_peak"):
        con.execute(f"DELETE FROM {table} WHERE measurement_id = ?", (measurement_id,))
    for table, df in tables(run).items():
        db.insert(con, table, df.assign(measurement_id=measurement_id))
    return True


def refresh(con: sqlite3.Connection, name: str, folder: Path) -> bool:
    """Read again the calibration of an imported measurement from its .alpinac folder (e.g.
    after ``esper calibrate``), without importing it again. False: not imported, or no
    calibration (nothing written)."""
    row = con.execute(
        "SELECT measurement_id FROM measurement WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return False
    run = read_run(Path(folder), name)
    with con:
        stored = store(con, row[0], run)
    for w in run.warnings:
        logger.warning(f"{name}: {w}")
    return stored
