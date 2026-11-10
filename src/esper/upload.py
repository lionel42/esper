"""Import the runs of an instrument (:mod:`esper.sources`) into an esper data directory."""

from __future__ import annotations

import contextlib
import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from esper import db, mass_cal
from esper.sources import (
    InstrumentError,
    MeasurementImport,
    get_source,
    parse_measurement_name,
    resolve_instrument,
)

logger = logging.getLogger(__name__)


@dataclass
class UploadSummary:
    imported: list[str]
    skipped_unchanged: list[str]
    skipped_incomplete: list[str]
    failed: list[tuple[str, str]]
    # Imported, but the fit failed
    fit_failed: list[tuple[str, str]] = field(default_factory=list)


def _is_unchanged(con: sqlite3.Connection, name: str, file: Path) -> bool:
    row = con.execute(
        "SELECT source_mtime, source_size FROM measurement WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return False
    stat = file.stat()
    return row == (stat.st_mtime, stat.st_size)


def _copy_if_changed(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` unless a copy of the same size and mtime is there."""
    src_stat = source.stat()
    if not (
        target.is_file()
        and target.stat().st_size == src_stat.st_size
        and target.stat().st_mtime == src_stat.st_mtime
    ):
        shutil.copy2(source, target)


def store_chromatograms(imported: MeasurementImport, data_dir: Path) -> str | None:
    """Write the chromatograms to ``chromatograms/<name>.parquet``.

    A ``chromatogram_file`` is copied, with the raw sums and the baselines of ``esper
    chromatograms`` next to it (``<name>.raw.parquet``, ``<name>.baseline.parquet``,
    :func:`esper.chromatograms.part_path`) when written (not by the legacy pipeline); a
    ``chromatogram_frame`` is written.

    :return: the path of the chromatograms, relative to the data directory (stored).
    """
    from esper.chromatograms import part_path

    source, frame = imported.chromatogram_file, imported.chromatogram_frame
    if source is None and frame is None:
        return None
    target = db.chromatograms_dir(data_dir) / f"{imported.name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        frame.to_parquet(target, compression="zstd")
    else:
        _copy_if_changed(source, target)
    for part in ("raw", "baseline"):
        part_source, copy = (
            None if source is None else part_path(source, part),
            part_path(target, part),
        )
        if part_source is not None and part_source.is_file():
            _copy_if_changed(part_source, copy)
        else:  # e.g. legacy chromatograms now: not the parts of an earlier import
            copy.unlink(missing_ok=True)
    return str(target.relative_to(data_dir))


def _none_if_nan(value):
    return None if value is None or value != value else value


def store_measurement(
    con: sqlite3.Connection, imported: MeasurementImport, chromatogram_file: str | None
) -> int:
    """Replace the measurement in the database (single transaction), returns its id."""
    run = imported.run
    stat = run.file.stat()
    tables = {
        "compound": imported.compounds,
        "compound_candidate": imported.candidates,
        "peak": imported.peaks,
    }
    start = imported.acq_start
    with con:
        con.execute("DELETE FROM measurement WHERE name = ?", (run.name,))
        cur = con.execute(
            """
            INSERT INTO measurement (
                name, acq_datetime, sample_type, sample_number,
                rt_start_s, rt_stop_s, cleaning, n_peaks, n_compounds,
                chromatogram_file, source_dir, source_mtime, source_size, imported_at,
                acq_start, acq_start_source, instrument, peak_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.name,
                *parse_measurement_name(run.name).values(),
                _none_if_nan(imported.rt_start_s),
                _none_if_nan(imported.rt_stop_s),
                imported.cleaning,
                0 if imported.peaks is None else len(imported.peaks),
                0 if imported.compounds is None else len(imported.compounds),
                chromatogram_file,
                str(Path(run.path).resolve()),
                stat.st_mtime,
                stat.st_size,
                datetime.now().isoformat(timespec="seconds"),
                None if start is None else start.isoformat(timespec="milliseconds"),
                imported.acq_start_source if start is not None else None,
                imported.instrument,
                imported.peak_source,
            ),
        )
        mid = cur.lastrowid

        if imported.calibration is not None:
            # Also the start of the acquisition, if there is a calibration
            mass_cal.store(con, mid, imported.calibration)
        for table, df in tables.items():
            if df is not None:
                db.insert(con, table, df.assign(measurement_id=mid))
    return mid


def check_instrument(con: sqlite3.Connection, instrument: str, data_dir: Path) -> None:
    """The measurements of the database are of ``instrument``."""
    others = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT instrument FROM measurement WHERE instrument != ?",
            (instrument,),
        )
    ]
    if others:
        raise InstrumentError(
            f"{data_dir} is declared for {instrument} but has measurements of"
            f" {', '.join(others)}"
        )


def _store_targets(
    con: sqlite3.Connection, instrument_source, source: Path, data_dir: Path
):
    """Replace the targets by those of the instrument (e.g. the GCwerks peak list)."""
    from esper.targets import store_targets

    targets = instrument_source.read_targets(source, db.read_config(data_dir))
    targets.check()
    store_targets(con, targets, str(Path(source).resolve()))
    logger.info(
        f"targets: {len(targets.table)}, {targets.rts['name'].nunique()} with retention"
        f" times ({len(targets.rts)} versions), {targets.ions['name'].nunique()} with ions"
    )


def upload(
    source: Path,
    data_dir: Path,
    force: bool = False,
    fit: bool = True,
    workers: int | None = None,
    instrument: str | None = None,
) -> UploadSummary:
    """Import the runs of ``source``, and fit their peaks (``esper.fitting``) if ``fit``.

    ``instrument``: of the runs; the one declared in the esper.yaml of the data directory,
    declared there if not yet (:func:`esper.sources.resolve_instrument`).
    ``workers``: processes of the fit (None: the number of CPUs).
    """
    data_dir = Path(data_dir)
    instrument = resolve_instrument(data_dir, instrument, source)
    instrument_source = get_source(instrument)
    # The fit (and its dependencies) only for the instruments with peaks
    if fit and instrument_source.PEAK_SOURCE is not None:
        from esper.fitting import FITTED_PEAK_SOURCES, fit_pool

        fit = instrument_source.PEAK_SOURCE in FITTED_PEAK_SOURCES
    else:
        fit = False
    summary = UploadSummary([], [], [], [])
    con = db.connect(data_dir, create=True)
    stack = contextlib.ExitStack()
    try:
        check_instrument(con, instrument, data_dir)
        if hasattr(instrument_source, "read_targets"):
            _store_targets(con, instrument_source, source, data_dir)
        runs = instrument_source.find_runs(source)
        logger.info(f"{len(runs)} runs of {instrument} in {source}")
        # The processes of the fits, for all the measurements (started at the first fit)
        pool = stack.enter_context(fit_pool(workers)) if fit else None
        for run in runs:
            name = run.name
            if run.file is None:
                logger.warning(f"{name}: not complete, skipped")
                summary.skipped_incomplete.append(name)
                continue
            if not force and _is_unchanged(con, name, run.file):
                logger.debug(f"{name}: unchanged, skipped")
                summary.skipped_unchanged.append(name)
                try:
                    instrument_source.refresh(con, run)
                except Exception as e:
                    logger.exception(f"{name}: refresh failed")
                    summary.failed.append((name, f"refresh: {e!r}"))
                continue
            try:
                imported = instrument_source.read(run)
                for w in imported.warnings:
                    logger.warning(f"{name}: {w}")
                chromatogram_file = store_chromatograms(imported, data_dir)
                mid = store_measurement(con, imported, chromatogram_file)
            except Exception as e:
                logger.debug(f"{name}: import failed", exc_info=True)
                logger.warning(f"{name}: import failed: {e!r}")
                summary.failed.append((name, repr(e)))
                continue
            n_peaks = 0 if imported.peaks is None else len(imported.peaks)
            n_compounds = 0 if imported.compounds is None else len(imported.compounds)
            logger.info(f"{name}: {n_peaks} peaks, {n_compounds} compounds")
            summary.imported.append(name)
            if fit:
                _fit(con, name, mid, data_dir, pool, summary)
    finally:
        stack.close()
        con.close()
    return summary


def _fit(
    con: sqlite3.Connection,
    name: str,
    mid: int,
    data_dir: Path,
    pool,
    summary: UploadSummary,
) -> None:
    from esper.fitting import fit_measurement
    from esper.fitting import summary as fit_summary

    try:
        table = fit_measurement(con, mid, data_dir, workers=1, pool=pool)
    except Exception as e:
        logger.exception(f"{name}: fit failed")
        summary.fit_failed.append((name, repr(e)))
        return
    if table is None:
        logger.warning(f"{name}: no chromatograms, not fitted")
        return
    logger.info(f"{name}: {fit_summary(table)}")
