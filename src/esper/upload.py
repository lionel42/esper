"""Import the runs of an instrument (:mod:`esper.sources`) into an esper data directory.

The runs imported with peaks (alpinac) are fitted (:mod:`esper.fitting`); those imported
without (GCwerks) are processed with the method of the data directory if it has one
(:mod:`esper.processing`: peaks found, identified and fitted), also when unchanged but
not processed yet.
"""

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
    # Processed (esper.processing): imported, or unchanged but not processed before
    processed: list[str] = field(default_factory=list)
    # Imported, but finding or identifying their peaks failed (esper.processing)
    processing_failed: list[tuple[str, str]] = field(default_factory=list)
    # Names of the targets of the instrument added to the esper.yaml (add_aliases)
    added_aliases: list[str] = field(default_factory=list)
    # Not imported: before ``after`` or not before ``before`` (or no date in the name)
    skipped_dates: list[str] = field(default_factory=list)


def in_dates(
    name: str, after: datetime | None = None, before: datetime | None = None
) -> bool:
    """Whether the run ``name`` (``YYMMDD.HHMM...``) was started on or after ``after``
    and before ``before`` (None: no limit). Without a date in its name: only without
    limits."""
    if after is None and before is None:
        return True
    started = parse_measurement_name(name)["acq_datetime"]
    if started is None:
        return False
    started = datetime.fromisoformat(started)
    return (after is None or started >= after) and (before is None or started < before)


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
                name, acq_datetime, sample_type, port,
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


# Comment of the names added to the esper.yaml by ``add_aliases``
ADDED_ALIAS_COMMENT = "added by esper upload: not a target, or map to its name"


def _store_targets(
    con: sqlite3.Connection,
    instrument_source,
    source: Path,
    data_dir: Path,
    add_aliases: bool = False,
) -> list[str]:
    """Replace the targets by those of the instrument (e.g. the GCwerks peak list).

    ``add_aliases``: the names that match no target are added to the esper.yaml as not
    targets (null), see ``upload``. Returns them.
    """
    from esper.targets import UnresolvedNamesError, store_targets

    added = []
    try:
        targets = instrument_source.read_targets(source, db.read_config(data_dir))
    except UnresolvedNamesError as e:
        if not add_aliases:
            raise
        db.add_config_entries(
            data_dir, e.key, dict.fromkeys(e.names), ADDED_ALIAS_COMMENT
        )
        added = e.names
        logger.warning(
            f"{len(added)} names added to the {e.key} of"
            f" {Path(data_dir) / db.CONFIG_FILENAME} as not targets (null):"
            f" {', '.join(added)}"
        )
        targets = instrument_source.read_targets(source, db.read_config(data_dir))
    targets.check()
    store_targets(con, targets, str(Path(source).resolve()))
    logger.info(
        f"targets: {len(targets.table)}, {targets.rts['name'].nunique()} with retention"
        f" times ({len(targets.rts)} versions), {targets.ions['name'].nunique()} with ions"
    )
    return added


def upload(
    source: Path,
    data_dir: Path,
    force: bool = False,
    fit: bool = True,
    workers: int | None = None,
    instrument: str | None = None,
    add_aliases: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
    method: str | None = None,
) -> UploadSummary:
    """Import the runs of ``source``, and fit their peaks (``esper.fitting``) if ``fit``;
    the runs without peaks are processed (``esper.processing``) with the method of the
    data directory (``method``: declared if not yet, see
    :func:`esper.processing.resolve_method`), and fitted if ``fit``.

    ``instrument``: of the runs; the one declared in the esper.yaml of the data directory,
    declared there if not yet (:func:`esper.sources.resolve_instrument`).
    ``workers``: processes of the fit (None: the number of CPUs).
    ``add_aliases``: the names of the instrument's targets that match none (GCwerks peakid
    names) are added to the esper.yaml as not targets (null, to map by hand), instead of
    stopping the upload.
    ``after``, ``before``: only the runs started on or after ``after`` and before
    ``before``, from the date in their name (``YYMMDD.HHMM...``; the targets are read
    all the same).
    """
    from esper.processing.config import (
        ProcessingConfigError,
        read_processing_config,
        resolve_method,
    )

    data_dir = Path(data_dir)
    instrument = resolve_instrument(data_dir, instrument, source)
    instrument_source = get_source(instrument)
    if method is not None and instrument_source.PEAK_SOURCE is not None:
        raise ProcessingConfigError(
            f"--method: the runs of {instrument} are imported with their peaks"
            f" ({instrument_source.PEAK_SOURCE}), esper does not find them"
        )
    resolve_method(data_dir, method)
    config = read_processing_config(data_dir)
    # The fit of the imported peaks (and its dependencies) only for the instruments with
    # peaks; the others are processed if the data directory has a method
    fit_imported = fit and instrument_source.PEAK_SOURCE is not None
    process = instrument_source.PEAK_SOURCE is None and config.method is not None
    if fit_imported:
        from esper.fitting import FITTED_PEAK_SOURCES

        fit_imported = instrument_source.PEAK_SOURCE in FITTED_PEAK_SOURCES
    if instrument_source.PEAK_SOURCE is None and config.method is None:
        logger.info(
            f"{data_dir}: no processing method, the peaks of the runs are not found"
            " (esper peaks --method ...)"
        )
    summary = UploadSummary([], [], [], [])
    con = db.connect(data_dir, create=True)
    stack = contextlib.ExitStack()
    try:
        check_instrument(con, instrument, data_dir)
        if hasattr(instrument_source, "read_targets"):
            summary.added_aliases = _store_targets(
                con, instrument_source, source, data_dir, add_aliases
            )
        processor, pending = None, set()
        if process:
            from esper.processing.pipeline import Processor, to_process

            processor = Processor(con, data_dir, config)
            pending = {name for _, name in to_process(con, config.method)}
        runs = instrument_source.find_runs(source)
        logger.info(f"{len(runs)} runs of {instrument} in {source}")
        # The processes of the fits, for all the measurements (started at the first fit)
        pool = None
        if fit_imported or (process and fit):
            from esper.fitting import fit_pool

            pool = stack.enter_context(fit_pool(workers))
        for run in runs:
            name = run.name
            if not in_dates(name, after, before):
                summary.skipped_dates.append(name)
                continue
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
                if name in pending:  # imported before, not processed yet
                    _process(con, processor, name, fit, pool, summary)
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
            if imported.peaks is None and processor is not None:
                _process(con, processor, name, fit, pool, summary, mid)
            elif imported.peaks is not None and fit_imported:
                _fit(con, name, mid, data_dir, pool, summary, config.fit)
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
    config=None,
) -> None:
    from esper.fitting import fit_measurement
    from esper.fitting import summary as fit_summary

    try:
        table = fit_measurement(con, mid, data_dir, workers=1, pool=pool, config=config)
    except Exception as e:
        logger.exception(f"{name}: fit failed")
        summary.fit_failed.append((name, repr(e)))
        return
    if table is None:
        logger.warning(f"{name}: no chromatograms, not fitted")
        return
    logger.info(f"{name}: {fit_summary(table)}")


def _process(
    con: sqlite3.Connection,
    processor,
    name: str,
    fit: bool,
    pool,
    summary: UploadSummary,
    mid: int | None = None,
) -> None:
    """Find and identify the peaks of a measurement (``esper.processing``), then fit them
    if ``fit`` (each failure counted apart)."""
    from esper.processing.pipeline import summary as process_summary

    if mid is None:
        (mid,) = con.execute(
            "SELECT measurement_id FROM measurement WHERE name = ?", (name,)
        ).fetchone()
    try:
        table = processor.process(mid, fit=False)
    except Exception as e:
        logger.exception(f"{name}: processing failed")
        summary.processing_failed.append((name, repr(e)))
        return
    if table is None:
        logger.warning(f"{name}: no chromatograms, not processed")
        return
    summary.processed.append(name)
    logger.info(f"{name}: {process_summary(table, con, mid)}")
    if fit:
        _fit(con, name, mid, processor.data_dir, pool, summary, processor.config.fit)
