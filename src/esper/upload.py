"""Import .alpinac folders into an esper data directory."""

from __future__ import annotations

import contextlib
import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from esper import db, mass_cal
from esper.alpinac import (
    ALPINAC_SUFFIX,
    DATED_PIPELINE_GLOB,
    PIPELINE_FILENAME,
    AlpinacResult,
    find_pipeline,
    is_complete,
    parse_measurement_name,
    read_alpinac_folder,
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


def find_alpinac_folders(source: Path) -> list[Path]:
    """The source can be a .alpinac folder or a directory containing some."""
    source = Path(source)
    if source.name.endswith(ALPINAC_SUFFIX):
        return [source]
    return sorted(p for p in source.rglob(f"*{ALPINAC_SUFFIX}") if p.is_dir())


def _is_unchanged(con: sqlite3.Connection, name: str, pipeline_file: Path) -> bool:
    row = con.execute(
        "SELECT source_mtime, source_size FROM measurement WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return False
    stat = pipeline_file.stat()
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


def _copy_chromatograms(result: AlpinacResult, data_dir: Path) -> str | None:
    """Copy the chromatograms to ``chromatograms/<name>.parquet``, with the raw sums and
    the baselines of ``esper chromatograms`` next to them (``<name>.raw.parquet``,
    ``<name>.baseline.parquet``, :func:`esper.chromatograms.part_path`) when written
    (not by the legacy pipeline).

    :return: the path of the chromatograms, relative to the data directory (stored).
    """
    from esper.chromatograms import part_path

    if result.chromatogram_file is None:
        return None
    target = db.chromatograms_dir(data_dir) / f"{result.name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    _copy_if_changed(result.chromatogram_file, target)
    for part in ("raw", "baseline"):
        source, copy = part_path(result.chromatogram_file, part), part_path(target, part)
        if source.is_file():
            _copy_if_changed(source, copy)
        else:  # e.g. legacy chromatograms now: not the parts of an earlier import
            copy.unlink(missing_ok=True)
    return str(target.relative_to(data_dir))


def store_result(
    con: sqlite3.Connection, result: AlpinacResult, chromatogram_file: str | None
) -> int:
    """Replace the measurement in the database (single transaction), returns its id."""
    stat = result.pipeline_file.stat()
    with con:
        con.execute("DELETE FROM measurement WHERE name = ?", (result.name,))
        cur = con.execute(
            """
            INSERT INTO measurement (
                name, acq_datetime, sample_type, sample_number,
                rt_start_s, rt_stop_s, cleaning, n_peaks, n_compounds,
                chromatogram_file, source_dir, source_mtime, source_size, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.name,
                *parse_measurement_name(result.name).values(),
                result.attrs["rt_start_s"],
                result.attrs["rt_stop_s"],
                result.attrs["cleaning"],
                len(result.peaks),
                len(result.compounds),
                chromatogram_file,
                str(result.folder.resolve()),
                stat.st_mtime,
                stat.st_size,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        mid = cur.lastrowid

        mass_cal.store(con, mid, result.run)

        db.insert(con, "compound", result.compounds.assign(measurement_id=mid))
        db.insert(
            con, "compound_candidate", result.candidates.assign(measurement_id=mid)
        )
        db.insert(con, "peak", result.peaks.assign(measurement_id=mid))
    return mid


def upload(
    source: Path,
    data_dir: Path,
    force: bool = False,
    fit: bool = True,
    workers: int | None = None,
) -> UploadSummary:
    """Import the .alpinac folders of ``source``, and fit their peaks (``esper.fitting``) if ``fit``.

    ``workers``: processes of the fit (None: the number of CPUs).
    """
    from esper.fitting import fit_pool

    data_dir = Path(data_dir)
    summary = UploadSummary([], [], [], [])
    con = db.connect(data_dir, create=True)
    stack = contextlib.ExitStack()
    # The processes of the fits, for all the measurements (started at the first fit)
    pool = stack.enter_context(fit_pool(workers)) if fit else None
    try:
        for folder in find_alpinac_folders(source):
            name = folder.name.removesuffix(ALPINAC_SUFFIX)
            if not is_complete(folder):
                logger.warning(
                    f"{name}: no {PIPELINE_FILENAME} or {DATED_PIPELINE_GLOB}, skipped"
                )
                summary.skipped_incomplete.append(name)
                continue
            if not force and _is_unchanged(con, name, find_pipeline(folder)):
                logger.info(f"{name}: unchanged, skipped")
                summary.skipped_unchanged.append(name)
                # The calibration may have been made again (esper calibrate)
                try:
                    mass_cal.refresh(con, name, folder)
                except Exception as e:
                    logger.exception(f"{name}: mass calibration failed")
                    summary.failed.append((name, f"mass calibration: {e!r}"))
                continue
            try:
                result = read_alpinac_folder(folder)
                for w in result.warnings:
                    logger.warning(f"{name}: {w}")
                chromatogram_file = _copy_chromatograms(result, data_dir)
                mid = store_result(con, result, chromatogram_file)
            except Exception as e:
                logger.exception(f"{name}: import failed")
                summary.failed.append((name, repr(e)))
                continue
            logger.info(
                f"{name}: {len(result.peaks)} peaks, {len(result.compounds)} compounds"
            )
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
