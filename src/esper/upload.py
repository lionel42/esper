"""Import .alpinac folders into an esper data directory."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from esper import db
from esper.alpinac import (
    ALPINAC_SUFFIX,
    PIPELINE_FILENAME,
    AlpinacResult,
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


def _copy_chromatograms(result: AlpinacResult, data_dir: Path) -> str | None:
    if result.chromatogram_file is None:
        return None
    target = db.chromatograms_dir(data_dir) / f"{result.name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    src_stat = result.chromatogram_file.stat()
    if not (
        target.is_file()
        and target.stat().st_size == src_stat.st_size
        and target.stat().st_mtime == src_stat.st_mtime
    ):
        shutil.copy2(result.chromatogram_file, target)
    return str(target.relative_to(data_dir))


def _insert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.astype(object).where(df.notna(), None)
    columns = ", ".join(df.columns)
    placeholders = ", ".join("?" * len(df.columns))
    con.executemany(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        df.itertuples(index=False, name=None),
    )


def store_result(con: sqlite3.Connection, result: AlpinacResult, chromatogram_file: str | None) -> None:
    """Replace the measurement in the database (single transaction)."""
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

        if result.mass_cal is not None:
            mc = result.mass_cal
            _insert(con, "mass_cal", pd.DataFrame([{"measurement_id": mid, "ionization": mc.ionization, **mc.params}]))
            _insert(con, "mass_cal_rt", mc.rt.assign(measurement_id=mid))
            _insert(con, "mass_cal_mass", mc.masses.assign(measurement_id=mid))

        _insert(con, "compound", result.compounds.assign(measurement_id=mid))
        _insert(con, "compound_candidate", result.candidates.assign(measurement_id=mid))
        _insert(con, "peak", result.peaks.assign(measurement_id=mid))


def upload(source: Path, data_dir: Path, force: bool = False) -> UploadSummary:
    data_dir = Path(data_dir)
    summary = UploadSummary([], [], [], [])
    con = db.connect(data_dir, create=True)
    try:
        for folder in find_alpinac_folders(source):
            name = folder.name.removesuffix(ALPINAC_SUFFIX)
            if not is_complete(folder):
                logger.warning(f"{name}: no {PIPELINE_FILENAME}, skipped")
                summary.skipped_incomplete.append(name)
                continue
            if not force and _is_unchanged(con, name, folder / PIPELINE_FILENAME):
                logger.info(f"{name}: unchanged, skipped")
                summary.skipped_unchanged.append(name)
                continue
            try:
                result = read_alpinac_folder(folder)
                for w in result.warnings:
                    logger.warning(f"{name}: {w}")
                chromatogram_file = _copy_chromatograms(result, data_dir)
                store_result(con, result, chromatogram_file)
            except Exception as e:
                logger.exception(f"{name}: import failed")
                summary.failed.append((name, repr(e)))
                continue
            logger.info(f"{name}: {len(result.peaks)} peaks, {len(result.compounds)} compounds")
            summary.imported.append(name)
    finally:
        con.close()
    return summary
