"""What an esper data directory holds: where its files are, its database, measurements and
targets, as sections of (label, value) lines (shown by the GUI, File > Data directory).

Read from the database and the esper.yaml only (a few queries and file stats, no
chromatogram file), so that it is quick on a large data directory too. Works on the
databases of older esper versions (columns added since: shown when present).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from esper import __version__, db


@dataclass
class Section:
    title: str
    rows: list[tuple[str, str]] = field(default_factory=list)

    def add(self, label: str, value) -> None:
        self.rows.append((label, str(value)))


def as_text(sections: list[Section]) -> str:
    """The sections as plain text (to copy)."""
    lines = []
    for section in sections:
        lines.append(section.title)
        lines += [f"  {label}: {value}" for label, value in section.rows]
    return "\n".join(lines)


def _when(iso: str | None) -> str:
    """An ISO date of the database, to the minute."""
    return iso[:16].replace("T", " ") if iso else "-"


def _size(n_bytes: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n_bytes < 1000 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1000
    raise AssertionError


def _counts(rows) -> str:
    """'air 120, tank 40' of (value, count) rows."""
    return ", ".join(f"{value or 'none'} {count}" for value, count in rows) or "-"


def describe(data_dir: Path, con: sqlite3.Connection) -> list[Section]:
    """The sections describing the data directory ``data_dir``, of database ``con``."""
    data_dir = Path(data_dir)
    try:
        config = db.read_config(data_dir)
    except ValueError as e:  # an invalid esper.yaml: say it, describe the rest
        config = {"error": str(e)}
    return [
        _location(data_dir, config),
        _database(con),
        _measurements(con, config),
        _targets(con, config),
    ]


def _location(data_dir: Path, config: dict) -> Section:
    section = Section("Location")
    section.add("Path", data_dir.resolve())
    config_path = data_dir / db.CONFIG_FILENAME
    if "error" in config:
        section.add("esper.yaml", config["error"])
    else:
        section.add("esper.yaml", config_path if config_path.is_file() else "none")
    path = db.db_path(data_dir)
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes")
    section.add(
        "Database", f"{path} ({_size(stat.st_size)}, modified {_when(modified)})"
    )
    section.add("Chromatograms", db.chromatograms_dir(data_dir))
    return section


def _database(con: sqlite3.Connection) -> Section:
    section = Section("Database")
    version = con.execute("PRAGMA user_version").fetchone()[0]
    min_reader = db.min_reader_version(con)
    section.add(
        "Schema version",
        f"{version} (this esper: {db.SCHEMA_VERSION})"
        + (
            f", readable by schema version {min_reader} and newer" if min_reader else ""
        ),
    )
    section.add(
        "esper version", f"{__version__} (this one; the database does not record it)"
    )
    return section


def _measurements(con: sqlite3.Connection, config: dict) -> Section:
    from esper.fitting import FIT_VERSION, FITTED_PEAK_SOURCES

    section = Section("Measurements")
    columns = set(db.table_columns(con, "measurement"))
    section.add("Instrument", config.get("instrument") or "not declared")
    n, first, last, imported = con.execute(
        "SELECT count(*), min(acq_datetime), max(acq_datetime), max(imported_at)"
        " FROM measurement"
    ).fetchone()
    section.add("Measurements", n)
    if not n:
        return section
    if "instrument" in columns:
        instruments = con.execute(
            "SELECT instrument, count(*) FROM measurement GROUP BY instrument"
        ).fetchall()
        if len(instruments) > 1 or instruments[0][0] != config.get("instrument"):
            section.add("Of the instruments", _counts(instruments))
    section.add("Acquired", f"{_when(first)} to {_when(last)}")
    section.add("Last import", _when(imported))
    section.add(
        "Sample types",
        _counts(
            con.execute(
                "SELECT sample_type, count(*) FROM measurement GROUP BY sample_type"
                " ORDER BY count(*) DESC, sample_type"
            )
        ),
    )
    processing = config.get("processing")
    method = processing.get("method") if isinstance(processing, dict) else None
    section.add("Processing method", method or "none (esper peaks --method)")
    if "peak_source" in columns:
        sources = con.execute(
            "SELECT peak_source, count(*) FROM measurement GROUP BY peak_source"
            " ORDER BY count(*) DESC"
        ).fetchall()
        section.add(
            "Peaks from",
            ", ".join(
                f"{source or 'no peaks yet'} {count}" for source, count in sources
            ),
        )
    if "identification" in columns:
        methods = con.execute(
            "SELECT identification, count(*) FROM measurement"
            " WHERE identification IS NOT NULL GROUP BY identification"
        ).fetchall()
        if methods:
            section.add("Identified by", _counts(methods))
    with_peaks, n_peaks = con.execute(
        "SELECT count(*), sum(n_peaks) FROM measurement WHERE n_peaks > 0"
    ).fetchone()
    if with_peaks:
        section.add("With peaks", f"{with_peaks} ({n_peaks or 0} peaks)")
    n_cal = con.execute("SELECT count(*) FROM mass_cal").fetchone()[0]
    if n_cal:
        section.add("With mass calibration", n_cal)
    if {"fit_version", "peak_source"} <= columns:
        sources = ", ".join("?" for _ in FITTED_PEAK_SOURCES)
        current, older, never, last_fit = con.execute(
            "SELECT sum(fit_version >= ?), sum(fit_version < ?),"
            " sum(fit_version IS NULL), max(fitted_at)"
            f" FROM measurement WHERE peak_source IN ({sources})",
            (FIT_VERSION, FIT_VERSION, *FITTED_PEAK_SOURCES),
        ).fetchone()
        if current or older or never:
            section.add(
                "Fitted",
                f"{current or 0} (fit version {FIT_VERSION}), {older or 0} by an older"
                f" esper, {never or 0} not fitted (esper fit)",
            )
            section.add("Last fit", _when(last_fit))
    return section


def _targets(con: sqlite3.Connection, config: dict) -> Section:
    section = Section("Targets")
    n = con.execute("SELECT count(*) FROM target").fetchone()[0]
    n_ri = con.execute("SELECT count(*) FROM target WHERE ri IS NOT NULL").fetchone()[0]
    n_ions = con.execute("SELECT count(DISTINCT name) FROM target_ion").fetchone()[0]
    section.add(
        "Targets", f"{n} ({n_ri} with retention index, {n_ions} with expected ions)"
    )
    section.add(
        "Aliases of the targets",
        con.execute("SELECT count(*) FROM target_alias").fetchone()[0],
    )
    meta = dict(con.execute("SELECT key, value FROM meta"))
    if "targets_source" in meta:
        section.add("Source", meta["targets_source"])
        section.add("Imported", _when(meta.get("targets_imported_at")))
    aliases = config.get("aliases")
    if isinstance(aliases, dict) and aliases:
        unmapped = sum(value is None for value in aliases.values())
        section.add(
            "Aliases of esper.yaml",
            f"{len(aliases)} names ({unmapped} of them not targets: null)",
        )
    return section
