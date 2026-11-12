"""SQLite database layout of an esper data directory.

An esper data directory looks like::

    <data_dir>/
        esper.yaml                  # "instrument: tofwerk|gcwerks" (esper.sources); optional
                                    # "database: <path>": where the database is
        esper.sqlite                # the database, unless esper.yaml says otherwise
        chromatograms/
            <measurement>.parquet   # unit mass chromatograms (index: rt [s], columns: masses + 'tic')

Units are part of the column names: ``_s`` seconds, ``_scans`` time bins.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

DB_FILENAME = "esper.sqlite"
CONFIG_FILENAME = "esper.yaml"
CHROMATOGRAMS_DIRNAME = "chromatograms"

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 8
# Versions that are upgraded by creating the missing tables and columns (3: the area
# columns of an unreleased version, kept)
COMPATIBLE_VERSIONS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
# Oldest SCHEMA_VERSION of esper that can still use a database of this version, stored in
# the database so that older esper versions know whether they can use it. A schema change
# that only adds tables or nullable columns keeps it, a breaking change sets it to SCHEMA_VERSION.
MIN_READER_VERSION = 4  # 4: the peaks split by the fit have no extraction bounds
# (6 keeps it: older versions read the targets of the target lists as before; the
# versioned ions of the GCwerks targets look to them like several ions per rank)
# (8 keeps it: measurement.identification, nullable)
# MIN_READER_VERSION of the schema versions written before schema_info existed
LEGACY_MIN_READER_VERSIONS = {3: 2}  # 3 only adds nullable columns

# Columns added after version 2: not in the CREATE TABLE statements, added by ALTER
# TABLE to new and old databases alike (nullable or with a default: older versions can
# still use the database)
ADDED_COLUMNS = {
    "measurement": [
        ("fit_version", "INTEGER"),  # esper.fitting.FIT_VERSION, NULL: not fitted
        ("fitted_at", "TEXT"),
        # Start of the acquisition (rt 0), UTC, ISO to the ms (esper.mass_cal)
        ("acq_start", "TEXT"),
        (
            "acq_start_source",
            "TEXT",
        ),  # 'h5': read in the h5 file; 'name': to the minute; 'gcwerks': its file, to the s
        # Instrument of the run (esper.sources: 'tofwerk', 'gcwerks'), and where its peaks
        # come from ('alpinac'; NULL: no peaks, e.g. GCwerks chromatograms only). The
        # defaults are those of the measurements imported before these columns.
        ("instrument", "TEXT NOT NULL DEFAULT 'tofwerk'"),
        ("peak_source", "TEXT DEFAULT 'alpinac'"),
        # Method of esper.processing that identified the compounds ('target': target
        # search, 'suspect': suspect screening); NULL: by the peak source ('alpinac'), or
        # no peaks. peak_source 'esper': the peaks found by esper.processing
        ("identification", "TEXT"),
    ],
    "mass_cal": [
        ("source_file", "TEXT"),  # the calibration file read
        ("segments", "TEXT"),  # segment range calibrated ('seg0'), masscal files only
    ],
    "mass_cal_rt": [
        # Absolute time of the point: acq_start + rt_s (UTC, ISO to the ms)
        ("time", "TEXT"),
        # 1: used by the calibration; 0: rt window whose fit was rejected (masscal files)
        ("ok", "INTEGER NOT NULL DEFAULT 1"),
        # rt window of esper calibrate (NULL for the legacy files)
        ("rt_start_s", "REAL"),
        ("rt_stop_s", "REAL"),
        ("n_calibrants", "INTEGER"),  # calibrant peaks found in the window
        ("n_used", "INTEGER"),  # used by the fit
        ("rms_ppm", "REAL"),  # residuals of the calibrants used
        ("max_ppm", "REAL"),
    ],
    "peak": [
        # 'extraction': found by alpinac; 'fit': split or added by the fit (esper.fitting)
        ("source", "TEXT NOT NULL DEFAULT 'extraction'"),
        # Chromatographic fit (peaxart, peak function fit_shape on a background), NULL: not fitted.
        # fit_rt_s / fit_height: the apex; fit_shape_rt_s, fit_shape_height, fit_sigma_s, fit_tailing: the
        # parameters of the function (peaxart.shapes; NULL fit_shape: Pap-Papai, whose apex they are)
        ("fit_rt_s", "REAL"),
        ("fit_rt_u_s", "REAL"),  # uncertainty (1 sigma)
        ("fit_height", "REAL"),  # above the baseline [counts]
        ("fit_height_u", "REAL"),
        ("fit_sigma_s", "REAL"),  # width (standard deviation of the Gaussian)
        # tail parameter (0: Gaussian): Pap-Papai a in ]0, 2[; EMG tau / sigma
        ("fit_tailing", "REAL"),
        ("fit_baseline", "REAL"),  # the background at the apex [counts]
        ("fit_area", "REAL"),  # [counts * s]
        ("fit_area_u", "REAL"),
        ("fit_sn", "REAL"),  # signal to noise
        ("fit_shape", "TEXT"),  # name of the peak function ('emg', 'pap_papai'), peaxart.shapes.SHAPES
        ("fit_shape_rt_s", "REAL"),  # its location
        ("fit_shape_height", "REAL"),  # its amplitude
        # derivative of the background at the apex [counts / s] (NULL: constant)
        ("fit_baseline_slope", "REAL"),
    ],
    "compound_identification": [
        # Best library spectrum of the name given by hand, and its scores
        ("spectrum_ref", "TEXT"),
        ("score", "REAL"),
        ("score_combined", "REAL"),
    ],
    "target_ion": [
        # Library spectrum the ion was taken from (target without ions in the target
        # list, e.g. 'nist_CFC-12'), NULL: from the target list
        ("spectrum_ref", "TEXT"),
    ],
}

TARGET_ION = """
-- Ions expected in the spectrum of a target, by decreasing importance. Versioned for
-- the instruments whose identification changes over time (GCwerks peakid): the ions of
-- a target are those of its latest valid_from before the run; NULL: always (target lists)
CREATE TABLE IF NOT EXISTS target_ion (
    name                TEXT REFERENCES target ON DELETE CASCADE,
    valid_from          TEXT,                   -- ISO, as measurement.acq_datetime
    rank                INTEGER,                -- 0: main (quantification) ion
    ion                 TEXT,                   -- e.g. 'CF2Cl+'
    exact_mass          REAL,
    mass                INTEGER,                -- unit mass
    PRIMARY KEY (name, valid_from, rank)
);
"""

SCHEMA_INFO = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurement (
    measurement_id      INTEGER PRIMARY KEY,
    name                TEXT UNIQUE NOT NULL,   -- e.g. '241007.0423.air.1'
    acq_datetime        TEXT,                   -- ISO format, parsed from the name
    sample_type         TEXT,                   -- 'air', 'tank', ...
    sample_number       INTEGER,
    rt_start_s          REAL,
    rt_stop_s           REAL,
    cleaning            TEXT,
    n_peaks             INTEGER,
    n_compounds         INTEGER,
    chromatogram_file   TEXT,                   -- relative to the data directory
    source_dir          TEXT NOT NULL,          -- the .alpinac folder (the file for gcwerks)
    source_mtime        REAL NOT NULL,          -- mtime of pipeline.nc (or the file), to detect reprocessing
    source_size         INTEGER NOT NULL,
    imported_at         TEXT NOT NULL
);

-- Scalar mass calibration and peak shape parameters (header of the *_mc.txt file)
CREATE TABLE IF NOT EXISTS mass_cal (
    measurement_id      INTEGER PRIMARY KEY REFERENCES measurement ON DELETE CASCADE,
    ionization          TEXT,                   -- 'EI', ...
    mode                INTEGER,                -- tofwerk mass calibration mode
    n_params            INTEGER,
    sigma_tofidx_slope  REAL,                   -- peak shape, sigma slope vs TOF index
    sigma_tofidx_b      REAL,                   -- peak shape, sigma intercept vs TOF index
    sigma_m_slope       REAL,                   -- peak shape, sigma slope vs mass
    sigma_m_b           REAL,                   -- peak shape, sigma intercept vs mass
    alpha_pseudo_voigt  REAL,
    s_per_bin           REAL
);

-- Mass calibration parameters fitted at several retention times
CREATE TABLE IF NOT EXISTS mass_cal_rt (
    measurement_id      INTEGER REFERENCES measurement ON DELETE CASCADE,
    rt_idx              INTEGER,                -- centre of the rt bin, in scans
    rt_s                REAL,                   -- rt_idx * s_per_bin
    p1 REAL, p2 REAL, p3 REAL, p4 REAL, p5 REAL,
    PRIMARY KEY (measurement_id, rt_idx)
);

-- Exact masses used for the mass calibration and their uncertainty
CREATE TABLE IF NOT EXISTS mass_cal_mass (
    measurement_id      INTEGER REFERENCES measurement ON DELETE CASCADE,
    exact_mass          REAL,
    u_ppm               REAL,
    PRIMARY KEY (measurement_id, exact_mass)
);

-- Calibrant peaks of the mass calibration: one per calibrant and rt window
-- (masscal.read_calibrant_peaks)
CREATE TABLE IF NOT EXISTS mass_cal_peak (
    measurement_id      INTEGER REFERENCES measurement ON DELETE CASCADE,
    calibrant           TEXT,                   -- e.g. 'CF3', 'N2'
    mz_exact            REAL,
    rt_s                REAL,                   -- centre of the rt window
    tof                 REAL,                   -- fitted position, tof index
    area                REAL,
    sigma               REAL,                   -- tof index
    alpha               REAL,                   -- Lorentzian fraction
    ppm                 REAL,                   -- residual of the calibration of the window
    used                INTEGER                 -- in the fit of the window
);
CREATE INDEX IF NOT EXISTS mass_cal_peak_measurement ON mass_cal_peak (measurement_id);

-- Group of co-eluting peaks. Summary columns are as written by alpinac.
CREATE TABLE IF NOT EXISTS compound (
    measurement_id      INTEGER REFERENCES measurement ON DELETE CASCADE,
    compound_idx        INTEGER,                -- 'compound' coordinate of pipeline.nc (with gaps)
    rt_mean_s           REAL,
    rt_std_s            REAL,
    n_peaks             INTEGER,                -- peak_rt_count
    prominence_sum      REAL,
    start_mean_s        REAL,
    start_std_s         REAL,
    stop_mean_s         REAL,
    stop_std_s          REAL,
    main_ion_mass       INTEGER,                -- mai_
    retention_index     REAL,
    is_ref_ri           INTEGER,
    valid_candidate     INTEGER,
    compound_name       TEXT,
    PRIMARY KEY (measurement_id, compound_idx)
);

-- The three best library matches of each compound
CREATE TABLE IF NOT EXISTS compound_candidate (
    measurement_id      INTEGER,
    compound_idx        INTEGER,
    rank                INTEGER,                -- 1, 2, 3
    candidate_name      TEXT,
    spectrum_ref        TEXT,                   -- e.g. 'nist_Xe'
    library             TEXT,                   -- prefix of spectrum_ref, e.g. 'nist'
    score               REAL,
    ri                  REAL,
    ri_diff             REAL,
    score_combined      REAL,
    is_best             INTEGER,                -- the candidate selected by alpinac (bc_*)
    PRIMARY KEY (measurement_id, compound_idx, rank),
    FOREIGN KEY (measurement_id, compound_idx)
        REFERENCES compound (measurement_id, compound_idx) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS peak (
    measurement_id      INTEGER,
    peak_idx            INTEGER,                -- 'peak' coordinate of pipeline.nc
    compound_idx        INTEGER,
    mass                INTEGER,                -- unit mass
    rt_s                REAL,
    height              REAL,
    prominence          REAL,
    width_scans         REAL,
    width_s             REAL,
    width_height        REAL,
    left_base_scan      INTEGER,
    right_base_scan     INTEGER,
    start_s             REAL,
    stop_s              REAL,
    flatness            REAL,
    tailing_ratio       REAL,
    is_max_of_cmp       INTEGER,
    PRIMARY KEY (measurement_id, peak_idx),
    FOREIGN KEY (measurement_id, compound_idx)
        REFERENCES compound (measurement_id, compound_idx) ON DELETE CASCADE
);

-- Identification of a compound edited by hand, replacing alpinac's compound_name and best
-- candidate (name NULL: marked as not identified). Lost when the measurement is imported again.
CREATE TABLE IF NOT EXISTS compound_identification (
    measurement_id      INTEGER,
    compound_idx        INTEGER,
    name                TEXT,                   -- a target, alias or any name
    edited_at           TEXT NOT NULL,
    spectrum_ref        TEXT,                   -- best library spectrum of the name, e.g. 'nist_CFC-12'
    score               REAL,                   -- as compound_candidate (NULL: no spectrum)
    score_combined      REAL,
    PRIMARY KEY (measurement_id, compound_idx),
    FOREIGN KEY (measurement_id, compound_idx)
        REFERENCES compound (measurement_id, compound_idx) ON DELETE CASCADE
);

-- Values of the extraction replaced by the fit (esper.fitting), to fit the measurement
-- again without importing it again: the peaks the fit changed, as extracted
CREATE TABLE IF NOT EXISTS extraction_peak (
    measurement_id      INTEGER REFERENCES measurement ON DELETE CASCADE,
    peak_idx            INTEGER,
    compound_idx        INTEGER,
    mass                INTEGER,
    rt_s                REAL,
    height              REAL,
    prominence          REAL,
    width_scans         REAL,
    width_s             REAL,
    width_height        REAL,
    left_base_scan      INTEGER,
    right_base_scan     INTEGER,
    start_s             REAL,
    stop_s              REAL,
    flatness            REAL,
    tailing_ratio       REAL,
    is_max_of_cmp       INTEGER,
    PRIMARY KEY (measurement_id, peak_idx)
);

-- ... and the summary columns of the compounds the fit changed, as extracted
CREATE TABLE IF NOT EXISTS extraction_compound (
    measurement_id      INTEGER REFERENCES measurement ON DELETE CASCADE,
    compound_idx        INTEGER,
    n_peaks             INTEGER,
    prominence_sum      REAL,
    rt_mean_s           REAL,
    rt_std_s            REAL,
    main_ion_mass       INTEGER,
    PRIMARY KEY (measurement_id, compound_idx)
);

CREATE INDEX IF NOT EXISTS peak_mass_rt ON peak (mass, rt_s);
CREATE INDEX IF NOT EXISTS peak_compound ON peak (measurement_id, compound_idx);
CREATE INDEX IF NOT EXISTS compound_name_idx ON compound (compound_name);

-- Target compounds: the substances we look for, imported with `esper targets`
CREATE TABLE IF NOT EXISTS target (
    name                TEXT PRIMARY KEY,
    rt_ref_s            REAL,                   -- rt in the reference run of the target list
    ri                  REAL,                   -- retention index (from rt_ref_s)
    formula             TEXT,
    chemical_name       TEXT,
    cas                 TEXT,
    cmp_class           TEXT,
    pubchem_cid         INTEGER,
    inchi               TEXT,
    smiles              TEXT
);

{TARGET_ION}
-- Expected retention time of a target in the runs from valid_from on (GCwerks peakid;
-- a row only when it changes). rt_s NULL: not identified from then on
CREATE TABLE IF NOT EXISTS target_rt (
    name                TEXT REFERENCES target ON DELETE CASCADE,
    valid_from          TEXT,                   -- ISO, as measurement.acq_datetime
    rt_s                REAL,
    window_s            REAL,                   -- full width of the identification window
    PRIMARY KEY (name, valid_from)
);

-- Other names of a target (e.g. as used in the spectrum libraries)
CREATE TABLE IF NOT EXISTS target_alias (
    alias               TEXT PRIMARY KEY,
    name                TEXT REFERENCES target ON DELETE CASCADE
);

-- Key/value information about the database, e.g. where the targets come from
CREATE TABLE IF NOT EXISTS meta (
    key                 TEXT PRIMARY KEY,
    value               TEXT
);

CREATE VIEW IF NOT EXISTS peak_full AS
SELECT
    m.name AS measurement,
    m.acq_datetime,
    m.sample_type,
    m.sample_number,
    p.*,
    c.compound_name,
    c.retention_index,
    c.valid_candidate,
    cc.spectrum_ref AS best_spectrum,
    cc.score AS best_score,
    cc.score_combined AS best_score_combined
FROM peak p
JOIN measurement m USING (measurement_id)
-- LEFT: the peaks of the fit that match no compound (compound_idx NULL) are kept
LEFT JOIN compound c USING (measurement_id, compound_idx)
LEFT JOIN compound_candidate cc
    ON cc.measurement_id = p.measurement_id
    AND cc.compound_idx = p.compound_idx
    AND cc.is_best = 1;
"""

SCHEMA = SCHEMA.replace("{TARGET_ION}", TARGET_ION)


def read_config(data_dir: Path) -> dict:
    """The esper.yaml of a data directory, empty if there is none."""
    path = Path(data_dir) / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        config = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid {path}: {e}") from e
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid {path}: expected keys like 'database: <path>'")
    return config


def add_config_entries(
    data_dir: Path, key: str, entries: dict, comment: str = ""
) -> None:
    """Add ``entries`` to the mapping ``key`` of the esper.yaml of a data directory (created
    if needed), its other lines and comments kept: after the last line of a block mapping
    ``key:``, else as a new block at the end. ``comment``: on each added line.

    Raises ValueError if ``key`` is not a block mapping (e.g. ``key: {a: b}``) or if an
    entry is already there.
    """
    path = Path(data_dir) / CONFIG_FILENAME
    config = read_config(data_dir)
    current = config.get(key)
    if current is not None and not isinstance(current, dict):
        raise ValueError(f"{path}: {key} is not a mapping")
    repeated = sorted(set(entries) & set(current or {}))
    if repeated:
        raise ValueError(f"{path}: already in {key}: {', '.join(map(str, repeated))}")
    suffix = f"  # {comment}" if comment else ""
    added = [
        f"  {yaml.safe_dump({name: value}, default_flow_style=True, width=1000)[1:-2]}"
        f"{suffix}"
        for name, value in entries.items()
    ]
    lines = path.read_text().splitlines() if path.is_file() else []
    start = next(
        (i for i, line in enumerate(lines) if line.rstrip() == f"{key}:"), None
    )
    if start is None:
        if key in config:  # e.g. a flow mapping on one line
            raise ValueError(
                f"{path}: {key} must be a block mapping ('{key}:' then one entry per line)"
            )
        lines += [f"{key}:", *added]
    else:
        end = start + 1  # after the indented lines (and comments) of the block
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
            end += 1
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = added
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    written = read_config(data_dir).get(key) or {}
    if any(k not in written or written[k] != v for k, v in entries.items()):
        raise ValueError(f"{path}: the entries added to {key} do not read back")


def db_path(data_dir: Path) -> Path:
    """The database of a data directory: ``database`` of its esper.yaml, or esper.sqlite in it.

    A relative ``database`` is relative to the data directory.
    """
    database = read_config(data_dir).get("database")
    if not database:
        return Path(data_dir) / DB_FILENAME
    return Path(data_dir) / Path(database).expanduser()


def chromatograms_dir(data_dir: Path) -> Path:
    return Path(data_dir) / CHROMATOGRAMS_DIRNAME


class SchemaVersionError(RuntimeError):
    """The database has a schema version that this esper cannot use."""


def connect(
    data_dir: Path, create: bool = False, readonly: bool = False
) -> sqlite3.Connection:
    """Open the database of a data directory, creating the schema if asked.

    Every connection checks that the schema version is usable; only ``create`` upgrades it.
    """
    path = db_path(data_dir)
    if not create and not path.is_file():
        raise FileNotFoundError(f"No esper database at {path}")
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.execute("PRAGMA foreign_keys = ON")
    try:
        version = _check_version(con, path)
        if create and not readonly and version <= SCHEMA_VERSION:
            _upgrade(con, path, version)
    except BaseException:
        con.close()
        raise
    return con


def min_reader_version(con: sqlite3.Connection) -> int | None:
    """The oldest SCHEMA_VERSION that can use the database, None if it does not say."""
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_info'"
    ).fetchone():
        return None
    row = con.execute(
        "SELECT value FROM schema_info WHERE key = 'min_reader_version'"
    ).fetchone()
    return None if row is None else row[0]


def _check_version(con: sqlite3.Connection, path: Path) -> int:
    """The schema version of the database, if this esper can use it."""
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        # Written by a newer esper: used as is (never downgraded) if it is compatible
        min_reader = min_reader_version(con) or LEGACY_MIN_READER_VERSIONS.get(version)
        if min_reader is None or min_reader > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{path} has schema version {version}, which needs a newer esper"
                f" (this one has schema version {SCHEMA_VERSION})"
            )
        logger.info(
            f"{path} has the newer schema version {version}, compatible with {SCHEMA_VERSION}"
        )
    return version


def _upgrade(con: sqlite3.Connection, path: Path, version: int) -> None:
    """Bring the schema of the database from an older or the same version to SCHEMA_VERSION."""
    if version not in COMPATIBLE_VERSIONS:
        raise SchemaVersionError(
            f"{path} has schema version {version}, expected one of {COMPATIBLE_VERSIONS}"
        )
    # Views are created again: CREATE VIEW IF NOT EXISTS keeps an older definition
    con.execute("DROP VIEW IF EXISTS peak_full")
    _rebuild_target_ion(con)
    con.executescript(SCHEMA + SCHEMA_INFO)
    add_columns(con)
    con.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('min_reader_version', ?)",
        (MIN_READER_VERSION,),
    )
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()


def _rebuild_target_ion(con: sqlite3.Connection) -> None:
    """Add valid_from to the key of a target_ion of before version 6 (SQLite cannot change
    a primary key in place): renamed, created again with its columns, rows copied. One
    transaction: nothing changes if it fails."""
    columns = table_columns(con, "target_ion")
    if not columns or "valid_from" in columns:
        return
    con.commit()
    con.execute("BEGIN")
    try:
        con.execute("ALTER TABLE target_ion RENAME TO target_ion_v5")
        con.execute(TARGET_ION)
        existing = set(table_columns(con, "target_ion"))
        for name, sql_type in ADDED_COLUMNS.get("target_ion", []):
            if name not in existing:
                con.execute(f"ALTER TABLE target_ion ADD COLUMN {name} {sql_type}")
        copied = ", ".join(columns)
        con.execute(f"INSERT INTO target_ion ({copied}) SELECT {copied} FROM target_ion_v5")
        con.execute("DROP TABLE target_ion_v5")
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def add_columns(con: sqlite3.Connection) -> None:
    """Add the missing ``ADDED_COLUMNS``."""
    for table, columns in ADDED_COLUMNS.items():
        existing = set(table_columns(con, table))
        for name, sql_type in columns:
            if name not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def insert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    """Insert the rows of a data frame (columns named as in the table, NaN/NA as NULL)."""
    if df.empty:
        return
    df = df.astype(object).where(df.notna(), None)
    columns = ", ".join(df.columns)
    placeholders = ", ".join("?" * len(df.columns))
    con.executemany(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        df.itertuples(index=False, name=None),
    )
