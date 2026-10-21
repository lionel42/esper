"""SQLite database layout of an esper data directory.

An esper data directory looks like::

    <data_dir>/
        esper.yaml                  # optional, "database: <path>": where the database is
        esper.sqlite                # the database, unless esper.yaml says otherwise
        chromatograms/
            <measurement>.parquet   # unit mass chromatograms (index: rt [s], columns: masses + 'tic')

Units are part of the column names: ``_s`` seconds, ``_scans`` time bins.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import yaml

DB_FILENAME = "esper.sqlite"
CONFIG_FILENAME = "esper.yaml"
CHROMATOGRAMS_DIRNAME = "chromatograms"

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
# Versions that are upgraded by creating the missing tables
COMPATIBLE_VERSIONS = (0, 1, 2)
# Oldest SCHEMA_VERSION of esper that can still use a database of this version, stored in
# the database so that older esper versions know whether they can use it. A schema change
# that only adds tables or nullable columns keeps it, a breaking change sets it to SCHEMA_VERSION.
MIN_READER_VERSION = 2
# MIN_READER_VERSION of the schema versions written before schema_info existed
LEGACY_MIN_READER_VERSIONS = {3: 2}  # 3 only adds nullable columns

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
    source_dir          TEXT NOT NULL,          -- the .alpinac folder
    source_mtime        REAL NOT NULL,          -- mtime of pipeline.nc, to detect reprocessing
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

-- Ions expected in the spectrum of a target, by decreasing importance
CREATE TABLE IF NOT EXISTS target_ion (
    name                TEXT REFERENCES target ON DELETE CASCADE,
    rank                INTEGER,                -- 0: main (quantification) ion
    ion                 TEXT,                   -- e.g. 'CF2Cl+'
    exact_mass          REAL,
    mass                INTEGER,                -- unit mass
    PRIMARY KEY (name, rank)
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
JOIN compound c USING (measurement_id, compound_idx)
LEFT JOIN compound_candidate cc
    ON cc.measurement_id = p.measurement_id
    AND cc.compound_idx = p.compound_idx
    AND cc.is_best = 1;
"""


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


def connect(data_dir: Path, create: bool = False, readonly: bool = False) -> sqlite3.Connection:
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
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_info'").fetchone():
        return None
    row = con.execute("SELECT value FROM schema_info WHERE key = 'min_reader_version'").fetchone()
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
        logger.info(f"{path} has the newer schema version {version}, compatible with {SCHEMA_VERSION}")
    return version


def _upgrade(con: sqlite3.Connection, path: Path, version: int) -> None:
    """Bring the schema of the database from an older or the same version to SCHEMA_VERSION."""
    if version not in COMPATIBLE_VERSIONS:
        raise SchemaVersionError(
            f"{path} has schema version {version}, expected one of {COMPATIBLE_VERSIONS}"
        )
    con.executescript(SCHEMA + SCHEMA_INFO)
    con.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('min_reader_version', ?)",
        (MIN_READER_VERSION,),
    )
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
