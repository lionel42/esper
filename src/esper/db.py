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

import sqlite3
from pathlib import Path

import yaml

DB_FILENAME = "esper.sqlite"
CONFIG_FILENAME = "esper.yaml"
CHROMATOGRAMS_DIRNAME = "chromatograms"

SCHEMA_VERSION = 3
# Versions that are upgraded by creating the missing tables and columns
COMPATIBLE_VERSIONS = (0, 1, 2, 3)

# Number of masses of a compound with their area (q1 has the largest area)
N_QUALIFIERS = 5

# Uncertainties (1 sigma) are in a column named after the value with UNC_SUFFIX
UNC_SUFFIX = "_unc"

# Columns added in version 3, filled by esper.integration (NULL until integrated).
# Not in the CREATE TABLE statements: added by ALTER TABLE, to new and old databases alike.
ADDED_COLUMNS = {
    "measurement": [
        ("integration_version", "INTEGER"),     # esper.integration.INTEGRATION_VERSION, NULL: not integrated
        ("integrated_at", "TEXT"),
        ("noise_gain", "REAL"),                 # noise of the chromatograms (esper.integration.NoiseParameters)
        ("noise_factor", "REAL"),
        ("noise_d2_ratio", "REAL"),
    ],
    "peak": [
        ("int_start_s", "REAL"),                # integration bounds
        ("int_stop_s", "REAL"),
        ("base_start", "REAL"),                 # baseline at the bounds [counts]
        ("base_stop", "REAL"),
        ("area", "REAL"),                       # above the baseline [counts * s]
        ("area_unc", "REAL"),                   # its statistical uncertainty (1 sigma)
    ],
    "compound": [("area_sum", "REAL"), ("area_sum_unc", "REAL")]  # area of all its peaks
    + [                                         # its masses by decreasing area
        (f"q{i}_{kind}", sql_type)
        for i in range(1, N_QUALIFIERS + 1)
        for kind, sql_type in (("mass", "INTEGER"), ("area", "REAL"), ("area_unc", "REAL"))
    ],
}

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


def connect(data_dir: Path, create: bool = False, readonly: bool = False) -> sqlite3.Connection:
    """Open the database of a data directory, creating the schema if asked."""
    path = db_path(data_dir)
    if not create and not path.is_file():
        raise FileNotFoundError(f"No esper database at {path}")
    if readonly:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    if create:
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version not in COMPATIBLE_VERSIONS:
            raise RuntimeError(
                f"{path} has schema version {version}, expected one of {COMPATIBLE_VERSIONS}"
            )
        con.executescript(SCHEMA)
        _add_columns(con)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return con


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def _add_columns(con: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        existing = set(table_columns(con, table))
        for name, sql_type in columns:
            if name not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    con.commit()
