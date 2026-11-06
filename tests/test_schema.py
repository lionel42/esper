"""Opening databases of older and newer schema versions."""

from __future__ import annotations

import sqlite3

import pytest

from esper import db


def _version(tmp_path) -> int:
    with sqlite3.connect(db.db_path(tmp_path)) as con:
        return con.execute("PRAGMA user_version").fetchone()[0]


def _make_newer(
    tmp_path, min_reader: int | None, version: int = db.SCHEMA_VERSION + 10
) -> None:
    """Turn the database into one written by a newer esper."""
    con = db.connect(tmp_path, create=True)
    if min_reader is None:
        con.execute("DROP TABLE schema_info")
    else:
        con.execute(
            "UPDATE schema_info SET value = ? WHERE key = 'min_reader_version'",
            (min_reader,),
        )
    con.execute(f"PRAGMA user_version = {version}")
    con.commit()
    con.close()


def test_new_database(tmp_path):
    con = db.connect(tmp_path, create=True)
    assert db.min_reader_version(con) == db.MIN_READER_VERSION
    con.close()
    assert _version(tmp_path) == db.SCHEMA_VERSION


def test_older_database_is_upgraded(tmp_path):
    with sqlite3.connect(db.db_path(tmp_path)) as con:
        con.execute("PRAGMA user_version = 1")
    con = db.connect(tmp_path, create=True)
    assert db.min_reader_version(con) == db.MIN_READER_VERSION
    con.close()
    assert _version(tmp_path) == db.SCHEMA_VERSION


def test_newer_compatible_database_is_not_downgraded(tmp_path):
    _make_newer(tmp_path, min_reader=db.SCHEMA_VERSION)
    db.connect(tmp_path, create=True).close()
    assert _version(tmp_path) == db.SCHEMA_VERSION + 10


@pytest.mark.parametrize("kwargs", [{}, {"readonly": True}])
def test_newer_compatible_database_is_opened(tmp_path, kwargs):
    _make_newer(tmp_path, min_reader=db.SCHEMA_VERSION)
    db.connect(tmp_path, **kwargs).close()
    assert _version(tmp_path) == db.SCHEMA_VERSION + 10


@pytest.mark.parametrize("kwargs", [{"create": True}, {}, {"readonly": True}])
@pytest.mark.parametrize("min_reader", [None, db.SCHEMA_VERSION + 10])
def test_newer_incompatible_database_is_refused(tmp_path, min_reader, kwargs):
    _make_newer(tmp_path, min_reader=min_reader)
    with pytest.raises(db.SchemaVersionError, match="needs a newer esper"):
        db.connect(tmp_path, **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"readonly": True}])
def test_older_database_is_not_upgraded_without_create(tmp_path, kwargs):
    with sqlite3.connect(db.db_path(tmp_path)) as con:
        con.execute("PRAGMA user_version = 1")
    db.connect(tmp_path, **kwargs).close()
    assert _version(tmp_path) == 1


@pytest.mark.skipif(db.SCHEMA_VERSION >= 3, reason="version 3 is not newer")
def test_legacy_version_3_without_schema_info(tmp_path):
    _make_newer(tmp_path, min_reader=None, version=3)
    db.connect(tmp_path, create=True).close()
    assert _version(tmp_path) == 3


def test_upgrade_adds_the_fit_columns(tmp_path):
    # A database of version 2: the tables of SCHEMA, without ADDED_COLUMNS
    with sqlite3.connect(db.db_path(tmp_path)) as con:
        con.executescript(db.SCHEMA)
        con.execute(
            "INSERT INTO peak (measurement_id, peak_idx, mass, rt_s) VALUES (1, 0, 43, 100.0)"
        )
        con.execute("PRAGMA user_version = 2")
    con = db.connect(tmp_path, create=True)
    columns = db.table_columns(con, "peak")
    assert {"source", "fit_rt_s", "fit_area"} <= set(columns)
    assert "fit_version" in db.table_columns(con, "measurement")
    assert con.execute("SELECT source, fit_area FROM peak").fetchone() == (
        "extraction",
        None,
    )
    con.close()
    # again: nothing to add
    db.connect(tmp_path, create=True).close()
    assert _version(tmp_path) == db.SCHEMA_VERSION


def _peak_full_count(con) -> int:
    con.execute(
        "INSERT INTO measurement (measurement_id, name, source_dir, source_mtime,"
        " source_size, imported_at) VALUES (1, 'm', 'd', 0, 0, 'now')"
    )
    con.execute("INSERT INTO compound (measurement_id, compound_idx) VALUES (1, 0)")
    # A peak of the fit that matches no compound: compound_idx NULL
    con.execute(
        "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass)"
        " VALUES (1, 0, 0, 43), (1, 1, NULL, 43)"
    )
    return con.execute("SELECT COUNT(*) FROM peak_full").fetchone()[0]


def test_peak_full_keeps_peaks_without_compound(tmp_path):
    con = db.connect(tmp_path, create=True)
    try:
        assert _peak_full_count(con) == 2
    finally:
        con.close()


def test_upgrade_replaces_peak_full(tmp_path):
    con = db.connect(tmp_path, create=True)
    # The view of older versions (inner join on compound)
    con.execute("DROP VIEW peak_full")
    con.execute(
        "CREATE VIEW peak_full AS SELECT p.* FROM peak p"
        " JOIN compound c USING (measurement_id, compound_idx)"
    )
    con.commit()
    con.close()
    con = db.connect(tmp_path, create=True)
    try:
        assert _peak_full_count(con) == 2
    finally:
        con.close()
