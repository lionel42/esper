"""Opening databases of older and newer schema versions."""

from __future__ import annotations

import sqlite3

import pytest

from esper import db


def _version(tmp_path) -> int:
    with sqlite3.connect(db.db_path(tmp_path)) as con:
        return con.execute("PRAGMA user_version").fetchone()[0]


def _make_newer(tmp_path, min_reader: int | None, version: int = db.SCHEMA_VERSION + 10) -> None:
    """Turn the database into one written by a newer esper."""
    con = db.connect(tmp_path, create=True)
    if min_reader is None:
        con.execute("DROP TABLE schema_info")
    else:
        con.execute("UPDATE schema_info SET value = ? WHERE key = 'min_reader_version'", (min_reader,))
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
