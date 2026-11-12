"""The information on a data directory (File > Data directory > Info in the GUI)."""

from __future__ import annotations

import os

import pytest

from esper import db
from esper.data_dir_info import as_text, describe
from esper.fitting import FIT_VERSION


def _measurement(con, name, sample_type, fit_version, peak_source="alpinac"):
    con.execute(
        "INSERT INTO measurement (name, acq_datetime, sample_type, n_peaks, source_dir,"
        " source_mtime, source_size, imported_at, fit_version, fitted_at, peak_source)"
        " VALUES (?, ?, ?, 10, 'src', 0, 0, '2026-11-01T10:00:00', ?, ?, ?)",
        (
            name,
            f"20{name[:2]}-{name[2:4]}-{name[4:6]}T{name[7:9]}:{name[9:11]}:00",
            sample_type,
            fit_version,
            None if fit_version is None else "2026-11-02T08:30:00",
            peak_source,
        ),
    )


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "esper.yaml").write_text(
        "instrument: tofwerk\naliases:\n  a: b\n  c: null\n"
    )
    con = db.connect(tmp_path, create=True)
    with con:
        _measurement(con, "241007.0423.air.1", "air", FIT_VERSION)
        _measurement(con, "241008.0900.air.2", "air", FIT_VERSION - 1)
        _measurement(con, "241009.1200.std.1", "std", None)
        con.execute("INSERT INTO target (name, ri) VALUES ('CFC-11', 512)")
        con.execute("INSERT INTO target (name) VALUES ('CFC-12')")
        con.execute("INSERT INTO target_alias VALUES ('R11', 'CFC-11')")
    con.close()
    return tmp_path


def test_describe(data_dir):
    con = db.connect(data_dir, readonly=True)
    sections = describe(data_dir, con)
    con.close()
    info = {s.title: dict(s.rows) for s in sections}
    assert info["Location"]["Path"] == str(data_dir.resolve())
    assert info["Location"]["Database"].startswith(str(data_dir / "esper.sqlite"))
    assert info["Database"]["Schema version"].startswith(f"{db.SCHEMA_VERSION} ")
    m = info["Measurements"]
    assert m["Instrument"] == "tofwerk"
    assert m["Measurements"] == "3"
    assert m["Acquired"] == "2024-10-07 04:23 to 2024-10-09 12:00"
    assert m["Sample types"] == "air 2, std 1"
    assert m["With peaks"] == "3 (30 peaks)"
    assert m["Fitted"].startswith(f"1 (fit version {FIT_VERSION}), 1 by an older")
    assert "1 not fitted" in m["Fitted"]
    assert m["Last fit"] == "2026-11-02 08:30"
    t = info["Targets"]
    assert t["Targets"].startswith("2 (1 with retention index")
    assert t["Aliases of the targets"] == "1"
    assert t["Aliases of esper.yaml"].startswith("2 names (1 of them")
    text = as_text(sections)
    assert text.splitlines()[0] == "Location"
    assert "  Measurements: 3" in text


def test_no_peaks_lines_for_gcwerks(tmp_path):
    (tmp_path / "esper.yaml").write_text("instrument: gcwerks\n")
    con = db.connect(tmp_path, create=True)
    with con:
        _measurement(con, "250101.0013.air.1", "air", None, peak_source=None)
        con.execute("UPDATE measurement SET n_peaks = 0")  # chromatograms only
    info = {s.title: dict(s.rows) for s in describe(tmp_path, con)}
    con.close()
    assert "With peaks" not in info["Measurements"]
    assert "Fitted" not in info["Measurements"]
    assert info["Measurements"]["Peaks from"] == "no peaks yet 1"
    assert info["Measurements"]["Processing method"].startswith("none")


def test_dialog_copies(data_dir):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from esper.gui.data_dir_dialog import DataDirDialog, copy_path

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    con = db.connect(data_dir, readonly=True)
    dialog = DataDirDialog(data_dir, con)
    con.close()
    assert "CFC" not in dialog.label.text()  # names of targets: not listed
    assert "Measurements" in dialog.label.text()
    assert copy_path(data_dir) == str(data_dir.resolve())
    assert app.clipboard().text() == str(data_dir.resolve())
    dialog.copy_all()
    assert app.clipboard().text().startswith("Location\n  Path: ")


def test_esper_info(data_dir, tmp_path_factory, capsys):
    from esper.cli.main import main

    assert main(["info", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Location\n  Path: {data_dir.resolve()}\n")
    assert "  Measurements: 3\n" in out
    empty = tmp_path_factory.mktemp("empty")
    assert main(["info", str(empty)]) == 1  # no database: said, no traceback
    assert "No esper database" in capsys.readouterr().err
