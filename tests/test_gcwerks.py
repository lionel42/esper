"""Import of the chromatograms of a GCwerks directory (esper.sources.gcwerks), and the
instrument of a data directory (esper.sources)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gcwerks.io import chromatogram as gc_chromatogram
from gcwerks.io.chromatogram import Chromatogram, IonChromatogram

from esper import db
from esper.fitting import to_fit
from esper.gui.chromatogram_store import ChromatogramStore
from esper.sources import (
    InstrumentError,
    declared_instrument,
    gcwerks,
    parse_measurement_name,
)
from esper.upload import upload

START = pd.Timestamp("2026-01-12 21:14:05")


def _chromatogram() -> Chromatogram:
    times = np.arange(10) * 0.5
    return Chromatogram(
        start=START,
        times=times,
        tic=np.arange(10, dtype=float) * 10,
        ions=[
            IonChromatogram(43.0, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])),
            IonChromatogram(85.0, np.array([3, 4]), np.array([5.0, 6.0])),
            # a second window of m/z 43
            IonChromatogram(43.0, np.array([7, 8]), np.array([7.0, 8.0])),
        ],
    )


def _gcdir(tmp_path, names):
    root = tmp_path / "gcdir"
    (root / "config").mkdir(parents=True)
    (root / "config" / "gcwerks.conf").write_text("")
    (root / "config" / "peak.list").write_text("CFC-12\n")
    for name in names:
        folder = root / name[:2] / "chromatograms" / "channel0"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"")
    # not chromatograms of channel0
    chromatograms = root / "26" / "chromatograms"
    for folder in ("channel0", "channel0.integration"):
        (chromatograms / folder).mkdir(parents=True, exist_ok=True)
    (chromatograms / "channel0.integration" / "260112.2114.air.7").write_bytes(b"")
    (chromatograms / "channel0" / "notes.txt").write_text("")
    return root


def test_chromatogram_frame():
    frame, warnings = gcwerks.chromatogram_frame(_chromatogram())
    assert warnings == []
    assert frame.index.name == "rt"
    assert list(frame.columns) == ["43", "85", "tic"]
    np.testing.assert_array_equal(
        frame["43"].to_numpy(), [1, 2, 3, np.nan, np.nan, np.nan, np.nan, 7, 8, np.nan]
    )
    assert frame["85"].count() == 2
    np.testing.assert_array_equal(frame["tic"], np.arange(10) * 10)


def test_masses_kept_as_measured():
    """The masses of the scans are not unit masses: one column each, not rounded."""
    from esper.gui.chromatogram_store import column_mass, mass_column

    chromatogram = _chromatogram()
    chromatogram.ions += [
        IonChromatogram(84.9, np.array([5]), np.array([9.0])),
        IonChromatogram(85.2, np.array([5]), np.array([10.0])),
        IonChromatogram(85.0000001, np.array([6]), np.array([11.0])),  # m/z 85
    ]
    frame, warnings = gcwerks.chromatogram_frame(chromatogram)
    assert warnings == []
    assert list(frame.columns) == ["43", "84.9", "85", "85.2", "tic"]
    assert frame["85"].count() == 3
    assert [column_mass(c) for c in frame.columns] == [43, 84.9, 85, 85.2, None]
    assert [mass_column(m) for m in (43, 84.9, 85.0)] == ["43", "84.9", "85"]


def test_windows_are_not_continuous():
    """A row of NaN between two time windows (layout B), where the chromatograms are not
    drawn continuous; one NaN in each gap of a drawn chromatogram."""
    from esper.gui.chromatogram_store import with_breaks

    chromatogram = Chromatogram(
        start=START,
        times=np.array([0.0, 0.5, 1.0, 1.6, 1.85]),
        tic=np.arange(5, dtype=float),
        ions=[
            IonChromatogram(69.0, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])),
            IonChromatogram(69.0, np.array([3, 4]), np.array([4.0, 5.0])),
            IonChromatogram(50.0, np.array([0, 1, 2]), np.array([6.0, 7.0, 8.0])),
        ],
        windows=[0, 3],
    )
    frame, warnings = gcwerks.chromatogram_frame(chromatogram)
    assert warnings == []
    np.testing.assert_allclose(frame.index, [0, 0.5, 1.0, 1.3, 1.6, 1.85])
    assert frame.loc[1.3].isna().all()
    np.testing.assert_array_equal(frame["69"].dropna(), [1, 2, 3, 4, 5])
    drawn = with_breaks(frame["69"])
    assert drawn.isna().tolist() == [False] * 3 + [True] + [False] * 2
    # 50: not in the second window: no NaN after its last value
    assert with_breaks(frame["50"]).tolist() == [6.0, 7.0, 8.0]
    # the same mass twice in the same scans: warned unless the same values
    chromatogram.ions.append(IonChromatogram(69.0, np.array([4]), np.array([5.0])))
    assert gcwerks.chromatogram_frame(chromatogram)[1] == []
    chromatogram.ions.append(IonChromatogram(69.0, np.array([4]), np.array([6.0])))
    (warning,) = gcwerks.chromatogram_frame(chromatogram)[1]
    assert warning.startswith("mass 69 measured twice in 1 scans (1.9 s")


def test_with_breaks():
    from esper.gui.chromatogram_store import with_breaks

    s = pd.Series([np.nan, 1, np.nan, np.nan, 2, 3, np.nan, 4, np.nan, np.nan])
    assert with_breaks(s).index.tolist() == [1, 2, 4, 5, 6, 7]
    assert with_breaks(pd.Series([np.nan, np.nan])).empty
    assert with_breaks(pd.Series([1.0, 2.0])).tolist() == [1.0, 2.0]


def test_find_chromatogram_files(tmp_path):
    names = ["250101.0013.std.5", "260112.2114.air.7", "260112.1907.tune"]
    root = _gcdir(tmp_path, names)
    expected = ["250101.0013.std.5", "260112.2114.air.7"]  # no tune file
    assert [p.name for p in gcwerks.find_chromatogram_files(root)] == expected
    assert [p.name for p in gcwerks.find_chromatogram_files(root / "26")] == expected[
        1:
    ]
    channel = root / "26" / "chromatograms" / "channel0"
    assert [
        p.name for p in gcwerks.find_chromatogram_files(channel.parent)
    ] == expected[1:]
    assert gcwerks.gcwerks_root(channel) == root.resolve()
    assert not gcwerks.is_gcwerks_source(tmp_path)


def test_name_without_number():
    assert parse_measurement_name("260112.1614.leak-test") == {
        "acq_datetime": "2026-01-12T16:14:00",
        "sample_type": "leak-test",
        "sample_number": None,
    }


def test_upload(tmp_path, monkeypatch):
    root = _gcdir(tmp_path, ["260112.2114.air.7", "260112.2224.air.7"])
    monkeypatch.setattr(
        gc_chromatogram, "read_chromatogram", lambda path: _chromatogram()
    )
    data_dir = tmp_path / "esper"

    summary = upload(root, data_dir, instrument="gcwerks")
    assert declared_instrument(data_dir) == "gcwerks"
    assert summary.imported == ["260112.2114.air.7", "260112.2224.air.7"]
    con = db.connect(data_dir)
    row = con.execute(
        "SELECT sample_type, sample_number, rt_start_s, rt_stop_s, n_peaks, acq_start,"
        " acq_start_source, instrument, peak_source, chromatogram_file FROM measurement"
        " WHERE name = ?",
        ("260112.2114.air.7",),
    ).fetchone()
    assert row == (
        "air",
        7,
        0.0,
        4.5,
        0,
        "2026-01-12T21:14:05.000",
        "gcwerks",
        "gcwerks",
        None,
        "chromatograms/260112.2114.air.7.parquet",
    )
    # chromatograms only: not fitted
    assert to_fit(con) == []
    con.close()

    # the GUI reads a mass without the scans where it was not measured
    store = ChromatogramStore(lambda mid: data_dir / row[-1])
    assert store.series(1, "85").tolist() == [5.0, 6.0]

    assert upload(root, data_dir).skipped_unchanged == summary.imported
    assert upload(root, data_dir, force=True).imported == summary.imported


def test_unreadable_file_fails(tmp_path, monkeypatch):
    root = _gcdir(tmp_path, ["110101.0000.air.7"])

    def read(path):
        raise ValueError("old format")

    monkeypatch.setattr(gc_chromatogram, "read_chromatogram", read)
    summary = upload(root, tmp_path / "esper", instrument="gcwerks")
    assert summary.imported == []
    assert [name for name, _ in summary.failed] == ["110101.0000.air.7"]


def test_new_data_dir_needs_an_instrument(tmp_path):
    with pytest.raises(InstrumentError, match="--instrument"):
        upload(tmp_path / "runs", tmp_path / "esper")


def test_gcwerks_source_needs_a_declared_instrument(tmp_path):
    root = _gcdir(tmp_path, [])
    data_dir = tmp_path / "esper"
    db.connect(data_dir, create=True).close()  # existing data directory, undeclared
    with pytest.raises(InstrumentError, match="--instrument gcwerks"):
        upload(root, data_dir)


def test_legacy_data_dir_is_tofwerk(tmp_path):
    data_dir = tmp_path / "esper"
    data_dir.mkdir()
    (data_dir / "esper.yaml").write_text("# shared\ndatabase: esper.sqlite")
    db.connect(data_dir, create=True).close()
    upload(tmp_path / "runs", data_dir, fit=False)
    assert declared_instrument(data_dir) == "tofwerk"
    # the other lines kept
    assert (
        (data_dir / "esper.yaml")
        .read_text()
        .startswith("# shared\ndatabase: esper.sqlite\n")
    )
    assert db.db_path(data_dir) == data_dir / "esper.sqlite"


def test_other_instrument_is_refused(tmp_path):
    data_dir = tmp_path / "esper"
    upload(tmp_path / "runs", data_dir, fit=False, instrument="tofwerk")
    with pytest.raises(InstrumentError, match="has the runs of tofwerk"):
        upload(tmp_path / "runs", data_dir, instrument="gcwerks")
    # declared: no need to give it again
    upload(tmp_path / "runs", data_dir, fit=False)


def test_measurements_of_another_instrument_are_refused(tmp_path, monkeypatch):
    root = _gcdir(tmp_path, ["260112.2114.air.7"])
    monkeypatch.setattr(
        gc_chromatogram, "read_chromatogram", lambda path: _chromatogram()
    )
    data_dir = tmp_path / "esper"
    upload(root, data_dir, instrument="gcwerks")
    (data_dir / "esper.yaml").write_text("instrument: tofwerk\n")
    with pytest.raises(InstrumentError, match="measurements of gcwerks"):
        upload(tmp_path / "runs", data_dir)


def test_masses_of_a_unit_mass():
    """A unit mass stands for the masses measured within +- 0.5 (69.5: 69 and 70)."""
    from esper.gui.chromatogram_store import masses_of_unit, unit_masses

    measured = [68.4, 68.5, 69.2, 69.5, 69.6, 85]
    assert masses_of_unit(measured, 69) == [68.5, 69.2, 69.5]
    assert masses_of_unit(measured, 70) == [69.5, 69.6]
    assert masses_of_unit(measured, 85) == [85]
    assert masses_of_unit(measured, 84) == []
    assert unit_masses(measured) == [68, 69, 70, 85]


def test_scan_runs_are_drawn_continuous(tmp_path, monkeypatch):
    """A scan run sweeps the whole mass range: a mass without a value in a scan was not
    found there, its curve links the values around (no gap, as between SIM windows)."""
    root = _gcdir(tmp_path, ["260112.2114.air.7", "260112.2224.std.9.scan"])

    def read(path):
        chromatogram = _chromatogram()
        chromatogram.kind = 5 if path.name.endswith(".scan") else 4
        return chromatogram

    monkeypatch.setattr(gc_chromatogram, "read_chromatogram", read)
    data_dir = tmp_path / "esper"
    upload(root, data_dir, instrument="gcwerks")
    files = {
        mid: data_dir / "chromatograms" / f"{name}.parquet"
        for mid, name in enumerate(["260112.2114.air.7", "260112.2224.std.9.scan"])
    }
    store = ChromatogramStore(files.get)
    assert not store.continuous(0) and store.continuous(1)
    # 43: scans 0-2 and 7-8
    assert store.drawn(0, "43").isna().sum() == 1
    assert store.drawn(1, "43").tolist() == [1.0, 2.0, 3.0, 7.0, 8.0]
    assert store.series(1, "43").tolist() == [1.0, 2.0, 3.0, 7.0, 8.0]


def test_upload_between_dates(tmp_path, monkeypatch, capsys):
    """--after (on or after) and --before (before): from the date in the names."""
    from datetime import datetime

    from esper.cli.main import main
    from esper.upload import in_dates

    names = ["260111.2300.air.7", "260112.0000.air.7", "260112.2114.air.7", "260113.0800.air.7"]
    root = _gcdir(tmp_path, names)
    monkeypatch.setattr(
        gc_chromatogram, "read_chromatogram", lambda path: _chromatogram()
    )
    data_dir = tmp_path / "esper"
    summary = upload(
        root, data_dir, instrument="gcwerks",
        after=datetime(2026, 1, 12), before=datetime(2026, 1, 13),
    )
    assert summary.imported == ["260112.0000.air.7", "260112.2114.air.7"]
    assert summary.skipped_dates == ["260111.2300.air.7", "260113.0800.air.7"]
    assert in_dates("no-date") and not in_dates("no-date", before=datetime(2026, 1, 1))

    assert main(["upload", str(root), str(data_dir), "--after", "2026-01-12 21:14"]) == 0
    out = capsys.readouterr().out
    assert "imported: 1," in out and "outside the dates: 2" in out
    with pytest.raises(SystemExit):
        main(["upload", str(root), str(data_dir), "--before", "12.01.2026"])
    # a year or a month: its start (--after includes it, --before leaves it out)
    from esper.cli.main import _date_argument

    assert _date_argument("2008") == datetime(2008, 1, 1)
    assert _date_argument("2008-08") == datetime(2008, 8, 1)
    assert _date_argument("2008-08-10 12:30") == datetime(2008, 8, 10, 12, 30)
    assert main(["upload", str(root), str(data_dir), "--after", "2026", "--before", "2027"]) == 0
    assert "outside the dates: 0" in capsys.readouterr().out
    assert main(["upload", str(root), str(data_dir), "--before", "2026"]) == 0
    assert "outside the dates: 4" in capsys.readouterr().out
