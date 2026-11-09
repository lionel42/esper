"""Import of the chromatograms of a GCwerks directory (esper.gcwerks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gcwerks.io import chromatogram as gc_chromatogram
from gcwerks.io.chromatogram import Chromatogram, IonChromatogram

from esper import db, gcwerks
from esper.alpinac import parse_measurement_name
from esper.fitting import to_fit
from esper.gui.chromatogram_store import ChromatogramStore
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


def test_find_chromatogram_files(tmp_path):
    names = ["250101.0013.std.5", "260112.2114.air.7", "260112.1907.tune"]
    root = _gcdir(tmp_path, names)
    expected = ["250101.0013.std.5", "260112.2114.air.7"]  # no tune file
    assert [p.name for p in gcwerks.find_chromatogram_files(root)] == expected
    assert [p.name for p in gcwerks.find_chromatogram_files(root / "26")] == expected[1:]
    channel = root / "26" / "chromatograms" / "channel0"
    assert [p.name for p in gcwerks.find_chromatogram_files(channel.parent)] == expected[1:]
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
    monkeypatch.setattr(gc_chromatogram, "read_chromatogram", lambda path: _chromatogram())
    data_dir = tmp_path / "esper"

    summary = upload(root, data_dir)
    assert summary.imported == ["260112.2114.air.7", "260112.2224.air.7"]
    con = db.connect(data_dir)
    row = con.execute(
        "SELECT sample_type, sample_number, rt_start_s, rt_stop_s, n_peaks, acq_start,"
        " acq_start_source, source_format, chromatogram_file FROM measurement WHERE name = ?",
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
    summary = upload(root, tmp_path / "esper")
    assert summary.imported == []
    assert [name for name, _ in summary.failed] == ["110101.0000.air.7"]


@pytest.mark.parametrize("source_format", ["alpinac", "gcwerks"])
def test_forced_format(tmp_path, source_format):
    summary = upload(tmp_path, tmp_path / "esper", fit=False, source_format=source_format)
    assert summary.imported == []
