"""Mass calibration of the runs (masscal files) in the database, with absolute times."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from masscal import MassCalibration

from esper import db, mass_cal
from esper.gui.data import EsperData
from esper.gui.panels.mass_cal_plot import (
    calibrant_drift,
    calibrant_residuals,
    drift,
    run_drift,
)

NAME = "240315.2211.air.1"
S_PER_BIN = 0.2
# 2024-03-15T22:11:39.758Z as a Windows FILETIME
ACQUISITION_TIME_ZERO = 133550142997580000

LEGACY = """3\tNumber of points, mass cal. vs RT
2\tNumber of points, mass cal. u vs mass
2\tMass cal. mode
3.6e-05\tPeak shape, sigma_slope vs TOF index
0.51\tPeak shape, sigma_b vs TOF index
9.6e-05\tPeak shape, sigma_slope vs mass
0.0013\tPeak shape, sigma_b vs mass
0.6\talpha, for tofidx Pseudo-Voigt fit
0.2\ts_per_bin
1000\t2829.2\t-3731.5\t0.49994
2000\t2829.3\t-3731.6\t0.49994
3000\t2829.1\t-3731.4\t0.49995
19.0\t6.7
69.0\t3.0
"""


def write_h5(path, time_zero: int | None = ACQUISITION_TIME_ZERO):
    with h5py.File(path, "w") as f:
        f.attrs["NbrWrites"] = np.array([2], dtype=np.int32)
        f.attrs["NbrBufs"] = np.array([3], dtype=np.int32)
        f.create_dataset(
            "FullSpectra/TofData", data=np.zeros((2, 3, 1, 10), np.float32)
        )
        td = f.create_group("TimingData")
        td.attrs["BlockPeriod"] = np.array([S_PER_BIN * 1e9])
        if time_zero is not None:
            td.attrs["AcquisitionTimeZero"] = np.array([time_zero], dtype=np.uint64)


@pytest.fixture
def folder(tmp_path):
    folder = tmp_path / f"{NAME}.alpinac"
    folder.mkdir()
    (folder / f"{NAME}.EI_mc.txt").write_text(LEGACY)
    return folder


def test_legacy_file_with_h5(folder):
    write_h5(folder.parent / f"{NAME}.h5")
    run = mass_cal.read_run(folder, NAME)
    assert run.start_source == mass_cal.START_H5
    assert run.warnings == ["No calibrant peaks of the mass calibration"]
    assert (
        run.start_time.isoformat(timespec="milliseconds") == "2024-03-15T22:11:39.758"
    )
    assert run.ionization == "EI" and run.s_per_bin == pytest.approx(S_PER_BIN)
    points = mass_cal.tables(run)["mass_cal_rt"]
    assert points["rt_idx"].tolist() == [1000, 2000, 3000]
    assert points["rt_s"].tolist() == pytest.approx([200.0, 400.0, 600.0])
    # start + rt, to the ms
    assert points["time"].tolist() == [
        "2024-03-15T22:14:59.758",
        "2024-03-15T22:18:19.758",
        "2024-03-15T22:21:39.758",
    ]
    assert points["ok"].tolist() == [1, 1, 1]


def test_without_h5_start_from_the_name(folder):
    run = mass_cal.read_run(folder, NAME)
    assert run.start_source == mass_cal.START_NAME
    assert run.start_time.isoformat() == "2024-03-15T22:11:00"
    assert run.s_per_bin == pytest.approx(S_PER_BIN)  # from the legacy file
    assert any("to the minute" in w for w in run.warnings)


def test_no_calibration(tmp_path):
    folder = tmp_path / f"{NAME}.alpinac"
    folder.mkdir()
    run = mass_cal.read_run(folder, NAME)
    assert run.calibration is None and "No mass calibration file" in run.warnings
    assert mass_cal.tables(run) == {}


def test_masscal_file_preferred_with_rejected_windows(folder):
    write_h5(folder.parent / f"{NAME}.h5")
    windows = [
        {"rt_start": 0, "rt_stop": 120, "rt": 60.0, "n_calibrants": 9, "n_used": 9,
         "rms_ppm": 1.2, "max_ppm": 2.5, "ok": True, "p1": 2829.2, "p2": -3731.5, "p3": 0.5},
        {"rt_start": 60, "rt_stop": 180, "rt": 120.0, "n_calibrants": 4, "n_used": 0,
         "rms_ppm": None, "max_ppm": None, "ok": False, "p1": None, "p2": None, "p3": None},
    ]  # fmt: skip
    cal = MassCalibration(
        mode="power", rt=[60.0], params=[[2829.2, -3731.5, 0.5]], segments=(0, 1),
        metadata={"windows": windows},
    )  # fmt: skip
    cal.to_json(folder / f"{NAME}.seg0_mc.json")
    run = mass_cal.read_run(folder, NAME)
    t = mass_cal.tables(run)
    assert t["mass_cal"].loc[0, "segments"] == "seg0"
    assert t["mass_cal"].loc[0, "ionization"] is None
    points = t["mass_cal_rt"]
    assert points["ok"].tolist() == [1, 0]
    assert points["rms_ppm"].iloc[0] == pytest.approx(1.2)
    assert points["n_calibrants"].tolist() == [9, 4]
    assert points["rt_idx"].tolist() == [300, 600]


def _measurement(con, name, source_dir):
    cur = con.execute(
        "INSERT INTO measurement (name, acq_datetime, sample_type, source_dir, source_mtime,"
        " source_size, imported_at) VALUES (?, '2024-03-15T22:11:00', 'air', ?, 0, 0, '')",
        (name, str(source_dir)),
    )
    return cur.lastrowid


def test_refresh_and_gui_points(folder, tmp_path):
    write_h5(folder.parent / f"{NAME}.h5")
    data_dir = tmp_path / "data"
    con = db.connect(data_dir, create=True)
    mid = _measurement(con, NAME, folder)
    # Measurement of an older esper: a point without time (start from the name + rt)
    old = _measurement(con, "240316.0000.air.1", tmp_path / "missing.alpinac")
    con.execute("INSERT INTO mass_cal (measurement_id, mode) VALUES (?, 2)", (old,))
    con.execute(
        "INSERT INTO mass_cal_rt (measurement_id, rt_idx, rt_s, p1, p2, p3)"
        " VALUES (?, 5, 1.0, 2829.2, -3731.5, 0.5)",
        (old,),
    )
    con.commit()

    assert mass_cal.refresh(con, NAME, folder)
    assert con.execute(
        "SELECT acq_start, acq_start_source FROM measurement WHERE measurement_id = ?",
        (mid,),
    ).fetchone() == ("2024-03-15T22:11:39.758", "h5")
    # Again: replaced, not duplicated
    assert mass_cal.refresh(con, NAME, folder)
    assert con.execute("SELECT COUNT(*) FROM mass_cal_rt").fetchone() == (4,)
    # No calibration file any more: nothing written, the stored calibration is kept
    (folder / f"{NAME}.EI_mc.txt").unlink()
    assert not mass_cal.refresh(con, NAME, folder)
    assert con.execute("SELECT COUNT(*) FROM mass_cal_rt").fetchone() == (4,)
    assert not mass_cal.refresh(con, "not.imported", folder)
    con.close()

    data = EsperData(data_dir)
    points = data.mass_cal_points
    assert len(points) == 4 and points["ok"].all()
    old_point = points[points["measurement_id"] == old].iloc[0]
    assert old_point["time"] == pd.Timestamp("2024-03-15T22:11:01")
    assert points.loc[points["measurement_id"] == mid, "time"].iloc[0] == pd.Timestamp(
        "2024-03-15T22:14:59.758"
    )
    data.close()


def test_drift_over_time():
    """The shift is the m/z error with the median calibration: 0 for the median point, and
    the drift of the ions (later: positive) for the others."""
    cal = {"mode": 2, "p3": 0.5, "p4": np.nan, "p5": np.nan, "ok": True}
    df = pd.DataFrame(
        [cal | {"p1": 2829.0, "p2": p2} for p2 in (-3731.0, -3730.0, -3732.0)]
    )
    shift = drift(df, 69.0, "ppm")
    assert shift[0] == pytest.approx(0.0)
    assert shift[1] > 0 > shift[2]
    # A larger p2 moves the ions to a larger tof index: tof = p1 sqrt(m) + p2
    expected = (np.sqrt(69.0) + 1 / 2829.0) ** 2 / 69.0 - 1
    assert shift[1] == pytest.approx(expected * 1e6, rel=1e-6)
    tof = drift(df, 69.0, "tof")
    assert tof.tolist() == pytest.approx([0.0, 1.0, -1.0])


def test_calibrant_peaks_stored_and_drift_in_run(folder, tmp_path):
    """The calibrant peaks of the legacy file are stored; within the run, the drift of a
    calibrant is measured against the calibration of the first window."""
    write_h5(folder.parent / f"{NAME}.h5")
    cal = MassCalibration.from_file(folder / f"{NAME}.EI_mc.txt")
    mz = 68.995
    # Peaks exactly on the calibration, but the second 0.5 tof index late
    tof = cal.tof(cal.rt, mz) + np.array([0.0, 0.5, 0.0])
    pd.DataFrame(
        {"rt": [1000, 2000, 3000], "tof": tof, "area": 1.0, "sigma": 0.8, "alpha": 0.5,
         "ppm": [0.0, 9.9, 0.0], "fragment": "CF3", "mass_exact": mz, "rt_s": cal.rt}
    ).to_csv(folder / f"{NAME}.calibration_peaks.csv", index=False)  # fmt: skip

    data_dir = tmp_path / "data"
    con = db.connect(data_dir, create=True)
    mid = _measurement(con, NAME, folder)
    assert mass_cal.refresh(con, NAME, folder)
    con.close()

    data = EsperData(data_dir)
    run_cal, peaks = data.mass_cal_run(mid)
    data.close()
    assert peaks["calibrant"].tolist() == ["CF3"] * 3 and peaks["used"].all()
    np.testing.assert_allclose(run_cal.rt, cal.rt)
    measured = calibrant_drift(run_cal, peaks, "tof")
    expected = run_drift(run_cal, mz, peaks["rt_s"].to_numpy(), "tof") + [0, 0.5, 0]
    np.testing.assert_allclose(measured, expected, atol=1e-9)
    np.testing.assert_allclose(
        calibrant_residuals(run_cal, peaks, "tof"), [0, 0.5, 0], atol=1e-9
    )
    assert calibrant_residuals(run_cal, peaks, "ppm").tolist() == [0.0, 9.9, 0.0]
