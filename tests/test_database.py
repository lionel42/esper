"""Integration of the measurements of a small synthetic data directory, through the database."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from synthetic import T, gaussian, gaussian_area, noisy, peaks_of

from esper import db
from esper.cli.main import main
from esper.gui.data import EsperData
from esper.integration import INTEGRATION_VERSION, integrate_measurements
from esper.noise import NOISE_COLUMNS

# Two compounds: A (masses 44 and 45) and B (mass 44, overlapping A on 44)
PEAKS = pd.concat([
    peaks_of((300, 2.0, 1000), (306, 2.0, 500), mass=44),
    peaks_of((300, 2.0, 400), mass=45),
], ignore_index=True).assign(compound_idx=[0, 1, 0])
NAMES = {0: "A", 1: "B"}
MEASUREMENTS = ["240101.1200.air.1", "240201.1200.air.1"]


def _chromatograms(rng) -> pd.DataFrame:
    signals = {"44": gaussian(300, 2.0, 1000) + gaussian(306, 2.0, 500) + 50, "45": gaussian(300, 2.0, 400) + 20}
    return pd.DataFrame({m: noisy(s, rng) for m, s in signals.items()}, index=pd.Index(T, name="rt"))


def _fill(con: sqlite3.Connection, data_dir) -> None:
    """Two measurements of the same compounds, with their chromatogram files."""
    rng = np.random.default_rng(0)
    db.chromatograms_dir(data_dir).mkdir(parents=True, exist_ok=True)
    for mid, name in enumerate(MEASUREMENTS, start=1):
        file = f"{db.CHROMATOGRAMS_DIRNAME}/{name}.parquet"
        _chromatograms(rng).to_parquet(data_dir / file)
        con.execute(
            "INSERT INTO measurement (measurement_id, name, acq_datetime, sample_type, chromatogram_file,"
            " source_dir, source_mtime, source_size, imported_at) VALUES (?, ?, ?, 'air', ?, '', 0, 0, '')",
            (mid, name, pd.Timestamp(f"20{name[:2]}-{name[2:4]}-{name[4:6]} 12:00").isoformat(), file),
        )
        for idx, cname in NAMES.items():
            con.execute(
                "INSERT INTO compound (measurement_id, compound_idx, rt_mean_s, valid_candidate, compound_name)"
                " VALUES (?, ?, ?, 1, ?)",
                (mid, idx, float(PEAKS[PEAKS.compound_idx == idx].rt_s.mean()), cname),
            )
        for peak_idx, p in PEAKS.iterrows():
            con.execute(
                "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass, rt_s, height, prominence,"
                " start_s, stop_s, left_base_scan, right_base_scan) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?)",
                (mid, peak_idx, int(p.compound_idx), int(p.mass), p.rt_s, p.start_s, p.stop_s,
                 int(p.left_base_scan), int(p.right_base_scan)),
            )
    con.commit()


def _data_dir(tmp_path):
    con = db.connect(tmp_path, create=True)
    _fill(con, tmp_path)
    return con


def _peaks(con) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM peak ORDER BY measurement_id, peak_idx", con)


def test_integrate_all_then_skip_then_force(tmp_path):
    con = _data_dir(tmp_path)
    summary = integrate_measurements(con, tmp_path)
    assert summary.integrated == MEASUREMENTS and not summary.failed

    peaks = _peaks(con)
    assert peaks["area"].notna().all() and (peaks["area_unc"] > 0).all()
    # Areas within a few uncertainties of the true ones
    true = np.tile([gaussian_area(2.0, 1000), gaussian_area(2.0, 500), gaussian_area(2.0, 400)], 2)
    assert (np.abs(peaks["area"] - true) < 5 * peaks["area_unc"] + 0.05 * true).all()
    measurements = pd.read_sql("SELECT * FROM measurement", con)
    assert (measurements["integration_version"] == INTEGRATION_VERSION).all()
    assert measurements[list(NOISE_COLUMNS.values())].notna().all().all()
    compounds = pd.read_sql("SELECT * FROM compound WHERE compound_idx = 0", con)
    assert (compounds["q1_mass"] == 44).all() and (compounds["q2_mass"] == 45).all()
    assert np.allclose(compounds["area_sum"], compounds["q1_area"] + compounds["q2_area"])

    summary = integrate_measurements(con, tmp_path)
    assert summary.integrated == [] and summary.skipped_integrated == MEASUREMENTS
    summary = integrate_measurements(con, tmp_path, force=True)
    assert summary.integrated == MEASUREMENTS
    assert np.allclose(_peaks(con)["area"], peaks["area"])  # same results


def test_partial_integration_does_not_mark_the_measurement(tmp_path):
    con = _data_dir(tmp_path)
    summary = integrate_measurements(con, tmp_path, measurement_ids=[1], compound_name="B")
    assert summary.integrated == [MEASUREMENTS[0]]
    peaks = _peaks(con)
    integrated = peaks["area"].notna()
    assert (peaks.loc[integrated, "measurement_id"] == 1).all()
    assert (peaks.loc[integrated, "compound_idx"] == 1).all()
    assert pd.read_sql("SELECT integration_version FROM measurement", con)["integration_version"].isna().all()

    # rt range: compound A only (mean rt 300 s)
    integrate_measurements(con, tmp_path, measurement_ids=[2], rt_range=(295, 302))
    peaks = _peaks(con)
    assert peaks.loc[peaks.measurement_id == 2, "area"].notna().tolist() == [True, False, True]

    # The complete integration still does both
    assert integrate_measurements(con, tmp_path).integrated == MEASUREMENTS


def test_command_line_since(tmp_path):
    con = _data_dir(tmp_path)
    con.close()
    assert main(["integrate", str(tmp_path), "--since", "2024-01-15"]) == 0
    con = db.connect(tmp_path, readonly=True)
    versions = pd.read_sql("SELECT name, integration_version FROM measurement ORDER BY name", con)
    assert versions["integration_version"].isna().tolist() == [True, False]


def test_old_database_is_upgraded(tmp_path):
    # Schema version 2: no integration columns
    con = sqlite3.connect(db.db_path(tmp_path))
    con.executescript(db.SCHEMA)
    con.execute("PRAGMA user_version = 2")
    _fill(con, tmp_path)
    con.close()

    # The GUI reads it (read-only, no area columns)
    data = EsperData(tmp_path)
    assert data.results("A", valid_only=False)["area_sum"].isna().all()
    data.close()

    con = db.connect(tmp_path, create=True)
    assert con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert integrate_measurements(con, tmp_path).integrated == MEASUREMENTS
    con.close()
    data = EsperData(tmp_path)
    assert data.results("A", valid_only=False)["area_sum_unc"].notna().all()


def test_gui_reads_a_database_with_some_integration_columns(tmp_path):
    # Integrated by a first version, without the uncertainties: they read as missing
    con = sqlite3.connect(db.db_path(tmp_path))
    con.executescript(db.SCHEMA)
    for table, columns in db.ADDED_COLUMNS.items():
        for name, sql_type in columns:
            if not name.endswith(db.UNC_SUFFIX) and not name.startswith("noise_"):
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    _fill(con, tmp_path)
    con.execute("UPDATE compound SET area_sum = 1")
    con.commit()
    con.close()

    data = EsperData(tmp_path)
    results = data.results("A", valid_only=False)
    assert (results["area_sum"] == 1).all() and results["area_sum_unc"].isna().all()
    # Noise parameters not stored: estimated from the chromatograms
    assert data.noise_parameters(1).factor > 1
