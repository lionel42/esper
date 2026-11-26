"""The parts of the GUI data layer that work without a database."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esper.gui.chromatogram_store import ChromatogramStore
from esper.target_index import TargetIndex


def _index() -> TargetIndex:
    targets = pd.DataFrame(
        {"ri": [500.0, 600.0]}, index=pd.Index(["CFC-12", "HFC-23"], name="name")
    )
    ions = pd.DataFrame(
        {
            "name": ["CFC-12", "CFC-12", "CFC-12"],
            "rank": [0, 1, 2],
            "mass": [85, 87, 85],
        }
    )
    aliases = pd.DataFrame({"alias": ["CFC12"], "name": ["CFC-12"]})
    return TargetIndex(targets, ions, aliases)


def test_target_index_names():
    index = _index()
    assert index.target_of("CFC12") == "CFC-12"
    assert index.target_of("Unknown") is None
    assert index.main_name("CFC12") == "CFC-12"
    assert index.main_name("Unknown") == "Unknown"
    assert index.main_name(np.nan) is None
    assert sorted(index.names_of("CFC-12")) == ["CFC-12", "CFC12"]
    assert index.names_of("Unknown") == ["Unknown"]


def test_target_index_masses():
    index = _index()
    # Main ion first, without repetition; none for a target without ions
    assert index.target_masses("CFC-12") == [85, 87]
    assert index.target_masses("HFC-23") == []
    assert index.target_masses("Unknown") == []
    assert index.target_main_ion("CFC12") == 85
    assert index.target_main_ion("HFC-23") is None


def test_target_index_empty():
    index = TargetIndex(
        pd.DataFrame(columns=["name", "ri"]).set_index("name"),
        pd.DataFrame(columns=["name", "rank", "mass"]),
        pd.DataFrame(columns=["alias", "name"]),
    )
    assert index.table.empty
    assert index.main_name("CFC-12") == "CFC-12"


def test_chromatogram_store(tmp_path):
    frame = pd.DataFrame(
        {"85": [1.0, 2.0], "tic": [3.0, 4.0]}, index=pd.Index([0.0, 0.2], name="rt")
    )
    frame.to_parquet(tmp_path / "1.parquet")
    paths = {1: tmp_path / "1.parquet", 2: None, 3: tmp_path / "missing.parquet"}
    store = ChromatogramStore(paths.get, cache_size=1)
    assert store.columns(1) == {"85", "tic", "rt"}
    assert store.columns(2) == frozenset() and store.columns(3) == frozenset()
    series = store.series(1, "85")
    assert series.tolist() == [1.0, 2.0] and series.index.tolist() == [0.0, 0.2]
    assert store.series(1, "85") is series  # cached
    assert store.series(1, "69") is None
    assert store.series(2, "85") is None
    store.series(1, "tic")  # cache of one: the m/z 85 is read again
    assert store.series(1, "85") is not series
    pd.testing.assert_frame_equal(store.frame(1), frame)
    assert store.frame(3) is None
    store.clear()
    assert store.series(1, "tic").tolist() == [3.0, 4.0]


def test_main_ion_of_a_measured_mass(tmp_path):
    # GCwerks scan runs: measured masses, e.g. 69.4
    from esper import db
    from esper.gui.data import EsperData

    con = db.connect(tmp_path, create=True)
    con.execute(
        "INSERT INTO measurement (measurement_id, name, source_dir, source_mtime,"
        " source_size, imported_at) VALUES (1, '250101.0000.air.1', '', 0, 0, '')"
    )
    con.execute(
        "INSERT INTO compound (measurement_id, compound_idx, compound_name)"
        " VALUES (1, 0, 'Unknown')"
    )
    con.executemany(
        "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass, rt_s,"
        " prominence) VALUES (1, ?, 0, ?, 100, ?)",
        [(0, 69.4, 10.0), (1, 70.0, 5.0)],
    )
    con.commit()
    con.close()
    data = EsperData(tmp_path)
    try:
        # the unit mass of its most prominent peak
        assert data.main_ions(1).tolist() == [69]
    finally:
        data.close()


def test_ips_bounds_of_a_mass_measured_in_a_window(tmp_path):
    """The bounds of the ips integration are times: also in a chromatogram measured in
    a window only (GCwerks SIM: NaN outside, dropped by the GUI)."""
    from esper import db
    from esper.gui.data import EsperData
    from esper.integration import integration_bounds, peak_area

    t = np.arange(0, 100, 0.5)
    y = np.where(t >= 50, 10 + 1000 * np.exp(-0.5 * ((t - 70) / 1.0) ** 2), np.nan)
    frame = pd.DataFrame({"85": y}, index=pd.Index(t, name="rt"))
    db.chromatograms_dir(tmp_path).mkdir(parents=True)
    frame.to_parquet(db.chromatograms_dir(tmp_path) / "run.parquet")
    con = db.connect(tmp_path, create=True)
    con.execute(
        "INSERT INTO measurement (measurement_id, name, source_dir, source_mtime,"
        " source_size, imported_at, chromatogram_file) VALUES"
        " (1, '250101.0000.air.1', '', 0, 0, '', 'chromatograms/run.parquet')"
    )
    con.execute("INSERT INTO compound (measurement_id, compound_idx) VALUES (1, 0)")
    con.execute(
        "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass, rt_s, start_s,"
        " stop_s, left_base_s, right_base_s, prominence)"
        " VALUES (1, 0, 0, 85, 70, 68.8, 71.2, 60, 80, 1000)"
    )
    con.commit()
    con.close()
    data = EsperData(tmp_path)
    try:
        start, stop = integration_bounds(data.peaks(1))
        # 3 half widths on each side of the apex, within the bases
        assert start[0] == pytest.approx(66.4) and stop[0] == pytest.approx(73.6)
        series = data.chromatogram(1, 85)
        area = peak_area(series.index.to_numpy(), series.to_numpy(), start[0], stop[0])
        assert area == pytest.approx(1000 * np.sqrt(2 * np.pi), rel=0.01)
    finally:
        data.close()
