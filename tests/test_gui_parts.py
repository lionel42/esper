"""The parts of the GUI data layer that work without a database."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esper.gui.chromatogram_store import ChromatogramStore
from esper.gui.target_index import TargetIndex


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
