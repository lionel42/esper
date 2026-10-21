"""The selection keeps the detected compound (and its candidate) apart from the target."""

from __future__ import annotations

import pytest

from esper import db
from esper.gui.data import EsperData
from esper.gui.selection import Selection

# Detected compounds (compound_idx: candidate name, rt) of two measurements.
# "CFC12" is an alias of the target "CFC-12"; "HFC-23" is a target found nowhere.
COMPOUNDS = {
    1: {0: ("CFC12", 300.0), 1: ("Unknown X", 400.0), 2: (None, 500.0)},
    2: {0: ("CFC-12", 301.0)},
}


@pytest.fixture
def selection(tmp_path):
    con = db.connect(tmp_path, create=True)
    for mid, compounds in COMPOUNDS.items():
        con.execute(
            "INSERT INTO measurement (measurement_id, name, acq_datetime, source_dir, source_mtime,"
            " source_size, imported_at) VALUES (?, ?, ?, '', 0, 0, '')",
            (mid, f"24010{mid}.1200.air.1", f"2024-01-0{mid}T12:00:00"),
        )
        for idx, (name, rt) in compounds.items():
            con.execute(
                "INSERT INTO compound (measurement_id, compound_idx, rt_mean_s, valid_candidate, compound_name,"
                " prominence_sum) VALUES (?, ?, ?, 1, ?, 1)",
                (mid, idx, rt, name),
            )
            con.execute(
                "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass, rt_s, height, prominence)"
                " VALUES (?, ?, ?, 85, ?, 1, 1)",
                (mid, idx, idx, rt),
            )
    con.executemany("INSERT INTO target (name, ri) VALUES (?, NULL)", [("CFC-12",), ("HFC-23",)])
    con.executemany("INSERT INTO target_ion (name, rank, mass) VALUES (?, 0, ?)", [("CFC-12", 85), ("HFC-23", 69)])
    con.execute("INSERT INTO target_alias (alias, name) VALUES ('CFC12', 'CFC-12')")
    con.commit()
    con.close()
    data = EsperData(tmp_path)
    yield Selection(data)
    data.close()


def test_compound_with_target(selection):
    selection.select_compound(1, 0)
    # The alias is resolved
    assert (selection.compound_idx, selection.candidate, selection.target) == (0, "CFC-12", "CFC-12")
    assert selection.label == "CFC-12"


def test_compound_without_target(selection):
    selection.select_compound(1, 1)
    assert (selection.candidate, selection.target, selection.label) == ("Unknown X", None, "Unknown X")


def test_compound_without_candidate(selection):
    selection.select_compound(1, 2)
    assert (selection.compound_idx, selection.candidate, selection.label) == (2, None, None)


def test_target_not_found_is_not_a_candidate(selection):
    selection.select_measurement(1)
    selection.select_target("HFC-23")
    assert (selection.compound_idx, selection.candidate, selection.target) == (None, None, "HFC-23")
    assert selection.label == "HFC-23"
    assert selection.mass == 69


def test_target_followed_through_its_aliases(selection):
    selection.select_measurement(1)
    selection.select_target("CFC-12")
    assert (selection.compound_idx, selection.candidate) == (0, "CFC-12")
    selection.select_measurement(2)
    assert (selection.compound_idx, selection.candidate, selection.target) == (0, "CFC-12", "CFC-12")


def test_candidate_followed_across_measurements(selection):
    selection.select_compound(1, 1)
    selection.select_measurement(2)
    assert (selection.compound_idx, selection.candidate) == (None, "Unknown X")


def test_results_of_all_names_of_a_target(selection):
    data = selection.data
    results = data.results("CFC-12", valid_only=False)
    assert sorted(results["compound_name"]) == ["CFC-12", "CFC12"]


def test_aliases_are_not_shown(selection):
    data = selection.data
    assert data.candidate_names(valid_only=False).to_dict() == {"CFC-12": 2, "Unknown X": 1}
    assert list(data.compounds(1)["main_name"].fillna("-")) == ["CFC-12", "Unknown X", "-"]
    assert sorted(data.search([85], "rt", 0, 1000)["main_name"].fillna("-")) == ["-", "CFC-12", "CFC-12", "Unknown X"]
