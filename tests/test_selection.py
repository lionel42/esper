"""The selection keeps the detected compound (and its candidate) apart from the target."""

from __future__ import annotations

import pandas as pd
import pytest

from esper import db
from esper.gui.data import EsperData
from esper.gui.references import identification_candidate
from esper.gui.selection import Selection
from esper.libraries import combined_score

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


def test_compound_identified_as_another_target(selection):
    data = selection.data
    selection.select_compound(1, 1)
    fields = []
    selection.changed.connect(fields.append)
    data.set_identification(1, 1, "HFC-23")
    selection.identification_edited()
    assert fields == [{"candidate", "target", "identification"}]
    assert (selection.candidate, selection.target) == ("HFC-23", "HFC-23")
    compound = data.compounds(1).loc[1]
    assert (compound["compound_name"], compound["alpinac_name"]) == ("HFC-23", "Unknown X")
    assert compound["identified_at"] is not None
    assert data.targets_found(1) == {"CFC-12", "HFC-23"}
    assert list(data.results("HFC-23", valid_only=True)["compound_idx"]) == [1]
    assert "Unknown X" not in data.candidate_names(valid_only=False)


def test_compound_marked_not_identified(selection):
    data = selection.data
    data.set_identification(1, 0, None)
    compound = data.compounds(1).loc[0]
    assert compound["main_name"] is None and compound["valid_candidate"] == 0
    assert data.targets_found(1) == set()
    assert data.find_target(2, "CFC-12") == 0


def test_identification_reset_to_alpinac(selection):
    data = selection.data
    data.set_identification(1, 1, "HFC-23")
    data.reset_identification(1, 1)
    compound = data.compounds(1).loc[1]
    assert compound["compound_name"] == "Unknown X" and compound["identified_at"] is None


def test_identification_lost_when_measurement_imported_again(selection):
    data = selection.data
    data.set_identification(1, 1, "HFC-23")
    con = db.connect(data.data_dir)
    con.execute("DELETE FROM measurement WHERE measurement_id = 1")
    con.commit()
    assert con.execute("SELECT COUNT(*) FROM compound_identification").fetchone()[0] == 0
    con.close()


def test_identification_in_database_without_its_table(selection):
    data = selection.data
    con = db.connect(data.data_dir)
    con.execute("DROP TABLE compound_identification")
    con.commit()
    con.close()
    data = EsperData(data.data_dir)
    assert data.compounds(1).loc[1, "compound_name"] == "Unknown X"
    data.set_identification(1, 1, "HFC-23")
    assert data.compounds(1).loc[1, "compound_name"] == "HFC-23"
    data.close()


def test_compounds_selected_together(selection):
    selection.select_compound(1, 0)
    selection.select_reference(("nist", "CFC-12"))
    fields = []
    selection.changed.connect(fields.append)
    selection.select_compounds(1, [2, 0])
    # The main compound (lowest index) is unchanged, and so is the reference
    assert fields == [{"compound_idxs"}]
    assert (selection.compound_idx, selection.compound_idxs) == (0, (0, 2))
    assert selection.reference == ("nist", "CFC-12")
    assert sorted(selection.data.compound_peaks(1, selection.compound_idxs).index) == [0, 2]
    selection.select_compound(1, 1)
    assert selection.compound_idxs == (1,)
    selection.select_target("HFC-23")
    assert selection.compound_idxs == ()


def test_compounds_merged(selection):
    data = selection.data
    data.set_identification(1, 1, "HFC-23")
    assert data.merge_compounds(1, [2, 1]) == 1
    compounds = data.compounds(1)
    assert list(compounds.index) == [0, 1]
    assert (compounds.loc[1, "compound_name"], compounds.loc[1, "n_peaks"]) == ("HFC-23", 2)
    assert compounds.loc[1, "rt_mean_s"] == 450.0
    assert list(data.compound_peaks(1, 1).index) == [1, 2]


def test_identification_keeps_its_scores(selection):
    data = selection.data
    data.set_identification(1, 1, "HFC-23", "nist_HFC-23", 0.8, 0.72)
    compound = data.compounds(1).loc[1]
    assert (compound["best_spectrum"], compound["best_score"], compound["best_score_combined"]) == (
        "nist_HFC-23", 0.8, 0.72
    )
    result = data.results("HFC-23", valid_only=True).iloc[0]
    assert (result["best_score"], result["best_score_combined"]) == (0.8, 0.72)
    # Not identified: no scores, not alpinac's
    data.set_identification(1, 1, None)
    assert pd.isna(data.compounds(1).loc[1, "best_score"])


def test_identification_as_an_alpinac_candidate(selection):
    data = selection.data
    con = db.connect(data.data_dir)
    con.executemany(
        "INSERT INTO compound_candidate (measurement_id, compound_idx, rank, candidate_name, spectrum_ref, score,"
        " score_combined, is_best) VALUES (1, 1, ?, ?, ?, ?, ?, ?)",
        [(1, "Unknown X", "nist_Unknown X", 0.9, 0.81, 1), (2, "CFC12", "mygu_CFC12", 0.7, 0.63, 0)],
    )
    con.commit()
    con.close()
    data._candidates.cache_clear()
    data.compounds.cache_clear()
    assert data.compounds(1).loc[1, "best_score"] == 0.9
    # Through an alias of the target
    assert identification_candidate(data, 1, 1, "CFC-12") == ("mygu_CFC12", 0.7, 0.63)


def test_identification_table_without_scores(selection):
    data = selection.data
    con = db.connect(data.data_dir)
    con.executescript(
        """
        DROP TABLE compound_identification;
        CREATE TABLE compound_identification (
            measurement_id INTEGER, compound_idx INTEGER, name TEXT, edited_at TEXT NOT NULL,
            PRIMARY KEY (measurement_id, compound_idx)
        );
        INSERT INTO compound_identification VALUES (1, 0, 'HFC-23', '2026-01-01T00:00:00');
        """
    )
    con.close()
    data = EsperData(data.data_dir)
    assert pd.isna(data.compounds(1).loc[0, "best_score"])
    data.set_identification(1, 1, "HFC-23", "nist_HFC-23", 0.8, 0.72)
    assert data.compounds(1).loc[1, "best_score"] == 0.8
    data.close()


@pytest.mark.parametrize(
    "score, ri_diff, combined", [(0.8, 0.0, 0.8), (0.8, -25.0, 0.6), (0.8, 150.0, 0.0), (0.8, None, 0.72)]
)
def test_combined_score_as_alpinac(score, ri_diff, combined):
    assert combined_score(score, ri_diff) == pytest.approx(combined)


def test_search_with_identification(selection):
    data = selection.data
    data.set_identification(1, 1, "HFC-23")
    data.set_identification(1, 0, None)
    df = data.search([85], "rt", 0, 1000).set_index(["measurement_id", "compound_idx"])
    assert (df.loc[(1, 1), "main_name"], df.loc[(1, 1), "valid_candidate"]) == ("HFC-23", 1)
    assert (df.loc[(1, 0), "main_name"], df.loc[(1, 0), "valid_candidate"]) == (None, 0)
    assert df.loc[(2, 0), "main_name"] == "CFC-12"
