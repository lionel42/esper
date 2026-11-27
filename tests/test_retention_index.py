"""Retention index references and the RIs of the compounds of a run (esper.retention_index)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esper import db
from esper.processing.config import SuspectScreeningConfig
from esper.processing.identification import CANDIDATE_COLUMNS, retention_indices
from esper.retention_index import (
    RetentionIndexConfig,
    default_references,
    find_references,
    read_reference_csv,
    read_references,
    rt_to_ri,
    store_references,
    target_ris,
    update_compound_ris,
    write_reference_csv,
)
from esper.target_index import TargetIndex

REFS = pd.DataFrame(
    {
        "name": ["Kr", "Xe-A", "Xe-B", "n-butane", "HFC-365mfc"],
        "ri": [25.0, 75.0, 175.0, 400.0, 644.0],
        "names": [None, "Xe; Xe-B", "Xe; Xe-A", "butane; C4H10-Butane", None],
        "occurrence": [None, 1, 2, None, None],
        "enabled": [1, 1, 1, 1, 0],
        "note": None,
    }
)


def _targets(ri=None) -> TargetIndex:
    names = ["Kr", "Xe-A", "Xe-B", "n-butane", "CFC-11", "benzene"]
    table = pd.DataFrame(
        {"rt_ref_s": [100.0, 120.0, 300.0, 500.0, 600.0, np.nan]},
        index=pd.Index(names, name="name"),
    )
    table["ri"] = target_ris(table["rt_ref_s"], REFS) if ri is None else ri
    ions = pd.DataFrame(
        columns=["name", "valid_from", "rank", "ion", "exact_mass", "mass"]
    )
    aliases = pd.DataFrame({"alias": ["R11"], "name": ["CFC-11"]})
    return TargetIndex(table, ions, aliases)


def test_the_seed_is_the_legacy_list(tmp_path):
    """The seed: rt_refs_v2.csv of the legacy pipeline, its aliases as names and its
    double calibrants as occurrences; the legacy format is read too."""
    seed = read_reference_csv(default_references())
    enabled = seed[seed["enabled"] == 1]
    assert enabled["name"].tolist()[:4] == ["Kr", "Xe-A", "Xe-B", "ethane"]
    assert seed.set_index("name").at["HFC-365mfc", "enabled"] == 0
    legacy = tmp_path / "rt_refs.csv"
    legacy.write_text(
        "# refs\ncompound,ri\nKr,25.0\nXe-A,75.0\n#HFC-365mfc,644\nn-butane,400\n"
    )
    refs = read_reference_csv(legacy).set_index("name")
    assert refs.loc["Xe-A", ["names", "occurrence"]].tolist() == ["Xe; Xe-B", 1]
    assert refs.at["n-butane", "names"] == "butane; C4H10-Butane"
    assert "HFC-365mfc" not in refs.index
    # written and read again
    write_reference_csv(seed, tmp_path / "out.csv")
    pd.testing.assert_frame_equal(read_reference_csv(tmp_path / "out.csv"), seed)


def test_target_ris():
    """Between the enabled references that are targets with an rt, extrapolated beyond."""
    rt = pd.Series({"Kr": 100.0, "Xe-A": 120.0, "x": 110.0, "y": 50.0, "z": 0.0})
    ri = target_ris(rt, REFS)
    assert ri["x"] == 50.0 and ri["y"] == -100.0 and np.isnan(ri["z"])
    assert target_ris(rt[["Kr", "x"]], REFS).isna().all()  # one reference only


def _run():
    """Compounds of a run with their best candidate: Kr, two Xe peaks (the first one
    named Xe-B by the library), a small butane before the large one, CFC-11."""
    compounds = pd.DataFrame(
        {
            "rt_mean_s": [90.0, 110.0, 290.0, 480.0, 495.0, 590.0, 700.0],
            "prominence_sum": [1e5, 1e4, 1e4, 50.0, 1e5, 2e4, 1e3],
        },
        index=pd.Index(range(7), name="compound_idx"),
    )
    best = pd.DataFrame(
        {
            "name": ["Kr", "Xe-B", "Xe", "butane", "C4H10-Butane", "R11", "benzene"],
            "score": [0.99, 0.97, 0.98, 0.99, 0.96, 0.95, 0.99],
        },
        index=compounds.index,
    )
    return compounds, best


def test_find_references():
    compounds, best = _run()
    found = find_references(compounds, best, _enabled(REFS), _targets())
    # Xe-A / Xe-B: the first Xe after the previous reference; n-butane: the largest
    assert found[["name", "compound_idx"]].values.tolist() == [
        ["Kr", 0],
        ["Xe-A", 1],
        ["Xe-B", 2],
        ["n-butane", 4],
    ]
    # a score not above the minimum: not a reference (Xe-B at 110 s: Xe-A is the next Xe,
    # no Xe-B after it; the small butane is the only one left)
    strict = find_references(
        compounds, best, _enabled(REFS), None, RetentionIndexConfig(min_score=0.97)
    )
    assert strict[["name", "compound_idx"]].values.tolist() == [
        ["Kr", 0],
        ["Xe-A", 2],
        ["n-butane", 3],
    ]
    ri = rt_to_ri(found)
    assert ri(np.array([110.0, 700.0])).tolist() == pytest.approx(
        [75.0, 400 + 205 * 225 / 205]
    )


def _enabled(refs: pd.DataFrame) -> pd.DataFrame:
    """The enabled references, as read_references gives them."""
    return refs[refs["enabled"] == 1].reset_index(drop=True)


def test_retention_indices_of_the_suspects():
    """The candidates get the RI of their target; the best combined score names the
    compound, identified within max_ri_diff; the references are named and identified."""
    compounds, best = _run()
    compounds = compounds.assign(compound_name=None, valid_candidate=0, is_ref_ri=0)
    rows = [
        (i, 1, name, f"nist_{name}", "nist", score, np.nan, np.nan, np.nan, 1)
        for i, (name, score) in enumerate(zip(best["name"], best["score"], strict=True))
    ]
    # CFC-11 (R11) is second by its spectrum, first with its RI
    rows[5] = (5, 1, "benzene", "nist_benzene", "nist", 0.97, np.nan, np.nan, np.nan, 1)
    rows.append((5, 2, "R11", "nist_R11", "nist", 0.96, np.nan, np.nan, np.nan, 0))
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    targets = _targets()
    out, candidates, found = retention_indices(
        compounds,
        candidates,
        _enabled(REFS),
        targets,
        suspect=SuspectScreeningConfig(min_score=0.9, max_ri_diff=10),
    )
    assert found["name"].tolist() == ["Kr", "Xe-A", "Xe-B", "n-butane"]
    assert out.loc[[0, 1, 2, 4], "compound_name"].tolist() == [
        "Kr",
        "Xe-A",
        "Xe-B",
        "n-butane",
    ]
    assert out.loc[[0, 1, 2, 4], "is_ref_ri"].tolist() == [1] * 4
    assert out.loc[[0, 1, 2, 4], "valid_candidate"].tolist() == [1] * 4
    # CFC-11 (R11) at 590 s: RI 504.27, its target 512.5 (rt 600 s): combined 0.96 x 0.918
    # = 0.881, benzene without RI 0.97 x 0.9 = 0.873
    assert out.at[5, "retention_index"] == pytest.approx(504.27)
    assert out.at[5, "compound_name"] == "R11"
    best5 = candidates[(candidates["compound_idx"] == 5) & (candidates["is_best"] == 1)]
    assert best5["candidate_name"].tolist() == ["R11"]
    assert abs(best5["ri_diff"].iloc[0]) < 10 and out.at[5, "valid_candidate"] == 1
    # benzene: no rt in the target list, no RI: not identified
    assert out.at[6, "compound_name"] == "benzene" and out.at[6, "valid_candidate"] == 0
    # without references: no RI, no compound identified by suspect screening
    out, _, found = retention_indices(
        compounds, candidates, _enabled(REFS).iloc[:0], targets
    )
    assert found.empty and out["retention_index"].isna().all()
    assert out["valid_candidate"].sum() == 0


def test_references_in_the_database(tmp_path):
    con = db.connect(tmp_path, create=True)
    with con:
        db.insert(
            con,
            "target",
            pd.DataFrame(
                {"name": ["Kr", "Xe-A", "CFC-11"], "rt_ref_s": [100.0, 120.0, 600.0]}
            ),
        )
    store_references(con, REFS)
    assert read_references(con)["name"].tolist() == ["Kr", "Xe-A", "Xe-B", "n-butane"]
    assert len(read_references(con, enabled=False)) == 5
    ri = dict(con.execute("SELECT name, ri FROM target").fetchall())
    assert ri == {
        "Kr": 25.0,
        "Xe-A": 75.0,
        "CFC-11": 1275.0,
    }  # Xe-B, n-butane: no targets
    # the RIs of the compounds of a run again from its references (e.g. after the fit)
    with con:
        con.execute(
            "INSERT INTO measurement (measurement_id, name, source_dir, source_mtime,"
            " source_size, imported_at) VALUES (1, 'run', '', 0, 0, '')"
        )
        db.insert(
            con,
            "compound",
            pd.DataFrame(
                {
                    "measurement_id": 1,
                    "compound_idx": [0, 1, 2],
                    "rt_mean_s": [90.0, 110.0, 130.0],
                }
            ),
        )
        db.insert(
            con,
            "measurement_ri_reference",
            pd.DataFrame(
                {
                    "measurement_id": 1,
                    "name": ["Kr", "Xe-A"],
                    "compound_idx": [0, 1],
                    "rt_s": [90.0, 110.0],
                    "score": 0.99,
                }
            ),
        )
        assert update_compound_ris(con, 1)
    ris = [
        r[0]
        for r in con.execute(
            "SELECT retention_index FROM compound ORDER BY compound_idx"
        )
    ]
    assert ris == [25.0, 75.0, 125.0]
    con.close()


def test_cli(tmp_path, capsys):
    from esper.cli.main import main

    db.connect(tmp_path, create=True).close()
    assert main(["ri-refs", "list", str(tmp_path)]) == 0
    assert "no references" in capsys.readouterr().out
    assert main(["ri-refs", "import", str(default_references()), str(tmp_path)]) == 0
    assert main(["ri-refs", "list", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "C4H10-Butane" in out and "HFC-365mfc" in out
    assert main(["ri-refs", "export", str(tmp_path), str(tmp_path / "refs.csv")]) == 0
    pd.testing.assert_frame_equal(
        read_reference_csv(tmp_path / "refs.csv"),
        read_reference_csv(default_references()),
    )
