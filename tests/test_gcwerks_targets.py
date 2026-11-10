"""The targets of a GCwerks directory (esper.sources.gcwerks_targets) and their versions
in the GUI (TargetIndex)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esper import db
from esper.gui.target_index import TargetIndex
from esper.sources.gcwerks_targets import read_targets, unit_mass
from esper.targets import TargetError, store_targets

PEAK_LIST = """\
CFC-12
HFC-32
SF6
#1-3butdiene
HFC-41
"""
# yymmdd versions: what changes from one to the next
VERSIONS = {
    # half masses of 2007-2009; CFC12: the openghg name of CFC-12
    "080101": """\
CFC12     1213.0  16.0  85.5  87.5  0.0
CH2F2     1070.0   8.0  51.5  33.5  0.0
O2         202.0 200.0  32.5
SF6       1060.0  12.0 127.5  89.5 108.5
""",
    # same as before for all but the rt of SF6; CFC-12 twice (the first line kept)
    "090703": """\
CFC-12    1213.0  16.0  85.0  87.0  0.0  10.0 35.0 10.0 25.0 0.0 0.0
CFC-12    1300.0  16.0  85.0  87.0  0.0
CH2F2     1070.0   8.0  51.0  33.0  0.0
O2         202.0 200.0  32.0
SF6       1062.0  12.0 127.0  89.0 108.0
""",
    # SF6 quantified on 89; HFC-32 no longer identified
    "220210.1200": """\
CFC-12    1220.6  16.0  85.0  87.0 101.0
SF6       1071.2  12.0  89.0 127.0 108.0
""",
}
ALIASES = {"CH2F2": "HFC-32", "O2": None}


def _gcdir(tmp_path, versions=VERSIONS, peak_list=PEAK_LIST):
    root = tmp_path / "gcdir"
    peakid = root / "integrator" / "channel0" / "peakid"
    peakid.mkdir(parents=True)
    (root / "config").mkdir()
    (root / "config" / "peak.list").write_text(peak_list)
    for name, text in versions.items():
        (peakid / name).write_text(text)
    return root


def _rows(df, name, columns):
    return [
        tuple(r) for r in df.loc[df["name"] == name, columns].itertuples(index=False)
    ]


def test_unit_mass():
    assert [unit_mass(m) for m in (32.5, 67.5, 67.0, 85.0)] == [32, 67, 67, 85]


def test_targets(tmp_path):
    targets = read_targets(_gcdir(tmp_path), ALIASES)
    targets.check()
    assert targets.table.index.tolist() == ["CFC-12", "HFC-32", "SF6", "HFC-41"]
    # the latest rt, for display; none for a target not identified
    assert targets.table["rt_ref_s"].tolist()[:3] == [1220.6, 1070.0, 1071.2]
    assert np.isnan(targets.table.at["HFC-41", "rt_ref_s"])

    rts = targets.rts
    cols = ["valid_from", "rt_s", "window_s"]
    # 090703: no change of CFC-12 (its second line ignored): no row
    assert _rows(rts, "CFC-12", cols) == [
        ("2008-01-01T00:00:00", 1213.0, 16.0),
        ("2022-02-10T12:00:00", 1220.6, 16.0),
    ]
    assert _rows(rts, "SF6", ["valid_from", "rt_s"]) == [
        ("2008-01-01T00:00:00", 1060.0),
        ("2009-07-03T00:00:00", 1062.0),
        ("2022-02-10T12:00:00", 1071.2),
    ]
    # HFC-32 (alias CH2F2) no longer identified from 2022-02-10
    hfc32 = _rows(rts, "HFC-32", cols)
    assert hfc32[0] == ("2008-01-01T00:00:00", 1070.0, 8.0)
    assert hfc32[1][0] == "2022-02-10T12:00:00" and np.isnan(hfc32[1][1])
    assert "O2" not in set(rts["name"])

    ions = targets.ions
    # the half masses are the unit masses below: same ions in 2009, no new rows
    assert _rows(ions, "SF6", ["valid_from", "rank", "mass"]) == [
        ("2008-01-01T00:00:00", 0, 127),
        ("2008-01-01T00:00:00", 1, 89),
        ("2008-01-01T00:00:00", 2, 108),
        ("2022-02-10T12:00:00", 0, 89),
        ("2022-02-10T12:00:00", 1, 127),
        ("2022-02-10T12:00:00", 2, 108),
    ]
    assert _rows(ions, "CFC-12", ["valid_from", "mass"])[-1] == (
        "2022-02-10T12:00:00",
        101,
    )


def test_unresolved_names_raise(tmp_path):
    root = _gcdir(tmp_path)
    with pytest.raises(TargetError, match="2 peakid names.*CH2F2, O2"):
        read_targets(root)
    with pytest.raises(TargetError, match="O2"):
        read_targets(root, {"CH2F2": "HFC-32"})
    with pytest.raises(TargetError, match="not peak.list names: CH2F2: HFC32"):
        read_targets(root, {"CH2F2": "HFC32", "O2": None})


def test_no_peak_list(tmp_path):
    root = _gcdir(tmp_path)
    (root / "config" / "peak.list").unlink()
    with pytest.raises(TargetError, match="No peak list"):
        read_targets(root, ALIASES)


def test_stored_and_found_by_time(tmp_path):
    targets = read_targets(_gcdir(tmp_path), ALIASES)
    con = db.connect(tmp_path / "esper", create=True)
    store_targets(con, targets, "gcdir")
    rts = pd.read_sql("SELECT * FROM target_rt", con)
    ions = pd.read_sql("SELECT * FROM target_ion", con)
    table = pd.read_sql("SELECT * FROM target", con).set_index("name")
    aliases = pd.DataFrame(columns=["alias", "name"])
    con.close()
    index = TargetIndex(table, ions, aliases, rts)

    assert index.rt_at("SF6", "2010-05-01") == (1062.0, 12.0)
    assert index.rt_at("SF6", "2026-01-01") == (1071.2, 12.0)
    assert index.rt_at("SF6", "2007-06-01") is None  # before the first version
    assert index.rt_at("HFC-32", "2015-01-01") == (1070.0, 8.0)
    assert index.rt_at("HFC-32", "2023-01-01") is None  # no longer identified
    assert index.masses_at("SF6", "2015-01-01") == [127, 89, 108]
    assert index.masses_at("SF6", "2023-01-01") == [89, 127, 108]
    # without time, and in the table: the latest version
    assert index.masses_at("SF6") == [89, 127, 108] == index.target_masses("SF6")
    assert index.has_rts("SF6") and not index.has_rts("HFC-41")
    assert (
        index.rt_at("HFC-41", "2023-01-01") is None and index.masses_at("HFC-41") == []
    )


def test_target_lists_are_not_versioned():
    """Targets of a target list: ions without valid_from, no rts."""
    table = pd.DataFrame({"ri": [500.0]}, index=pd.Index(["CFC-12"], name="name"))
    ions = pd.DataFrame(
        {
            "name": ["CFC-12"] * 2,
            "valid_from": [None, None],
            "rank": [0, 1],
            "mass": [85, 87],
        }
    )
    index = TargetIndex(table, ions, pd.DataFrame(columns=["alias", "name"]))
    assert (
        index.masses_at("CFC-12", "2020-01-01")
        == [85, 87]
        == index.target_masses("CFC-12")
    )
    assert not index.has_rts("CFC-12")


def test_upload_adds_the_unresolved_names_to_the_esper_yaml(tmp_path):
    from esper.upload import upload

    root = _gcdir(tmp_path)
    (root / "config" / "gcwerks.conf").write_text("")
    data_dir = tmp_path / "esper"
    with pytest.raises(TargetError, match="--add-aliases"):
        upload(root, data_dir, instrument="gcwerks")
    # one name mapped by hand: only the other one added, after it
    config = data_dir / "esper.yaml"
    config.write_text(
        "instrument: gcwerks\n# names of the peakid\naliases:\n  CH2F2: HFC-32\n\n"
        "database: esper.sqlite\n"
    )
    summary = upload(root, data_dir, add_aliases=True)
    assert summary.added_aliases == ["O2"]
    text = config.read_text()
    assert text.startswith(
        "instrument: gcwerks\n# names of the peakid\naliases:\n  CH2F2: HFC-32\n  O2: null"
    )
    assert text.endswith("\n\ndatabase: esper.sqlite\n")
    assert db.read_config(data_dir)["aliases"] == ALIASES
    con = db.connect(data_dir)
    assert con.execute("SELECT count(*) FROM target").fetchone() == (4,)
    con.close()
    # resolved now: nothing more to add
    assert upload(root, data_dir, add_aliases=True).added_aliases == []


def test_add_config_entries(tmp_path):
    db.add_config_entries(tmp_path, "aliases", {"1,2-DCA": None, "O2": "Kr"}, "check")
    assert (tmp_path / "esper.yaml").read_text() == (
        "aliases:\n  '1,2-DCA': null  # check\n  O2: Kr  # check\n"
    )
    db.add_config_entries(tmp_path, "aliases", {"N2": None})
    assert db.read_config(tmp_path)["aliases"] == {
        "1,2-DCA": None,
        "O2": "Kr",
        "N2": None,
    }
    with pytest.raises(ValueError, match="already in aliases: O2"):
        db.add_config_entries(tmp_path, "aliases", {"O2": None})
    (tmp_path / "esper.yaml").write_text("aliases: {O2: null}\n")
    with pytest.raises(ValueError, match="block mapping"):
        db.add_config_entries(tmp_path, "aliases", {"N2": None})
