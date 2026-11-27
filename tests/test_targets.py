"""Ions of the targets taken from their reference spectrum."""

from __future__ import annotations

import pandas as pd

from esper.libraries import Library, Spectrum, write_jdx
from esper.targets import ION_COLUMNS, TARGET_COLUMNS, Targets, spectrum_ions


def _targets() -> Targets:
    table = pd.DataFrame(
        index=pd.Index(["CFC-12", "HFC-134a", "unknown"], name="name"),
        columns=TARGET_COLUMNS,
    )
    ions = pd.DataFrame(
        [("CFC-12", None, 0, "CF2Cl+", 84.966, 85, None)], columns=ION_COLUMNS
    )
    return Targets(table, ions, aliases={"R134a": "HFC-134a"})


def test_ions_from_the_reference_spectrum(tmp_path):
    (tmp_path / "low").mkdir()
    (tmp_path / "high").mkdir()
    spectrum = Spectrum.from_pairs([33, 69, 83, 102], [400.0, 100.0, 999.0, 50.0])
    write_jdx(tmp_path / "high" / "R134a.jdx", spectrum, "R134a")
    write_jdx(tmp_path / "low" / "HFC-134a.jdx", Spectrum.from_pairs([1], [1.0]), "x")
    libraries = [
        Library("nist_ei", tmp_path / "high"),
        Library("off", tmp_path / "low", enabled=False),
    ]
    ions = spectrum_ions(_targets(), libraries)
    # CFC-12 has ions in the target list, unknown has no spectrum; HFC-134a: under its
    # alias, the disabled library ignored, largest first
    assert set(ions["name"]) == {"HFC-134a"}
    assert ions["mass"].tolist() == [83, 33, 69, 102]
    assert ions["rank"].tolist() == [0, 1, 2, 3]
    assert set(ions["spectrum_ref"]) == {"nist_R134a"}


def _substances(path):
    """A substances Excel file as /shared/data/substances.xlsx (sheet "all")."""
    from esper.targets import EXCEL_ION_COLUMNS, EXCEL_NAME_COLUMN, EXCEL_RT_COLUMN

    rows = [
        # name, rt, formula, CAS, pubchem, alias 1, alias 2, Q0 ion, Q0 mass
        ("Kr", 100.0, "Kr", None, None, None, None, "Kr+", 83.911),
        ("CFC-12", 200.0, "CCl2F2 ", "75-71-8", 6391, "R12", None, "CF2Cl+", 84.966),
        ("benzene", 300.0, "C6H6", "71-43-2", 241.0, None, "Benzol", "C6H6+", 78.047),
        (None, 250.0, None, None, None, None, None, None, None),  # no name: skipped
        ("CFC-12", 999.0, "x", None, None, "ignored", None, None, None),  # again: ignored
        ("HFC-1234yf", 0.0, "C3H2F4", None, None, None, None, None, None),  # rt unknown
        ("octane", 400.0, "C8H18", None, None, None, None, None, None),
    ]
    data = {
        EXCEL_NAME_COLUMN: [r[0] for r in rows],
        EXCEL_RT_COLUMN: [r[1] for r in rows],
        "Formula ": [r[2] for r in rows],
        "CAS": [r[3] for r in rows],
        "Chemical name": None,
        "Pubchem CID": [r[4] for r in rows],
        "InChI": None,
        "SMILES": None,
        "Class": "test",
        "Alias 1": [r[5] for r in rows],
        "Alias 2": [r[6] for r in rows],
    }
    for ion, mass in EXCEL_ION_COLUMNS:
        if ion:
            data[ion] = None
        data[mass] = None
    data["Q0 Ion"] = [r[7] for r in rows]
    data["Q0 Mass"] = [r[8] for r in rows]
    pd.DataFrame(data).to_excel(path, sheet_name="all", index=False)
    return path


def test_read_substances(tmp_path):
    from esper.targets import read_substances

    df, aliases = read_substances(_substances(tmp_path / "substances.xlsx"))
    assert list(df.index) == ["Kr", "CFC-12", "benzene", "HFC-1234yf", "octane"]
    assert aliases == {"R12": "CFC-12", "Benzol": "benzene"}
    assert df.at["CFC-12", "formula"] == "CCl2F2 " and df.at["Kr", "cas"] == ""
    assert df["pubchem_cid"].tolist() == [-1, 6391, 241, -1, -1]
    assert df.at["CFC-12", "rt"] == 200.0 and df.at["HFC-1234yf", "rt"] == 0.0


def test_load_excel_retention_index(tmp_path):
    from esper.targets import load_excel

    refs = tmp_path / "refs.csv"
    refs.write_text("# references\ncompound,ri\nKr,25.0\n#benzene,727\nCFC-12,350.0\n")
    targets = load_excel(_substances(tmp_path / "substances.xlsx"), refs)
    table = targets.table
    assert table.at["CFC-12", "formula"] == "CCl2F2"
    assert pd.isna(table.at["HFC-1234yf", "rt_ref_s"]) and pd.isna(table.at["HFC-1234yf", "ri"])
    # between and beyond the references (Kr 100 s: 25, CFC-12 200 s: 350): extrapolated
    assert table.loc[["Kr", "CFC-12", "benzene", "octane"], "ri"].tolist() == [25, 350, 675, 1000]
    assert targets.ions.set_index("name").loc["CFC-12", "mass"] == 85
    assert targets.aliases == {"R12": "CFC-12", "Benzol": "benzene"}
