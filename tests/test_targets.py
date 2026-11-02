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
        [("CFC-12", 0, "CF2Cl+", 84.966, 85, None)], columns=ION_COLUMNS
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
