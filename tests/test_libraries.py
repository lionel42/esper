"""Spectra written as JCAMP-DX are read back like the NIST library files."""

from __future__ import annotations

import numpy as np

from esper.libraries import JDX_MAX_Y, Spectrum, read_jdx, write_jdx


def test_write_jdx_round_trip(tmp_path):
    # 11 peaks: several lines of the peak table; m/z 60 rounds to 0 and m/z 70 is empty
    mz = [31, 35, 47, 50, 51, 60, 69, 70, 85, 87, 101]
    intensities = [
        404.0,
        20.0,
        3.5,
        76.0,
        9999.0,
        0.01,
        5848.0,
        0.0,
        211.0,
        158.0,
        12.0,
    ]
    file = tmp_path / "241007.0423.air.1_CFC-12.jdx"
    write_jdx(
        file,
        Spectrum.from_pairs(mz, intensities),
        title="CFC-12",
        comments={"MEASUREMENT": "241007.0423.air.1"},
    )

    text = file.read_text()
    assert "##TITLE=CFC-12\n" in text
    assert "##$MEASUREMENT=241007.0423.air.1\n" in text
    assert "##NPOINTS=9\n" in text
    assert text.endswith("##END=\n")

    spectrum = read_jdx(file)
    expected = [m for m, i in zip(mz, intensities) if round(JDX_MAX_Y * i / 9999.0) > 0]
    np.testing.assert_array_equal(spectrum.mz, expected)
    assert spectrum.intensities.max() == JDX_MAX_Y
    np.testing.assert_allclose(spectrum.intensities[spectrum.mz == 69], 5848)


def _library(folder, spectra):
    folder.mkdir(parents=True)
    for name, peaks in spectra.items():
        write_jdx(folder / f"{name}.jdx", Spectrum.from_pairs(*zip(*peaks)), title=name)


def test_selected_libraries_matched_by_name(tmp_path):
    """The libraries of the processing config (select_libraries), only the spectra of
    some names (targets_only), one candidate per name (distinct)."""
    import pytest

    from esper.libraries import LibraryCollection, Processing, select_libraries

    _library(
        tmp_path / "nist_db",
        {
            "CF4": [(69, 100), (50, 12)],
            "C2F6": [(69, 100), (119, 40)],
            "X": [(69, 99), (31, 5)],
        },
    )
    _library(tmp_path / "myriam_ei", {"CF4": [(69, 100), (50, 10)]})
    libraries = select_libraries(["nist_ei", "mygu"], tmp_path)
    assert [(lib.name, lib.path.name) for lib in libraries] == [
        ("nist_ei", "nist_db"),
        ("mygu", "myriam_ei"),
    ]
    with pytest.raises(ValueError, match="unknown libraries nist"):
        select_libraries(["nist"], tmp_path)
    query = Spectrum.from_pairs([69, 50], [100, 11])
    collection = LibraryCollection(libraries, Processing())
    names = [m.name for m in collection.best_matches(query, 3)]
    assert names[:2] == ["CF4", "CF4"]  # the same compound in both libraries
    names = [m.name for m in collection.best_matches(query, 3, distinct=True)]
    assert names == ["CF4", "X", "C2F6"]
    restricted = LibraryCollection(libraries, Processing(), {"CF4", "C2F6"})
    assert [m.name for m in restricted.best_matches(query, 3, distinct=True)] == [
        "CF4",
        "C2F6",
    ]
    # a search: all the spectra
    assert len(restricted.best_matches(query, None, search="x")) == 1
