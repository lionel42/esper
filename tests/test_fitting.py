"""Chromatographic fit of the peaks: split coeluting peaks and their compounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from peaxart.shapes import chromato, chromato_area

from esper import db
from esper.fitting import (
    SOURCE_EXTRACTION,
    SOURCE_FIT,
    compound_shapes,
    fit_measurement,
    fit_peaks,
    fitted_peak_table,
    match_compounds,
)

DT = 1 / 6
SIGMA = 1.4
TAILING = 0.3
# Around 1800 s, where peaxart expects peaks of this width (ExtractionConfig.rt_sigma_at)
T0 = 1700.0
# Two coeluting compounds 3.2 s apart, each alone on one mass (2- and 3-methylhexane
# are 2.9 s apart: the fit of these synthetic peaks then finds a biased local minimum)
RT_A, RT_B = T0 + 100.0, T0 + 103.2
# Compounds of the extraction: A, B, the unresolved peak of both (C), and E, whose
# only peak is a coelution of two unknown compounds
A, B, C, E = 1, 2, 3, 4

# Extraction peaks: (peak_idx, compound_idx, mass, rt, start, stop) [s]
EXTRACTION = [
    (0, A, 85, RT_A, T0 + 98.4, T0 + 101.9),
    (1, A, 84, RT_A, T0 + 98.4, T0 + 101.9),
    (2, B, 70, RT_B, T0 + 101.6, T0 + 105.1),
    (3, B, 71, RT_B, T0 + 101.6, T0 + 105.1),
    # both: split by the fit. Started between the two (from some other starts, the
    # fit of peaxart ends 0.3 s off, with the areas split -26/+32 %)
    (4, C, 43, T0 + 101.6, T0 + 98.4, T0 + 105.1),
    # two unknown compounds: split, no compound
    (5, E, 31, T0 + 141.5, T0 + 138.4, T0 + 144.8),
]
# m/z 85 has a peak not extracted at this rt
NOT_EXTRACTED = T0 + 112.0
# Signal: mass -> [(height, rt)]
SIGNAL = {
    85: [(1e5, RT_A), (4e4, NOT_EXTRACTED)],
    84: [(5e4, RT_A)],
    70: [(8e4, RT_B)],
    71: [(6e4, RT_B)],
    43: [(1e5, RT_A), (8e4, RT_B)],
    31: [(5e4, T0 + 140.0), (5e4, T0 + 142.9)],
}


def chromatograms(noise: float = 30.0) -> pd.DataFrame:
    t = T0 + np.arange(0, 200, DT)
    rng = np.random.default_rng(0)
    columns = {
        str(mass): 500.0
        + sum(chromato(t, h, rt, SIGMA, TAILING) for h, rt in peaks)
        + rng.normal(0, noise, len(t))
        for mass, peaks in SIGNAL.items()
    }
    return pd.DataFrame(columns, index=pd.Index(t, name="rt"))


def extraction_peaks() -> pd.DataFrame:
    df = pd.DataFrame(
        EXTRACTION,
        columns=["peak_idx", "compound_idx", "mass", "rt_s", "start_s", "stop_s"],
    ).set_index("peak_idx")
    df["height"] = df["prominence"] = 1e5
    df["left_base_scan"] = ((df["rt_s"] - T0 - 10) / DT).astype(int)
    df["right_base_scan"] = ((df["rt_s"] - T0 + 10) / DT).astype(int)
    df["width_s"] = df["stop_s"] - df["start_s"]
    df["is_max_of_cmp"] = 0
    return df


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    peaks = extraction_peaks()
    return fitted_peak_table(peaks, fit_peaks(chromatograms(), peaks, workers=1))


def area(height: float) -> float:
    return chromato_area(height, SIGMA, TAILING)


def test_peaks_fitted_as_extracted(table):
    clean = table.loc[[0, 1, 2, 3]]
    assert (clean["source"] == SOURCE_EXTRACTION).all()
    assert clean["compound_idx"].tolist() == [A, A, B, B]
    assert clean["fit_rt_s"].tolist() == pytest.approx(
        [RT_A, RT_A, RT_B, RT_B], abs=0.1
    )
    assert clean["fit_area"].tolist() == pytest.approx(
        [area(1e5), area(5e4), area(8e4), area(6e4)], rel=0.03
    )
    # the extraction columns are kept
    assert clean["start_s"].notna().all()


def test_coelution_is_split_between_the_compounds(table):
    on_43 = table[table["mass"] == 43].sort_values("fit_rt_s")
    assert len(on_43) == 2
    assert (on_43["source"] == SOURCE_FIT).all()
    assert on_43["compound_idx"].tolist() == [A, B]
    assert on_43["rt_s"].tolist() == pytest.approx([RT_A, RT_B], abs=0.15)
    assert on_43["fit_area"].tolist() == pytest.approx([area(1e5), area(8e4)], rel=0.05)
    # the row of the extraction peak takes the nearest piece, its ips are gone
    assert 4 in on_43.index
    assert on_43["start_s"].isna().all()
    assert (table["compound_idx"] == C).sum() == 0


def test_no_compound_matches(table):
    on_31 = table[table["mass"] == 31]
    assert len(on_31) == 2
    assert on_31["compound_idx"].isna().all()


def test_peak_not_extracted_is_added(table):
    on_85 = table[table["mass"] == 85].sort_values("fit_rt_s")
    assert on_85["source"].tolist() == [SOURCE_EXTRACTION, SOURCE_FIT]
    assert on_85.index[0] == 0  # the extracted peak keeps its row
    assert on_85["fit_rt_s"].iloc[1] == pytest.approx(NOT_EXTRACTED, abs=0.1)
    assert pd.isna(on_85["compound_idx"].iloc[1])
    assert on_85.index[1] > max(p[0] for p in EXTRACTION)


def test_match_compounds_thresholds():
    shapes = pd.DataFrame(
        {"rt_s": [100.0, 103.0], "sigma_s": [1.5, 1.5], "n_peaks": [3, 3]},
        index=[A, B],
    )
    peaks = pd.DataFrame(
        {
            "fit_rt_s": [100.1, 102.8, 101.5, 100.0],
            "fit_sigma_s": [1.6, 1.4, 1.5, 4.0],
            "fit_rt_u_s": [0.02, 0.02, 0.02, 0.02],
        }
    )
    # the last two: between the compounds, and three times too broad
    matched = match_compounds(peaks, shapes)
    assert matched.iloc[:2].tolist() == [A, B]
    assert matched.iloc[2:].isna().all()
    # an uncertain rt widens the tolerance
    peaks["fit_rt_u_s"] = 0.6
    assert match_compounds(peaks, shapes).iloc[2] in (A, B)


def test_compound_shapes_weighted_by_height():
    reference = pd.DataFrame(
        {
            "compound_idx": [A, A, A],
            "fit_rt_s": [100.0, 100.1, 105.0],
            "fit_sigma_s": [1.5, 1.5, 3.0],
            "fit_height": [1e5, 1e5, 1e3],
        }
    )
    shapes = compound_shapes(reference)
    assert shapes.loc[A, "rt_s"] == pytest.approx(100.0, abs=0.11)
    assert shapes.loc[A, "sigma_s"] == 1.5


def test_fit_measurement(tmp_path):
    con = db.connect(tmp_path, create=True)
    db.chromatograms_dir(tmp_path).mkdir()
    chromatograms().to_parquet(db.chromatograms_dir(tmp_path) / "m.parquet")
    con.execute(
        "INSERT INTO measurement (measurement_id, name, n_peaks, chromatogram_file,"
        " source_dir, source_mtime, source_size, imported_at)"
        " VALUES (1, 'm', ?, 'chromatograms/m.parquet', '', 0, 0, '')",
        (len(EXTRACTION),),
    )
    for idx in (A, B, C, E):
        con.execute(
            "INSERT INTO compound (measurement_id, compound_idx, n_peaks, prominence_sum, rt_mean_s)"
            " VALUES (1, ?, 1, 1, 0)",
            (idx,),
        )
    peaks = extraction_peaks().reset_index().assign(measurement_id=1)
    db.insert(con, "peak", peaks)
    con.commit()

    fit_measurement(con, 1, tmp_path, workers=1)
    stored = pd.read_sql("SELECT * FROM peak", con, index_col="peak_idx")
    # one more piece on m/z 43 and 31, the peak not extracted on m/z 85
    assert len(stored) == len(EXTRACTION) + 3
    assert set(stored["source"]) == {SOURCE_EXTRACTION, SOURCE_FIT}
    on_43 = stored[stored["mass"] == 43].sort_values("rt_s")
    assert on_43["compound_idx"].tolist() == [A, B]
    n_peaks = dict(con.execute("SELECT compound_idx, n_peaks FROM compound"))
    assert n_peaks == {A: 3, B: 3, C: 0, E: 0}
    assert con.execute("SELECT n_peaks, fit_version FROM measurement").fetchone() == (
        len(stored),
        1,
    )
    with pytest.raises(ValueError, match="already fitted"):
        fit_measurement(con, 1, tmp_path, workers=1)
    con.close()
