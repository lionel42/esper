"""Chromatographic fit of the peaks: split coeluting peaks and their compounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from peaxart.shapes import peak_shape as shape_of

from esper import db
from esper.fitting import (
    FIT_COLUMNS,
    FIT_VERSION,
    SOURCE_EXTRACTION,
    SOURCE_FIT,
    ExpectedTarget,
    compound_shapes,
    fit_changes,
    fit_measurement,
    fit_peaks,
    fitted_peak_table,
    match_compounds,
    merge_compounds,
    new_compounds,
    peak_baseline,
    peak_groups,
    peak_shape,
    to_fit,
)
from esper.processing.identification import TargetSearch
from esper.target_index import TargetIndex

DT = 1 / 6
SIGMA = 1.4
TAILING = 0.3
# The synthetic peaks are Pap-Papai peaks, fitted with that function (pap_papai_fit)
FIT_SHAPE = "pap_papai"
SHAPE = shape_of(FIT_SHAPE)
# Around 1800 s, where peaxart expects peaks of this width (ExtractionConfig.rt_sigma_at)
T0 = 1700.0
# Two coeluting compounds 5 s apart (FWHM 3.3 s), each alone on one mass. Closer, the
# default fit of peaxart (exponentially modified Gaussians, tail up to 3 sigma) can give
# the tail of A the peak of B (EMG peaks 4.5 s apart: B at 60 % of its area, the baseline
# at a quarter)
RT_A, RT_B = T0 + 100.0, T0 + 105.0
# Compounds of the extraction: A, B, the unresolved peak of both (C), and E, whose
# only peak is a coelution of two unknown compounds
A, B, C, E = 1, 2, 3, 4

# Extraction peaks: (peak_idx, compound_idx, mass, rt, start, stop) [s]
EXTRACTION = [
    (0, A, 85, RT_A, RT_A - 1.6, RT_A + 1.9),
    (1, A, 84, RT_A, RT_A - 1.6, RT_A + 1.9),
    (2, B, 70, RT_B, RT_B - 1.6, RT_B + 1.9),
    (3, B, 71, RT_B, RT_B - 1.6, RT_B + 1.9),
    # both: split by the fit. At the apex of the larger one, as the extraction finds
    # it (started between the two, the fit keeps one broad peak)
    (4, C, 43, RT_A, RT_A - 1.6, RT_B + 1.9),
    # two unknown compounds: split; the piece at its apex stays in E, the other in none
    (5, E, 31, T0 + 140.0, T0 + 138.4, T0 + 146.9),
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
    31: [(5e4, T0 + 140.0), (5e4, T0 + 145.0)],
}


def _amplitude(height: float, rt: float) -> tuple[float, float]:
    """(amplitude, location) of the shape of a peak of this apex height and rt."""
    apex_rt, apex_height = SHAPE.apex(1.0, 0.0, SIGMA, TAILING)
    return height / apex_height, rt - apex_rt


def peak(t: np.ndarray, height: float, rt: float) -> np.ndarray:
    """A peak of apex ``height`` at ``rt`` (s)."""
    amplitude, location = _amplitude(height, rt)
    return SHAPE.value(t, amplitude, location, SIGMA, TAILING)
@pytest.fixture(autouse=True, scope="module")
def pap_papai_fit():
    """The synthetic peaks are Pap-Papai peaks (``SHAPE``): fitted with that function (the default,
    the EMG, is chosen on real data; these tests are about the peak table, not the peak function).
    Module scope: before the module-scoped ``table``."""
    import peaxart as px

    from esper import fitting

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitting, "FIT_CONFIG", px.ChromatogramFitConfig(peak_shape=FIT_SHAPE))
        yield


def chromatograms(noise: float = 30.0) -> pd.DataFrame:
    t = T0 + np.arange(0, 200, DT)
    rng = np.random.default_rng(0)
    columns = {
        str(mass): 500.0
        + sum(peak(t, h, rt) for h, rt in peaks)
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
    """Area of a peak of apex ``height``."""
    return SHAPE.area(_amplitude(height, 0.0)[0], SIGMA, TAILING)


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
    on_31 = table[table["mass"] == 31].sort_values("fit_rt_s")
    assert len(on_31) == 2
    # E has no shape (its only peak was split): the piece keeping the row of its peak
    # (at its apex) stays in E, it does not lose its peak;
    # the other matches another compound
    assert on_31.index[0] == 5
    assert on_31["compound_idx"].iloc[0] == E
    assert pd.isna(on_31["compound_idx"].iloc[1])


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


def test_prominence_of_the_pieces_is_scipys(table):
    on_43 = table[table["mass"] == 43].sort_values("fit_rt_s")
    a, b = on_43.iloc[0], on_43.iloc[1]
    # Both are local maxima of the fitted signal: scipy's prominences. A, the higher,
    # from the baseline; B from the valley between them, less than its fitted height
    # (a shoulder, without a local maximum, gets its height alone:
    # peaxart.deconvolution.fitted_signal_peaks)
    assert a["prominence"] == pytest.approx(a["height"] - a["fit_baseline"], rel=0.01)
    assert 0 < b["prominence"] < b["fit_height"]
    # heights: the fitted signal at the apex, above zero as find_peaks
    assert (on_43["height"] > on_43["fit_baseline"] + on_43["fit_height"]).all()


def test_largest_peak_of_the_changed_compounds(table):
    largest = table[table["is_max_of_cmp"] == 1]
    # A and B gained a piece of m/z 43, E kept a piece of its split peak: their largest
    # peak is recomputed (C untouched: no peak left)
    assert set(largest["compound_idx"].dropna()) == {A, B, E}
    for compound_idx in (A, B, E):
        peaks = table[table["compound_idx"] == compound_idx]
        assert peaks.loc[peaks["is_max_of_cmp"] == 1].index.tolist() == [
            peaks["prominence"].idxmax()
        ]


def test_fitted_peak_just_outside_the_crossings_takes_the_row():
    # The half maximum crossings of the extraction are narrow on the tail side: the
    # fitted apex is 0.2 s beyond them, it is still the same peak
    peaks = extraction_peaks().loc[[0]]
    # (the fit moves a peak by up to 3 scans, 0.5 s, from its start)
    peaks["rt_s"] = RT_A - 0.4
    peaks["stop_s"] = RT_A - 0.2
    table = fitted_peak_table(peaks, fit_peaks(chromatograms(), peaks, workers=1))
    on_85 = table[table["mass"] == 85].sort_values("fit_rt_s")
    assert on_85.index[0] == 0
    assert on_85["source"].iloc[0] == SOURCE_EXTRACTION
    # the fit may move a peak by 3 scans (0.5 s) from its start (RT_A - 0.6 here): it stops at that
    # bound, 0.1 s before the true apex (exactly at the edge of a 0.1 s tolerance: up to the last bits)
    assert on_85["fit_rt_s"].iloc[0] == pytest.approx(RT_A, abs=0.15)
    near = on_85[(on_85["fit_rt_s"] - RT_A).abs() < 3]
    assert len(near) == 1
    # the other peaks of m/z 85 (not given as extracted here) are found by the added seeds: new rows
    others = on_85.drop(near.index)
    assert (others["source"] == SOURCE_FIT).all()


def test_peak_shape_of_the_stored_fits(table):
    t = T0 + np.arange(80, 120, DT)
    p = table.loc[0]  # m/z 85 at RT_A
    assert p["fit_shape"] == FIT_SHAPE
    shape = peak_shape(t, p)
    # its apex: fit_rt_s and fit_height
    assert t[shape.argmax()] == pytest.approx(p["fit_rt_s"], abs=DT)
    assert shape.max() == pytest.approx(p["fit_height"], rel=1e-3)
    # the fits of FIT_VERSION 2: Pap-Papai, without the shape columns
    from peaxart.shapes import chromato

    legacy = p.drop(["fit_shape", "fit_shape_rt_s", "fit_shape_height"])
    np.testing.assert_allclose(
        peak_shape(t, legacy),
        chromato(t, p["fit_height"], p["fit_rt_s"], p["fit_sigma_s"], p["fit_tailing"]),
    )
    # ... or with them NULL (read from the database)
    null = p.copy()
    null[["fit_shape", "fit_shape_rt_s", "fit_shape_height"]] = [None, np.nan, np.nan]
    np.testing.assert_allclose(peak_shape(t, null), peak_shape(t, legacy))


def test_peak_groups_sorted_by_their_start():
    # A broad peak whose window starts before the narrow ones eluting before it
    peaks = pd.DataFrame(
        {
            "mass": 1,
            "rt_s": [100.0, 105.0, 110.0],
            "start_s": [99.7, 104.7, 100.0],
            "stop_s": [100.3, 105.3, 110.3],
        }
    )
    ((group, lo, hi),) = peak_groups(peaks)
    assert group.index.tolist() == [0, 1, 2]
    assert lo == pytest.approx(110.0 - 6 * 10.0)


def _measurement(tmp_path):
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
            "INSERT INTO compound (measurement_id, compound_idx, n_peaks, prominence_sum,"
            " rt_mean_s, main_ion_mass) VALUES (1, ?, 1, 1, 0, 1)",
            (idx,),
        )
    peaks = extraction_peaks().reset_index().assign(measurement_id=1)
    db.insert(con, "peak", peaks)
    con.commit()
    return con


def test_fit_measurement(tmp_path):
    con = _measurement(tmp_path)
    assert [mid for mid, _ in to_fit(con)] == [1]
    fit_measurement(con, 1, tmp_path, workers=1)
    stored = pd.read_sql("SELECT * FROM peak", con, index_col="peak_idx")
    # one more piece on m/z 43 and 31, the peak not extracted on m/z 85
    assert len(stored) == len(EXTRACTION) + 3
    assert set(stored["source"]) == {SOURCE_EXTRACTION, SOURCE_FIT}
    on_43 = stored[stored["mass"] == 43].sort_values("rt_s")
    assert on_43["compound_idx"].tolist() == [A, B]
    compounds = pd.read_sql(
        "SELECT * FROM compound", con, index_col="compound_idx"
    ).sort_index()
    # E keeps the piece of its split peak at its apex; the pieces matching no compound
    # (the other piece of E's peak, the peak not extracted on m/z 85, at other rts) are
    # compounds of their own, made by the fit, after the last one
    made = compounds[compounds["source"] == SOURCE_FIT]
    assert list(made.index) == [E + 1, E + 2]
    assert compounds["n_peaks"].to_dict() == {
        A: 3, B: 3, C: 0, E: 1, E + 1: 1, E + 2: 1
    }  # fmt: skip
    assert compounds.loc[[A, B, C, E], "source"].isna().all()
    assert made["compound_name"].isna().all() and (made["valid_candidate"] == 0).all()
    made_peaks = stored[stored["compound_idx"].isin(made.index)].sort_values("rt_s")
    assert made_peaks["mass"].tolist() == [85, 31]
    assert made_peaks["rt_s"].tolist() == pytest.approx(
        [NOT_EXTRACTED, T0 + 145.0], abs=0.2
    )
    assert stored["compound_idx"].notna().all()  # every peak in a compound
    # the main ion of A, B, E and the new ones: their largest peak; C without peaks
    # keeps its own
    assert compounds["main_ion_mass"].to_dict() == {
        **{
            idx: stored.loc[
                stored[stored["compound_idx"] == idx]["prominence"].idxmax(), "mass"
            ]
            for idx in (A, B, E, *made.index)
        },
        C: 1,
    }
    assert con.execute(
        "SELECT n_peaks, n_compounds, fit_version FROM measurement"
    ).fetchone() == (len(stored), len(compounds), FIT_VERSION)
    assert to_fit(con) == []
    assert set(stored["fit_shape"].dropna()) == {FIT_SHAPE}
    # the extraction values replaced by the fit are kept: the split peaks (4, 5) and
    # the peaks whose is_max_of_cmp changed, and the compounds changed
    kept = pd.read_sql("SELECT * FROM extraction_peak", con, index_col="peak_idx")
    assert {4, 5} <= set(kept.index)
    assert kept.loc[4, "compound_idx"] == C
    assert kept.loc[4, "start_s"] == pytest.approx(T0 + 98.4)
    kept_compounds = pd.read_sql("SELECT * FROM extraction_compound", con)
    assert set(kept_compounds["compound_idx"]) == {A, B, C, E}
    assert (kept_compounds["n_peaks"] == 1).all()

    # Fitted again (forced): from the extraction, same result
    assert [mid for mid, _ in to_fit(con, force=True)] == [1]
    fit_measurement(con, 1, tmp_path, workers=1)
    again = pd.read_sql("SELECT * FROM peak", con, index_col="peak_idx")
    columns = ["compound_idx", "mass", "source", "rt_s", "is_max_of_cmp", *FIT_COLUMNS]
    pd.testing.assert_frame_equal(again[columns], stored[columns])
    # the compounds made by the fit made again, not added to those of the first fit
    pd.testing.assert_frame_equal(
        pd.read_sql("SELECT * FROM compound", con, index_col="compound_idx")
        .sort_index(),
        compounds,
    )
    assert (
        pd.read_sql("SELECT * FROM extraction_peak", con, index_col="peak_idx")
        .sort_index()
        .equals(kept.sort_index())
    )
    con.close()


def test_merge_then_fit_again(tmp_path):
    con = _measurement(tmp_path)
    fit_measurement(con, 1, tmp_path, workers=1)
    # C: a peak split by the fit, kept as extracted (extraction_peak 4)
    assert con.execute(
        "SELECT count(*) FROM extraction_peak WHERE compound_idx = ?", (C,)
    ).fetchone() == (1,)
    n_peaks = con.execute("SELECT count(*) FROM peak").fetchone()[0]
    assert merge_compounds(con, 1, [C, B], retention_index=lambda rt: None) == B

    def compound_idxs(table: str) -> set:
        rows = con.execute(f"SELECT DISTINCT compound_idx FROM {table}")
        return {idx for (idx,) in rows}

    # Nothing left of C, also as extracted
    for table in ("compound", "peak", "extraction_peak", "extraction_compound"):
        assert C not in compound_idxs(table)
    stored = pd.read_sql("SELECT * FROM peak", con, index_col="peak_idx")
    of_b = stored[stored["compound_idx"] == B]
    row = con.execute(
        "SELECT n_peaks, prominence_sum, main_ion_mass FROM compound"
        " WHERE compound_idx = ?",
        (B,),
    ).fetchone()
    largest = of_b.loc[of_b["prominence"].idxmax(), "mass"]
    assert row == (len(of_b), pytest.approx(of_b["prominence"].sum()), largest)
    assert of_b["is_max_of_cmp"].sum() == 1
    # As extracted: the peaks of B (70, 71) and C (43)
    (extracted,) = con.execute(
        "SELECT n_peaks FROM extraction_compound WHERE compound_idx = ?", (B,)
    ).fetchone()
    assert extracted == 3
    n_compounds = con.execute("SELECT count(*) FROM compound").fetchone()[0]
    assert con.execute("SELECT n_peaks, n_compounds FROM measurement").fetchone() == (
        n_peaks,
        n_compounds,
    )
    # Fitted again: from the merged extraction, C's peaks in B
    fit_measurement(con, 1, tmp_path, workers=1)
    again = pd.read_sql("SELECT * FROM peak", con, index_col="peak_idx")
    assert C not in set(again["compound_idx"])
    assert (again["compound_idx"] == B).sum() >= len(of_b) - 1
    con.close()


def test_fitted_without_the_extraction_must_be_imported_again(tmp_path):
    con = _measurement(tmp_path)
    fit_measurement(con, 1, tmp_path, workers=1)
    con.execute("UPDATE measurement SET fit_version = 1")
    con.execute("DELETE FROM extraction_peak")
    con.commit()
    with pytest.raises(ValueError, match="import it again"):
        fit_measurement(con, 1, tmp_path, workers=1)
    con.close()


def _extraction_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Peaks and compound summaries as extracted, for ``fit_changes``.

    Compound 0: peaks 0 (m/z 43, main) and 1 (m/z 57); 1: peak 2; 2: peak 3; 3: peak 4.
    """
    peaks = pd.DataFrame(
        {
            "compound_idx": [0, 0, 1, 2, 3],
            "mass": [43, 57, 69, 85, 101],
            "rt_s": [100.0, 100.2, 200.0, 300.0, 400.0],
            "prominence": [10.0, 5.0, 8.0, 3.0, 7.0],
            "is_max_of_cmp": [1, 0, 1, 1, 1],
        },
        index=pd.Index(range(5), name="peak_idx"),
    )
    compounds = pd.DataFrame(
        {
            "n_peaks": [2, 1, 1, 1],
            "prominence_sum": [15.0, 8.0, 3.0, 7.0],
            "rt_mean_s": [100.1, 200.0, 300.0, 400.0],
            "rt_std_s": [0.14, np.nan, np.nan, np.nan],
            "main_ion_mass": [43, 69, 85, 101],
        },
        index=pd.Index(range(4), name="compound_idx"),
    )
    return peaks, compounds


def test_fit_changes():
    peaks, compounds = _extraction_tables()
    table = peaks.assign(source=SOURCE_EXTRACTION)
    table["compound_idx"] = table["compound_idx"].astype("Int64")
    # Compound 0: its m/z 57 peak becomes the largest (main ion only)
    table.loc[1, "prominence"] = 20.0
    table.loc[[0, 1], "is_max_of_cmp"] = [0, 1]
    # Compound 2: its only peak is given to no compound (changed)
    table.loc[3, "compound_idx"] = pd.NA
    # Compound 3: changed by an earlier fit, as extracted now (restored)
    changes = fit_changes(peaks, compounds, table, earlier=[3])

    assert sorted(changes.extraction_peaks.index) == [0, 1, 3]
    assert sorted(changes.extraction_compounds.index) == [0, 2]
    pd.testing.assert_frame_equal(changes.extraction_compounds, compounds.loc[[0, 2]])
    summaries = changes.summaries.sort_index()
    assert list(summaries.index) == [0, 2, 3]
    # Main ion only: the other columns as extracted
    assert summaries.loc[0, "main_ion_mass"] == 57
    assert summaries.loc[0, "n_peaks"] == 2
    assert summaries.loc[0, "prominence_sum"] == 15.0
    # Without peaks: none counted, rt and main ion kept
    assert summaries.loc[2, "n_peaks"] == 0
    assert summaries.loc[2, "prominence_sum"] == 0.0
    assert summaries.loc[2, "rt_mean_s"] == 300.0
    assert summaries.loc[2, "main_ion_mass"] == 85
    # Restored: as extracted
    pd.testing.assert_series_equal(
        summaries.loc[3], compounds.loc[3], check_dtype=False
    )


def test_fit_changes_nothing_changed():
    peaks, compounds = _extraction_tables()
    table = peaks.assign(source=SOURCE_EXTRACTION)
    changes = fit_changes(peaks, compounds, table)
    assert changes.extraction_peaks.empty
    assert changes.extraction_compounds.empty
    assert changes.summaries.empty


def test_peak_baseline():
    """The background under a fitted peak: its value at the apex and its slope (constant
    for rows fitted before the slope was stored)."""
    t = np.array([10.0, 20.0, 30.0])
    row = pd.Series({"fit_rt_s": 20.0, "fit_baseline": 100.0, "fit_baseline_slope": -2.0})
    np.testing.assert_allclose(peak_baseline(t, row), [120.0, 100.0, 80.0])
    old = pd.Series({"fit_rt_s": 20.0, "fit_baseline": 100.0, "fit_baseline_slope": np.nan})
    np.testing.assert_allclose(peak_baseline(t, old), 100.0)


def test_target_keeps_the_piece_nearest_its_expected_rt():
    # E's only peak (m/z 31, apex at 140 s) is split into 140 s and 145 s: named by
    # target search with an expected rt of 145 s, E keeps the 145 s piece (whatever the
    # extracted apex), the other is left to the compounds it matches (none here)
    peaks = extraction_peaks()
    fitted = fit_peaks(chromatograms(), peaks, workers=1)
    target = ExpectedTarget(T0 + 145.0, frozenset({31}))
    table = fitted_peak_table(peaks, fitted, targets={E: target})
    on_31 = table[table["mass"] == 31].sort_values("fit_rt_s")
    assert pd.isna(on_31["compound_idx"].iloc[0])
    assert on_31["compound_idx"].iloc[1] == E
    assert on_31.index[1] == 5  # takes the row of the extraction peak
    # without: the piece at the extracted apex
    table = fitted_peak_table(peaks, fitted)
    assert table.at[5, "compound_idx"] == E
    assert table.at[5, "fit_rt_s"] == pytest.approx(T0 + 140.0, abs=0.2)
    # m/z 31 not one of the target's ions: judged as without target
    other_ions = ExpectedTarget(T0 + 145.0, frozenset({85}))
    table = fitted_peak_table(peaks, fitted, targets={E: other_ions})
    assert table.at[5, "compound_idx"] == E
    assert table.at[5, "fit_rt_s"] == pytest.approx(T0 + 140.0, abs=0.2)


def test_new_compounds_one_peak_per_mass_and_no_chain():
    # pieces matching no compound (sigma 1 s: tolerance 0.3 s from the first of a group)
    rows = [
        (31, 100.0),
        (43, 100.2),  # with the 31 one
        (31, 100.25),  # m/z 31 again: a compound of its own (its first piece)
        (85, 100.5),  # 0.25 s from that first piece: with it
        (69, 100.6),  # 0.1 s from the previous one but 0.35 s from the first: its own
        (85, 110.0),
    ]
    table = pd.DataFrame(
        {
            "mass": [m for m, _ in rows],
            "fit_rt_s": [rt for _, rt in rows],
            "fit_sigma_s": 1.0,
            "source": SOURCE_FIT,
            "compound_idx": pd.array([pd.NA] * len(rows), dtype="Int64"),
        }
    )
    assert new_compounds(table, 7) == [7, 8, 9, 10]
    assert table["compound_idx"].tolist() == [7, 7, 8, 8, 9, 10]


def test_compounds_made_by_the_fit_named_by_target_search(tmp_path):
    con = _measurement(tmp_path)
    # Identified by target search: E is the target "E-target"; "Other" is a target
    # on m/z 31 at 145 s, which the detection did not see (one peak with E)
    con.execute("UPDATE measurement SET identification = 'target'")
    con.execute("UPDATE compound SET compound_name = 'E-target' WHERE compound_idx = ?", (E,))
    con.executemany(
        "INSERT INTO target (name, rt_ref_s) VALUES (?, ?)",
        [("E-target", T0 + 140.0), ("Other", T0 + 145.0)],
    )
    con.executemany(
        "INSERT INTO target_ion (name, rank, mass) VALUES (?, 0, 31)",
        [("E-target",), ("Other",)],
    )
    con.commit()
    search = TargetSearch(TargetIndex.from_db(con))
    fit_measurement(con, 1, tmp_path, workers=1, target_search=search)
    compounds = pd.read_sql("SELECT * FROM compound", con, index_col="compound_idx")
    peaks = pd.read_sql("SELECT * FROM peak", con, index_col="peak_idx")
    on_31 = peaks[peaks["mass"] == 31].sort_values("rt_s")
    names = compounds.loc[on_31["compound_idx"], "compound_name"].tolist()
    # E keeps its piece (nearest its expected rt); the other piece is a new compound,
    # named "Other" (E-target, found already, is not looked for again)
    assert names == ["E-target", "Other"]
    other = compounds[compounds["compound_name"] == "Other"]
    assert other["source"].tolist() == [SOURCE_FIT]
    assert other["valid_candidate"].tolist() == [1]
    con.close()
