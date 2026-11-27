"""Screening rules after the identification (esper.processing.rules)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esper import db
from esper.libraries import Library, LibraryCollection, Processing, Spectrum
from esper.processing.config import processing_config
from esper.processing.rules import drop_unfitted_single_peaks, merge, noise_merges


class _Libraries(LibraryCollection):
    """One library in memory: CF4 (69, 50, 31) and N2O (44, 30)."""

    def __init__(self):
        super().__init__([Library("nist_ei", None)], Processing(cutoff_mass=0))
        self._spectra["nist_ei"] = {
            "CF4": Spectrum.from_pairs([69, 50, 31], [100, 12, 5]),
            "N2O": Spectrum.from_pairs([44, 30], [100, 31]),
        }


def _run():
    """CF4 at 100 s split in three compounds: its 69 (0, identified), its 50 (1, in
    range: merged), a 44 (2, in range, not of CF4: not merged); its 31 out of range (3)."""
    peaks = pd.DataFrame(
        {
            "compound_idx": [0, 1, 2, 3],
            "mass": [69, 50, 44, 31],
            "height": [1000.0, 120.0, 300.0, 50.0],
            "rt_s": [100.0, 100.5, 100.2, 104.0],
            "prominence": [1000.0, 120.0, 300.0, 50.0],
            "is_max_of_cmp": [1, 1, 1, 1],
        }
    )
    compounds = pd.DataFrame(
        {
            "rt_mean_s": [100.0, 100.5, 100.2, 104.0],
            "start_mean_s": [98.0, 99.0, 99.0, 103.0],
            "stop_mean_s": [102.0, 102.0, 101.0, 105.0],
            "n_peaks": 1,
            "prominence_sum": [1000.0, 120.0, 300.0, 50.0],
            "rt_std_s": np.nan,
            "main_ion_mass": [69, 50, 44, 31],
            "valid_candidate": [1, 0, 0, 0],
        },
        index=pd.Index(range(4), name="compound_idx"),
    )
    candidates = pd.DataFrame(
        {"compound_idx": [0, 2], "is_best": 1, "spectrum_ref": ["nist_CF4", "nist_N2O"]}
    )
    return peaks, compounds, candidates


def test_noise_merge():
    peaks, compounds, candidates = _run()
    merges = noise_merges(peaks, compounds, candidates, _Libraries())
    assert merges == {0: [1]}
    peaks, compounds = merge(peaks, compounds, merges)
    assert compounds.index.tolist() == [0, 2, 3]
    assert peaks["compound_idx"].tolist() == [0, 0, 2, 3]
    assert compounds.loc[
        0, ["n_peaks", "prominence_sum", "main_ion_mass"]
    ].tolist() == [
        2,
        1120.0,
        69,
    ]
    assert compounds.at[0, "rt_mean_s"] == 100.25
    assert compounds.at[0, "start_mean_s"] == 98.0  # the range kept (legacy)


def test_saturated_compound_takes_all_its_pieces():
    """With at least saturation_pieces unidentified compounds in its range, an identified
    compound takes them all, also those that make its match worse (the 44 here)."""
    peaks, compounds, candidates = _run()
    libraries = _Libraries()
    assert noise_merges(peaks, compounds, candidates, libraries, 3) == {0: [1]}
    assert noise_merges(peaks, compounds, candidates, libraries, 2) == {0: [1, 2]}
    # unless it would no longer be identified: the rule of the match then
    assert noise_merges(peaks, compounds, candidates, libraries, 2, 0.99) == {0: [1]}


def test_rules_config():
    assert processing_config(None).rules.noise_merge
    config = processing_config({"rules": {"noise_merge": False}})
    assert not config.rules.noise_merge and config.rules.drop_unfitted_single_peaks


def test_drop_unfitted_single_peaks(tmp_path):
    """Compounds of one peak the fit rejected: deleted with their peak, not those of
    more peaks, fitted, identified or identified by hand; nothing before the fit."""
    con = db.connect(tmp_path, create=True)
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
                    "compound_idx": [0, 1, 2, 3, 4],
                    "rt_mean_s": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "valid_candidate": [0, 0, 0, 0, 1],
                }
            ),
        )
        db.insert(
            con,
            "peak",
            pd.DataFrame(
                {
                    "measurement_id": 1,
                    "peak_idx": [0, 1, 2, 3, 4, 5],
                    # 0: one peak, not fitted; 1: one peak, fitted; 2: two peaks, one
                    # fitted; 3: one peak not fitted, identified by hand; 4: one peak
                    # not fitted, identified
                    "compound_idx": [0, 1, 2, 2, 3, 4],
                    "mass": 44,
                    "fit_rt_s": [np.nan, 20.0, 30.0, np.nan, np.nan, np.nan],
                }
            ),
        )
        con.execute(
            "INSERT INTO compound_identification (measurement_id, compound_idx, name,"
            " edited_at) VALUES (1, 3, 'x', '2026-11-27')"
        )
    assert drop_unfitted_single_peaks(con, 1) == 0  # not fitted
    with con:
        con.execute("UPDATE measurement SET fit_version = 5")
    assert drop_unfitted_single_peaks(con, 1) == 1
    compounds = [r[0] for r in con.execute("SELECT compound_idx FROM compound")]
    assert compounds == [1, 2, 3, 4]
    assert con.execute("SELECT count(*) FROM peak").fetchone() == (5,)
    counts = con.execute("SELECT n_peaks, n_compounds FROM measurement").fetchone()
    assert counts == (5, 4)
    con.close()
