"""Upload of .alpinac folders (instrument tofwerk), with a small synthetic pipeline.nc."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from esper import db
from esper.alpinac import _COMPOUND_COLUMNS, _PEAK_COLUMNS, _RANKS
from esper.sources import declared_instrument
from esper.upload import upload

NAME = "241007.0423.air.1"


def _pipeline(path):
    n_peaks, n_compounds = 3, 2
    peaks = {var: ("peak", np.arange(n_peaks, dtype=float)) for var in _PEAK_COLUMNS}
    peaks["compound_id"] = ("peak", np.array([0, 0, 1]))
    peaks["mass_of_peak"] = ("peak", np.array([85, 87, 43]))
    peaks["is_max_of_cmp"] = ("peak", np.array([1, 0, 1]))
    compounds = {
        var: ("compound", np.arange(n_compounds, dtype=float))
        for var in _COMPOUND_COLUMNS
    }
    compounds["compound_name"] = ("compound", np.array(["CFC-12", ""], dtype=object))
    for col in ("is_ref_ri", "valid_candidate"):
        compounds[col] = ("compound", np.array([1, 0]))
    candidates = {
        "bc_spectra": ("compound", np.array(["nist_CFC-12", ""], dtype=object))
    }
    for prefix in _RANKS.values():
        candidates[f"{prefix}_spectra"] = (
            "compound",
            np.array(["nist_CFC-12", ""], dtype=object),
        )
        candidates[f"{prefix}_candidate"] = (
            "compound",
            np.array(["CFC-12", ""], dtype=object),
        )
        for var in ("score", "ri", "ri_diff", "score_combined"):
            candidates[f"{prefix}_{var}"] = ("compound", np.array([0.9, np.nan]))
    ds = xr.Dataset(
        {**peaks, **compounds, **candidates},
        coords={"peak": np.arange(n_peaks), "compound": np.arange(n_compounds)},
        attrs={"rt_start": 10.0, "rt_stopp": 100.0, "cleaning": "none"},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_upload_alpinac_folders(tmp_path):
    runs = tmp_path / "runs"
    folder = runs / f"{NAME}.alpinac"
    folder.mkdir(parents=True)
    _pipeline(folder / "pipeline.nc")
    pd.DataFrame(
        {"43": [1.0, 2.0], "tic": [3.0, 4.0]}, index=pd.Index([0.0, 0.2], name="rt")
    ).to_parquet(folder / f"{NAME}.unit_mass_chromatos.parquet")
    (runs / "241007.0533.air.2.alpinac").mkdir()  # not complete: no pipeline
    data_dir = tmp_path / "esper"

    summary = upload(runs, data_dir, fit=False, instrument="tofwerk")
    assert summary.imported == [NAME]
    assert summary.skipped_incomplete == ["241007.0533.air.2"]
    assert declared_instrument(data_dir) == "tofwerk"
    con = db.connect(data_dir)
    row = con.execute(
        "SELECT instrument, peak_source, sample_type, sample_number, rt_start_s, rt_stop_s,"
        " cleaning, n_peaks, n_compounds, chromatogram_file, source_dir FROM measurement"
    ).fetchone()
    assert row == (
        "tofwerk",
        "alpinac",
        "air",
        1,
        10.0,
        100.0,
        "none",
        3,
        2,
        f"chromatograms/{NAME}.parquet",
        str(folder.resolve()),
    )
    assert con.execute("SELECT count(*) FROM peak").fetchone() == (3,)
    assert con.execute(
        "SELECT compound_name FROM compound WHERE compound_idx = 0"
    ).fetchone() == ("CFC-12",)
    assert con.execute(
        "SELECT count(*) FROM compound_candidate WHERE is_best = 1"
    ).fetchone() == (1,)
    con.close()
    assert (data_dir / f"chromatograms/{NAME}.parquet").is_file()

    again = upload(runs, data_dir, fit=False)
    assert again.skipped_unchanged == [NAME] and again.failed == []
