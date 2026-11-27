"""Processing of the chromatograms (esper.processing): the configuration of a data
directory, the detection per measured window, the identification by target search or
suspect screening, and the pipeline (esper process, esper upload)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import peaxart as px
import pytest

from esper import db
from esper.processing import PEAK_SOURCE
from esper.processing.config import (
    DetectionConfig,
    ProcessingConfigError,
    SuspectScreeningConfig,
    TargetSearchConfig,
    processing_config,
    read_processing_config,
    resolve_method,
)
from esper.processing.detection import detect, find_peaks
from esper.processing.identification import screen_suspects, search_targets
from esper.processing.pipeline import Processor, to_process
from esper.sources import CONTINUOUS
from esper.target_index import TargetIndex
from esper.windows import cuts_of, measured_windows

# --- configuration ----------------------------------------------------------------------


def test_defaults():
    config = processing_config(None)
    assert config.method is None
    assert config.baseline == px.BaselineConfig()
    assert config.detection == DetectionConfig()
    assert (config.detection.width, config.detection.min_snr) == (5, 10)
    assert config.fit.peak_shape == "emg"


def test_overrides_of_the_yaml():
    config = processing_config(
        {
            "method": "target",
            "baseline": {"lam": "1e6"},  # YAML reads 1e6 as a string
            "detection": {"prominence": 50, "height": 20},
            "fit": {"peak_shape": "pap_papai", "sigma_bounds": [0.3, 4]},
            "target": {"window_s": 12},
        }
    )
    assert config.baseline.lam == 1e6
    assert (config.detection.prominence, config.detection.height) == (50, 20)
    assert config.fit.peak_shape == px.ShapeName("pap_papai")
    assert config.fit.sigma_bounds == (0.3, 4.0)
    assert config.target.window_s == 12.0
    assert processing_config({"baseline": False}).baseline is None


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"method": "both"}, "method: 'both'"),
        ({"detection": {"prominance": 50}}, "unknown prominance"),
        ({"fit": {"peak_shape": "square"}}, "fit.peak_shape.*one of pap_papai, emg"),
        ({"fit": {"convention": "legacy"}}, "only 'standard'"),
        ({"detection": {"height": "high"}}, "detection.height"),
        ({"steps": {}}, "unknown steps"),
    ],
)
def test_invalid_configuration(section, message):
    with pytest.raises(ProcessingConfigError, match=message):
        processing_config(section)


def test_method_declared_once(tmp_path):
    (tmp_path / "esper.yaml").write_text("instrument: gcwerks\n")
    assert resolve_method(tmp_path) is None
    assert resolve_method(tmp_path, "target") == "target"
    assert read_processing_config(tmp_path).method == "target"
    assert "instrument: gcwerks" in (tmp_path / "esper.yaml").read_text()
    assert resolve_method(tmp_path) == "target"
    with pytest.raises(ProcessingConfigError, match="uses target"):
        resolve_method(tmp_path, "suspect")


# --- detection ----------------------------------------------------------------------------


def test_measured_windows():
    y = np.array([np.nan, 1, 2, np.nan, np.nan, 3, 4, 5, np.nan])
    assert [w.tolist() for w in measured_windows(y)] == [[1, 2], [5, 6, 7]]
    assert [w.tolist() for w in measured_windows(y, min_points=3)] == [[5, 6, 7]]
    # scans: the NaN are not gaps
    assert [w.tolist() for w in measured_windows(y, continuous=True)] == [
        [1, 2, 5, 6, 7]
    ]
    assert [w.tolist() for w in measured_windows(np.arange(4.0))] == [[0, 1, 2, 3]]


def _gauss(t, rt, height, sigma=1.0):
    return height * np.exp(-0.5 * ((t - rt) / sigma) ** 2)


T = np.arange(0, 200, 0.2)


def _sim_frame() -> pd.DataFrame:
    """Two masses measured in SIM windows, on a baseline of 80 counts, NaN between."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 2, len(T))
    m69 = 80 + noise + _gauss(T, 50, 2000) + _gauss(T, 150, 1500)
    m69[(T > 70) & (T < 130)] = np.nan  # not measured between its two windows
    m85 = 80 + noise + _gauss(T, 50.3, 900) + _gauss(T, 60, 700)
    m85[T > 90] = np.nan
    return pd.DataFrame(
        {"69": m69, "85": m85, "tic": 80 + noise}, index=pd.Index(T, name="rt")
    )


def test_peaks_found_in_each_window():
    config = processing_config(None)
    peaks = find_peaks(_sim_frame(), config.baseline, config.detection)
    found = sorted(zip(peaks["mass"], peaks["peak_rt"].round(0)))
    assert found == [(69, 50), (69, 150), (85, 50), (85, 60)]
    # scans of the run, not of the window
    second = peaks[(peaks["mass"] == 69) & (peaks["peak_rt"] > 100)].iloc[0]
    assert second["left_bases"] < 150 < second["right_bases"]  # times [s]


def test_peaks_grouped_into_compounds():
    config = processing_config(None)
    peaks, compounds = detect(_sim_frame(), config.baseline, config.detection)
    assert len(compounds) == 3  # 69 + 85 at 50 s, 85 at 60 s, 69 at 150 s
    first = peaks.loc[peaks["rt_s"].between(49, 51), "compound_idx"]
    assert first.nunique() == 1 and len(first) == 2
    assert compounds["compound_name"].isna().all()
    assert set(compounds["valid_candidate"]) == {0}
    assert peaks["width_s"].between(1.5, 4).all()  # sigma 1 s: FWHM 2.35 s


def test_prominence_relative_to_the_noise():
    """A small peak on a quiet mass is found (legacy: 120 counts), the same peak in
    noise of its size is not."""
    rng = np.random.default_rng(1)
    quiet = 48 + rng.normal(0, 0.7, len(T)) + _gauss(T, 100, 60, sigma=2)
    noisy = 48 + rng.normal(0, 15, len(T)) + _gauss(T, 100, 60, sigma=2)
    frame = pd.DataFrame({"173": quiet, "174": noisy}, index=pd.Index(T, name="rt"))
    config = processing_config(None)
    peaks = find_peaks(frame, config.baseline, config.detection)
    assert sorted(zip(peaks["mass"], peaks["peak_rt"].round(0))) == [(173, 100)]
    legacy = dataclasses.replace(config.detection, prominence=120, min_snr=0)
    assert find_peaks(frame, config.baseline, legacy).empty
    # the floor: no noise, no threshold of 0
    assert DetectionConfig().min_prominence(0.0) == 20


def test_prominence_relative_to_the_local_noise():
    """The noise of a TOF grows with the signal: a run quiet, then 20 x noisier (a high
    background). Relative to the noise of the whole run (its median: of the quiet part),
    the noise of the loud part gives peaks; relative to the noise around each peak, only
    the real one (in the quiet part)."""
    rng = np.random.default_rng(2)
    t = np.arange(0, 600, 0.2)
    scale = np.where(t < 400, 1.0, 20.0)
    y = 100 + scale * rng.normal(0, 2, len(t)) + _gauss(t, 150, 60, sigma=2)
    frame = pd.DataFrame({"57": y}, index=pd.Index(t, name="rt"))
    config = processing_config(None)
    assert config.detection.noise_window_s == 20
    peaks = find_peaks(frame, config.baseline, config.detection)
    assert list(peaks["peak_rt"].round(0)) == [150]
    whole = dataclasses.replace(config.detection, noise_window_s=None)
    noisy = find_peaks(frame, config.baseline, whole)
    assert 150 in set(noisy["peak_rt"].round(0)) and (noisy["peak_rt"] > 400).sum() > 3


def test_scan_runs_one_window_per_mass():
    frame = _sim_frame()
    frame.attrs[CONTINUOUS] = "1"
    # a value missing in some scans (no peak found there): not a gap
    frame.loc[frame.index[::7], "69"] = np.nan
    config = processing_config(None)
    peaks = find_peaks(frame, config.baseline, config.detection)
    assert sorted(peaks.loc[peaks["mass"] == 69, "peak_rt"].round(0)) == [50, 150]
    del frame.attrs[CONTINUOUS]  # SIM: each missing value splits the windows
    peaks = find_peaks(frame, config.baseline, config.detection)
    assert (peaks["mass"] == 69).sum() == 0  # windows of 6 values: too short


# --- target search ------------------------------------------------------------------------


def _targets(rts: dict, ions: dict, ref_rts: dict | None = None) -> TargetIndex:
    names = sorted(set(ions))
    table = pd.DataFrame(
        {
            "rt_ref_s": [(ref_rts or {}).get(n, np.nan) for n in names],
            "ri": np.nan,
        },
        index=pd.Index(names, name="name"),
    )
    ion_rows = pd.DataFrame(
        [(n, r, m) for n, masses in ions.items() for r, m in enumerate(masses)],
        columns=["name", "rank", "mass"],
    )
    rt_rows = pd.DataFrame(
        [(n, "2020-01-01", rt, w) for n, (rt, w) in rts.items()],
        columns=["name", "valid_from", "rt_s", "window_s"],
    )
    aliases = pd.DataFrame(columns=["alias", "name"])
    return TargetIndex(table, ion_rows, aliases, rt_rows)


def _detected(rows) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Peaks (compound_idx, mass, rt_s, height) and their compounds."""
    peaks = pd.DataFrame(rows, columns=["compound_idx", "mass", "rt_s", "height"])
    peaks = peaks.assign(
        prominence=peaks["height"],
        start_s=peaks["rt_s"] - 1,
        stop_s=peaks["rt_s"] + 1,
        is_max_of_cmp=0,
    )
    peaks.index.name = "peak_idx"
    compounds = (
        peaks.groupby("compound_idx")
        .agg(rt_mean_s=("rt_s", "mean"), n_peaks=("rt_s", "size"))
        .assign(compound_name=None, valid_candidate=0, is_ref_ri=0)
    )
    return peaks, compounds


def test_target_found_on_its_main_ion_nearest_the_expected_rt():
    peaks, compounds = _detected(
        [
            (0, 69, 100.0, 500),
            (1, 69, 103.5, 900),  # also in the window, farther
            (2, 85, 200.0, 300),  # outside every window: kept, not named
        ]
    )
    targets = _targets({"CF4": (100.5, 10.0)}, {"CF4": [69, 50]})
    peaks, compounds = search_targets(peaks, compounds, targets, "2025-01-01")
    assert compounds.loc[0, "compound_name"] == "CF4"
    assert compounds.loc[0, "valid_candidate"] == 1
    assert compounds.loc[[1, 2], "compound_name"].isna().all()
    assert len(compounds) == 3


def test_target_on_a_measured_half_mass_and_a_secondary_ion():
    # 2008: SIM masses set 0.5 higher (69.5 is m/z 69); the main ion not measured
    peaks, compounds = _detected([(0, 69.5, 100.0, 500), (1, 51.5, 300.0, 400)])
    targets = _targets(
        {"CF4": (100.0, 10.0), "HFC-23": (299.0, 10.0)},
        {"CF4": [69], "HFC-23": [70, 51]},
    )
    _, compounds = search_targets(peaks, compounds, targets)
    assert compounds["compound_name"].tolist() == ["CF4", "HFC-23"]


def test_targets_not_expected_are_not_searched():
    peaks, compounds = _detected([(0, 69, 100.0, 500), (1, 85, 50.0, 500)])
    rts = pd.DataFrame(
        [("A", "2020-01-01", 100.0, 10.0), ("A", "2024-01-01", np.nan, np.nan)],
        columns=["name", "valid_from", "rt_s", "window_s"],
    )
    table = pd.DataFrame(
        {"rt_ref_s": [np.nan, 50.0], "ri": np.nan},
        index=pd.Index(["A", "B"], name="name"),
    )
    ions = pd.DataFrame([("A", 0, 69), ("B", 0, 85)], columns=["name", "rank", "mass"])
    targets = TargetIndex(table, ions, pd.DataFrame(columns=["alias", "name"]), rts)
    # A: not identified from 2024 on (peakid); B: from its rt in the target list
    _, before = search_targets(peaks, compounds, targets, "2023-06-01")
    assert before["compound_name"].tolist() == ["A", "B"]
    _, after = search_targets(peaks, compounds, targets, "2025-06-01")
    assert after["compound_name"].tolist() == [None, "B"]


def test_coeluting_targets_get_a_compound_each():
    # The detection grouped HFC-152a (65) and HFC-134a (83) into one compound
    peaks, compounds = _detected(
        [(0, 65, 1169.7, 900), (0, 83, 1169.8, 600), (0, 69, 1169.8, 100)]
    )
    targets = _targets(
        {"HFC-152a": (1170.0, 12.0), "HFC-134a": (1171.0, 12.0)},
        {"HFC-152a": [65], "HFC-134a": [83, 69]},
    )
    peaks, compounds = search_targets(peaks, compounds, targets)
    assert sorted(compounds["compound_name"]) == ["HFC-134a", "HFC-152a"]
    of_134a = compounds.index[compounds["compound_name"] == "HFC-134a"][0]
    assert sorted(peaks.loc[peaks["compound_idx"] == of_134a, "mass"]) == [69, 83]
    assert compounds.loc[of_134a, "n_peaks"] == 2
    assert compounds.loc[of_134a, "main_ion_mass"] == 83  # its largest peak
    assert peaks.groupby("compound_idx")["is_max_of_cmp"].sum().eq(1).all()


def test_window_of_a_target_without_one():
    peaks, compounds = _detected([(0, 69, 100.0, 500)])
    targets = _targets({}, {"CF4": [69]}, ref_rts={"CF4": 108.0})
    config = TargetSearchConfig(window_s=10.0)
    _, found = search_targets(peaks, compounds, targets, config=config)
    assert found["compound_name"].isna().all()  # 8 s from its rt, window ±5 s
    config = TargetSearchConfig(window_s=20.0)
    _, found = search_targets(peaks, compounds, targets, config=config)
    assert found["compound_name"].tolist() == ["CF4"]


# --- suspect screening ------------------------------------------------------------------


class _Libraries:
    """best_matches of a LibraryCollection: the spectra closest to a fixed one."""

    def best_matches(self, spectrum, n):
        from esper.libraries import Match

        if 69 in spectrum.mz.tolist():
            matches = [Match("nist_ei", "CF4", 0.95, 2), Match("mygu", "C2F6", 0.7, 1)]
        else:
            matches = [Match("nist_ei", "N2O", 0.5, 1)]
        return matches[:n]


def test_suspect_screening():
    peaks, compounds = _detected(
        [(0, 69, 100.0, 1000), (0, 50, 100.1, 120), (1, 44, 300.0, 400)]
    )
    compounds, candidates = screen_suspects(
        peaks, compounds, _Libraries(), SuspectScreeningConfig(min_score=0.8)
    )
    assert compounds["compound_name"].tolist() == ["CF4", "N2O"]
    assert compounds["valid_candidate"].tolist() == [1, 0]  # 0.5 < min_score
    first = candidates[candidates["compound_idx"] == 0]
    assert first["rank"].tolist() == [1, 2]
    assert first["spectrum_ref"].tolist() == ["nist_CF4", "mygu_C2F6"]
    assert first["is_best"].tolist() == [1, 0]


# --- pipeline ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path):
    """A data directory with one measurement imported without peaks (as GCwerks)."""
    (tmp_path / "esper.yaml").write_text("instrument: gcwerks\n")
    con = db.connect(tmp_path, create=True)
    (tmp_path / "chromatograms").mkdir()
    _sim_frame().to_parquet(tmp_path / "chromatograms" / "run.parquet")
    with con:
        con.execute(
            "INSERT INTO measurement (name, acq_datetime, chromatogram_file, source_dir,"
            " source_mtime, source_size, imported_at, instrument, peak_source)"
            " VALUES ('250101.0000.air.1', '2025-01-01T00:00:00',"
            " 'chromatograms/run.parquet', 'src', 0, 0, '2025-01-02', 'gcwerks', NULL)"
        )
        con.execute("INSERT INTO target (name) VALUES ('CF4')")
        con.execute(
            "INSERT INTO target_ion (name, valid_from, rank, mass)"
            " VALUES ('CF4', '2020-01-01', 0, 69)"
        )
        con.execute("INSERT INTO target_rt VALUES ('CF4', '2020-01-01', 149.0, 10.0)")
    con.close()
    return tmp_path


def test_process_a_measurement(data_dir):
    resolve_method(data_dir, "target")
    con = db.connect(data_dir, create=True)
    processor = Processor(con, data_dir)
    ((mid, _),) = to_process(con, "target")
    table = processor.process(mid, workers=1)
    assert table["fit_rt_s"].notna().all()
    row = con.execute(
        "SELECT peak_source, identification, n_compounds, fit_version FROM measurement"
    ).fetchone()
    assert row[:3] == (PEAK_SOURCE, "target", 3) and row[3] is not None
    named = con.execute(
        "SELECT compound_name, rt_mean_s FROM compound WHERE compound_name IS NOT NULL"
    ).fetchall()
    assert [n for n, _ in named] == ["CF4"] and abs(named[0][1] - 150) < 0.5
    # the fit is on the stored chromatogram: its background is the baseline (80)
    cf4 = table[(table["mass"] == 69) & (table["rt_s"] > 100)].iloc[0]
    assert abs(cf4["fit_baseline"] - 80) < 10 and abs(cf4["fit_height"] - 1500) < 100
    assert to_process(con, "target") == []
    assert len(to_process(con, "target", force=True)) == 1
    assert len(to_process(con, "suspect")) == 1  # identified with another method
    con.close()


def test_no_method_no_processing(data_dir):
    con = db.connect(data_dir)
    with pytest.raises(ProcessingConfigError, match="--method"):
        Processor(con, data_dir)
    con.close()


def test_esper_process(data_dir, capsys, caplog):
    from esper.cli.main import main

    def run(*args) -> tuple[int, str]:
        code = main(["process", str(data_dir), "-j", "1", *args])
        return code, capsys.readouterr().out

    def refused(*args) -> str:
        assert main(["process", str(data_dir), *args]) == 2
        return capsys.readouterr().err

    # no method
    assert "--method" in refused("--steps", "peaks")
    assert "--method" in refused("--compounds", "CF4")
    # checked before the method is declared
    args = ("--method", "target", "--compounds", "CF4")
    assert "needs the peaks step" in refused(*args, "--steps", "fit")
    assert "--overrides" in refused(*args, "--overrides")
    assert "target search" in refused("--method", "suspect", "--compounds", "CF4")
    assert "not targets: CF-4" in refused("--method", "target", "--compounds", "CF-4")
    assert read_processing_config(data_dir).method is None
    # no method: the fit only, and a warning about the runs left without peaks
    assert run() == (0, "runs to process: 0\n")
    assert "1 runs without peaks" in caplog.text
    code, out = run("--method", "target")
    assert code == 0
    assert "250101.0000.air.1:   3 compounds,   1 identified" in out
    assert "peaks found and identified: 1, failed: 0" in out
    assert "fitted: 1, failed: 0" in out
    assert run() == (0, "runs to process: 0\n")
    # the fit alone, again: of the peaks found by esper too
    code, out = run("--steps", "fit", "-f")
    assert code == 0 and out.startswith("runs to process: 1\n")
    assert out.endswith("fitted: 1, failed: 0\n") and "peaks found" not in out
    assert run("-f", "--before", "2025-01-01") == (0, "runs to process: 0\n")


def test_esper_process_compounds(data_dir, capsys):
    from esper.cli.main import main

    assert main(["process", str(data_dir), "--method", "target", "-j", "1"]) == 0
    con = db.connect(data_dir)
    before = _compounds(con)
    con.close()
    # processed: left as it is without -f
    assert main(["process", str(data_dir), "--compounds", "CF4", "-j", "1"]) == 0
    assert "runs to process: 0" in capsys.readouterr().out
    assert main(["process", str(data_dir), "--compounds", "CF4", "-f", "-j", "1"]) == 0
    assert "peaks found and identified: 1" in capsys.readouterr().out
    con = db.connect(data_dir)
    after = _compounds(con)
    con.close()
    assert [c for c in after if c[1] != "CF4"] == [c for c in before if c[1] != "CF4"]
    assert [c[1] for c in after].count("CF4") == 1


def test_upload_processes_the_runs_without_peaks(tmp_path, monkeypatch):
    from gcwerks.io import chromatogram as gc_chromatogram
    from gcwerks.io.chromatogram import Chromatogram, IonChromatogram

    from esper.upload import upload

    frame = _sim_frame()
    ions = []
    for column in ("69", "85"):
        y = frame[column].to_numpy()
        for index in measured_windows(y):
            ions.append(IonChromatogram(float(column), index, y[index]))
    run = Chromatogram(
        start=pd.Timestamp("2026-01-12 21:14:05"),
        times=T,
        tic=frame["tic"].to_numpy(),
        ions=ions,
    )
    root = tmp_path / "gcdir"
    (root / "config").mkdir(parents=True)
    (root / "config" / "gcwerks.conf").write_text("")
    (root / "config" / "peak.list").write_text("CFC-12\n")
    folder = root / "26" / "chromatograms" / "channel0"
    folder.mkdir(parents=True)
    for name in ("260112.2114.air.7", "260112.2224.air.7"):
        (folder / name).write_bytes(b"")
    monkeypatch.setattr(gc_chromatogram, "read_chromatogram", lambda path: run)
    data_dir = tmp_path / "esper"

    # without a method: imported only
    summary = upload(root, data_dir, instrument="gcwerks", workers=1)
    assert summary.processed == [] and len(summary.imported) == 2
    # with one: the runs imported before are processed, though unchanged
    summary = upload(root, data_dir, method="target", workers=1)
    assert summary.imported == []
    assert summary.processed == ["260112.2114.air.7", "260112.2224.air.7"]
    con = db.connect(data_dir)
    rows = con.execute(
        "SELECT peak_source, identification, n_peaks > 0, fit_version IS NOT NULL"
        " FROM measurement"
    ).fetchall()
    assert rows == [(PEAK_SOURCE, "target", 1, 1)] * 2
    con.close()
    assert upload(root, data_dir, workers=1).processed == []
    assert upload(root, data_dir, force=True, workers=1).processed == [
        "260112.2114.air.7",
        "260112.2224.air.7",
    ]


# --- edge cases -------------------------------------------------------------------------


def test_run_without_peaks_is_processed_once(data_dir):
    flat = pd.DataFrame({"69": 80.0, "tic": 80.0}, index=pd.Index(T, name="rt"))
    flat.to_parquet(data_dir / "chromatograms" / "run.parquet")
    resolve_method(data_dir, "target")
    con = db.connect(data_dir, create=True)
    ((mid, _),) = to_process(con, "target")
    assert Processor(con, data_dir).process(mid, workers=1).empty
    assert con.execute("SELECT n_peaks, peak_source FROM measurement").fetchone() == (
        0,
        PEAK_SOURCE,
    )
    assert to_process(con, "target") == []  # not again at each upload
    con.close()


def test_peak_at_a_gap_is_fitted_in_the_nearest_window():
    from esper.fitting import mass_windows

    y = np.ones(40)
    y[10] = np.nan
    frame = pd.DataFrame({"69": y}, index=np.arange(40) * 0.5)
    peaks = pd.DataFrame({"mass": [69, 69], "rt_s": [2.0, 5.0]})  # 5.0 on the NaN
    windows = mass_windows(frame, peaks)
    assert sum(len(group) for _, _, group in windows) == 2
    # without NaN: one window, as the TOF chromatograms
    windows = mass_windows(frame.fillna(1.0), peaks)
    assert [len(t) for t, _, _ in windows] == [40]


def test_width_of_a_scan_run_peak_in_seconds():
    frame = _sim_frame()
    frame.attrs[CONTINUOUS] = "1"
    frame.loc[frame.index[::7], "69"] = np.nan
    config = processing_config(None)
    peaks, _ = detect(frame, config.baseline, config.detection)
    # sigma 1 s: width at half prominence 2.35 s, whatever the missing values
    assert peaks.loc[peaks["mass"] == 69, "width_s"].between(2.0, 2.8).all()


def test_processing_failures_counted_apart(data_dir, monkeypatch):
    from esper.process import RunProcessing

    def fail(self, mid, **kwargs):
        raise RuntimeError("detection")

    resolve_method(data_dir, "target")
    monkeypatch.setattr(Processor, "process", fail)
    con = db.connect(data_dir)
    runs = RunProcessing(con, data_dir, read_processing_config(data_dir))
    runs.run("run", 1)
    con.close()
    assert runs.summary.processing_failed == [("run", "RuntimeError('detection')")]
    assert runs.summary.fit_failed == [] and runs.summary.processed == []


def _alpinac_folders(root, names=("250101.0000.air.1", "250101.0130.air.1"), raw=True):
    """.alpinac folders with their chromatograms only (TOF: no gaps), corrected and raw
    sums (``raw``; the raw ones x10 after 100 s: a step of the emission, recorded in the
    provenance of the corrected ones); the second also has a pipeline.nc that cannot be
    read (it must not be)."""
    import json

    from esper.chromatograms import PROVENANCE

    corrected = _sim_frame().ffill().bfill()
    corrected.attrs[PROVENANCE] = json.dumps(
        {"steps": {"config": {}, "emission": [float(T[500])], "ejection": []}}
    )
    sums = corrected.astype("float32")
    sums.loc[sums.index >= T[500]] *= 10
    for i, name in enumerate(names):
        folder = root / f"{name}.alpinac"
        folder.mkdir(parents=True)
        corrected.to_parquet(folder / f"{name}.unit_mass_chromatos.parquet")
        if raw:
            sums.to_parquet(folder / f"{name}.unit_mass_chromatos.raw.parquet")
        if i:
            (folder / "pipeline.nc").write_text("not a netCDF file")
    return root


def test_tofwerk_runs_without_peaks_with_a_method(tmp_path):
    from esper.upload import upload

    runs = _alpinac_folders(tmp_path / "runs")
    data_dir = tmp_path / "esper"
    summary = upload(runs, data_dir, instrument="tofwerk", method="target", workers=1)
    assert summary.failed == []
    assert summary.processed == ["250101.0000.air.1", "250101.0130.air.1"]
    con = db.connect(data_dir)
    rows = con.execute(
        "SELECT peak_source, identification, n_peaks > 0, fit_version IS NOT NULL,"
        " rt_start_s, rt_stop_s, acq_start_source FROM measurement ORDER BY name"
    ).fetchall()
    con.close()
    assert rows == [(PEAK_SOURCE, "target", 1, 1, 0.0, T[-1], "name")] * 2
    # unchanged chromatograms: not again
    assert upload(runs, data_dir, workers=1).processed == []
    # the chromatograms stored: the raw sums, cut at the step of their background
    stored = pd.read_parquet(data_dir / "chromatograms" / "250101.0000.air.1.parquet")
    assert stored["69"].max() > 10_000  # x10 after the step
    assert cuts_of(stored, "69").tolist() == [500]
    assert [len(w) for w in measured_windows(stored["69"].to_numpy(), cuts=[500])] == [
        500,
        500,
    ]


def test_tofwerk_runs_without_raw_sums_are_not_complete(tmp_path):
    """The chromatograms of the legacy pipeline (corrected only): not imported without
    their peaks."""
    from esper.upload import upload

    runs = _alpinac_folders(tmp_path / "runs", names=("250101.0000.air.1",), raw=False)
    summary = upload(
        runs, tmp_path / "esper", instrument="tofwerk", method="target", workers=1
    )
    assert summary.skipped_incomplete == ["250101.0000.air.1"]


def test_tofwerk_runs_without_a_method_need_their_pipeline(tmp_path):
    from esper.upload import upload

    runs = _alpinac_folders(tmp_path / "runs", names=("250101.0000.air.1",))
    summary = upload(runs, tmp_path / "esper", instrument="tofwerk", workers=1)
    assert summary.skipped_incomplete == ["250101.0000.air.1"]


def test_no_method_where_runs_have_imported_peaks(tmp_path):
    """A method would import the runs again without their peaks (and their
    identifications): refused for a data directory with runs of alpinac."""
    from esper.process import process
    from esper.upload import upload

    (tmp_path / "esper.yaml").write_text("instrument: tofwerk\n")
    con = db.connect(tmp_path, create=True)
    with con:
        con.execute(
            "INSERT INTO measurement (name, source_dir, source_mtime, source_size,"
            " imported_at, instrument, peak_source)"
            " VALUES ('250101.0000.air.1', 'src', 0, 0, '2025-01-02', 'tofwerk', 'alpinac')"
        )
    con.close()
    with pytest.raises(ProcessingConfigError, match="1 runs imported with the peaks"):
        upload(tmp_path / "runs", tmp_path, method="suspect")
    with pytest.raises(ProcessingConfigError, match="another data directory"):
        process(tmp_path, method="suspect")
    assert read_processing_config(tmp_path).method is None


def test_method_of_the_instruments():
    from esper.process import check_method

    # tofwerk: the runs can be read without their peaks; gcwerks: never with
    check_method("tofwerk", "suspect")
    check_method("gcwerks", "target")
    # without a method: nothing to check
    check_method("tofwerk", None)


def test_method_in_a_processing_on_one_line(tmp_path):
    (tmp_path / "esper.yaml").write_text("processing: {fit: {peak_shape: emg}}\n")
    with pytest.raises(ProcessingConfigError, match="add 'method: target'"):
        resolve_method(tmp_path, "target")


def _compounds(con) -> list[tuple]:
    return con.execute(
        "SELECT compound_idx, compound_name, n_peaks, round(rt_mean_s, 3), source"
        " FROM compound ORDER BY compound_idx"
    ).fetchall()


def test_process_compounds_of_a_run_not_processed(data_dir):
    resolve_method(data_dir, "target")
    con = db.connect(data_dir, create=True)
    processor = Processor(con, data_dir)
    ((mid, _),) = to_process(con, "target")
    table = processor.process_compounds(mid, ["CF4"], workers=1)
    # CF4 alone: found on its ion (m/z 69) only, fitted
    assert [(n, k) for _, n, k, _, _ in _compounds(con)] == [("CF4", 1)]
    assert set(table["mass"]) == {69} and table["fit_rt_s"].notna().all()
    row = con.execute(
        "SELECT peak_source, identification, n_compounds, n_peaks, fit_version"
        " FROM measurement"
    ).fetchone()
    assert row[:4] == (PEAK_SOURCE, "target", 1, len(table)) and row[4] is not None
    assert to_process(con, "target") == []  # processed (with CF4 only)
    con.close()


def test_process_compounds_keeps_the_others(data_dir):
    resolve_method(data_dir, "target")
    con = db.connect(data_dir, create=True)
    processor = Processor(con, data_dir)
    ((mid, _),) = to_process(con, "target")
    processor.process(mid, workers=1)
    before = _compounds(con)
    (n_compounds,) = con.execute("SELECT n_compounds FROM measurement").fetchone()
    processor.process_compounds(mid, ["CF4"], workers=1)
    after = _compounds(con)
    # the other compounds as they were; CF4 found again, numbered after them
    assert [c for c in after if c[1] != "CF4"] == [c for c in before if c[1] != "CF4"]
    (cf4,) = [c for c in after if c[1] == "CF4"]
    assert cf4[0] > max(c[0] for c in before if c[1] != "CF4")
    assert abs(cf4[3] - 150) < 0.5
    row = con.execute("SELECT n_compounds, n_peaks, fit_version FROM measurement")
    n_compounds_after, n_peaks, fit_version = row.fetchone()
    assert n_compounds_after == n_compounds and fit_version is not None
    assert n_peaks == con.execute("SELECT count(*) FROM peak").fetchone()[0]
    con.close()


def test_target_names(data_dir):
    resolve_method(data_dir, "target")
    con = db.connect(data_dir, create=True)
    with con:
        con.execute("INSERT INTO target_alias (alias, name) VALUES ('CF-4', 'CF4')")
    processor = Processor(con, data_dir)
    assert processor.target_names(["CF-4", "CF4"]) == ["CF4"]
    with pytest.raises(ProcessingConfigError, match="not targets: Unknown"):
        processor.target_names(["CF4", "Unknown"])
    con.close()
    suspect = data_dir / "suspect"
    suspect.mkdir()
    (suspect / "esper.yaml").write_text("processing:\n  method: suspect\n")
    con = db.connect(suspect, create=True)
    with pytest.raises(ProcessingConfigError, match="only with target search"):
        Processor(con, suspect).target_names(["CF4"])
    con.close()


def test_upload_compounds_only_when_esper_finds_the_peaks(tmp_path):
    # tofwerk runs are imported with their peaks
    from esper.upload import upload

    with pytest.raises(ProcessingConfigError, match="--compounds"):
        upload(tmp_path, tmp_path / "esper", instrument="tofwerk", compounds=["CF4"])


def test_retention_index_step_alone(data_dir, capsys):
    """esper process --steps ri: the references, RIs and identification of the runs whose
    peaks esper found, from their compounds and candidates as stored; the peaks, the fit
    and the identifications edited by hand are kept."""
    from esper.cli.main import main
    from esper.process import process
    from esper.retention_index import store_references

    (data_dir / "esper.yaml").write_text(
        "instrument: gcwerks\nprocessing:\n  method: suspect\n"
    )
    con = db.connect(data_dir, create=True)
    with con:
        con.execute(
            "UPDATE measurement SET peak_source = ?, identification = 'suspect',"
            " fit_version = 5",
            (PEAK_SOURCE,),
        )
        for name, rt in (("Kr", 100.0), ("Xe-A", 120.0), ("CFC-11", 600.0)):
            con.execute("INSERT INTO target (name, rt_ref_s) VALUES (?, ?)", (name, rt))
        db.insert(
            con,
            "compound",
            pd.DataFrame(
                {
                    "measurement_id": 1,
                    "compound_idx": [0, 1, 2],
                    "rt_mean_s": [90.0, 110.0, 590.0],
                    "prominence_sum": 1000.0,
                    "compound_name": ["x", "y", "z"],
                    "valid_candidate": 0,
                }
            ),
        )
        db.insert(
            con,
            "compound_candidate",
            pd.DataFrame(
                {
                    "measurement_id": 1,
                    "compound_idx": [0, 1, 2, 2],
                    "rank": [1, 1, 1, 2],
                    "candidate_name": ["Kr", "Xe-A", "benzene", "CFC-11"],
                    "score": [0.99, 0.99, 0.97, 0.96],
                    "is_best": [1, 1, 1, 0],
                }
            ),
        )
        con.execute(
            "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass, rt_s,"
            " fit_rt_s) VALUES (1, 0, 2, 101, 590.0, 590.1)"
        )
        con.execute(
            "INSERT INTO compound_identification (measurement_id, compound_idx, name,"
            " edited_at) VALUES (1, 2, 'by hand', '2026-11-27')"
        )
    refs = pd.DataFrame({"name": ["Kr", "Xe-A"], "ri": [25.0, 75.0]})
    store_references(con, refs)
    con.close()

    summary = process(data_dir, steps=["ri"])
    assert summary.indexed == ["250101.0000.air.1"] and not summary.processed
    con = db.connect(data_dir)
    rows = con.execute(
        "SELECT compound_name, valid_candidate, is_ref_ri, retention_index FROM compound"
        " ORDER BY compound_idx"
    ).fetchall()
    # CFC-11 at 590 s: RI 1275, its target the same (600 s in the reference run):
    # combined 0.96 > benzene without RI 0.97 x 0.9
    assert rows == [("Kr", 1, 1, 25.0), ("Xe-A", 1, 1, 75.0), ("CFC-11", 1, 0, 1275.0)]
    best = con.execute(
        "SELECT candidate_name, ri, ri_diff FROM compound_candidate WHERE is_best = 1"
        " AND compound_idx = 2"
    ).fetchall()
    assert best == [("CFC-11", 1275.0, 0.0)]
    found = con.execute(
        "SELECT name, compound_idx FROM measurement_ri_reference ORDER BY rt_s"
    ).fetchall()
    assert found == [("Kr", 0), ("Xe-A", 1)]
    # kept: the fit, the peaks, the identification by hand
    assert con.execute("SELECT fit_version FROM measurement").fetchone() == (5,)
    assert con.execute("SELECT fit_rt_s FROM peak").fetchone() == (590.1,)
    assert con.execute("SELECT name FROM compound_identification").fetchone() == (
        "by hand",
    )
    con.close()
    # the command line; refused without retention indices
    assert main(["process", str(data_dir), "--steps", "ri"]) == 0
    assert "retention indices: 1, failed: 0" in capsys.readouterr().out
    (data_dir / "esper.yaml").write_text(
        "instrument: gcwerks\nprocessing:\n  method: suspect\n  retention_index: false\n"
    )
    assert main(["process", str(data_dir), "--steps", "ri"]) == 2
    assert "uses no retention indices" in capsys.readouterr().err
