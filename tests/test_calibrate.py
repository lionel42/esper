"""esper calibrate: the spectra of an h5 file read with twutils, fitted with masscal."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import twutils
from masscal import MassCalibration, mass_to_tof
from peaxart.shapes import LEGACY, pseudo_voigt_height
from peaxart.shapes import pseudo_voigt as pv

from esper.calibrate import calibrate_run, calibrate_to_file
from esper.cli.main import main

NAME = "241007.0423.air.1"
N_SCANS, N_BUFS, N_POINTS, S_PER_BIN = 600, 10, 9400, 0.2
SIGMA_B, SIGMA_SLOPE, ALPHA = 0.5, 4e-5, 0.5
CALIBRANTS = [("A", 28.0061), ("B", 43.9898), ("C", 68.9947), ("D", 99.9931),
              ("E", 130.9915), ("F", 168.9883), ("G", 218.9851)]
MZ = np.array([mz for _, mz in CALIBRANTS])
WINDOWS = {"averaging": 30.0, "spacing": 15.0}


def drift_params(rt):
    """True calibration (power mode) at rt: the tof indices shift by 0.8 over the run."""
    return {"p1": 600.0, "p2": -200.0 + 0.8 * np.asarray(rt) / (N_SCANS * S_PER_BIN), "p3": 0.5}


def write_drift_h5(path, n_segments: int = 1):
    """A run where the calibrants are present all along and the calibration drifts.

    The h5 calibration (TOFWERK attributes) is the one of the start, off by 0.5 tof index.
    """
    rng = np.random.default_rng(1)
    n_writes = N_SCANS // N_BUFS
    rt = (np.arange(N_SCANS) + 1) * S_PER_BIN
    tof = np.arange(N_POINTS, dtype=float)
    data = rng.normal(1.0, 0.3, (len(rt), N_POINTS)).astype(np.float32)
    for i, t in enumerate(rt):
        for mz, height in [(mz, 2000.0) for _, mz in CALIBRANTS] + [(68.83, 300.0)]:
            mu = float(mass_to_tof(mz, drift_params(t), "power"))
            sigma = SIGMA_B + SIGMA_SLOPE * mu
            area = height / pseudo_voigt_height(1.0, sigma, ALPHA)
            data[i] += pv(tof, area, mu, sigma, ALPHA, LEGACY).astype(np.float32)
    data = np.repeat(data.reshape(n_writes, N_BUFS, 1, N_POINTS), n_segments, axis=2)
    initial = {**drift_params(0.0), "p2": -200.5}
    with h5py.File(path, "w") as f:
        fs = f.create_group("FullSpectra")
        fs.create_dataset("TofData", data=data, chunks=(1, 1, 1, N_POINTS))
        fs.create_dataset("MassAxis", data=((tof - initial["p2"]) / initial["p1"]) ** 2)
        fs.attrs["MassCalibMode"] = np.array([2], dtype=np.int32)
        for k, v in initial.items():
            fs.attrs[f"MassCalibration {k}"] = np.array([v])
        td = f.create_group("TimingData")
        td.create_dataset("BufTimes", data=rt.reshape(n_writes, N_BUFS), chunks=(1, 1))
        td.attrs["BlockPeriod"] = np.array([S_PER_BIN * 1e9])
        f.attrs["NbrWrites"] = np.array([n_writes], dtype=np.int32)
        f.attrs["NbrBufs"] = np.array([N_BUFS], dtype=np.int32)
        f.attrs["NbrBlocks"] = np.array([1], dtype=np.int32)
    return path


@pytest.fixture(scope="module")
def drift_file(tmp_path_factory):
    return write_drift_h5(tmp_path_factory.mktemp("drift") / f"{NAME}.h5")


@pytest.fixture(scope="module")
def drift_result(drift_file):
    return calibrate_run(drift_file, CALIBRANTS, block=70, **WINDOWS)


def test_calibrate_drift(drift_result):
    """The drift is followed (the h5 calibration is off by 0.5 tof index)."""
    result, cal = drift_result, drift_result.calibration
    assert result.windows["ok"].all() and len(result.windows) == 7
    assert (result.windows["n_used"] == len(CALIBRANTS)).all()
    rt = cal.rt[:, None]
    true_tof = mass_to_tof(MZ[None, :], drift_params(rt), "power")
    assert np.abs((cal.mz(rt, true_tof) - MZ) / MZ * 1e6).max() < 0.5
    assert cal.alpha == pytest.approx(ALPHA, abs=0.03)
    assert cal.segments == (0, 1) and result.name == NAME
    # the masses are right at the end of the run, the h5 calibration is ~100 ppm off
    rt_end = cal.rt[-1]
    true_tof = mass_to_tof(MZ, drift_params(rt_end), "power")
    h5 = twutils.h5_calibration(twutils.TofFile(result.calibration.metadata["source"]["file"]))
    assert np.abs((h5.mz(rt_end, true_tof) - MZ) / MZ * 1e6).max() > 50


def test_calibrate_to_file(drift_file, drift_result):
    """Written where twutils finds it, with how it was made."""
    _, path = calibrate_to_file(drift_file, calibrants=CALIBRANTS, **WINDOWS)
    assert path == drift_file.parent / f"{NAME}.alpinac" / f"{NAME}.seg0_mc.json"
    assert path.with_suffix(".peaks.csv").is_file()
    cal = twutils.calibration_for(drift_file)
    assert cal.source == str(path)
    np.testing.assert_array_equal(cal.params, drift_result.calibration.params)
    source = cal.metadata["source"]
    assert source["file"] == str(drift_file.absolute()) and source["config"]["averaging"] == 30
    assert set(source["versions"]) == {"masscal", "peaxart", "twutils", "esper"}
    assert [c["mz"] for c in source["calibrants"]] == list(MZ)
    assert "h5 attributes" in source["initial"]
    path.unlink()
    path.with_suffix(".peaks.csv").unlink()


def test_calibrate_spikes(drift_file):
    result = calibrate_run(drift_file, calibrants=[], spikes={60.0: CALIBRANTS, 500.0: CALIBRANTS})
    assert list(result.windows["rt"]) == [60.0] and result.windows["ok"].all()  # 500 s: after the run
    true_tof = mass_to_tof(MZ, drift_params(60.0), "power")
    assert np.abs((result.calibration.mz(60.0, true_tof) - MZ) / MZ * 1e6).max() < 0.5


def test_calibrate_spike_with_windows(drift_file):
    """A spike is fitted with the closest window, at its own rt."""
    spike = ("spike", CALIBRANTS[3][1])
    result = calibrate_run(drift_file, CALIBRANTS, spikes={61.3: [spike]}, **WINDOWS)
    peaks = result.peaks[result.peaks["calibrant"] == "spike"]
    assert len(peaks) and (peaks["rt"] == 61.3).all()


def test_calibrate_yaml_and_initial(drift_file, tmp_path):
    """Calibrants and initial calibration of the yaml file of the pipeline."""
    path = tmp_path / "config_calib.yml"
    lines = ['"default": {"mode": 2, "p1": 600.0, "p2": -200.3, "p3": 0.5}', '"0.0":']
    path.write_text("\n".join(lines + [f"  - {mz}" for mz in MZ]) + "\n")
    result = calibrate_run(drift_file, path, **WINDOWS)
    assert result.calibration.metadata["source"]["initial"].endswith("(default)")
    assert result.windows["ok"].all()
    initial = MassCalibration(mode="power", rt=[0.0], params=[[600.0, -200.2, 0.5]], source="mine")
    result = calibrate_run(drift_file, path, initial=initial, **WINDOWS)
    assert result.calibration.metadata["source"]["initial"] == "mine"


def test_calibrate_arguments(drift_file, tmp_path):
    with pytest.raises(ValueError, match="cubic"):
        calibrate_run(drift_file, CALIBRANTS, mode="cubic")
    with pytest.raises(TypeError):
        calibrate_run(drift_file, CALIBRANTS, averagin=30)  # not a config field
    with pytest.raises(ValueError, match="No calibrant"):
        calibrate_run(drift_file, [], spikes={500.0: CALIBRANTS})
    two = write_drift_h5(tmp_path / "two.h5", n_segments=2)
    with pytest.raises(ValueError, match="2 segments"):  # which segments: not guessed
        calibrate_run(two, CALIBRANTS)
    result = calibrate_run(two, CALIBRANTS, segments=(1, 2), **WINDOWS)
    assert result.calibration.segments == (1, 2) and result.windows["ok"].all()


def test_cli(drift_file, tmp_path, capsys):
    out = tmp_path / "run.seg0_mc.json"
    args = ["calibrate", str(drift_file), "--averaging", "60", "--spacing", "30", "-o", str(out)]
    assert main(args + ["--calibrant", *[f"{mz}" for mz in MZ]]) == 0
    assert "3/3 windows" in capsys.readouterr().out
    assert len(MassCalibration.from_file(out).rt) == 3 and out.with_suffix(".peaks.csv").is_file()
    assert main(["calibrate", str(drift_file), "--averaging", "60", "--output-dir", str(tmp_path / "cal"),
                 "--no-peaks-csv", "--mode", "sqrt", "--calibrant", *[f"{mz}" for mz in MZ]]) == 0
    written = tmp_path / "cal" / f"{NAME}.seg0_mc.json"
    assert MassCalibration.from_file(written).mode == "sqrt"
    assert not written.with_suffix(".peaks.csv").exists()
    # a bad file: reported, the exit code is 1
    assert main(["calibrate", str(tmp_path / "missing.h5")]) == 1


def test_cli_plot_calibration(drift_file, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    path = tmp_path / "run.EI_mc.json"
    assert main(["calibrate", str(drift_file), "--averaging", "60", "--spacing", "30", "-o", str(path),
                 "--no-peaks-csv", "--calibrant", *[f"{mz}" for mz in MZ]]) == 0
    # without the peaks: the plots of the calibration only
    assert main(["plot-calibration", str(path)]) == 0
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["run.EI_mc.drift.png", "run.EI_mc.parameters.png"]
    assert main(["calibrate", str(drift_file), "--averaging", "60", "--spacing", "30", "-o", str(path),
                 "--plots", "--calibrant", *[f"{mz}" for mz in MZ]]) == 0
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert main(["plot-calibration", str(path), "--plot", "residuals"]) == 0
    assert "run.EI_mc.residuals.png" in capsys.readouterr().out


# --- a real run, against the legacy alpinac calibration (~25 s) -----------------------

RUN = Path("/media/Shuttle/TOF-Data/2024/240315.2211.air.1.h5")
LEGACY_MC = RUN.parent / "240315.2211.air.1.alpinac" / "240315.2211.air.1.EI_mc.txt"
PIPELINE_CALIBRANTS = Path("/shared/scripts/tof-processing/config_calib.yml")
real_data = pytest.mark.skipif(
    not (RUN.is_file() and LEGACY_MC.is_file() and PIPELINE_CALIBRANTS.is_file()),
    reason=f"{RUN} or {PIPELINE_CALIBRANTS} not available",
)


@pytest.fixture(scope="module")
def real_result():
    pytest.importorskip("alpinac.formula", reason="needs the alpinac of gitlabs/alpinac")
    return calibrate_run(RUN, PIPELINE_CALIBRANTS)


@real_data
def test_real_calibration(real_result):
    windows = real_result.windows
    assert windows["ok"].all() and len(windows) == 24
    assert (windows["n_used"] >= 18).all() and (windows["rms_ppm"] < 10).all()
    # most calibrants are used in every window (some air ions only part of the run)
    used = real_result.peaks.groupby("calibrant")["used"].sum()
    assert (used >= 20).sum() >= 18


@real_data
def test_real_against_legacy(real_result):
    """Same calibration as the legacy alpinac, except the electron mass (it used neutral masses).

    Compared away from the jump of the calibration at ~1800 s, where the
    windows of the two calibrations are not at the same rts. From 400 to 800
    s only on average: there N2, O2 and F are 40-80 ppm off the power
    calibration (both codes leave N2 and O2 out), and the calibration depends
    on which of them is kept (up to ~16 ppm at m/z 28, ~5 ppm at m/z 131).
    Below m/z 69 the calibration is extrapolated from few ions: only on average.
    """
    from alpinac.elements import ELECTRON_MASS

    legacy = MassCalibration.from_legacy_file(LEGACY_MC)
    cal = real_result.calibration
    rt = np.r_[np.linspace(200, 1200, 6), np.linspace(2000, 2700, 4)][:, None]
    mz = np.array([28.0, 44.0, 69.0, 131.0, 219.0, 293.0])
    tof = legacy.tof(rt, mz)
    neutral = cal.mz(rt, tof) + ELECTRON_MASS  # ions of charge 1
    diff = np.abs((neutral - mz) / mz * 1e6)
    clear = ((rt < 400) | (rt > 800)).ravel()
    assert np.median(diff) < 1.5 and np.median(diff[:, :2]) < 2
    assert diff[clear][:, 2:].max() < 4
    # the drift over the run is the same (~70 ppm at m/z 150)
    np.testing.assert_allclose(np.ptp(cal.drift_ppm(150.0, rt)), np.ptp(legacy.drift_ppm(150.0, rt)), atol=5)
    # peak shape (the legacy file has narrower high-mass peaks: there its fits of the
    # Re and PFTBA peaks stopped at the lower sigma bound)
    np.testing.assert_allclose(cal.sigma([20000, 40000]), legacy.sigma([20000, 40000]), rtol=0.08)
