"""esper chromatograms: unit mass chromatograms of a run (read with twutils, peaxart)."""

from __future__ import annotations

import numpy as np
import peaxart as px
import pytest
import twutils
from conftest import CAL, RT_APEX, S_PER_BIN, STEP_EJECTION, STEP_EMISSION, write_h5
from masscal import tof_to_mass

from esper.chromatograms import (
    part_path,
    provenance,
    read_chromatograms,
    unit_mass_chromatograms,
)
from esper.cli.main import main


@pytest.fixture
def result(h5_file):
    return unit_mass_chromatograms(h5_file, masses=[68, 69, 70, 100], block=37)


def test_chromatograms(result):
    c = result.chromatograms
    assert list(c.columns) == ["68", "69", "70", "100", "tic"] and c.index.name == "rt"
    assert len(c) == result.file.n_scans
    for mass in ("69", "100"):  # the compound of the synthetic run
        assert c.index[np.argmax(c[mass].to_numpy())] == pytest.approx(RT_APEX, abs=2 * S_PER_BIN)
    assert c["69"].max() > 20 * c["68"].max()  # only noise around
    np.testing.assert_allclose(c["tic"], c.drop(columns="tic").sum(axis=1))
    assert (c.drop(columns="tic") >= 0).all().all()


def test_same_as_peaxart_in_memory(result, h5_file):
    """Read in blocks and summed with the bounds of each scan = peaxart on the whole run in memory
    (no TPS2 in the synthetic run: no cut)."""
    assert result.steps == {"emission": [], "ejection": []}
    run = twutils.TofFile(h5_file)
    intensity, _ = run.read()
    mz = tof_to_mass(np.arange(run.n_points), CAL, "power")  # the calibration is the same at every rt
    expected = px.unit_mass_chromatograms(intensity, mz, [68, 69, 70, 100])
    np.testing.assert_allclose(result.chromatograms.drop(columns="tic").to_numpy(), expected, rtol=1e-6, atol=1e-6)
    end = unit_mass_chromatograms(h5_file, masses=[68, 69, 70, 100], mass_axis="end")
    np.testing.assert_allclose(end.chromatograms.to_numpy(), result.chromatograms.to_numpy(), rtol=1e-9)


def test_write_and_read(result, h5_file):
    path = result.write()
    name = "241007.0423.air.1.unit_mass_chromatos.parquet"
    assert path == h5_file.parent / "241007.0423.air.1.alpinac" / name
    back = read_chromatograms(path)
    np.testing.assert_array_equal(back.to_numpy(), result.chromatograms.to_numpy())
    info = provenance(back)
    assert info["segments"] == [0, 1] and info["mass_axis"] == "drift"
    assert info["config"]["baseline"]["padding"] == "edge"
    assert info["calibration"].endswith("EI_mc.txt")
    raw, baseline = read_chromatograms(path, "raw"), read_chromatograms(path, "baseline")
    assert list(raw.columns) == list(baseline.columns) == ["68", "69", "70", "100", "tic"]
    assert (raw.to_numpy() > 0).all() and raw.dtypes.eq(np.float32).all()
    # corrected = smoothed raw - baseline (clipped): the baseline is below the raw signal, near it away from the peak
    assert (baseline["100"] <= raw["100"].max()).all()
    np.testing.assert_allclose(baseline["100"].median(), raw["100"].median(), rtol=0.05)


def test_rt_range_and_unwritten_scans(tmp_path):
    run = write_h5(tmp_path / "a.h5", unwritten=7)
    c = unit_mass_chromatograms(run, masses=[69]).chromatograms
    assert len(c) == twutils.TofFile(run).n_scans  # the empty scans are left out
    part = unit_mass_chromatograms(run, masses=[69], rt_start=10, rt_stop=50).chromatograms
    assert part.index[0] >= 10 and part.index[-1] <= 50


def test_default_masses_and_segments(tmp_path):
    run = write_h5(tmp_path / "a.h5", n_segments=2)
    with pytest.raises(ValueError, match="2 segments"):
        unit_mass_chromatograms(run)
    result = unit_mass_chromatograms(run, segments=(0, 2))
    assert result.default_path().name == "a.unit_mass_chromatos.parquet"
    columns = list(result.chromatograms.columns)
    assert all(c.startswith("seg0-1:") for c in columns) and columns[-1] == "seg0-1:tic"
    masses = [int(c.removeprefix("seg0-1:")) for c in columns[:-1]]
    assert masses[0] == 1 and masses == list(range(1, masses[-1] + 1))


def test_cli(h5_file, tmp_path, capsys):
    out = tmp_path / "c.parquet"
    args = ["chromatograms", str(h5_file), "--mass", "69", "100", "-o", str(out), "--legacy", "--no-raw"]
    assert main(args) == 0
    assert str(out) in capsys.readouterr().out
    info = provenance(read_chromatograms(out))
    assert info["mass_axis"] == "end" and info["config"]["baseline"]["padding"] == "zeros"
    assert info["config"]["baseline"]["lam"] == 1e5 and info["steps"] is None
    assert not part_path(out, "raw").exists()
    assert set(info["versions"]) == {"esper", "twutils", "peaxart", "masscal"}
    # several files and --output: refused; a bad file: exit code 1
    with pytest.raises(SystemExit):
        main(["chromatograms", str(h5_file), str(write_h5(tmp_path / "b.h5")), "-o", str(out)])
    assert main(["chromatograms", str(tmp_path / "missing.h5")]) == 1


def test_twutils_provenance(result, tmp_path):
    """The files written by twutils (before esper) still give their provenance."""
    c = result.chromatograms.copy()
    c.attrs = {"twutils": c.attrs["esper"]}
    c.to_parquet(tmp_path / "old.parquet")
    assert provenance(read_chromatograms(tmp_path / "old.parquet"))["mass_axis"] == "drift"


def test_steps_of_the_acquisition(tmp_path):
    """Emission step (all masses x10) and ejection of m/z 100, from TPS2: cuts at the scans of the steps."""
    run = write_h5(tmp_path / "a.h5", steps=True)
    changes = twutils.TofFile(run).tps2_changes(["FILAMENT_EMISSION target", "QUADA_REXAMPL1"])
    assert [(name[:10], round(old, 2), new) for name, _, old, new in changes] == [
        ("FILAMENT_E", 0.05, 0.5), ("QUADA_REXA", 0.0, 3.0), ("QUADA_REXA", 3.0, 0.0)]
    result = unit_mass_chromatograms(run, masses=[68, 69, 70, 100])
    assert result.steps["emission"] == [pytest.approx(STEP_EMISSION, abs=0.01)]
    [ejection] = result.steps["ejection"]
    assert ejection["masses"] == [100.0] and ejection["channels"] == ["QUADA_REXAMPL1"]
    assert ejection["rt"] == [pytest.approx(t + 0.4, abs=0.21) for t in STEP_EJECTION]
    c, raw, base = result.chromatograms, result.raw, result.baseline
    # the baseline follows the steps: only noise left in the corrected chromatograms around them
    for t in (STEP_EMISSION, *STEP_EJECTION):
        near = (c.index > t - 5) & (c.index < t + 5)
        assert c.loc[near, "100"].max() < 0.1 * raw.loc[near, "100"].max()
    inside = (base.index > STEP_EJECTION[0] + 1) & (base.index < STEP_EJECTION[1] - 1)
    assert base.loc[inside, "100"].median() < 0.05 * base.loc[~inside & (base.index > 25), "100"].median()
    # the peak is kept
    assert c.index[np.argmax(c["69"].to_numpy())] == pytest.approx(RT_APEX, abs=2 * S_PER_BIN)
    # without the cuts: the whole-run baseline lags behind the steps
    whole = unit_mass_chromatograms(run, masses=[100], steps=None).chromatograms
    after = (c.index > STEP_EJECTION[1]) & (c.index < STEP_EJECTION[1] + 5)
    assert whole.loc[after, "100"].sum() > 5 * c.loc[after, "100"].sum()
    info = provenance(c)
    assert info["steps"]["ejection"][0]["masses"] == [100.0] and info["steps"]["config"]["emission"]


def test_upload_copies(result, tmp_path):
    """Upload: the chromatograms copied with their raw sums and baselines; parts of an earlier
    import removed when the new source has none (legacy pipeline)."""
    from esper.alpinac import find_chromatograms
    from esper.sources import MeasurementImport, Run
    from esper.upload import store_chromatograms

    def store(chromatogram_file):
        run = Run(name, folder, None)
        imported = MeasurementImport(run, "tofwerk", "alpinac", chromatogram_file=chromatogram_file)
        return store_chromatograms(imported, data_dir)

    name = "241007.0423.air.1"
    folder = tmp_path / "upload" / f"{name}.alpinac"
    folder.mkdir(parents=True)
    assert find_chromatograms(folder, name) is None
    new = result.write(folder / f"{name}.unit_mass_chromatos.parquet")
    assert find_chromatograms(folder, name) == new
    legacy = tmp_path / "legacy" / f"{name}.unit_mass_chromatos.parquet"  # without raw sums and baselines
    legacy.parent.mkdir()
    result.chromatograms.to_parquet(legacy)

    data_dir = tmp_path / "data"
    stored = store(new)
    assert stored == f"chromatograms/{name}.parquet"
    copy = data_dir / stored
    for part in ("corrected", "raw", "baseline"):
        np.testing.assert_array_equal(read_chromatograms(copy, part).to_numpy(),
                                      read_chromatograms(new, part).to_numpy())
    assert provenance(read_chromatograms(copy))["segments"] == [0, 1]
    # back to the legacy file: no raw sums or baselines left from the earlier copy
    store(legacy)
    assert not part_path(copy, "raw").exists() and not part_path(copy, "baseline").exists()
    assert store(None) is None
