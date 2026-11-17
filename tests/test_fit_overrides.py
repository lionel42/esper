"""Fit overrides: other fit parameters for the peaks of a target on one mass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import peaxart as px
import pytest
from test_fitting import (  # the synthetic run of the fit tests
    DT,
    FIT_SHAPE,
    T0,
    _measurement,
    peak,
)

from esper import db
from esper.fit_overrides import (
    FitOverride,
    OverrideParams,
    delete_override,
    fit_window,
    read_overrides,
    run_overrides,
    runs_covered,
    save_override,
)
from esper.fitting import fit_measurement

BASE = px.ChromatogramFitConfig(peak_shape=FIT_SHAPE)
E_RT = T0 + 140.0  # the run of test_fitting: two peaks on m/z 31, 140 and 145 s


@pytest.fixture(autouse=True)
def pap_papai_fit(monkeypatch):
    """As the fit tests: the synthetic peaks are Pap-Papai peaks."""
    from esper import fitting

    monkeypatch.setattr(fitting, "FIT_CONFIG", BASE)


def test_params_json():
    params = OverrideParams(
        backgrounds=("linear",), positions=(-1.5, 2.0), position_tolerance=0.0, min_peaks=2
    )
    assert OverrideParams.from_json(params.to_json()) == params
    assert params.keeps_all_peaks
    assert not OverrideParams().keeps_all_peaks
    # unknown keys (a later version) ignored; saved before the absolute widths: the
    # relative ones (as they were fitted)
    assert OverrideParams.from_json('{"peak_shape": "emg", "new": 1}') == OverrideParams(
        sigma_range_s=None
    )
    assert OverrideParams.from_config(BASE).peak_shape == FIT_SHAPE


def test_applies_to_runs():
    override = FitOverride(
        "T", 31, -5, 5, valid_from="2025-01-01T00:00:00", sample_types=("air",)
    )
    assert override.applies("2025-03-01 10:00:00", "air")
    assert override.applies(pd.Timestamp("2030-01-01"), "air")  # no end: future runs
    assert not override.applies("2024-12-31 23:59:00", "air")
    assert not override.applies("2025-03-01 10:00:00", "std")
    assert not override.applies(None, "air")  # date unknown
    bounded = FitOverride("T", 31, -5, 5, valid_to="2025-02-01T00:00:00")
    assert bounded.applies("2025-01-31 23:00:00", "std")
    assert bounded.applies("2025-02-01 00:00:00", "std")  # the last run: included
    assert not bounded.applies("2025-02-01 00:00:01", "std")
    assert FitOverride("T", 31, -5, 5).applies(None, None)
    # one per target and mass: the latest (first in the list)
    newer = FitOverride("T", 31, -2, 2, override_id=2)
    older = FitOverride("T", 31, -5, 5, override_id=1)
    other = FitOverride("T", 32, -5, 5, override_id=3)
    assert run_overrides([newer, older, other], None, "air") == [newer, other]


def _two_peaks():
    t = T0 + np.arange(120, 160, DT)
    rng = np.random.default_rng(1)
    y = 500 + peak(t, 5e4, E_RT) + peak(t, 3e4, E_RT + 5) + rng.normal(0, 30, len(t))
    return t, y


def test_fit_window_number_of_peaks():
    t, y = _two_peaks()
    # one start: at least 2 peaks, the second where the residual is largest
    params = OverrideParams(peak_shape=FIT_SHAPE, backgrounds=("constant",), min_peaks=2)
    fitted = fit_window(t, y, [E_RT], params, BASE)
    assert len(fitted) == 2
    assert sorted(fitted["rt"]) == pytest.approx([E_RT, E_RT + 5], abs=0.3)
    assert {"signal_height", "signal_prominence", "shape"} <= set(fitted.columns)
    # at most 1: one peak
    params = OverrideParams(peak_shape=FIT_SHAPE, backgrounds=("constant",), max_peaks=1)
    assert len(fit_window(t, y, [E_RT, E_RT + 5], params, BASE)) == 1


def test_fit_window_positions():
    t, y = _two_peaks()
    # positions given: exactly them, each kept near its start
    params = OverrideParams(
        peak_shape=FIT_SHAPE,
        backgrounds=("constant",),
        positions=(0.0, 5.0),
        position_tolerance=0.2,
    )
    fitted = fit_window(t, y, [E_RT + 0.1, E_RT + 4.9], params, BASE)
    assert len(fitted) == 2
    assert sorted(fitted["rt"]) == pytest.approx([E_RT, E_RT + 5], abs=0.5)


def test_overrides_stored_and_applied(tmp_path):
    con = _measurement(tmp_path)
    con.execute("UPDATE measurement SET acq_datetime = '2025-01-01 00:00:00',"
                " sample_type = 'air', peak_source = 'esper'")
    con.execute("UPDATE compound SET compound_name = 'E-target', rt_mean_s = ?"
                " WHERE compound_idx = 4", (E_RT,))
    con.execute("INSERT INTO target (name, rt_ref_s) VALUES ('E-target', ?)", (E_RT,))
    con.commit()
    fit_measurement(con, 1, tmp_path, workers=1)
    default = pd.read_sql("SELECT * FROM peak WHERE mass = 31", con)
    assert len(default) == 2 and default["fit_override_id"].isna().all()

    # one peak on m/z 31 around E: the window replaced by the fit of the override
    override = FitOverride(
        "E-target",
        31,
        -10,
        15,
        OverrideParams(peak_shape=FIT_SHAPE, backgrounds=("constant",), max_peaks=1),
        sample_types=("air",),
    )
    override_id = save_override(con, override)
    (stored,) = read_overrides(con)
    assert stored.override_id == override_id and stored.params == override.params
    assert runs_covered(con, [stored]) == [(1, "m")]
    assert runs_covered(con, [FitOverride("E-target", 31, -1, 1, sample_types=("std",))]) == []
    fit_measurement(con, 1, tmp_path, workers=1)
    on_31 = pd.read_sql("SELECT * FROM peak WHERE mass = 31", con)
    assert len(on_31) == 1
    assert on_31["fit_override_id"].tolist() == [override_id]
    # the other masses: the default fit
    others = pd.read_sql("SELECT * FROM peak WHERE mass != 31", con)
    assert others["fit_override_id"].isna().all() and others["fit_rt_s"].notna().any()

    # deleted: fitted again, the default fit back
    delete_override(con, override_id)
    fit_measurement(con, 1, tmp_path, workers=1)
    again = pd.read_sql("SELECT * FROM peak WHERE mass = 31", con)
    assert len(again) == 2 and again["fit_override_id"].isna().all()
    con.close()


def test_override_without_reference_rt_not_applied(tmp_path):
    con = _measurement(tmp_path)
    con.execute("INSERT INTO target (name) VALUES ('Absent')")
    save_override(con, FitOverride("Absent", 31, -10, 10, OverrideParams(max_peaks=1)))
    fit_measurement(con, 1, tmp_path, workers=1)
    assert pd.read_sql("SELECT * FROM peak WHERE mass = 31", con)[
        "fit_override_id"
    ].isna().all()
    assert db.SCHEMA_VERSION >= 11
    con.close()


def test_fit_window_keeps_every_position():
    """The positions are approximate starts, one peak each: kept even where a peak is
    not significant (three starts on one peak), their S/N as peaxart computes it."""
    t = T0 + np.arange(120, 160, DT)
    rng = np.random.default_rng(2)
    y = 500 + peak(t, 5e4, E_RT) + rng.normal(0, 30, len(t))
    params = OverrideParams(
        peak_shape=FIT_SHAPE, backgrounds=("constant",), positions=(-1.5, 0.0, 1.5)
    )
    fitted = fit_window(t, y, [E_RT - 1.5, E_RT, E_RT + 1.5], params, BASE)
    assert len(fitted) == 3
    assert np.isfinite(fitted["sn"]).all() and (fitted["sn"] > 0).all()
    # without positions, the default tests: one peak
    single = fit_window(t, y, [E_RT], OverrideParams(peak_shape=FIT_SHAPE), BASE)
    assert len(single) == 1


def test_fit_window_diagnostics():
    """What peaxart drops is recorded: the background kept, the iterations; peaxart's
    functions are restored after."""
    import peaxart.rtfit as rtfit

    from esper.fit_overrides import DIAGNOSTIC_COLUMNS

    original = rtfit.fit
    t, y = _two_peaks()
    params = OverrideParams(peak_shape=FIT_SHAPE, backgrounds=("constant", "linear"))
    fitted = fit_window(t, y, [E_RT, E_RT + 5], params, BASE)
    assert set(DIAGNOSTIC_COLUMNS) <= set(fitted.columns)
    assert set(fitted["background"]) <= {"constant", "linear"}
    assert (fitted["iterations"] > 0).all()
    # peaxart leaves out the end of the window (and outliers)
    assert fitted["n_points"].between(len(t) - 5, len(t)).all()
    assert fitted.attrs["n_least_squares"] >= 1
    assert fitted.attrs["iterations_total"] >= fitted["iterations"].max()
    assert rtfit.fit is original


def test_broad_peak_late_in_the_run():
    """A broad peak after 1000 s: peaxart's expected width at 1500 s is ~1 s, its bounds
    (x 5) too narrow for sigma 6 s; the absolute range of the override fits it."""
    t = np.arange(1440, 1560, DT)
    rng = np.random.default_rng(3)
    y = 500 + 2e4 * np.exp(-((t - 1500) ** 2) / (2 * 6.0**2)) + rng.normal(0, 30, len(t))
    absolute = OverrideParams(peak_shape=FIT_SHAPE, backgrounds=("constant",))
    fitted = fit_window(t, y, [1500.0], absolute, BASE)
    assert fitted["rt_sigma"].iloc[0] == pytest.approx(6.0, rel=0.1)
    assert fitted.attrs["start_sigma"] == pytest.approx(6.0, rel=0.15)
    relative = OverrideParams(peak_shape=FIT_SHAPE, backgrounds=("constant",), sigma_range_s=None)
    capped = fit_window(t, y, [1500.0], relative, BASE)
    assert capped["rt_sigma"].iloc[0] < 5.0
