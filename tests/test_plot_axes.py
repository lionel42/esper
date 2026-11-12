"""The axes of the results plot: a variable of an element (measurement, compound, ion)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from esper import db  # noqa: E402
from esper.gui.data import EsperData  # noqa: E402
from esper.gui.plot_axes import (  # noqa: E402
    COMPOUND,
    MEASUREMENT,
    Axis,
    AxisSelector,
    ion_element,
    results_elements,
)

# (mass, prominence) of the peaks of the compounds of the run, by name.
# CFC-12 is a target with the ions 85, 87, 101 (main ion first)
PEAKS = {
    "CFC-12": [(85.0, 10.0), (84.9, 5.0), (87.0, 20.0), (35.0, 50.0)],
    "Unknown X": [(35.0, 50.0), (85.0, 10.0), (87.0, 20.0)],
}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def data(tmp_path):
    con = db.connect(tmp_path, create=True)
    con.execute(
        "INSERT INTO measurement (measurement_id, name, acq_datetime, source_dir,"
        " source_mtime, source_size, imported_at)"
        " VALUES (1, '240101.1200.air.1', '2024-01-01T12:00:00', '', 0, 0, '')"
    )
    peak_idx = 0
    for idx, (name, peaks) in enumerate(PEAKS.items()):
        con.execute(
            "INSERT INTO compound (measurement_id, compound_idx, rt_mean_s,"
            " valid_candidate, compound_name, prominence_sum) VALUES (1, ?, 300, 1, ?, 1)",
            (idx, name),
        )
        for mass, prominence in peaks:
            con.execute(
                "INSERT INTO peak (measurement_id, peak_idx, compound_idx, mass, rt_s,"
                " height, prominence) VALUES (1, ?, ?, ?, 300, ?, ?)",
                (peak_idx, idx, mass, 2 * prominence, prominence),
            )
            peak_idx += 1
    con.execute("INSERT INTO target (name, ri) VALUES ('CFC-12', NULL)")
    con.executemany(
        "INSERT INTO target_ion (name, rank, mass) VALUES ('CFC-12', ?, ?)",
        [(0, 85), (1, 87), (2, 101)],
    )
    con.commit()
    con.close()
    data = EsperData(tmp_path)
    yield data
    data.close()


def test_ions_of_a_target(data):
    df = data.results("CFC-12", valid_only=False)
    row = df.iloc[0]
    # Its ions in rank order, on their most prominent peak (84.9 is on 85); none on 101
    assert (row["ion1_mass"], row["ion1_prominence"]) == (85.0, 10.0)
    assert (row["ion2_mass"], row["ion2_prominence"]) == (87.0, 20.0)
    assert "ion3_mass" not in df  # no peak in any run
    assert row["ion2_ratio_prominence"] == 2.0
    assert row["n_peaks"] == 4


def test_ions_of_another_compound(data):
    row = data.results("Unknown X", valid_only=False).iloc[0]
    # Its masses by prominence
    assert [row[f"ion{k}_mass"] for k in (1, 2, 3)] == [35.0, 87.0, 85.0]
    assert row["ion3_ratio_height"] == pytest.approx(0.2)


def test_axis_values_and_labels(data):
    df = data.results("CFC-12", valid_only=False)
    time = Axis(MEASUREMENT, MEASUREMENT.variable("time"))
    assert time.time and time.label == "measurement time"
    assert time.values(df)[0] == pd.Timestamp("2024-01-01T12:00:00").timestamp()
    assert time.format(time.values(df)[0]) == "2024-01-01 12:00"
    ratio = Axis(ion_element(2), ion_element(2).variable("prominence / main ion"))
    assert ratio.label == "2nd ion prominence / main ion"
    assert ratio.values(df)[0] == 2.0


def test_variables_of_the_elements():
    # A ratio to the main ion: only of the other ions
    assert ion_element(1).variable("fit area / main ion") is None
    assert ion_element(3).variable("fit area / main ion") is not None
    # Of an ion, not of the compound, and the other way round
    assert COMPOUND.variable("fit sigma [s]") is None
    assert ion_element(1).variable("score") is None


ELEMENTS = (MEASUREMENT, COMPOUND, *(ion_element(k) for k in (1, 2, 3)))


def test_elements_of_the_compound(data):
    # The target: its ions with a peak (none on 101), with their mass
    elements = results_elements(data.results("CFC-12", valid_only=False))
    assert [e.label for e in elements] == [
        "Measurement",
        "Compound",
        "Main ion (m/z 85)",
        "2nd ion (m/z 87)",
    ]
    # Another compound: its masses
    elements = results_elements(data.results("Unknown X", valid_only=False))
    assert [e.key for e in elements[2:]] == ["ion1", "ion2", "ion3"]
    # Nothing loaded: the main ion still
    assert [e.key for e in results_elements(None)] == [
        "measurement",
        "compound",
        "ion1",
    ]


def test_selector_follows_the_elements(app, data):
    ratio = Axis(ion_element(3), ion_element(3).variable("fit area / main ion"))
    selector = AxisSelector("Y:", ELEMENTS, ratio)
    changes = []
    selector.changed.connect(changes.append)
    # The ion shown is not of the new compound: the main ion, no ratio: its first variable
    selector.set_elements(results_elements(data.results("CFC-12", valid_only=False)))
    assert selector.axis.label == "Main ion (m/z 85) prominence"
    assert selector.element_combo.count() == 4
    # Kept when the new compound has it (with the label of the new one)
    selector.set_axis(Axis(ion_element(2), ion_element(2).variable("height")))
    selector.set_elements(ELEMENTS)
    assert selector.axis.label == "2nd ion height"
    # Not an ion: kept as is
    selector.set_axis(Axis(MEASUREMENT, MEASUREMENT.variable("port")))
    selector.set_elements(results_elements(None))
    assert selector.axis.label == "measurement port"
    assert len(changes) == 4


def test_selector(app):
    selector = AxisSelector("Y:", ELEMENTS, Axis(COMPOUND, COMPOUND.variables[0]))
    changes = []
    selector.changed.connect(changes.append)
    # Another element with the same variable: kept
    selector.set_axis(
        Axis(ion_element(2), ion_element(2).variable("fit area / main ion"))
    )
    selector.element_combo.setCurrentIndex(selector.element_combo.findData("ion3"))
    assert selector.axis.label == "3rd ion fit area / main ion"
    # Not of the new element: its first variable
    selector.element_combo.setCurrentIndex(selector.element_combo.findData("ion1"))
    assert selector.axis == Axis(ion_element(1), ion_element(1).variables[0])
    assert [a.label for a in changes] == [
        "2nd ion fit area / main ion",
        "3rd ion fit area / main ion",
        "Main ion prominence",
    ]
    # Shown again: no change
    selector.set_axis(selector.axis)
    assert len(changes) == 3


def test_optional_selector(app):
    selector = AxisSelector(
        "Y2:", ELEMENTS, Axis(COMPOUND, COMPOUND.variables[0]), optional=True
    )
    changes = []
    selector.changed.connect(changes.append)
    assert selector.value is None and not selector.variable_combo.isEnabled()
    selector.set_checked(True)
    assert selector.value == Axis(COMPOUND, COMPOUND.variables[0])
    selector.set_checked(False)
    assert changes == [Axis(COMPOUND, COMPOUND.variables[0]), None]


def test_other_y_axes(app, data):
    from esper.gui.panels.results_plot import ResultsPlotPanel
    from esper.gui.selection import Selection

    selection = Selection(data)
    panel = ResultsPlotPanel(data, selection, title="Results 2")
    assert panel.title == "Results 2"
    selection.select_candidate("CFC-12")
    panel.redraw()  # loaded: the ions of CFC-12 in the selectors
    ratio = Axis(ion_element(2), ion_element(2).variable("prominence / main ion"))
    panel.selectors["y2"].set_axis(ratio)
    panel.selectors["y2"].set_checked(True)
    panel.redraw()
    right = panel.right_axes["y2"]
    assert right.axis is not None and list(right.plotted["y"]) == [2.0]
    assert panel.right_axes["y3"].axis is None
    assert "prominence / main ion" in panel.plot_title()
    # Unchecked: its axis removed
    panel.selectors["y2"].set_checked(False)
    panel.redraw()
    assert right.axis is None and right.plotted is None


def test_results_by_run(data):
    df = data.results_by_run(("CFC-12", "Unknown X"), valid_only=False)
    # One row per measurement, the columns of each compound
    assert len(df) == 1
    assert df.loc[0, "CFC-12|ion1_mass"] == 85.0
    assert df.loc[0, "Unknown X|ion1_mass"] == 35.0
    assert df.loc[0, "name"] == "240101.1200.air.1"
    axis = Axis(ion_element(1), ion_element(1).variable("prominence"), "Unknown X")
    assert axis.column == "Unknown X|ion1_prominence"
    assert axis.label == "Unknown X Main ion prominence"
    assert axis.values(df)[0] == 50.0


def test_compare_compounds(app, data, tmp_path, monkeypatch):
    from esper.gui import settings as settings_module
    from esper.gui.panels.results_plot import ResultsPlotPanel
    from esper.gui.selection import Selection
    from esper.gui.settings import SettingsManager

    manager = SettingsManager(tmp_path / "settings.json")
    monkeypatch.setattr(settings_module, "_manager", manager)
    selection = Selection(data)
    selection.select_candidate("CFC-12")
    panel = ResultsPlotPanel(data, selection)
    panel.compare_check.setChecked(True)
    assert manager.compare_compounds("Results")
    panel.redraw()
    # The axes of a compound got the one selected; the measurement none
    assert panel.options["y"].compound == "CFC-12"
    assert panel.options["x"].compound is None
    # Another compound on y: its own ions
    panel.selectors["y"].set_compound("Unknown X")
    assert panel.selectors["y"].element_combo.count() == 5
    panel.redraw()
    assert panel._compounds() == ("Unknown X",)
    assert "Unknown X" in panel.plot_title() and len(panel.plotted) == 1
    # A click: that measurement and the compound of the series
    points = panel._of_compound(panel.plotted, "Unknown X")
    panel.select_point(tuple(points.iloc[0][["measurement_id", "compound_idx"]]))
    assert selection.label == "Unknown X"
    # Off: the compound selected again, no compound on the axes
    panel.compare_check.setChecked(False)
    panel.redraw()
    assert panel.options["y"].compound is None
    assert not manager.compare_compounds("Results")
