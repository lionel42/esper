"""The overlays of the chromatogram panels: one state, a 'Masses' menu in each plot."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pg = pytest.importorskip("pyqtgraph")

from esper.gui.panels.chromatogram import DEFAULT_OVERLAYS, OverlayOptions  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_rank3_checks_rank2():
    overlays = OverlayOptions()
    assert overlays.checked == DEFAULT_OVERLAYS
    changes = []
    overlays.changed.connect(lambda: changes.append(set(overlays.checked)))
    overlays.set("rank3", True)
    assert {"rank2", "rank3"} <= overlays.checked
    # The 2nd mass can then be unchecked alone
    overlays.set("rank2", False)
    assert "rank3" in overlays and "rank2" not in overlays
    # No change, no signal
    overlays.set("rank2", False)
    assert len(changes) == 2


def test_menus_in_sync(app):
    overlays = OverlayOptions()
    plots = [pg.PlotWidget(), pg.PlotWidget()]
    first, second = (overlays.add_menu(plot) for plot in plots)
    second["all"].setChecked(True)
    assert "all" in overlays
    assert first["all"].isChecked()
    first["rank3"].setChecked(True)
    assert second["rank2"].isChecked() and second["rank3"].isChecked()


def test_menu_of_a_deleted_plot(app):
    overlays = OverlayOptions()
    kept = pg.PlotWidget()
    checks = overlays.add_menu(kept)
    gone = pg.PlotWidget()
    overlays.add_menu(gone)
    gone.deleteLater()
    del gone
    QtWidgets.QApplication.sendPostedEvents(None, 0)  # deleteLater
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    overlays.set("tic", True)  # no RuntimeError from the deleted checkboxes
    assert checks["tic"].isChecked()


def test_centered_text_lines(app):
    """The lines of a peak label are centered on its position (Qt centers them only in a
    fixed text width)."""
    from esper.gui.plotting import centered_text

    item = centered_text("a long first line<br>x")
    document = item.textItem.document()
    assert item.textItem.textWidth() == pytest.approx(document.idealWidth())
    rect = item.textItem.mapRectToParent(item.textItem.boundingRect())
    assert rect.center().x() == pytest.approx(0, abs=1)


def test_masses_instead_of_main():
    """The masses of a compound drawn as the main one when it has no peak there (a target
    found on another of its ions), most prominent first."""
    import pandas as pd

    from esper.gui.panels.chromatogram import masses_instead_of_main

    peaks = pd.DataFrame(
        {
            "compound_idx": [0, 0, 1, 1],
            "mass": [117.0, 119.0, 82.0, 117.0],
            "prominence": [100.0, 300.0, 50.0, 200.0],
        }
    )
    # compound 0: no peak on the main ion 82
    assert masses_instead_of_main(peaks, (0,), {82}) == [119, 117]
    # compound 1 has one: nothing instead
    assert masses_instead_of_main(peaks, (1,), {82}) == []
    # a GCwerks scan: the main unit mass stands for the masses measured around it
    assert masses_instead_of_main(peaks, (0,), {117, 116.9, 117.1}) == []
    assert masses_instead_of_main(peaks, (2,), {82}) == []
