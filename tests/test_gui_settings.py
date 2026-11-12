"""The GUI settings: their choices (enums) and the panels shown, kept across sessions."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtCore = pytest.importorskip("PySide6.QtCore")

from esper.gui import settings as settings_module  # noqa: E402
from esper.gui.panels.base import BasePanel  # noqa: E402
from esper.gui.settings import (  # noqa: E402
    CurveStyle,
    Integration,
    Intensity,
    Settings,
    SettingsManager,
)
from esper.gui.views.base import BaseView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """The settings of the application, in a temporary file."""
    manager = SettingsManager(tmp_path / "settings.json")
    monkeypatch.setattr(settings_module, "_manager", manager)
    return manager


def test_choices_from_json():
    loaded = Settings.from_json(
        json.dumps(
            {
                "chromatogram_integration": "ips",  # name of an earlier version
                "chromatogram_style": "bogus",
                "spectrum_intensity": "fit_area",
            }
        )
    )
    assert loaded.chromatogram_integration is Integration.EXTRACTION
    assert loaded.chromatogram_style is CurveStyle.LINES  # the default
    assert loaded.spectrum_intensity is Intensity.FIT_AREA
    assert Settings.from_json(loaded.to_json()) == loaded


def test_panels_from_json():
    loaded = Settings.from_json(
        json.dumps({"panels": {"View": {"A": False, "B": "yes"}, "Other": 3}})
    )
    assert loaded.panels == {"View": {"A": False}}
    assert Settings.from_json(json.dumps({"panels": []})).panels == {}


class _Selection(QtCore.QObject):
    changed = QtCore.Signal(object)


class _Panel(BasePanel):
    def __init__(self, title: str, selection) -> None:
        self.title = title
        super().__init__(None, selection)

    def redraw(self) -> None:
        pass


class _View(BaseView):
    title = "View"

    def __init__(self) -> None:
        selection = _Selection()
        super().__init__(None, selection)
        self.main = self.add_panel(_Panel("Main", selection))
        self.shown = self.add_panel(_Panel("Shown", selection), optional=True)
        self.hidden = self.add_panel(
            _Panel("Hidden", selection), optional=True, visible=False
        )


def test_panels_remembered(app, manager):
    view = _View()
    assert not view.shown.isHidden() and view.hidden.isHidden()  # the defaults
    shown, hidden = view.panel_actions()
    shown.setChecked(False)
    hidden.setChecked(True)
    assert manager.settings.panels == {"View": {"Shown": False, "Hidden": True}}

    # Saved: the next session starts with them
    settings_module._manager = SettingsManager(manager.path)
    again = _View()
    assert again.shown.isHidden() and not again.hidden.isHidden()
    assert [a.isChecked() for a in again.panel_actions()] == [False, True]
