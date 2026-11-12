"""User settings of the GUI, saved as JSON in the user's config directory."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from esper.libraries import Library, LibraryCollection, Processing, default_libraries

logger = logging.getLogger(__name__)

SETTINGS_FILE = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "esper"
    / "settings.json"
)

# Number of data directories remembered
MAX_RECENT_DATA_DIRS = 10


class Intensity(StrEnum):
    """Intensity of the measured peaks used for the spectra (a column of the peaks)."""

    PROMINENCE = "prominence"
    HEIGHT = "height"
    AREA = "area"  # integrated in the chromatograms, from the ips
    FIT_AREA = "fit_area"


class Integration(StrEnum):
    """Peak integration drawn in the chromatograms."""

    FIT = "fit_integration"  # of the fit (esper.fitting): bases, baseline and area
    FIT_SHAPES = "fit_shapes"  # the fitted peak shapes, filled
    EXTRACTION = "extraction"  # of the extraction (alpinac), from its ips
    NONE = "none"

    @property
    def of_fit(self) -> bool:
        """Drawn from the fit (none for a measurement not fitted)."""
        return self in (Integration.FIT, Integration.FIT_SHAPES)

    @property
    def text(self) -> str:
        """In the menus."""
        return _INTEGRATION_TEXTS[self]


_INTEGRATION_TEXTS = {
    Integration.FIT: "fit: integration",
    Integration.FIT_SHAPES: "fit: peak shapes",
    Integration.EXTRACTION: "extraction (ips)",
    Integration.NONE: "none",
}
# Names of an earlier version
OLD_INTEGRATIONS = {
    "integration": Integration.FIT,
    "fit": Integration.FIT_SHAPES,
    "ips": Integration.EXTRACTION,
}


class CurveStyle(StrEnum):
    """How the chromatograms are drawn: lines, or each value as a point or a cross."""

    LINES = "lines"
    POINTS = "points"
    CROSSES = "crosses"

    @property
    def text(self) -> str:
        """In the menus."""
        return self.value


def _choice(value, default: StrEnum) -> StrEnum:
    """``value`` (of a settings file) as a member of the enum of ``default``, ``default``
    if it is none of them."""
    try:
        return type(default)(value)
    except ValueError:
        return default


@dataclass
class Settings:
    # Spectrum of the measured compounds
    spectrum_intensity: Intensity = Intensity.PROMINENCE
    # Peak integration drawn in the chromatograms
    chromatogram_integration: Integration = Integration.FIT
    # How the chromatograms are drawn
    chromatogram_style: CurveStyle = CurveStyle.LINES
    # Spectral libraries, by priority
    libraries: list[Library] = field(default_factory=default_libraries)
    # Matching (alpinac's processing)
    cutoff_mass: float = 19.0
    ignored_masses: list[float] = field(default_factory=list)
    normalize: bool = True
    n_best_matches: int = 20
    # m/z whose drift is shown in the mass calibration view
    mass_drift_masses: list[float] = field(
        default_factory=lambda: [20.0, 50.0, 100.0, 230.0]
    )
    # Data directories opened before, most recent first
    recent_data_dirs: list[str] = field(default_factory=list)
    # Optional panels shown (true) or hidden (false) by the user: view -> panel ->
    # shown (titles; the panels not in it have their default)
    panels: dict[str, dict[str, bool]] = field(default_factory=dict)
    # Results panels comparing compounds (true), by title
    compare: dict[str, bool] = field(default_factory=dict)

    @property
    def processing(self) -> Processing:
        return Processing(self.cutoff_mass, list(self.ignored_masses), self.normalize)

    def to_json(self) -> str:
        data = asdict(self)
        data["libraries"] = [
            {"name": lib.name, "path": str(lib.path), "enabled": lib.enabled}
            for lib in self.libraries
        ]
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, text: str) -> Settings:
        data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "libraries" in known:
            known["libraries"] = [
                Library(d["name"], Path(d["path"]), d.get("enabled", True))
                for d in known["libraries"]
            ]
        settings = cls(**known)
        if not isinstance(settings.panels, dict):
            settings.panels = {}
        settings.panels = {
            view: {p: v for p, v in shown.items() if isinstance(v, bool)}
            for view, shown in settings.panels.items()
            if isinstance(shown, dict)
        }
        if not isinstance(settings.compare, dict):
            settings.compare = {}
        settings.compare = {
            p: v for p, v in settings.compare.items() if isinstance(v, bool)
        }
        if not settings.mass_drift_masses:
            settings.mass_drift_masses = Settings().mass_drift_masses
        settings.chromatogram_integration = OLD_INTEGRATIONS.get(
            settings.chromatogram_integration, settings.chromatogram_integration
        )
        # the choices (fields with an enum default) as members, their default if invalid
        for f in fields(cls):
            if isinstance(f.default, StrEnum):
                setattr(settings, f.name, _choice(getattr(settings, f.name), f.default))
        return settings


class SettingsManager(QObject):
    """The current settings and the libraries built from them.

    ``changed`` is emitted after ``update``; the libraries are rebuilt (and
    read again lazily) only when the library list or the processing changed.
    """

    changed = Signal()

    def __init__(self, path: Path = SETTINGS_FILE) -> None:
        super().__init__()
        self.path = path
        self.settings = self._load()
        self._libraries: LibraryCollection | None = None

    def _load(self) -> Settings:
        if self.path.is_file():
            try:
                return Settings.from_json(self.path.read_text())
            except Exception as e:
                logger.error(f"Could not read the settings {self.path}: {e}")
        return Settings()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.settings.to_json())
        except OSError as e:
            logger.error(f"Could not save the settings {self.path}: {e}")

    def remember_data_dir(self, data_dir: Path) -> None:
        """Put a data directory first in the recent ones (saved, no ``changed`` signal)."""
        path = str(Path(data_dir).resolve())
        recent = [p for p in self.settings.recent_data_dirs if p != path]
        self.settings.recent_data_dirs = [path] + recent[: MAX_RECENT_DATA_DIRS - 1]
        self.save()

    def forget_data_dir(self, data_dir: str) -> None:
        self.settings.recent_data_dirs = [
            p for p in self.settings.recent_data_dirs if p != data_dir
        ]
        self.save()

    def panel_shown(self, view: str, panel: str) -> bool | None:
        """Whether the user showed a panel of a view (None: never toggled it)."""
        return self.settings.panels.get(view, {}).get(panel)

    def remember_panel(self, view: str, panel: str, shown: bool) -> None:
        """Remember that the user showed or hid a panel of a view (saved, no
        ``changed`` signal)."""
        self.settings.panels = {
            **self.settings.panels,
            view: {**self.settings.panels.get(view, {}), panel: shown},
        }
        self.save()

    def compare_compounds(self, panel: str) -> bool:
        """Whether a results panel compares compounds."""
        return self.settings.compare.get(panel, False)

    def remember_compare_compounds(self, panel: str, compare: bool) -> None:
        """Remember whether a results panel compares compounds (saved, no ``changed``
        signal)."""
        if self.compare_compounds(panel) != compare:
            self.settings.compare = {**self.settings.compare, panel: compare}
            self.save()

    def update(self, settings: Settings) -> None:
        old = self.settings
        self.settings = settings
        if self._libraries is not None:
            if [asdict(lib) for lib in old.libraries] != [
                asdict(lib) for lib in settings.libraries
            ]:
                self._libraries = None
            elif old.processing != settings.processing:
                self._libraries = self._libraries.with_processing(settings.processing)
        self.save()
        self.changed.emit()

    @property
    def libraries(self) -> LibraryCollection:
        if self._libraries is None:
            self._libraries = LibraryCollection(
                list(self.settings.libraries), self.settings.processing
            )
        return self._libraries


_manager: SettingsManager | None = None


def settings_manager() -> SettingsManager:
    """The settings of the application (created on first use)."""
    global _manager
    if _manager is None:
        _manager = SettingsManager()
    return _manager
