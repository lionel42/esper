"""User settings of the GUI, saved as JSON in the user's config directory."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from esper.libraries import Library, LibraryCollection, Processing, default_libraries

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "esper" / "settings.json"

# Number of data directories remembered
MAX_RECENT_DATA_DIRS = 10

# Intensity of the measured peaks used for the spectra
INTENSITIES = ["prominence", "height", "area"]


@dataclass
class Settings:
    # Spectrum of the measured compounds
    spectrum_intensity: str = "prominence"
    # Spectral libraries, by priority
    libraries: list[Library] = field(default_factory=default_libraries)
    # Matching (alpinac's processing)
    cutoff_mass: float = 19.0
    ignored_masses: list[float] = field(default_factory=list)
    normalize: bool = True
    n_best_matches: int = 20
    # Data directories opened before, most recent first
    recent_data_dirs: list[str] = field(default_factory=list)

    @property
    def processing(self) -> Processing:
        return Processing(self.cutoff_mass, list(self.ignored_masses), self.normalize)

    def to_json(self) -> str:
        data = asdict(self)
        data["libraries"] = [{"name": lib.name, "path": str(lib.path), "enabled": lib.enabled} for lib in self.libraries]
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, text: str) -> Settings:
        data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "libraries" in known:
            known["libraries"] = [Library(d["name"], Path(d["path"]), d.get("enabled", True)) for d in known["libraries"]]
        settings = cls(**known)
        if settings.spectrum_intensity not in INTENSITIES:
            settings.spectrum_intensity = INTENSITIES[0]
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
        self.settings.recent_data_dirs = [p for p in self.settings.recent_data_dirs if p != data_dir]
        self.save()

    def update(self, settings: Settings) -> None:
        old = self.settings
        self.settings = settings
        if self._libraries is not None:
            if [asdict(lib) for lib in old.libraries] != [asdict(lib) for lib in settings.libraries]:
                self._libraries = None
            elif old.processing != settings.processing:
                self._libraries = self._libraries.with_processing(settings.processing)
        self.save()
        self.changed.emit()

    @property
    def libraries(self) -> LibraryCollection:
        if self._libraries is None:
            self._libraries = LibraryCollection(list(self.settings.libraries), self.settings.processing)
        return self._libraries


_manager: SettingsManager | None = None


def settings_manager() -> SettingsManager:
    """The settings of the application (created on first use)."""
    global _manager
    if _manager is None:
        _manager = SettingsManager()
    return _manager
