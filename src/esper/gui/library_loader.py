"""Read the spectral libraries in a background thread, without blocking the GUI."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal

from esper.gui.settings import settings_manager
from esper.libraries import LibraryCollection

logger = logging.getLogger(__name__)


class _LoadThread(QThread):
    progress = Signal(str, int, int)  # library, files done, files total

    def __init__(self, collection: LibraryCollection, libraries: list[str]) -> None:
        super().__init__()
        self.collection = collection
        self.libraries = libraries

    def run(self) -> None:
        for name in self.libraries:
            try:
                self.collection.load(name, lambda done, total, name=name: self.progress.emit(name, done, total))
            except Exception:
                logger.exception(f"Could not read library {name}")


class LibraryLoader(QObject):
    """Loads the libraries of the current settings in the background.

    Panels call ``ensure()``: True if the libraries are ready, otherwise the
    loading starts (if not running yet) and ``loaded`` is emitted at the end,
    when they can redraw.
    """

    progress = Signal(str, int, int)  # library, files done, files total
    busy = Signal(bool)
    loaded = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._thread: _LoadThread | None = None
        self._pending_all = False

    @property
    def libraries(self) -> LibraryCollection:
        return settings_manager().libraries

    def ensure(self, all_libraries: bool = False) -> bool:
        """Are the enabled (or all) libraries read? If not, start reading them."""
        missing = self.libraries.missing(all_libraries)
        if not missing:
            return True
        if self._thread is not None:
            # Running: read the others afterwards
            self._pending_all |= all_libraries
            return False
        self._thread = _LoadThread(self.libraries, missing)
        self._thread.progress.connect(self.progress)
        self._thread.finished.connect(self._on_finished)
        self.busy.emit(True)
        self._thread.start()
        return False

    def _on_finished(self) -> None:
        self._thread.deleteLater()
        self._thread = None
        pending_all, self._pending_all = self._pending_all, False
        # Libraries changed in the settings meanwhile, or more were asked for
        if self.libraries.missing(pending_all):
            self.ensure(pending_all)
        if self._thread is None:
            self.busy.emit(False)
        self.loaded.emit()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.wait()


_loader: LibraryLoader | None = None


def library_loader() -> LibraryLoader:
    global _loader
    if _loader is None:
        _loader = LibraryLoader()
    return _loader
