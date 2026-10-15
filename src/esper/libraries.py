"""Reference mass spectra libraries (peaksdb) and spectral matching.

A library is a folder of spectra, read like peaksdb's ``db_reader`` does:
NIST ``.jdx`` files (JCAMP-DX) and alpinac fragment files (``.txt``). Libraries are ordered by
priority: when a spectrum exists in several libraries, the first one is used.
They are read the first time they are needed and kept in memory.

Spectra are compared like in alpinac: unit masses, masses below a cutoff and
ignored masses removed, intensities normalized, cosine similarity
(equivalent to matchms ``CosineGreedy`` for unit masses).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PEAKSDB = Path("/shared/data/peaksdb")


@dataclass
class Library:
    name: str  # the prefix before '_' is its family, e.g. 'nist' for 'nist_ei'
    path: Path
    enabled: bool = True

    @property
    def family(self) -> str:
        return self.name.split("_", 1)[0]


def default_libraries() -> list[Library]:
    """mygu and the main NIST download first, the other peaksdb folders disabled."""
    return [
        Library("mygu", PEAKSDB / "myriam_ei", True),
        Library("nist_ei", PEAKSDB / "nist_db", True),
        Library("nist_ch", PEAKSDB / "nist_CH", False),
        Library("nist_chon", PEAKSDB / "nist_CHON", False),
        Library("nist_cf", PEAKSDB / "nist_CF", False),
        Library("nist_chf", PEAKSDB / "nist_CHF", False),
        Library("nist_auto", PEAKSDB / "nist_autodownloads", False),
    ]


@dataclass(frozen=True)
class Spectrum:
    """Unit masses and intensities, sorted by mass."""

    mz: np.ndarray
    intensities: np.ndarray

    @classmethod
    def from_pairs(cls, mz, intensities) -> Spectrum:
        """Sum the intensities of the peaks on the same unit mass."""
        mz = np.round(np.asarray(mz, dtype=float)).astype(int)
        intensities = np.asarray(intensities, dtype=float)
        unique, inverse = np.unique(mz, return_inverse=True)
        summed = np.zeros(len(unique))
        np.add.at(summed, inverse, intensities)
        return cls(unique, summed)

    def __len__(self) -> int:
        return len(self.mz)


@dataclass
class Processing:
    """alpinac's processing before matching."""

    cutoff_mass: float = 19.0
    ignored_masses: list[float] = field(default_factory=list)
    normalize: bool = True

    def apply(self, spectrum: Spectrum) -> Spectrum:
        mz, i = spectrum.mz, spectrum.intensities
        # Like alpinac: no cutoff if all the peaks are below it
        if len(mz) and mz.max() >= self.cutoff_mass:
            keep = mz >= self.cutoff_mass
            mz, i = mz[keep], i[keep]
        keep = ~np.isin(mz, self.ignored_masses)
        mz, i = mz[keep], i[keep]
        if self.normalize and len(i) and i.max() > 0:
            i = i / i.max()
        return Spectrum(mz, i)


def read_jdx(file: Path) -> Spectrum:
    """NIST JCAMP-DX spectrum (as ``db_reader.JdxReader``)."""
    from jcamp import jcamp_readfile

    data = jcamp_readfile(str(file))
    return Spectrum.from_pairs(data["x"], data["y"])


def read_alpinac_fragments(file: Path) -> Spectrum:
    """alpinac fragment file: peaks assigned to a compound, area summed per unit mass
    (as ``db_reader.AlpinacFragmentsReader`` with ``integer_masses=True``)."""
    import pandas as pd

    df = pd.read_csv(file, sep=r"\s+", comment="#")
    df = df[df["compound_bin"] != -1]
    return Spectrum.from_pairs(df["mass"], df["area"])


# Reader by file suffix. peaksdb's db_reader is not used: importing it (matchms) takes seconds.
READERS: dict[str, Callable[[Path], Spectrum]] = {
    ".jdx": read_jdx,
    ".txt": read_alpinac_fragments,
}


def read_library(path: Path, progress: Callable[[int, int], None] | None = None) -> dict[str, Spectrum]:
    """All the spectra of a library folder, by name (file stem).

    ``progress(done, total)`` is called after each file. Unreadable files are
    skipped (with a warning).
    """
    path = Path(path)
    if not path.is_dir():
        logger.warning(f"Library folder not found: {path}")
        return {}
    files = sorted(f for f in path.iterdir() if f.suffix in READERS)
    spectra = {}
    for done, file in enumerate(files, 1):
        if progress is not None:
            progress(done, len(files))
        try:
            spectrum = READERS[file.suffix](file)
        except Exception as e:
            logger.warning(f"Skipping unreadable spectrum {file}: {e}")
            continue
        if len(spectrum):
            spectra[file.stem] = spectrum
    return spectra


def cosine(query: Spectrum, reference: Spectrum) -> tuple[float, int]:
    """Cosine score and number of matched masses of two processed spectra."""
    common, iq, ir = np.intersect1d(query.mz, reference.mz, return_indices=True)
    if not len(common):
        return 0.0, 0
    norm = np.linalg.norm(query.intensities) * np.linalg.norm(reference.intensities)
    if norm == 0:
        return 0.0, 0
    return float(query.intensities[iq] @ reference.intensities[ir] / norm), len(common)


@dataclass
class Match:
    library: str
    name: str
    score: float
    matched: int


class LibraryCollection:
    """The ordered libraries, read lazily and kept in memory."""

    def __init__(self, libraries: list[Library], processing: Processing) -> None:
        self.libraries = libraries
        self.processing = processing
        self._spectra: dict[str, dict[str, Spectrum]] = {}
        # Processed spectra on a unit mass grid, for the enabled / all libraries
        self._matrices: dict[bool, tuple] = {}

    def with_processing(self, processing: Processing) -> LibraryCollection:
        """Same libraries (spectra already read are kept) with another processing."""
        other = LibraryCollection(self.libraries, processing)
        other._spectra = self._spectra
        return other

    @property
    def enabled(self) -> list[Library]:
        return [lib for lib in self.libraries if lib.enabled]

    def missing(self, all_libraries: bool = False) -> list[str]:
        """Libraries (enabled ones, or all) not read yet."""
        libraries = self.libraries if all_libraries else self.enabled
        return [lib.name for lib in libraries if lib.name not in self._spectra]

    def is_loaded(self, library: str) -> bool:
        return library in self._spectra

    def load(self, library: str, progress: Callable[[int, int], None] | None = None) -> None:
        """Read a library (can run in another thread: the result is stored at the end)."""
        lib = next(lib for lib in self.libraries if lib.name == library)
        spectra = read_library(lib.path, progress)
        logger.info(f"Library {library}: {len(spectra)} spectra from {lib.path}")
        self._spectra[library] = spectra

    def spectra(self, library: str) -> dict[str, Spectrum]:
        """Spectra of a library, read now if needed (the GUI loads them in the background first)."""
        if library not in self._spectra:
            self.load(library)
        return self._spectra[library]

    def spectrum(self, library: str, name: str) -> Spectrum | None:
        return self.spectra(library).get(name)

    def libraries_with(self, name: str, family: str | None = None) -> list[str]:
        """Enabled libraries containing a spectrum, by priority (of a family if given)."""
        return [
            lib.name
            for lib in self.enabled
            if (family is None or lib.family == family) and name in self.spectra(lib.name)
        ]

    def resolve(self, spectrum_ref: str | None) -> tuple[str, str] | None:
        """(library, name) of an alpinac spectrum reference like 'nist_CFC-12' or 'mygu_CFC-12'.

        The prefix is the library family, the first enabled library of that
        family containing the spectrum is used (any library if none of them).
        """
        if not spectrum_ref:
            return None
        family, _, name = spectrum_ref.partition("_")
        if not name:
            family, name = None, spectrum_ref
        found = self.libraries_with(name, family) or self.libraries_with(name)
        return (found[0], name) if found else None

    def _reference_matrix(self, all_libraries: bool = False):
        """Processed spectra of the enabled (or all) libraries on a common unit mass grid."""
        if all_libraries not in self._matrices:
            keys, spectra = [], []
            for lib in self.libraries if all_libraries else self.enabled:
                for name, spec in self.spectra(lib.name).items():
                    keys.append((lib.name, name))
                    spectra.append(self.processing.apply(spec))
            max_mass = max((int(s.mz.max()) for s in spectra if len(s)), default=0)
            matrix = np.zeros((len(spectra), max_mass + 1), dtype=np.float32)
            for row, spec in enumerate(spectra):
                matrix[row, spec.mz] = spec.intensities
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1
            self._matrices[all_libraries] = (keys, matrix / norms[:, None], matrix > 0)
        return self._matrices[all_libraries]

    def best_matches(self, query: Spectrum, n: int | None, search: str = "") -> list[Match]:
        """The best scoring spectra of the enabled libraries for a (raw) spectrum.

        With ``search``, the spectra of all the libraries (also the disabled ones)
        whose 'library/name' contains it (case insensitive), all of them if ``n``
        is None.
        """
        query = self.processing.apply(query)
        keys, matrix, present = self._reference_matrix(all_libraries=bool(search))
        if not len(keys):
            return []
        vector = np.zeros(matrix.shape[1], dtype=np.float32)
        inside = query.mz < matrix.shape[1]
        vector[query.mz[inside]] = query.intensities[inside]
        norm = np.linalg.norm(query.intensities) if len(query) else 0.0
        scores = matrix @ vector / norm if norm > 0 else np.zeros(len(keys))
        matched = present[:, vector > 0].sum(axis=1)
        rows = np.arange(len(keys))
        if search:
            search = search.lower()
            rows = np.array([i for i, (lib, name) in enumerate(keys) if search in f"{lib}/{name}".lower()], dtype=int)
        best = rows[np.argsort(scores[rows])[::-1]][:n]
        return [Match(keys[i][0], keys[i][1], float(scores[i]), int(matched[i])) for i in best]

    def score(self, query: Spectrum, reference: Spectrum) -> tuple[float, int]:
        return cosine(self.processing.apply(query), self.processing.apply(reference))
