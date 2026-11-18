"""The processing of the measurements, step by step (see :mod:`esper.processing`)."""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import Executor
from pathlib import Path

import pandas as pd

from esper import db
from esper.processing import METHODS, PEAK_SOURCE, TARGET
from esper.processing.config import (
    ProcessingConfig,
    ProcessingConfigError,
    read_processing_config,
)
from esper.target_index import TargetIndex

logger = logging.getLogger(__name__)


def to_process(
    con: sqlite3.Connection, method: str, force: bool = False
) -> list[tuple[int, str]]:
    """(measurement_id, name) of the measurements to process, by acquisition time: those
    with chromatograms and no peaks, or whose peaks esper found and identified with
    another method (all of those if ``force``). Not those imported with peaks."""
    return con.execute(
        "SELECT measurement_id, name FROM measurement"
        " WHERE chromatogram_file IS NOT NULL AND (peak_source IS NULL"
        "  OR (peak_source = ? AND (? OR identification IS NOT ?)))"
        " ORDER BY acq_datetime, name",
        (PEAK_SOURCE, bool(force), method),
    ).fetchall()


def load_targets(con: sqlite3.Connection) -> TargetIndex:
    """The targets of the database, with their ions, aliases and retention times."""
    return TargetIndex.from_db(con)


def _store(
    con: sqlite3.Connection,
    mid: int,
    method: str,
    peaks: pd.DataFrame,
    compounds: pd.DataFrame,
    candidates: pd.DataFrame | None,
) -> None:
    """Replace the peaks, compounds and candidates of a measurement (single transaction);
    the fit and the identifications edited by hand are dropped."""
    peaks = peaks.assign(compound_idx=peaks["compound_idx"].astype("Int64"))
    with con:
        for table in (
            "peak",
            "extraction_peak",
            "extraction_compound",
            "compound_candidate",
            "compound_identification",
            "compound",
        ):
            con.execute(f"DELETE FROM {table} WHERE measurement_id = ?", (mid,))
        db.insert(con, "compound", compounds.reset_index().assign(measurement_id=mid))
        db.insert(con, "peak", peaks.reset_index().assign(measurement_id=mid))
        if candidates is not None:
            db.insert(con, "compound_candidate", candidates.assign(measurement_id=mid))
        con.execute(
            "UPDATE measurement SET peak_source = ?, identification = ?, n_peaks = ?,"
            " n_compounds = ?, fit_version = NULL, fitted_at = NULL"
            " WHERE measurement_id = ?",
            (PEAK_SOURCE, method, len(peaks), len(compounds), mid),
        )


def _replace_compounds(
    con: sqlite3.Connection,
    mid: int,
    method: str,
    names: list[str],
    peaks: pd.DataFrame,
    compounds: pd.DataFrame,
) -> list[int]:
    """Replace the compounds of a measurement named one of ``names`` (and their peaks,
    with their pieces given to other compounds by the fit) with ``compounds`` and
    ``peaks`` (renumbered after the others); the other compounds are kept. Single
    transaction. Returns the compound_idx given to ``compounds``."""
    name_marks = ", ".join("?" * len(names))
    with con:
        old = [
            row[0]
            for row in con.execute(
                "SELECT compound_idx FROM compound WHERE measurement_id = ?"
                f" AND compound_name IN ({name_marks})",
                (mid, *names),
            )
        ]
        if old:
            marks = ", ".join("?" * len(old))
            old_peaks = [
                row[0]
                for row in con.execute(
                    f"SELECT peak_idx FROM peak WHERE measurement_id = ?"
                    f" AND compound_idx IN ({marks})"
                    f" UNION SELECT peak_idx FROM extraction_peak WHERE measurement_id = ?"
                    f" AND compound_idx IN ({marks})",
                    (mid, *old, mid, *old),
                )
            ]
            peak_marks = ", ".join("?" * len(old_peaks))
            for table in ("peak", "extraction_peak"):
                con.execute(
                    f"DELETE FROM {table} WHERE measurement_id = ?"
                    f" AND peak_idx IN ({peak_marks})",
                    (mid, *old_peaks),
                )
            for table in ("extraction_compound", "compound"):  # compound: cascades
                con.execute(
                    f"DELETE FROM {table} WHERE measurement_id = ?"
                    f" AND compound_idx IN ({marks})",
                    (mid, *old),
                )
        (first_compound,) = con.execute(
            "SELECT coalesce(max(compound_idx) + 1, 0) FROM compound"
            " WHERE measurement_id = ?",
            (mid,),
        ).fetchone()
        first_peak = db.next_peak_idx(con, mid)
        numbers = {
            int(old_idx): first_compound + i
            for i, old_idx in enumerate(compounds.index)
        }
        compounds = compounds.rename(index=numbers).rename_axis("compound_idx")
        peaks = peaks.assign(
            compound_idx=peaks["compound_idx"].map(numbers).astype("Int64")
        )
        peaks.index = pd.RangeIndex(
            first_peak, first_peak + len(peaks), name="peak_idx"
        )
        db.insert(con, "compound", compounds.reset_index().assign(measurement_id=mid))
        db.insert(con, "peak", peaks.reset_index().assign(measurement_id=mid))
        con.execute(
            "UPDATE measurement SET peak_source = coalesce(peak_source, ?),"
            " identification = coalesce(identification, ?) WHERE measurement_id = ?",
            (PEAK_SOURCE, method, mid),
        )
        db.update_counts(con, mid)
    return list(numbers.values())


class Processor:
    """Processes the measurements of a data directory with its configuration (the targets
    and the libraries are read once, when first needed)."""

    def __init__(
        self,
        con: sqlite3.Connection,
        data_dir: Path,
        config: ProcessingConfig | None = None,
    ) -> None:
        self.con = con
        self.data_dir = Path(data_dir)
        self.config = config or read_processing_config(data_dir)
        if self.config.method is None:
            raise ProcessingConfigError(
                f"{data_dir}: no method to identify the compounds; declare it"
                f" (--method {'|'.join(METHODS)})"
            )
        self._targets: TargetIndex | None = None
        self._target_search = None
        self._libraries = None

    @property
    def method(self) -> str:
        return self.config.method

    @property
    def targets(self) -> TargetIndex:
        if self._targets is None:
            self._targets = load_targets(self.con)
            if self._targets.table.empty:
                logger.warning("no targets: the compounds are not identified")
        return self._targets

    @property
    def target_search(self):
        """The target search of the runs (``identification.TargetSearch``), for their
        fit; None with suspect screening."""
        from esper.processing.identification import TargetSearch

        if self.method != TARGET:
            return None
        if self._target_search is None:
            self._target_search = TargetSearch(self.targets, self.config.target)
        return self._target_search

    def target_names(self, names: list[str]) -> list[str]:
        """The targets of ``names`` (aliases resolved), for ``process_compounds``.

        Raises ProcessingConfigError if the method is not target search or a name is
        no target.
        """
        if self.method != TARGET:
            raise ProcessingConfigError(
                f"{self.data_dir}: compounds are processed alone only with target"
                f" search, not {self.method}"
            )
        found = {name: self.targets.target_of(name) for name in names}
        unknown = sorted(name for name, target in found.items() if target is None)
        if unknown:
            raise ProcessingConfigError(f"not targets: {', '.join(unknown)}")
        return list(dict.fromkeys(found.values()))

    @property
    def libraries(self):
        if self._libraries is None:
            from esper.libraries import LibraryCollection, Processing, default_libraries

            self._libraries = LibraryCollection(default_libraries(), Processing())
        return self._libraries

    def identify(
        self, peaks: pd.DataFrame, compounds: pd.DataFrame, time=None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """The peaks, compounds and candidates (None: none) identified by the method."""
        from esper.processing.identification import screen_suspects, search_targets

        if self.method == TARGET:
            peaks, compounds = search_targets(
                peaks, compounds, self.targets, time, self.config.target
            )
            return peaks, compounds, None
        compounds, candidates = screen_suspects(
            peaks, compounds, self.libraries, self.config.suspect
        )
        return peaks, compounds, candidates

    def _chromatograms(self, mid: int) -> tuple[pd.DataFrame, object] | None:
        """(chromatograms, acquisition time) of a measurement, None without them."""
        chromatogram_file, time = self.con.execute(
            "SELECT chromatogram_file, acq_datetime FROM measurement"
            " WHERE measurement_id = ?",
            (mid,),
        ).fetchone()
        if chromatogram_file is None:
            return None
        return pd.read_parquet(self.data_dir / chromatogram_file), time

    def process(
        self,
        measurement_id: int,
        workers: int | None = None,
        pool: Executor | None = None,
        fit: bool = True,
    ) -> pd.DataFrame | None:
        """Detect, identify, store and fit (if ``fit``) the peaks of a measurement.

        Returns the peak table (fitted: see ``esper.fitting.fit_measurement``), None if the
        measurement has no chromatograms.
        """
        from esper.fitting import fit_measurement
        from esper.processing.detection import detect

        mid = int(measurement_id)
        run = self._chromatograms(mid)
        if run is None:
            return None
        frame, time = run
        peaks, compounds = detect(frame, self.config.baseline, self.config.detection)
        peaks, compounds, candidates = self.identify(peaks, compounds, time)
        _store(self.con, mid, self.method, peaks, compounds, candidates)
        if not fit:
            return peaks
        return fit_measurement(
            self.con,
            mid,
            self.data_dir,
            workers,
            pool,
            self.config.fit,
            self.target_search,
        )

    def process_compounds(
        self,
        measurement_id: int,
        names: list[str],
        workers: int | None = None,
        pool: Executor | None = None,
        fit: bool = True,
    ) -> pd.DataFrame | None:
        """Find, identify and fit (if ``fit``) only the compounds of some targets in a
        measurement (``names``: targets, see ``target_names``), replacing those it has;
        its other compounds are kept as they are.

        The peaks are found on the ions of these targets only, and target search looks
        for these targets only; the compounds it names are kept. Returns their peaks
        (fitted: see ``esper.fitting.fit_compounds``), None if the measurement has no
        chromatograms.
        """
        from esper.deconvolution import chromatogram_masses
        from esper.fitting import fit_compounds
        from esper.processing.detection import detect
        from esper.processing.identification import search_targets
        from esper.target_index import unit_mass

        mid = int(measurement_id)
        run = self._chromatograms(mid)
        if run is None:
            return None
        frame, time = run
        ions = {mass for name in names for mass in self.targets.masses_at(name, time)}
        on_ions = [c for c in chromatogram_masses(frame) if unit_mass(float(c)) in ions]
        alone = frame[on_ions]
        alone.attrs = dict(frame.attrs)  # e.g. a scan run (esper.sources.CONTINUOUS)
        peaks, compounds = detect(alone, self.config.baseline, self.config.detection)
        others = set(self.targets.table.index) - set(names)
        peaks, compounds = search_targets(
            peaks, compounds, self.targets, time, self.config.target, skip=others
        )
        compounds = compounds[compounds["compound_name"].isin(names)]
        peaks = peaks[peaks["compound_idx"].isin(compounds.index)]
        made = _replace_compounds(self.con, mid, self.method, names, peaks, compounds)
        if not fit:
            # the run's fit, if any, is no longer of all its compounds
            with self.con:
                self.con.execute(
                    "UPDATE measurement SET fit_version = NULL, fitted_at = NULL"
                    " WHERE measurement_id = ?",
                    (mid,),
                )
            return peaks
        return fit_compounds(
            self.con,
            mid,
            self.data_dir,
            made,
            workers,
            pool,
            self.config.fit,
            self.target_search,
        )


def summary(table: pd.DataFrame, con: sqlite3.Connection, mid: int) -> str:
    """One line on a processed measurement."""
    from esper.fitting import summary as fit_summary

    n_compounds, n_named = con.execute(
        "SELECT count(*), count(compound_name) FROM compound WHERE measurement_id = ?",
        (int(mid),),
    ).fetchone()
    fitted = fit_summary(table) if "fit_rt_s" in table else f"{len(table)} peaks"
    return f"{n_compounds} compounds, {n_named} identified; {fitted}"
