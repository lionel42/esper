"""The processing of the measurements, step by step (see :mod:`esper.processing`)."""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import Executor
from pathlib import Path
from typing import NamedTuple

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


class Identified(NamedTuple):
    """The result of the identification step alone (``Processor.identify_stored``)."""

    #: the retention index references found (None: retention indices not used)
    found: pd.DataFrame | None
    #: compounds merged into others (noise merge)
    merged: int
    #: unfitted one-peak compounds deleted
    dropped: int


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


def no_method(data_dir: Path) -> ProcessingConfigError:
    """The error of a data directory without a method, for a step that needs one."""
    return ProcessingConfigError(
        f"{data_dir}: no method to identify the compounds; declare it"
        f" (--method {'|'.join(METHODS)})"
    )


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
    references: pd.DataFrame | None = None,
) -> None:
    """Replace the peaks, compounds, candidates and retention index references found
    (``references``, None: none) of a measurement (single transaction); the fit and the
    identifications edited by hand are dropped."""
    from esper.retention_index import FOUND_COLUMNS, store_run_references

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
        store_run_references(
            con,
            mid,
            pd.DataFrame(columns=FOUND_COLUMNS) if references is None else references,
        )
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
            raise no_method(data_dir)
        self._targets: TargetIndex | None = None
        self._target_search = None
        self._libraries = None
        self._references: pd.DataFrame | None = None

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
        """The spectral libraries of suspect screening (``processing: suspect:``): its
        ``libraries`` from ``peaksdb``, only the spectra of the targets with an rt (and of
        their aliases) if ``targets_only``.

        Raises ProcessingConfigError for an unknown library.
        """
        if self._libraries is None:
            from esper.libraries import LibraryCollection, Processing, select_libraries

            suspect = self.config.suspect
            try:
                libraries = select_libraries(suspect.libraries, Path(suspect.peaksdb))
            except ValueError as e:
                raise ProcessingConfigError(f"processing: suspect: {e}") from None
            names = None
            if suspect.targets_only:
                names = self.targets.names_with_rt()
                if not names:
                    logger.warning(
                        "targets_only: no targets with an rt, all the library spectra"
                        " are matched"
                    )
                    names = None
            self._libraries = LibraryCollection(libraries, Processing(), names)
        return self._libraries

    @property
    def references(self) -> pd.DataFrame:
        """The retention index references of the data directory (enabled)."""
        if self._references is None:
            from esper.retention_index import read_references

            self._references = read_references(self.con)
            if self._references.empty:
                logger.warning(
                    "no retention index references (esper targets, esper ri-refs):"
                    " the compounds have no retention index"
                )
        return self._references

    def identify(
        self, peaks: pd.DataFrame, compounds: pd.DataFrame, time=None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """The peaks, compounds and candidates (None: none) identified by the method, and
        the retention index references found (None: retention indices not used)."""
        from esper.processing.identification import (
            retention_indices,
            screen_suspects,
            search_targets,
        )

        candidates = None
        if self.method == TARGET:
            peaks, compounds = search_targets(
                peaks, compounds, self.targets, time, self.config.target
            )
        else:
            compounds, candidates = screen_suspects(
                peaks, compounds, self.libraries, self.config.suspect
            )
        if not self.config.uses_retention_index:
            return peaks, compounds, candidates, None
        compounds, candidates, found = retention_indices(
            compounds,
            candidates,
            self.references,
            self.targets,
            self.config.ri,
            self.config.suspect,
        )
        return peaks, compounds, candidates, found

    def _stored_compounds(self, mid: int) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT compound_idx, rt_mean_s, start_mean_s, stop_mean_s, prominence_sum,"
            " compound_name, valid_candidate, is_ref_ri, retention_index FROM compound"
            " WHERE measurement_id = ?",
            self.con,
            params=(mid,),
            index_col="compound_idx",
        )

    def identify_stored(self, measurement_id: int) -> Identified:
        """The identification step alone on a run whose peaks esper found (suspect
        screening): its compounds matched again to the libraries, from the heights of
        their peaks as found (``esper.fitting.extraction``: as the peaks step, before the
        fit; the compounds made by the fit get none), then their retention indices (if
        used, as ``retention_indices``, from the fitted rts), then the rules
        (``processing: rules:``): the noise merge (``esper.fitting.merge_compounds``, for
        good: only the peaks step splits them again; then identified again) and, if
        fitted, the unfitted one-peak compounds deleted. The peaks, the fit and the
        identifications edited by hand are kept otherwise.

        Raises ProcessingConfigError with target search (its identification is part of
        the peaks: ``esper process -f``).
        """
        from esper.fitting import extraction, merge_compounds
        from esper.processing.rules import drop_unfitted_single_peaks

        if self.method == TARGET:
            raise ProcessingConfigError(
                f"{self.data_dir}: the identification alone is of suspect screening;"
                " target search identifies with the peaks (esper process -f)"
            )
        mid = int(measurement_id)

        def identified():
            # the peaks as found (before the fit), as the identification of the peaks step
            peaks, _ = extraction(self.con, mid)
            compounds = self._stored_compounds(mid).assign(
                compound_name=None, valid_candidate=0, is_ref_ri=0, retention_index=None
            )
            return (peaks, *self._screen(peaks, compounds))

        peaks, compounds, candidates, found = identified()
        merges = self._noise_merges(peaks, compounds, candidates)
        for main, others in merges.items():
            merge_compounds(self.con, mid, [main, *others])
        if merges:
            peaks, compounds, candidates, found = identified()
        self._update_identification(mid, compounds, candidates, found, replace=True)
        dropped = 0
        if self.config.rules.drop_unfitted_single_peaks:
            dropped = drop_unfitted_single_peaks(self.con, mid)
        return Identified(found, sum(map(len, merges.values())), dropped)

    def _screen(self, peaks: pd.DataFrame, compounds: pd.DataFrame):
        """(compounds, candidates, references found or None) of suspect screening and
        the retention indices (if used)."""
        from esper.processing.identification import retention_indices, screen_suspects

        compounds, candidates = screen_suspects(
            peaks, compounds, self.libraries, self.config.suspect
        )
        found = None
        if self.config.uses_retention_index:
            compounds, candidates, found = retention_indices(
                compounds,
                candidates,
                self.references,
                self.targets,
                self.config.ri,
                self.config.suspect,
            )
        return compounds, candidates, found

    def _noise_merges(
        self,
        peaks: pd.DataFrame,
        compounds: pd.DataFrame,
        candidates: pd.DataFrame | None,
    ) -> dict[int, list[int]]:
        """The noise merges of a run (``esper.processing.rules.noise_merges``), none
        without suspect screening or if the rule is off."""
        from esper.processing.rules import noise_merges

        if (
            self.method == TARGET
            or candidates is None
            or not self.config.rules.noise_merge
        ):
            return {}
        return noise_merges(
            peaks,
            compounds,
            candidates,
            self.libraries,
            self.config.rules.saturation_pieces,
            self.config.suspect.min_score,
        )

    def _update_identification(
        self,
        mid: int,
        compounds: pd.DataFrame,
        candidates: pd.DataFrame | None,
        found: pd.DataFrame | None,
        replace: bool,
    ) -> None:
        """Store the identification of the compounds of a run (single transaction):
        their name, validation and RI, their candidates (``replace``: all of them again,
        else their RI, combined score and best) and its references found (None: kept)."""
        from esper.retention_index import store_run_references

        compounds = compounds.astype(object).where(compounds.notna(), None)
        with self.con:
            self.con.executemany(
                "UPDATE compound SET compound_name = ?, valid_candidate = ?,"
                " is_ref_ri = ?, retention_index = ?"
                " WHERE measurement_id = ? AND compound_idx = ?",
                [
                    (
                        r.compound_name,
                        r.valid_candidate,
                        r.is_ref_ri,
                        r.retention_index,
                        mid,
                        int(i),
                    )
                    for i, r in compounds.iterrows()
                ],
            )
            if candidates is not None and replace:
                self.con.execute(
                    "DELETE FROM compound_candidate WHERE measurement_id = ?", (mid,)
                )
                db.insert(
                    self.con,
                    "compound_candidate",
                    candidates.assign(measurement_id=mid),
                )
            elif candidates is not None:
                values = candidates.astype(object).where(candidates.notna(), None)
                self.con.executemany(
                    "UPDATE compound_candidate SET ri = ?, ri_diff = ?,"
                    " score_combined = ?, is_best = ?"
                    " WHERE measurement_id = ? AND compound_idx = ? AND rank = ?",
                    [
                        (
                            r.ri,
                            r.ri_diff,
                            r.score_combined,
                            r.is_best,
                            mid,
                            r.compound_idx,
                            r.rank,
                        )
                        for r in values.itertuples()
                    ],
                )
            if found is not None:
                store_run_references(self.con, mid, found)

    def retention_indices(self, measurement_id: int) -> pd.DataFrame:
        """The retention index step alone on a run whose peaks esper found: its
        references found again among its compounds as they are (fitted or not), the
        RI of its compounds, and with suspect screening the RI of the candidates, the
        best one and the validation (``identification.retention_indices``). The peaks,
        the fit and the identifications edited by hand are kept. Single transaction.

        Returns the references found (``esper.retention_index.FOUND_COLUMNS``).
        Raises ProcessingConfigError if the data directory does not use retention
        indices.
        """
        from esper.processing.identification import CANDIDATE_COLUMNS, retention_indices

        if not self.config.uses_retention_index:
            raise ProcessingConfigError(
                f"{self.data_dir}: no retention indices (processing: retention_index)"
            )
        mid = int(measurement_id)
        compounds = self._stored_compounds(mid)
        candidates = None
        if self.method != TARGET:
            candidates = pd.read_sql(
                "SELECT * FROM compound_candidate WHERE measurement_id = ?",
                self.con,
                params=(mid,),
            )[CANDIDATE_COLUMNS]
        compounds, candidates, found = retention_indices(
            compounds,
            candidates,
            self.references,
            self.targets,
            self.config.ri,
            self.config.suspect,
        )
        self._update_identification(mid, compounds, candidates, found, replace=False)
        return found

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
        as_found = list(compounds.columns)
        peaks, compounds, candidates, found = self.identify(peaks, compounds, time)
        merges = self._noise_merges(peaks, compounds, candidates)
        if merges:
            from esper.processing.rules import merge

            peaks, compounds = merge(peaks, compounds[as_found], merges)
            compounds = compounds.assign(compound_name=None, valid_candidate=0)
            peaks, compounds, candidates, found = self.identify(peaks, compounds, time)
        _store(self.con, mid, self.method, peaks, compounds, candidates, found)
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
    fitted = fit_summary(table) if "fit_rt_s" in table else f"{len(table):4} peaks"
    return f"{n_compounds:3} compounds, {n_named:3} identified; {fitted}"
