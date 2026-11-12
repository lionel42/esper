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
        chromatogram_file, time = self.con.execute(
            "SELECT chromatogram_file, acq_datetime FROM measurement"
            " WHERE measurement_id = ?",
            (mid,),
        ).fetchone()
        if chromatogram_file is None:
            return None
        frame = pd.read_parquet(self.data_dir / chromatogram_file)
        peaks, compounds = detect(frame, self.config.baseline, self.config.detection)
        peaks, compounds, candidates = self.identify(peaks, compounds, time)
        _store(self.con, mid, self.method, peaks, compounds, candidates)
        if not fit:
            return peaks
        return fit_measurement(
            self.con, mid, self.data_dir, workers, pool, self.config.fit
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
