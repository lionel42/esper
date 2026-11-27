"""The processing of the runs of a data directory once imported: their peaks found and
identified (:mod:`esper.processing`: the runs imported without peaks) and fitted
(:mod:`esper.fitting`). ``esper process`` on the runs of the database, and ``esper
upload`` on the runs it imports (:mod:`esper.upload`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from esper import db
from esper.processing import TARGET
from esper.processing.config import (
    ProcessingConfig,
    ProcessingConfigError,
    read_processing_config,
    resolve_method,
)
from esper.sources import get_source, in_dates, name_width, peak_source, run_label

logger = logging.getLogger(__name__)

PEAKS = "peaks"  # found and identified (esper.processing)
# the identification alone (suspect screening), of the runs whose peaks esper found (part
# of PEAKS; with the retention indices)
IDENTIFY = "identify"
# the retention indices alone, of the runs whose peaks esper found (part of PEAKS)
RI = "ri"
FIT = "fit"
STEPS = (PEAKS, IDENTIFY, RI, FIT)
# the steps by default (with a method)
DEFAULT_STEPS = (PEAKS, FIT)


@dataclass
class ProcessSummary:
    # Peaks found and identified (esper.processing)
    processed: list[str] = field(default_factory=list)
    processing_failed: list[tuple[str, str]] = field(default_factory=list)
    fitted: list[str] = field(default_factory=list)
    fit_failed: list[tuple[str, str]] = field(default_factory=list)
    # The identification step alone
    identified: list[str] = field(default_factory=list)
    identify_failed: list[tuple[str, str]] = field(default_factory=list)
    # The retention index step alone
    indexed: list[str] = field(default_factory=list)
    index_failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        return bool(
            self.processing_failed
            or self.fit_failed
            or self.identify_failed
            or self.index_failed
        )


class RunProcessing:
    """Processes runs one at a time with the configuration of a data directory, logs a
    line per step and counts the results in ``summary``.

    ``pool``: the processes of the fits (``esper.fitting.fit_pool``). ``compounds``:
    targets (or their aliases) whose peaks alone are found, identified and fitted
    (target search: ``Processor.process_compounds``; the data directory needs a
    method). ``width``: of the run names (``run_label``). ``report``: what is done to
    each run, one line (the failures: logged).

    Raises ProcessingConfigError if ``compounds`` are given without target search.
    """

    def __init__(
        self,
        con: sqlite3.Connection,
        data_dir: Path,
        config: ProcessingConfig,
        pool=None,
        compounds: list[str] | None = None,
        summary: ProcessSummary | None = None,
        width: int = 0,
        report: Callable[[str], None] = logger.info,
    ) -> None:
        from esper.processing.pipeline import Processor

        self.con = con
        self.data_dir = Path(data_dir)
        self.config = config
        self.pool = pool
        self.summary = ProcessSummary() if summary is None else summary
        self.width = width
        self.report = report
        self.processor = (
            None if config.method is None else Processor(con, data_dir, config)
        )
        self.names = self.processor.target_names(compounds) if compounds else None
        self._target_search = None

    @property
    def target_search(self):
        """The target search of the data directory, for the fit of the runs it
        identified (``esper.processing.identification.TargetSearch``)."""
        from esper.processing.identification import TargetSearch
        from esper.target_index import TargetIndex

        if self._target_search is None:
            targets = (
                TargetIndex.from_db(self.con)
                if self.processor is None
                else self.processor.targets
            )
            self._target_search = TargetSearch(targets, self.config.target)
        return self._target_search

    def _label(self, name: str) -> str:
        return run_label(name, self.width)

    def run(self, name: str, mid: int, steps: Iterable[str] = DEFAULT_STEPS) -> None:
        """The steps of a run: its peaks found and identified (and fitted with them;
        with its retention indices), or only fitted and / or identified (with its
        retention indices) or its retention indices (after the fit)."""
        steps = set(steps)
        if PEAKS in steps:
            self.find_peaks(name, mid, fit=FIT in steps)
            return
        if FIT in steps:
            self.fit(name, mid)
        if IDENTIFY in steps:
            self.identify(name, mid)
        elif RI in steps:
            self.retention_indices(name, mid)

    def find_peaks(self, name: str, mid: int, fit: bool = True) -> None:
        """Find and identify the peaks of a run (of the compounds only, if given), then
        fit them if ``fit`` (each failure counted apart; with compounds, they are fitted
        with them: a failure counted as of the processing)."""
        from esper.fitting import _compound_names
        from esper.processing.pipeline import summary as process_summary

        label = self._label(name)
        try:
            if self.names:
                table = self.processor.process_compounds(
                    mid, self.names, pool=self.pool, fit=fit
                )
            else:
                table = self.processor.process(mid, fit=False)
        except Exception as e:
            logger.exception(f"{label} processing failed")
            self.summary.processing_failed.append((name, repr(e)))
            return
        if table is None:
            logger.warning(f"{label} no chromatograms, not processed")
            return
        self.summary.processed.append(name)
        if self.names:
            if fit:
                self.summary.fitted.append(name)
            # one column per target (logged once, ``process``), "-" if not found
            found = set(_compound_names(self.con, mid).dropna())
            columns = "  ".join(
                f"{n if n in found else '-':<{len(n)}}" for n in self.names
            )
            n_found = len(found & set(self.names))
            self.report(
                f"{label} found {n_found}/{len(self.names)}: {columns.rstrip()}"
            )
            return
        self.report(f"{label} {process_summary(table, self.con, mid)}")
        if fit:
            self.fit(name, mid)

    def fit(self, name: str, mid: int) -> None:
        """Fit the peaks of a run (``esper.fitting.fit_measurement``)."""
        from esper.fitting import fit_measurement
        from esper.fitting import summary as fit_summary

        label = self._label(name)
        try:
            table = fit_measurement(
                self.con,
                mid,
                self.data_dir,
                workers=1,
                pool=self.pool,
                config=self.config.fit,
                target_search=self.target_search,
            )
        except Exception as e:
            logger.exception(f"{label} fit failed")
            self.summary.fit_failed.append((name, repr(e)))
            return
        if table is None:
            logger.warning(f"{label} no chromatograms, not fitted")
            return
        self.summary.fitted.append(name)
        dropped = self._drop_unfitted(mid)
        deleted = f"; {dropped} unfitted one-peak compounds deleted" if dropped else ""
        self.report(f"{label} {fit_summary(table)}{deleted}")

    def _drop_unfitted(self, mid: int) -> int:
        """The rule after the fit of a run whose peaks esper found
        (``esper.processing.rules.drop_unfitted_single_peaks``, if on)."""
        from esper.processing import PEAK_SOURCE
        from esper.processing.rules import drop_unfitted_single_peaks

        if self.processor is None or not self.config.rules.drop_unfitted_single_peaks:
            return 0
        (source,) = self.con.execute(
            "SELECT peak_source FROM measurement WHERE measurement_id = ?", (mid,)
        ).fetchone()
        if source != PEAK_SOURCE:
            return 0
        return drop_unfitted_single_peaks(self.con, mid)

    def identify(self, name: str, mid: int) -> None:
        """The identification step alone (``Processor.identify_stored``)."""

        label = self._label(name)
        try:
            found, merged, dropped = self.processor.identify_stored(mid)
        except Exception as e:
            logger.exception(f"{label} identification failed")
            self.summary.identify_failed.append((name, repr(e)))
            return
        self.summary.identified.append(name)
        (n_valid,) = self.con.execute(
            "SELECT count(*) FROM compound WHERE measurement_id = ? AND valid_candidate = 1",
            (mid,),
        ).fetchone()
        refs = "" if found is None else f"{len(found):2} retention index references, "
        self.report(
            f"{label} {refs}{n_valid:3} compounds identified; {merged:3} merged"
            f" (noise merge), {dropped:3} unfitted one-peak compounds deleted"
        )

    def retention_indices(self, name: str, mid: int) -> None:
        """The retention index step alone (``Processor.retention_indices``)."""

        label = self._label(name)
        try:
            found = self.processor.retention_indices(mid)
        except Exception as e:
            logger.exception(f"{label} retention indices failed")
            self.summary.index_failed.append((name, repr(e)))
            return
        self.summary.indexed.append(name)
        (n_valid,) = self.con.execute(
            "SELECT count(*) FROM compound WHERE measurement_id = ? AND valid_candidate = 1",
            (mid,),
        ).fetchone()
        self.report(
            f"{label} {len(found):2} retention index references, {n_valid} compounds"
            " identified"
        )


def to_run(
    con: sqlite3.Connection,
    method: str | None,
    steps: set[str],
    force: bool = False,
    overrides: bool = False,
    compounds: bool = False,
) -> list[tuple[int, str, set[str]]]:
    """(measurement_id, name, steps) of the runs to process, by acquisition time.

    The peaks (``method``): of the runs without, or whose peaks esper found with
    another method (all of those found by esper if ``force``), fitted with them if
    ``FIT`` is a step. The fit alone: of the runs not fitted, or fitted by an older
    esper (all if ``force``), and those a fit override applies to (``overrides``); not
    with ``compounds`` (only their compounds are processed, peaks and fit). The
    identification alone (``IDENTIFY``) and the retention indices alone (``RI``): of all
    the runs whose peaks esper found with ``method`` (and not found again here).
    """
    from esper.fitting import to_fit
    from esper.processing import PEAK_SOURCE
    from esper.processing.pipeline import to_process

    todo = {}
    if PEAKS in steps:
        for mid, _ in to_process(con, method, force):
            todo[mid] = steps & {PEAKS, FIT}
    alone = steps & {IDENTIFY, RI}
    if alone and not compounds:
        for (mid,) in con.execute(
            "SELECT measurement_id FROM measurement WHERE peak_source = ?"
            " AND identification = ?",
            (PEAK_SOURCE, method),
        ):
            todo.setdefault(mid, set()).update(alone)
    if FIT in steps and not compounds:
        fits = to_fit(con, force)
        if overrides and not force:
            from esper.fit_overrides import read_overrides, runs_covered

            fits += runs_covered(con, read_overrides(con))
        for mid, _ in fits:
            todo.setdefault(mid, set()).add(FIT)
    return [
        (mid, name, todo[mid])
        for mid, name in con.execute(
            "SELECT measurement_id, name FROM measurement ORDER BY acq_datetime, name"
        )
        if mid in todo
    ]


def check_method(
    instrument: str | None, method: str | None, data_dir: Path | None = None
) -> None:
    """Raises ProcessingConfigError if a ``method`` is given for an instrument whose runs
    are always imported with their peaks (esper does not find them); or if the data
    directory would import without their peaks (``method``, else the declared one) the
    runs of an instrument whose peaks are optional, but already has runs imported with
    them: importing them again would drop those peaks and their identifications."""
    if instrument is None:
        return
    source = get_source(instrument)
    if method is not None:
        imported = peak_source(source, method)
        if imported is not None:
            raise ProcessingConfigError(
                f"--method: the runs of {instrument} are imported with their peaks"
                f" ({imported}), esper does not find them"
            )
    if data_dir is None or source.PEAK_SOURCE is None:
        return
    method = method or read_processing_config(data_dir).method
    if method is None or peak_source(source, method) is not None:
        return
    n = _runs_with_peaks(data_dir, source.PEAK_SOURCE)
    if n:
        raise ProcessingConfigError(
            f"{data_dir} has {n} runs imported with the peaks of {source.PEAK_SOURCE}:"
            f" with a method its runs are imported without them (esper finds the peaks),"
            f" which would drop those peaks and their identifications. Process the runs"
            f" in another data directory"
        )


def _runs_with_peaks(data_dir: Path, peak_source: str) -> int:
    """The runs of the database of a data directory imported with the peaks of
    ``peak_source`` (0 without database)."""
    if not db.db_path(data_dir).is_file():
        return 0
    con = db.connect(data_dir, readonly=True)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM measurement WHERE peak_source = ?", (peak_source,)
        ).fetchone()[0]
    finally:
        con.close()


def process(
    data_dir: Path,
    steps: Iterable[str] | None = None,
    force: bool = False,
    workers: int | None = None,
    method: str | None = None,
    compounds: list[str] | None = None,
    overrides: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
    report: Callable[[str], None] = logger.info,
) -> ProcessSummary:
    """Process the runs of a data directory that need it (see ``to_run``).

    ``steps``: ``PEAKS`` (found and identified with the method of the data directory,
    ``method``: declared if not yet, see :func:`esper.processing.resolve_method`),
    ``FIT``, and alone ``IDENTIFY`` (suspect screening) or ``RI``; None: both, the peaks only with a method (the runs without peaks: a
    warning). ``force``: the steps of all the runs again. ``compounds``: see
    ``RunProcessing``. ``overrides``: also fit again the runs that a fit override
    applies to. ``after``, ``before``: only the runs started on or after ``after`` and
    before ``before`` (see ``esper.sources.in_dates``). ``report``: the progress, one
    line each (see ``RunProcessing``).

    Raises ProcessingConfigError (before anything is written) without a method for the
    ``PEAKS`` asked or ``compounds``, with ``compounds`` but not the ``PEAKS`` step or
    not targets (``Processor.target_names``), with ``overrides`` but not the fit alone
    (``compounds``, or no ``FIT`` step), or with a ``method`` the runs do not use
    (``check_method``).
    """
    from esper.fitting import fit_pool
    from esper.processing.pipeline import no_method
    from esper.sources import declared_instrument

    check_method(declared_instrument(data_dir), method, data_dir)
    config = read_processing_config(data_dir)
    if method is not None:  # as asked, declared once the arguments are checked
        config = dataclasses.replace(config, method=method)
    has_method = config.method is not None
    if not has_method and (
        compounds or (steps is not None and ({PEAKS, IDENTIFY, RI} & set(steps)))
    ):
        raise no_method(data_dir)
    if steps is None:
        steps = set(DEFAULT_STEPS) if has_method else {FIT}
    steps = set(steps)
    if compounds and PEAKS not in steps:
        raise ProcessingConfigError(f"--compounds: needs the {PEAKS} step")
    if IDENTIFY in steps and config.method == TARGET:
        raise ProcessingConfigError(
            f"--steps {IDENTIFY}: of suspect screening; target search identifies with"
            f" the peaks (--steps {PEAKS} {FIT} -f)"
        )
    if RI in steps and not config.uses_retention_index:
        raise ProcessingConfigError(
            f"--steps {RI}: {data_dir} uses no retention indices"
            " (processing: retention_index)"
        )
    if overrides and (compounds or FIT not in steps):
        raise ProcessingConfigError(
            "--overrides: fits whole runs again, needs the fit step and no --compounds"
        )
    # create: brings an older schema up to date
    con = db.connect(data_dir, create=True)
    try:
        # the compounds checked (targets of the database) before declaring the method
        runs = RunProcessing(con, data_dir, config, compounds=compounds, report=report)
        resolve_method(data_dir, method)
        if config.method is None:
            _warn_without_peaks(con, data_dir)
        todo = [
            run
            for run in to_run(con, config.method, steps, force, overrides, compounds)
            if in_dates(run[1], after, before)
        ]
        runs.width = name_width(name for _, name, _ in todo)
        with contextlib.ExitStack() as stack:
            if FIT in steps:
                runs.pool = stack.enter_context(fit_pool(workers))
            report(f"runs to process: {len(todo)}")
            if runs.names:
                report(f"compounds: {', '.join(runs.names)}")
            for mid, name, run_steps in todo:
                runs.run(name, mid, run_steps)
    finally:
        con.close()
    return runs.summary


def update_retention_indices(
    data_dir: Path, report: Callable[[str], None] = logger.info
) -> ProcessSummary | None:
    """The retention index step (``RI``) of all the runs whose peaks esper found, after
    the references or the targets (their RIs) of the data directory changed: their
    references, the RIs of their compounds and candidates and their identification
    follow (peaks, fit and identifications by hand kept). None if the data directory
    has no method or uses no retention indices (nothing to update).

    Raises ValueError if its esper.yaml is invalid.
    """
    config = read_processing_config(data_dir)
    if config.method is None or not config.uses_retention_index:
        return None
    return process(data_dir, steps={RI}, report=report)


def _warn_without_peaks(con: sqlite3.Connection, data_dir: Path) -> None:
    """Warn about the runs whose peaks are not found: the data directory has no method."""
    (n,) = con.execute(
        "SELECT count(*) FROM measurement"
        " WHERE chromatogram_file IS NOT NULL AND peak_source IS NULL"
    ).fetchone()
    if n:
        logger.warning(
            f"{n} runs without peaks: {data_dir} has no method to find them (--method)"
        )
