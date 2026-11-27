"""What the commands of the command line share: :class:`Command`, the options of several
commands (``add_*``) and the conversion of their values."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Command:
    """A command of ``esper`` (one module of ``esper.cli`` each, its ``COMMAND``).

    ``add_arguments`` is only called for the command given (its options may import heavy
    packages for their defaults); ``run`` returns the exit code.
    """

    name: str
    help: str
    """One line in ``esper --help``."""
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]
    description: str | None = None
    """Shown by ``esper <name> --help`` (None: ``help``)."""


DATE_FORMATS = "YYYY, YYYY-MM, YYYY-MM-DD or 'YYYY-MM-DD HH:MM'"


def date_argument(text: str) -> datetime:
    """A date of the command line: a year or a month (its first day), a day, or a day
    with a time."""
    text = text.strip()
    for fmt in ("%Y", "%Y-%m"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a date: {text!r} ({DATE_FORMATS})"
        ) from None


# --- options of several commands -------------------------------------------------------


def add_data_dir(
    parser: argparse.ArgumentParser, note: str | None = None, optional: bool = False
) -> None:
    """The positional ``data_dir``: an esper data directory (``note``: in parentheses)."""
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?" if optional else None,
        help="esper data directory" + (f" ({note})" if note else ""),
    )


def add_workers(parser: argparse.ArgumentParser) -> None:
    """``-j/--workers``: the processes of the fit."""
    parser.add_argument(
        "-j", "--workers", type=int, help="processes of the fit (default: all CPUs)"
    )


def add_method(parser: argparse.ArgumentParser) -> None:
    """``--method``: how the compounds are identified (``esper.processing``)."""
    from esper.processing import METHODS

    parser.add_argument(
        "--method",
        choices=METHODS,
        help="how the compounds of the runs imported without peaks (gcwerks) are "
        "identified, declared in the esper.yaml of the data directory (then optional, "
        "must be the same): target (expected rt and ions of the targets) or suspect "
        "(spectra in the libraries); without a method their peaks are not found",
    )


def add_compounds(parser: argparse.ArgumentParser, note: str) -> None:
    """``--compounds``: the targets whose peaks alone are found (``note``: added to the
    help, e.g. what ``-f`` does with them)."""
    parser.add_argument(
        "--compounds",
        nargs="+",
        metavar="NAME",
        help="find, identify and fit the peaks of these targets only (target search;"
        f" the other compounds of the runs are kept). {note}",
    )


def add_dates(parser: argparse.ArgumentParser) -> None:
    """``--after`` and ``--before``: the runs started between, from their name."""
    parser.add_argument(
        "--after",
        type=date_argument,
        metavar="DATE",
        help=f"only the runs started on or after DATE ({DATE_FORMATS}; a year or a"
        " month: from its start, included), from the date in their name",
    )
    parser.add_argument(
        "--before",
        type=date_argument,
        metavar="DATE",
        help=f"only the runs started before DATE ({DATE_FORMATS}; a year, month or day:"
        " before its start, itself left out)",
    )


def add_files(
    parser: argparse.ArgumentParser, help: str = "h5 files (glob patterns allowed)"
) -> None:
    """The positional ``files``, expanded by :func:`files`."""
    parser.add_argument("files", nargs="+", help=help)


def add_segments(parser: argparse.ArgumentParser, help: str) -> None:
    """``--segments`` of h5 files: ``1``, or ``1:3`` for 1 and 2 (as :func:`segments`:
    ``args.segments`` is the (start, stop) tuple, or None)."""
    parser.add_argument("--segments", type=segments, metavar="N[:M]", help=help)


def add_ionisation(parser: argparse.ArgumentParser) -> None:
    """``--ionisation``: of the legacy calibration file."""
    parser.add_argument(
        "--ionisation",
        choices=["EI", "CI"],
        default="EI",
        help="legacy calibration file <run>.<EI|CI>_mc.txt, used if there is no json file",
    )


def add_rt_range(
    parser: argparse.ArgumentParser,
    of: str = "the rt range",
    stop_note: str | None = None,
) -> None:
    """``--rt-start`` and ``--rt-stop``, s."""
    parser.add_argument("--rt-start", type=float, help=f"start of {of}, s")
    parser.add_argument(
        "--rt-stop",
        type=float,
        help=f"end of {of}, s" + (f" ({stop_note})" if stop_note else ""),
    )


def add_output(parser: argparse.ArgumentParser) -> None:
    """``-o/--output`` (one h5 file only) and ``--output-dir``."""
    parser.add_argument(
        "-o", "--output", type=Path, help="output file (one h5 file only)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <run>.alpinac next to the h5 file)",
    )


def add_block(parser: argparse.ArgumentParser) -> None:
    """``--block``: the scans of an h5 file read at once."""
    parser.add_argument(
        "--block",
        type=int,
        default=600,
        help="scans read at once (default %(default)s)",
    )


# --- values of the arguments ------------------------------------------------------------


def files(patterns: list[str]) -> list[Path]:
    """The files of the arguments (glob patterns expanded, in order, without duplicates)."""
    import glob

    found = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            print(f"No file matches {pattern}", file=sys.stderr)
        found += [Path(m) for m in matches if Path(m) not in found]
    return found


def segments(value: str | None) -> tuple[int, int] | None:
    """``1`` -> (1, 2), ``1:3`` -> (1, 3)."""
    if value is None:
        return None
    start, _, stop = value.partition(":")
    try:
        return (int(start), int(stop) if stop else int(start) + 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not segments: {value!r} (1, or 1:3 for 1 and 2)"
        ) from None


# --- output ----------------------------------------------------------------------------


def print_processing(summary) -> None:
    """The counts of the processing of the runs (``esper.process.ProcessSummary``)."""
    if summary.processed or summary.processing_failed:
        print(
            f"peaks found and identified: {len(summary.processed)},"
            f" failed: {len(summary.processing_failed)}"
        )
    if summary.fitted or summary.fit_failed:
        print(f"fitted: {len(summary.fitted)}, failed: {len(summary.fit_failed)}")
    indexed = getattr(summary, "indexed", [])
    index_failed = getattr(summary, "index_failed", [])
    if indexed or index_failed:
        print(f"retention indices: {len(indexed)}, failed: {len(index_failed)}")


def add_no_update(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="leave the retention indices of the runs as they are (later: esper process"
        " DATA_DIR --steps ri); by default the runs whose peaks esper found get them"
        " again from the new references and target RIs",
    )


def update_retention_indices(data_dir: Path) -> int:
    """The retention index step of the runs of a data directory after its references
    or targets changed (``esper.process.update_retention_indices``), the counts and
    failures printed. Returns the exit code: 1 if a run failed, 2 if the esper.yaml
    is invalid."""
    from esper.process import update_retention_indices as update

    try:
        summary = update(data_dir)
    except ValueError as e:
        print(f"retention indices of the runs not updated: {e}", file=sys.stderr)
        return 2
    if summary is None:
        return 0
    n = len(summary.indexed)
    print(
        f"retention indices of the runs updated: {n}"
        + (f", failed: {len(summary.index_failed)}" if summary.index_failed else "")
    )
    print_processing_failures(summary)
    return 1 if summary.any_failed else 0


def print_processing_failures(summary) -> None:
    """The runs whose processing failed (``esper.process.ProcessSummary``), stderr."""
    for name, error in summary.processing_failed:
        print(f"  PEAKS FAILED {name}: {error}", file=sys.stderr)
    for name, error in summary.fit_failed:
        print(f"  FIT FAILED {name}: {error}", file=sys.stderr)
    for name, error in getattr(summary, "index_failed", []):
        print(f"  RETENTION INDICES FAILED {name}: {error}", file=sys.stderr)
