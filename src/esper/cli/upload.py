"""``esper upload``: import the runs of an instrument into a data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from esper.cli.common import (
    DATE_FORMATS,
    Command,
    add_data_dir,
    add_workers,
    date_argument,
)

HELP = (
    "Import the runs of an instrument into an esper data directory: .alpinac "
    "folders (tofwerk) or the chromatograms of a GCwerks directory (gcwerks)"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from esper.sources import INSTRUMENTS

    parser.add_argument(
        "source",
        type=Path,
        help="a .alpinac folder or a directory containing some; or a GCwerks directory, "
        "one of its years (e.g. gcdir/26) or a chromatogram file",
    )
    add_data_dir(parser, "created if needed")
    parser.add_argument(
        "-f", "--force", action="store_true", help="reimport unchanged measurements"
    )
    parser.add_argument(
        "--no-fit", action="store_true", help="do not fit the peaks (see esper fit)"
    )
    add_workers(parser)
    parser.add_argument(
        "--instrument",
        choices=INSTRUMENTS,
        help="instrument of the runs, declared in the esper.yaml of a new data directory "
        "(then optional, must be the same). gcwerks: only the chromatograms, not fitted",
    )
    parser.add_argument(
        "--add-aliases",
        action="store_true",
        help="gcwerks: add the peakid names that match no peak.list name to the aliases "
        "of the esper.yaml, as not targets (null; map the targets among them to their "
        "peak.list name), instead of stopping",
    )
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


def run(args: argparse.Namespace) -> int:
    from esper.sources import InstrumentError
    from esper.targets import TargetError
    from esper.upload import upload

    try:
        summary = upload(
            args.source,
            args.data_dir,
            force=args.force,
            fit=not args.no_fit,
            workers=args.workers,
            instrument=args.instrument,
            add_aliases=args.add_aliases,
            after=args.after,
            before=args.before,
        )
    except (InstrumentError, TargetError) as e:
        print(e, file=sys.stderr)
        return 2
    print(
        f"imported: {len(summary.imported)}, "
        f"unchanged: {len(summary.skipped_unchanged)}, "
        f"incomplete: {len(summary.skipped_incomplete)}, "
        f"failed: {len(summary.failed)}, "
        f"fit failed: {len(summary.fit_failed)}"
    )
    if args.after is not None or args.before is not None:
        print(f"outside the dates: {len(summary.skipped_dates)}")
    if summary.added_aliases:
        print(
            f"{len(summary.added_aliases)} names added to the aliases of"
            f" {args.data_dir / 'esper.yaml'} as not targets (null): map those that are"
            f" targets to their peak.list name: {', '.join(summary.added_aliases)}"
        )
    for name, error in summary.failed:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    for name, error in summary.fit_failed:
        print(f"  FIT FAILED {name}: {error}", file=sys.stderr)
    return 1 if summary.failed or summary.fit_failed else 0


COMMAND = Command(
    name="upload",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
