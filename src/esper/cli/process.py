"""``esper process``: find, identify and fit the peaks of the runs of a data directory."""

from __future__ import annotations

import argparse
import sys

from esper.cli.common import (
    Command,
    add_compounds,
    add_data_dir,
    add_dates,
    add_method,
    add_workers,
    print_processing,
    print_processing_failures,
)

HELP = (
    "Process the runs of a data directory that need it: find and identify the peaks of "
    "the runs imported without (method of the data directory), and fit the peaks"
)
DESCRIPTION = (
    HELP + ". By default: the peaks of the runs not processed yet (or with another "
    "method), and the fit of the runs not fitted yet (or by an older esper). The method "
    "(target: target search, suspect: suspect screening) and the parameters of each "
    "step are in the 'processing:' of the esper.yaml of the data directory (see "
    "esper.processing.config). Unlike esper upload, the runs are not read again: only "
    "the database and the chromatograms of the data directory."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from esper.process import STEPS

    add_data_dir(parser)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        help="the steps: peaks (found and identified, then fitted if fit is a step too)"
        " and fit (default: both; the peaks only with a method)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="the steps of all the runs again (peaks: of the runs whose peaks esper"
        " found)",
    )
    add_method(parser)
    add_compounds(
        parser,
        "The runs not processed yet get only these compounds (the others: with -f"
        " later); those already processed are left as they are, unless -f: these"
        " compounds are found again in all of them",
    )
    parser.add_argument(
        "--overrides",
        action="store_true",
        help="also fit again the runs that a fit override applies to (made in the GUI:"
        " right click on a compound, Refit peaks)",
    )
    add_dates(parser)
    add_workers(parser)


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.process import process

    if not db.db_path(args.data_dir).is_file():
        print(f"No esper database at {db.db_path(args.data_dir)}", file=sys.stderr)
        return 1
    try:
        summary = process(
            args.data_dir,
            steps=args.steps,
            force=args.force,
            workers=args.workers,
            method=args.method,
            compounds=args.compounds,
            overrides=args.overrides,
            after=args.after,
            before=args.before,
            report=print,
        )
    except ValueError as e:  # ProcessingConfigError, or an invalid esper.yaml
        print(e, file=sys.stderr)
        return 2
    print_processing(summary)
    print_processing_failures(summary)
    return 1 if summary.any_failed else 0


COMMAND = Command(
    name="process",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
    description=DESCRIPTION,
)
