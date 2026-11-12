"""``esper peaks``: find, identify and fit the peaks of the measurements without peaks."""

from __future__ import annotations

import argparse
import logging
import sys

from esper.cli.common import Command, add_data_dir, add_workers

logger = logging.getLogger(__name__)

HELP = (
    "Find the peaks in the chromatograms of the measurements imported without peaks, "
    "identify them with the method of the data directory and fit them"
)
DESCRIPTION = (
    HELP + ". The method (target: target search, suspect: suspect screening) and the "
    "parameters of each step are in the 'processing:' of the esper.yaml of the data "
    "directory (see esper.processing.config)."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from esper.processing import METHODS

    add_data_dir(parser)
    parser.add_argument(
        "--method",
        choices=METHODS,
        help="how the compounds are identified, declared in the esper.yaml of the data "
        "directory (then optional, must be the same): target (expected rt and ions of "
        "the targets) or suspect (spectra in the libraries)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="process again all the measurements whose peaks esper found",
    )
    parser.add_argument(
        "--no-fit", action="store_true", help="do not fit the peaks (see esper fit)"
    )
    add_workers(parser)


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.fitting import fit_pool
    from esper.processing import METHODS
    from esper.processing.config import (
        ProcessingConfigError,
        read_processing_config,
        resolve_method,
    )
    from esper.processing.pipeline import Processor, summary, to_process

    if not db.db_path(args.data_dir).is_file():
        print(f"No esper database at {db.db_path(args.data_dir)}", file=sys.stderr)
        return 1
    try:
        resolve_method(args.data_dir, args.method)
        config = read_processing_config(args.data_dir)
        if config.method is None:
            raise ProcessingConfigError(
                f"{args.data_dir}: no method to identify the compounds; declare it"
                f" (--method {'|'.join(METHODS)})"
            )
    except ValueError as e:  # ProcessingConfigError, or an invalid esper.yaml
        print(e, file=sys.stderr)
        return 2
    con = db.connect(args.data_dir, create=True)
    processor = Processor(con, args.data_dir, config)
    failed = 0
    try:
        todo = to_process(con, processor.method, force=args.force)
        print(f"measurements to process ({processor.method}): {len(todo)}")
        with fit_pool(1 if args.no_fit else args.workers) as pool:
            for mid, name in todo:
                try:
                    table = processor.process(
                        mid, workers=1, pool=pool, fit=not args.no_fit
                    )
                except Exception as e:
                    logger.exception(f"{name}: processing failed")
                    print(f"  FAILED {name}: {e!r}", file=sys.stderr)
                    failed += 1
                    continue
                if table is None:
                    print(f"  {name}: no chromatograms")
                    continue
                print(f"  {name}: {summary(table, con, mid)}")
    finally:
        con.close()
    return 1 if failed else 0


COMMAND = Command(
    name="peaks",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
    description=DESCRIPTION,
)
