"""``esper fit``: fit the peaks of the measurements not fitted yet."""

from __future__ import annotations

import argparse
import logging
import sys

from esper.cli.common import Command, add_data_dir, add_workers

logger = logging.getLogger(__name__)


HELP = "Fit the peaks of the measurements not fitted yet, or fitted by an older esper"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f", "--force", action="store_true", help="fit all the measurements again"
    )
    add_data_dir(parser)
    add_workers(parser)


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.fitting import fit_measurement, fit_pool, summary, to_fit
    from esper.processing.config import read_processing_config

    if not db.db_path(args.data_dir).is_file():
        print(f"No esper database at {db.db_path(args.data_dir)}", file=sys.stderr)
        return 1
    try:  # the fit of the data directory (processing: fit: of its esper.yaml)
        config = read_processing_config(args.data_dir).fit
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    # create: brings an older schema up to date (fit columns)
    con = db.connect(args.data_dir, create=True)
    failed = 0
    try:
        todo = to_fit(con, force=args.force)
        print(f"measurements to fit: {len(todo)}")
        with fit_pool(args.workers) as pool:
            for mid, name in todo:
                try:
                    table = fit_measurement(
                        con, mid, args.data_dir, workers=1, pool=pool, config=config
                    )
                except Exception as e:
                    logger.exception(f"{name}: fit failed")
                    print(f"  FIT FAILED {name}: {e!r}", file=sys.stderr)
                    failed += 1
                    continue
                if table is None:
                    print(f"  {name}: no chromatograms")
                    continue
                print(f"  {name}: {summary(table)}")
    finally:
        con.close()
    return 1 if failed else 0


COMMAND = Command(
    name="fit",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
