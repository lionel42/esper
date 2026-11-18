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
    parser.add_argument(
        "--overrides",
        action="store_true",
        help="also fit again the measurements that a fit override applies to (made in"
        " the GUI: right click on a compound, Refit peaks)",
    )
    add_data_dir(parser)
    add_workers(parser)


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.fitting import fit_measurement, fit_pool, summary, to_fit
    from esper.processing.config import read_processing_config
    from esper.processing.identification import TargetSearch
    from esper.sources import run_label
    from esper.target_index import TargetIndex

    if not db.db_path(args.data_dir).is_file():
        print(f"No esper database at {db.db_path(args.data_dir)}", file=sys.stderr)
        return 1
    try:  # the fit of the data directory (processing: fit: of its esper.yaml)
        processing = read_processing_config(args.data_dir)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    # create: brings an older schema up to date (fit columns)
    con = db.connect(args.data_dir, create=True)
    failed = 0
    try:
        # the runs identified by target search: their targets in the fit (read once)
        target_search = TargetSearch(TargetIndex.from_db(con), processing.target)
        todo = to_fit(con, force=args.force)
        if args.overrides and not args.force:
            from esper.fit_overrides import read_overrides, runs_covered

            known = set(todo)
            covered = runs_covered(con, read_overrides(con))
            todo += [run for run in covered if run not in known]
        print(f"measurements to fit: {len(todo)}")
        width = max((len(name) for _, name in todo), default=0)
        with fit_pool(args.workers) as pool:
            for mid, name in todo:
                try:
                    table = fit_measurement(
                        con,
                        mid,
                        args.data_dir,
                        workers=1,
                        pool=pool,
                        config=processing.fit,
                        target_search=target_search,
                    )
                except Exception as e:
                    logger.exception(f"{name}: fit failed")
                    print(f"  FIT FAILED {name}: {e!r}", file=sys.stderr)
                    failed += 1
                    continue
                if table is None:
                    print(f"  {run_label(name, width)} no chromatograms")
                    continue
                print(f"  {run_label(name, width)} {summary(table)}")
    finally:
        con.close()
    return 1 if failed else 0


COMMAND = Command(
    name="fit",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
