"""``esper info``: describe a data directory."""

from __future__ import annotations

import argparse
import sys

from esper.cli.common import Command, add_data_dir

HELP = "Describe an esper data directory: its files, database, measurements and targets"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_data_dir(parser)


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.data_dir_info import as_text, describe

    try:
        con = db.connect(args.data_dir, readonly=True)
    except (FileNotFoundError, ValueError, db.SchemaVersionError) as e:
        print(e, file=sys.stderr)
        return 1
    try:
        print(as_text(describe(args.data_dir, con)))
    finally:
        con.close()
    return 0


COMMAND = Command(
    name="info",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
