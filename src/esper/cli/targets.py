"""``esper targets``: import the target compounds."""

from __future__ import annotations

import argparse
from pathlib import Path

from esper.cli.common import Command, add_data_dir

HELP = "Import the target compounds (replaces the current ones)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path, help="target list, e.g. substances.xlsx")
    add_data_dir(parser)
    parser.add_argument(
        "--ri-references",
        type=Path,
        help="retention index references (default: alpinac's)",
    )


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.targets import load_targets, store_targets

    options = {"ri_references": args.ri_references} if args.ri_references else {}
    targets = load_targets(args.source, **options)
    con = db.connect(args.data_dir, create=True)
    try:
        store_targets(con, targets, str(Path(args.source).resolve()))
    finally:
        con.close()
    n_ri = int(targets.table["ri"].notna().sum())
    n_ions = targets.ions["name"].nunique()
    n_spectrum = targets.ions.loc[
        targets.ions["spectrum_ref"].notna(), "name"
    ].nunique()
    print(
        f"targets: {len(targets.table)} ({n_ri} with retention index, {n_ions} with expected ions,"
        f" {n_spectrum} of them from their reference spectrum), aliases: {len(targets.aliases)}"
    )
    return 0


COMMAND = Command(
    name="targets",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
