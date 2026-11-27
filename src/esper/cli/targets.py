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
        help="retention index references (CSV, see esper ri-refs), stored in the data"
        " directory; default: those of the data directory, else esper's seed",
    )


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    from esper import db
    from esper.retention_index import (
        read_references,
        store_references,
        update_target_ris,
    )
    from esper.targets import load_targets, store_targets

    options = {"ri_references": args.ri_references} if args.ri_references else {}
    targets = load_targets(args.source, **options)
    con = db.connect(args.data_dir, create=True)
    try:
        store_targets(con, targets, str(Path(args.source).resolve()))
        # the references of the data directory, unless others are given
        if targets.ri_references is not None and (
            args.ri_references or read_references(con, enabled=False).empty
        ):
            store_references(con, targets.ri_references)
            print(f"retention index references: {len(targets.ri_references)} stored")
        else:
            update_target_ris(con)
        targets.table["ri"] = pd.read_sql(
            "SELECT name, ri FROM target", con, index_col="name"
        )["ri"].reindex(targets.table.index)
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
