"""``esper ri-refs``: the retention index references of a data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from esper.cli.common import (
    Command,
    add_data_dir,
    add_no_update,
    update_retention_indices,
)

HELP = "List, import or export the retention index references"
DESCRIPTION = """\
The retention index references of a data directory (esper.retention_index): compounds of
known RI found in every run, which give the RI of its compounds, and the RIs of the
targets (from their rt in the reference run). The CSV file has the columns name, ri,
names (other candidate names, separated by ';'), occurrence, enabled and note; the legacy
format (compound, ri) is read too. Importing replaces the references and the RIs of the
targets, then the runs whose peaks esper found get their references, RIs and
identification again (esper process --steps ri; peaks, fit and identifications by hand
kept)."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", help="show the references")
    add_data_dir(listing)
    importing = sub.add_parser(
        "import", help="replace the references by those of a CSV file"
    )
    importing.add_argument("file", type=Path, help="CSV file of references")
    add_data_dir(importing)
    add_no_update(importing)
    exporting = sub.add_parser("export", help="write the references to a CSV file")
    add_data_dir(exporting)
    exporting.add_argument("file", type=Path, help="CSV file to write")


def run(args: argparse.Namespace) -> int:
    from esper import db
    from esper.retention_index import (
        read_reference_csv,
        read_references,
        store_references,
        write_reference_csv,
    )

    con = db.connect(args.data_dir, create=args.action == "import")
    try:
        if args.action == "import":
            refs = read_reference_csv(args.file)
            store_references(con, refs)
            (n_ri,) = con.execute("SELECT count(ri) FROM target").fetchone()
            print(
                f"{len(refs)} references ({int(refs['enabled'].sum())} enabled);"
                f" {n_ri} targets with a retention index"
            )
        else:
            refs = read_references(con, enabled=False)
            if args.action == "export":
                write_reference_csv(refs, args.file)
                print(f"{len(refs)} references written to {args.file}")
            else:
                _print_references(refs)
            return 0
    finally:
        con.close()
    return 0 if args.no_update else update_retention_indices(args.data_dir)


def _print_references(refs) -> None:
    if refs.empty:
        print("no references (esper targets, or esper ri-refs import)")
        return
    table = refs.assign(
        occurrence=refs["occurrence"]
        .astype(object)
        .where(refs["occurrence"].notna(), ""),
        enabled=refs["enabled"].map({1: "yes", 0: "no"}),
    ).fillna("")
    print(table.to_string(index=False))


COMMAND = Command(
    name="ri-refs",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
    description=DESCRIPTION,
)
