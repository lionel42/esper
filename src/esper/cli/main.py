"""Command line interface: ``esper gui``, ``esper upload`` and ``esper targets``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _gui(args: argparse.Namespace) -> int:
    from esper.gui.main_window import run

    data_dir = args.data_dir or os.environ.get("ESPER_DATA_DIR")
    return run(Path(data_dir) if data_dir else None)


def _upload(args: argparse.Namespace) -> int:
    from esper.upload import upload

    summary = upload(args.source, args.data_dir, force=args.force)
    print(
        f"imported: {len(summary.imported)}, "
        f"unchanged: {len(summary.skipped_unchanged)}, "
        f"incomplete: {len(summary.skipped_incomplete)}, "
        f"failed: {len(summary.failed)}"
    )
    for name, error in summary.failed:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    return 1 if summary.failed else 0


def _targets(args: argparse.Namespace) -> int:
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
    print(
        f"targets: {len(targets.table)} ({n_ri} with retention index, {n_ions} with expected ions), "
        f"aliases: {len(targets.aliases)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="esper")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gui = sub.add_parser("gui", help="Open the GUI")
    p_gui.add_argument("data_dir", type=Path, nargs="?", help="esper data directory (default: $ESPER_DATA_DIR)")
    p_gui.set_defaults(func=_gui)

    p_upload = sub.add_parser("upload", help="Import .alpinac folders into an esper data directory")
    p_upload.add_argument("source", type=Path, help="a .alpinac folder or a directory containing some")
    p_upload.add_argument("data_dir", type=Path, help="esper data directory (created if needed)")
    p_upload.add_argument("-f", "--force", action="store_true", help="reimport unchanged measurements")
    p_upload.set_defaults(func=_upload)

    p_targets = sub.add_parser("targets", help="Import the target compounds (replaces the current ones)")
    p_targets.add_argument("source", type=Path, help="target list, e.g. substances.xlsx")
    p_targets.add_argument("data_dir", type=Path, help="esper data directory")
    p_targets.add_argument("--ri-references", type=Path, help="retention index references (default: alpinac's)")
    p_targets.set_defaults(func=_targets)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
