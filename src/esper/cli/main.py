"""Command line interface: ``esper gui``, ``esper upload``, ``esper integrate`` and ``esper targets``."""

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

    summary = upload(args.source, args.data_dir, force=args.force, integrate=not args.skip_integration)
    print(
        f"imported: {len(summary.imported)}, "
        f"unchanged: {len(summary.skipped_unchanged)}, "
        f"incomplete: {len(summary.skipped_incomplete)}, "
        f"failed: {len(summary.failed)}"
    )
    for name, error in summary.failed:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    return 1 if summary.failed else 0


def _integrate(args: argparse.Namespace) -> int:
    import pandas as pd

    from esper import db
    from esper.integration import integrate_measurements

    con = db.connect(args.data_dir, create=True)
    try:
        ids = None
        if args.since or args.until:
            m = pd.read_sql("SELECT measurement_id, acq_datetime FROM measurement", con)
            acq = pd.to_datetime(m["acq_datetime"])
            keep = pd.Series(True, index=m.index)
            if args.since:
                keep &= acq >= pd.Timestamp(args.since)
            if args.until:
                keep &= acq <= pd.Timestamp(args.until)
            ids = m.loc[keep, "measurement_id"].tolist()

        def progress(done: int, total: int, name: str) -> None:
            if sys.stderr.isatty():
                print(f"\r{done}/{total} {name:<40}", end="" if done < total else "\n", file=sys.stderr, flush=True)

        summary = integrate_measurements(
            con, args.data_dir, ids, compound_name=args.name, rt_range=args.rt, force=args.force, progress=progress
        )
    finally:
        con.close()
    print(
        f"integrated: {len(summary.integrated)}, "
        f"already integrated: {len(summary.skipped_integrated)}, "
        f"no chromatogram: {len(summary.no_chromatogram)}, "
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


def _install(args: argparse.Namespace) -> int:
    from esper.install import install, is_root, uninstall

    system = args.system
    if system and not is_root() and args.prefix is None:
        print("Installing for all users needs root, run it with sudo:", file=sys.stderr)
        print(f"  sudo {sys.argv[0]} install --system" + (f' --data-dir "{args.data_dir}"' if args.data_dir else ""), file=sys.stderr)
        return 1
    if args.uninstall:
        for path in uninstall(system, args.prefix):
            print(f"removed {path}")
        return 0
    data_dir = args.data_dir or os.environ.get("ESPER_DATA_DIR")
    for path in install(system, data_dir, args.desktop_shortcut, args.prefix):
        print(f"installed {path}")
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
    p_upload.add_argument("--skip-integration", action="store_true", help="do not compute the peak areas (see 'esper integrate')")
    p_upload.set_defaults(func=_upload)

    p_integrate = sub.add_parser("integrate", help="Compute the peak areas of the measurements not integrated yet")
    p_integrate.add_argument("data_dir", type=Path, help="esper data directory")
    p_integrate.add_argument("-f", "--force", action="store_true", help="also redo the measurements already integrated")
    p_integrate.add_argument("--since", help="only measurements acquired from this date/time (e.g. 2026-10-01)")
    p_integrate.add_argument("--until", help="only measurements acquired until this date/time")
    p_integrate.add_argument("--name", help="only the compounds with this name")
    p_integrate.add_argument("--rt", type=float, nargs=2, metavar=("FROM", "TO"), help="only the compounds with a mean rt in this range [s]")
    p_integrate.set_defaults(func=_integrate)

    p_targets = sub.add_parser("targets", help="Import the target compounds (replaces the current ones)")
    p_targets.add_argument("source", type=Path, help="target list, e.g. substances.xlsx")
    p_targets.add_argument("data_dir", type=Path, help="esper data directory")
    p_targets.add_argument("--ri-references", type=Path, help="retention index references (default: alpinac's)")
    p_targets.set_defaults(func=_targets)

    p_install = sub.add_parser(
        "install", help="Add esper to the application menu (and the 'esper' command to the PATH)"
    )
    p_install.add_argument("--system", action="store_true", help="for all users, in /usr/local (needs sudo)")
    p_install.add_argument("--data-dir", type=Path, help="data directory opened by the menu entry (default: $ESPER_DATA_DIR)")
    p_install.add_argument("--desktop-shortcut", action="store_true", help="also put a shortcut on your desktop")
    p_install.add_argument("--uninstall", action="store_true", help="remove what install added")
    p_install.add_argument("--prefix", type=Path, help=argparse.SUPPRESS)
    p_install.set_defaults(func=_install)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
