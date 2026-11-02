"""Command line interface: ``esper gui``, ``esper upload``, ``esper fit`` and ``esper targets``."""

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

    summary = upload(
        args.source,
        args.data_dir,
        force=args.force,
        fit=not args.no_fit,
        workers=args.workers,
    )
    print(
        f"imported: {len(summary.imported)}, "
        f"unchanged: {len(summary.skipped_unchanged)}, "
        f"incomplete: {len(summary.skipped_incomplete)}, "
        f"failed: {len(summary.failed)}, "
        f"fit failed: {len(summary.fit_failed)}"
    )
    for name, error in summary.failed:
        print(f"  FAILED {name}: {error}", file=sys.stderr)
    for name, error in summary.fit_failed:
        print(f"  FIT FAILED {name}: {error}", file=sys.stderr)
    return 1 if summary.failed or summary.fit_failed else 0


def _fit(args: argparse.Namespace) -> int:
    from esper import db
    from esper.fitting import SOURCE_FIT, fit_measurement

    con = db.connect(args.data_dir, create=True)
    failed = 0
    try:
        todo = con.execute(
            "SELECT measurement_id, name FROM measurement WHERE fit_version IS NULL"
            " ORDER BY acq_datetime, name"
        ).fetchall()
        print(f"measurements to fit: {len(todo)}")
        for mid, name in todo:
            try:
                table = fit_measurement(con, mid, args.data_dir, args.workers)
            except Exception as e:
                logging.exception(f"{name}: fit failed")
                print(f"  FIT FAILED {name}: {e!r}", file=sys.stderr)
                failed += 1
                continue
            if table is None:
                print(f"  {name}: no chromatograms")
                continue
            print(
                f"  {name}: {int(table['fit_rt_s'].notna().sum())} of {len(table)} peaks fitted,"
                f" {int((table['source'] == SOURCE_FIT).sum())} split or added by the fit"
            )
    finally:
        con.close()
    return 1 if failed else 0


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
        print(
            f"  sudo {sys.argv[0]} install --system"
            + (f' --data-dir "{args.data_dir}"' if args.data_dir else ""),
            file=sys.stderr,
        )
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
    p_gui.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        help="esper data directory (default: $ESPER_DATA_DIR)",
    )
    p_gui.set_defaults(func=_gui)

    p_upload = sub.add_parser(
        "upload", help="Import .alpinac folders into an esper data directory"
    )
    p_upload.add_argument(
        "source", type=Path, help="a .alpinac folder or a directory containing some"
    )
    p_upload.add_argument(
        "data_dir", type=Path, help="esper data directory (created if needed)"
    )
    p_upload.add_argument(
        "-f", "--force", action="store_true", help="reimport unchanged measurements"
    )
    p_upload.add_argument(
        "--no-fit", action="store_true", help="do not fit the peaks (see esper fit)"
    )
    p_upload.add_argument(
        "-j", "--workers", type=int, help="processes of the fit (default: all CPUs)"
    )
    p_upload.set_defaults(func=_upload)

    p_fit = sub.add_parser(
        "fit",
        help="Fit the peaks of the measurements not fitted yet (to fit one again: upload --force)",
    )
    p_fit.add_argument("data_dir", type=Path, help="esper data directory")
    p_fit.add_argument(
        "-j", "--workers", type=int, help="processes of the fit (default: all CPUs)"
    )
    p_fit.set_defaults(func=_fit)

    p_targets = sub.add_parser(
        "targets", help="Import the target compounds (replaces the current ones)"
    )
    p_targets.add_argument(
        "source", type=Path, help="target list, e.g. substances.xlsx"
    )
    p_targets.add_argument("data_dir", type=Path, help="esper data directory")
    p_targets.add_argument(
        "--ri-references",
        type=Path,
        help="retention index references (default: alpinac's)",
    )
    p_targets.set_defaults(func=_targets)

    p_install = sub.add_parser(
        "install",
        help="Add esper to the application menu (and the 'esper' command to the PATH)",
    )
    p_install.add_argument(
        "--system",
        action="store_true",
        help="for all users, in /usr/local (needs sudo)",
    )
    p_install.add_argument(
        "--data-dir",
        type=Path,
        help="data directory opened by the menu entry (default: $ESPER_DATA_DIR)",
    )
    p_install.add_argument(
        "--desktop-shortcut",
        action="store_true",
        help="also put a shortcut on your desktop",
    )
    p_install.add_argument(
        "--uninstall", action="store_true", help="remove what install added"
    )
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
