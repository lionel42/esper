"""``esper install``: add esper to the application menu."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from esper.cli.common import Command

HELP = "Add esper to the application menu (and the 'esper' command to the PATH)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--system",
        action="store_true",
        help="for all users, in /usr/local (needs sudo)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="data directory opened by the menu entry (default: $ESPER_DATA_DIR)",
    )
    parser.add_argument(
        "--desktop-shortcut",
        action="store_true",
        help="also put a shortcut on your desktop",
    )
    parser.add_argument(
        "--uninstall", action="store_true", help="remove what install added"
    )
    parser.add_argument("--prefix", type=Path, help=argparse.SUPPRESS)


def run(args: argparse.Namespace) -> int:
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


COMMAND = Command(
    name="install",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
