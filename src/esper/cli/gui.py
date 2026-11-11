"""``esper gui``: open the GUI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from esper.cli.common import Command, add_data_dir

HELP = "Open the GUI"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_data_dir(parser, "default: $ESPER_DATA_DIR", optional=True)


def run(args: argparse.Namespace) -> int:
    from esper.gui.main_window import run as run_gui

    data_dir = args.data_dir or os.environ.get("ESPER_DATA_DIR")
    return run_gui(Path(data_dir) if data_dir else None)


COMMAND = Command(
    name="gui",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
)
