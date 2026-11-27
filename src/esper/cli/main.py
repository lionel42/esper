"""Command line interface: ``esper <command>``, one module of ``esper.cli`` per command.

Each command module has a ``COMMAND`` (:class:`esper.cli.common.Command`: name, help,
``add_arguments(parser)``, ``run(args) -> int``); the options of several commands are
added by the ``add_*`` functions of :mod:`esper.cli.common`. To add a command, write its
module and add its ``COMMAND`` to :data:`COMMANDS`.

Only the options of the command given are built: some read their defaults from heavy
packages (masscal, peaxart: ~1 s to import), and the others do not need them. The heavy
imports of a command go inside its functions.
"""

from __future__ import annotations

import argparse
import logging
import sys

from esper.cli import (
    calibrate,
    chromatograms,
    gui,
    info,
    install,
    plot_calibration,
    process,
    ri_refs,
    targets,
    upload,
)
from esper.cli.common import Command

COMMANDS: list[Command] = [
    gui.COMMAND,
    upload.COMMAND,
    process.COMMAND,
    info.COMMAND,
    targets.COMMAND,
    ri_refs.COMMAND,
    calibrate.COMMAND,
    plot_calibration.COMMAND,
    chromatograms.COMMAND,
    install.COMMAND,
]
"""The commands, in the order of ``esper --help``."""

VERBOSE_HELP = "show the progress (log level INFO)"
DEBUG_HELP = "show the debug messages (log level DEBUG)"


def _command_name(argv: list[str]) -> str | None:
    """The command of the arguments: the first that is not an option (the options before
    the command take no value)."""
    return next((arg for arg in argv if not arg.startswith("-")), None)


def build_parser(argv: list[str]) -> argparse.ArgumentParser:
    """The parser of ``esper``, with the options of the command of ``argv`` only (the
    others: their name and help)."""
    parser = argparse.ArgumentParser(prog="esper")
    parser.add_argument("-v", "--verbose", action="store_true", help=VERBOSE_HELP)
    parser.add_argument("-d", "--debug", action="store_true", help=DEBUG_HELP)
    sub = parser.add_subparsers(dest="command", required=True)
    # -v and --debug also after the command (esper upload ... -v); not given there: as
    # before it
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help=VERBOSE_HELP,
    )
    common.add_argument(
        "-d", "--debug", action="store_true", default=argparse.SUPPRESS, help=DEBUG_HELP
    )
    name = _command_name(argv)
    for command in COMMANDS:
        command_parser = sub.add_parser(
            command.name,
            parents=[common],
            help=command.help,
            description=command.description or command.help,
        )
        if command.name == name:
            command.add_arguments(command_parser)
        command_parser.set_defaults(func=command.run)
    help_parser = sub.add_parser(
        "help",
        help="Show this help, or the help of a command (esper help upload)",
        description="Show the help of esper (as esper --help), or of a command (as "
        "esper <command> --help).",
    )
    help_parser.add_argument(
        "help_command",
        nargs="?",
        metavar="command",
        choices=[command.name for command in COMMANDS],
        help="the command to show the help of",
    )
    return parser


def _help_argv(argv: list[str]) -> list[str] | None:
    """The arguments that show the help asked by ``esper help [command]`` (None: not a help
    command or an invalid one, which the parser reports)."""
    rest = argv[argv.index("help") + 1 :]
    if not rest:
        return ["--help"]
    if len(rest) == 1 and rest[0] in {command.name for command in COMMANDS}:
        return [rest[0], "--help"]
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if _command_name(argv) == "help":
        argv = _help_argv(argv) or argv
    args = build_parser(argv).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG
        if args.debug
        else logging.INFO
        if args.verbose
        else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
