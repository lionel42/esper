"""The command line (esper.cli): one module per command, -v and --debug for all of
them."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import replace

import pytest

from esper.cli import main as cli


@pytest.mark.parametrize(
    ("argv", "level"),
    [
        (["-v", "fit", "data"], logging.INFO),
        (["fit", "data", "-v"], logging.INFO),
        (["fit", "--verbose", "data"], logging.INFO),
        (["fit", "data"], logging.WARNING),
        (["upload", "src", "data", "-v"], logging.INFO),
        (["chromatograms", "run.h5", "-v"], logging.INFO),
        (["--debug", "fit", "data"], logging.DEBUG),
        (["fit", "data", "-d"], logging.DEBUG),
        (["fit", "data", "-v", "--debug"], logging.DEBUG),
    ],
)
def test_verbose_before_or_after_the_command(monkeypatch, argv, level):
    """-v and --debug apply to every command, given before or after it."""
    levels = []
    monkeypatch.setattr(
        logging, "basicConfig", lambda **kwargs: levels.append(kwargs["level"])
    )
    commands = [replace(c, run=lambda args: 0) for c in cli.COMMANDS]
    monkeypatch.setattr(cli, "COMMANDS", commands)
    assert cli.main(argv) == 0
    assert levels == [level]


def test_every_command_is_listed_with_its_options(capsys):
    """esper --help lists all the commands; esper <command> --help gives its options."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    listed = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command.name in listed
        with pytest.raises(SystemExit):
            cli.main([command.name, "--help"])
        out = " ".join(capsys.readouterr().out.split())
        assert (command.description or command.help)[:60] in out  # what it does
        assert "-v, --verbose" in out
        assert "-d, --debug" in out
    with pytest.raises(SystemExit):
        cli.main(["uplod", "src", "data"])  # misspelled: argparse lists the commands
    assert "invalid choice: 'uplod'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["chromatograms", "run.h5", "--segments", "a"])
    assert "not segments: 'a'" in capsys.readouterr().err


def test_only_the_options_of_the_command_are_built():
    """The options of calibrate and chromatograms import masscal and peaxart (~1 s):
    not for another command."""
    code = (
        "import sys; from esper.cli.main import build_parser;"
        " build_parser(['upload', 'src', 'data']).parse_args(['upload', 'src', 'data']);"
        " print('masscal' in sys.modules, 'peaxart' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.split() == ["False", "False"]
