"""``esper plot-calibration``: plots of mass calibrations written by ``esper calibrate``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from esper.cli.common import (
    Command,
    add_files,
    add_ionisation,
    add_segments,
    files,
)

logger = logging.getLogger(__name__)


HELP = "Plots of mass calibrations written by esper calibrate"
DESCRIPTION = (
    "Plot *_mc.json files and their calibrant peaks (.peaks.csv next to "
    "them; without them only the parameters and the drift). The plots are written "
    "next to the calibration file."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_files(
        parser,
        "*_mc.json (or legacy *_mc.txt) files, or h5 files (their calibration file)",
    )
    parser.add_argument(
        "--peaks", type=Path, help="calibrant peaks file (one input file only)"
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        dest="plots",
        metavar="NAME",
        help="plots to make: parameters drift residuals peak_shape (default all)",
    )
    add_segments(parser, "segments of the calibration file of the h5 files (1, 1:3)")
    add_ionisation(parser)
    parser.add_argument("--show", action="store_true", help="show the plots")


def run(args: argparse.Namespace) -> int:
    import twutils
    from masscal.plots import PLOTS, plot_outputs

    paths = files(args.files)
    if not paths:
        return 1
    if args.peaks is not None and len(paths) > 1:
        raise SystemExit("--peaks can only be used with one file.")
    failed = 0
    for path in paths:
        mc_file = path
        if path.suffix == ".h5":
            candidates = [
                twutils.mc_path(path, args.segments),
                *twutils.legacy_mc_paths(path, args.ionisation),
            ]
            mc_file = next((c for c in candidates if c.is_file()), candidates[0])
        try:
            pngs = plot_outputs(
                mc_file, args.peaks, args.plots or PLOTS, show=args.show
            )
        except Exception as e:
            logger.exception(f"{mc_file}: plots failed")
            print(f"  PLOTS FAILED {mc_file}: {e!r}", file=sys.stderr)
            failed += 1
            continue
        print(mc_file)
        for png in pngs.values():
            print(f"  {png}")
    return 1 if failed else 0


COMMAND = Command(
    name="plot-calibration",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
    description=DESCRIPTION,
)
