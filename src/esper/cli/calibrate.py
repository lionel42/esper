"""``esper calibrate``: mass calibration of h5 files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from esper.cli.common import (
    Command,
    add_block,
    add_files,
    add_output,
    add_rt_range,
    add_segments,
    files,
)

logger = logging.getLogger(__name__)


HELP = "Mass calibration of h5 files, following the drift with the retention time"
DESCRIPTION = (
    "Read the spectra of h5 files (twutils), fit the mass calibration in rt "
    "windows on calibrant ions (masscal) and write <run>.alpinac/<run>.seg0_mc.json "
    "and the calibrant peaks (.peaks.csv), used by the extraction and esper upload."
)


def _calibrants(values: list[str] | None) -> list | None:
    """Formulas or m/z values."""
    if values is None:
        return None
    out = []
    for v in values:
        try:
            out.append(float(v))
        except ValueError:
            out.append(v)
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from masscal import CalibrationConfig, Mode

    default = CalibrationConfig()
    add_files(parser)
    parser.add_argument(
        "-c",
        "--calibrants-file",
        type=Path,
        help="yaml file of the calibrants (and initial calibration), as "
        "/shared/scripts/tof-processing/config_calib.yml",
    )
    parser.add_argument(
        "--calibrant",
        nargs="+",
        metavar="FORMULA",
        help="calibrants present during the run: formulas (CF3, Re+2, [13C]O) or m/z "
        "values (instead of those of --calibrants-file). Default: air, F, Re and PFTBA "
        "ions (masscal.DEFAULT_CALIBRANTS)",
    )
    parser.add_argument(
        "--spike",
        nargs="+",
        action="append",
        default=[],
        metavar=("RT", "FORMULA"),
        help="calibrants injected at RT s (repeatable): --spike 1200 CF3 C2F5",
    )
    parser.add_argument(
        "--mode",
        default=str(default.mode),
        choices=[m.value for m in Mode],
        help="calibration function (default %(default)s)",
    )
    parser.add_argument(
        "--averaging",
        type=float,
        default=default.averaging,
        help="duration of the rt windows, s (default %(default)s)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=default.spacing,
        help="interval between the rt windows, s (default %(default)s)",
    )
    add_rt_range(parser, "the calibrated rt range")
    parser.add_argument(
        "--mz-range",
        type=float,
        default=default.mz_range,
        help="half width of the calibrant domains, m/z (default %(default)s)",
    )
    parser.add_argument(
        "--max-ppm",
        type=float,
        default=default.max_ppm,
        help="calibrants with larger residuals are left out (default %(default)s)",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        help="calibration file to find the calibrants (default: the h5 calibration)",
    )
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="neutral masses of the formulas without charge (legacy alpinac)",
    )
    add_segments(
        parser,
        "segments calibrated (summed): 1, or 1:3 for 1 and 2. Required for files "
        "with several segments (EI/CI instruments); names the output (seg1, seg1-2)",
    )
    add_output(parser)
    parser.add_argument(
        "--no-peaks-csv", action="store_true", help="do not write the calibrant peaks"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="also write the plots (.<plot>.png next to the output, needs matplotlib)",
    )
    add_block(parser)


def run(args: argparse.Namespace) -> int:
    from masscal import Calibrants, CalibrationConfig

    from esper.calibrate import calibrate_to_file

    paths = files(args.files)
    if not paths:
        return 1
    if args.output is not None and len(paths) > 1:
        raise SystemExit("--output can only be used with one file.")
    try:
        spikes = {float(rt): _calibrants(cals) for rt, *cals in args.spike}
    except ValueError:
        raise SystemExit("--spike: expected RT FORMULA [FORMULA ...]") from None
    calibrants = _calibrants(args.calibrant)
    if args.calibrants_file is not None:
        from_file = Calibrants.from_yaml(args.calibrants_file)
        calibrants = (
            from_file
            if calibrants is None
            else Calibrants(calibrants, from_file.spikes, from_file.initial)
        )
    config = CalibrationConfig(
        mode=args.mode,
        averaging=args.averaging,
        spacing=args.spacing,
        mz_range=args.mz_range,
        max_ppm=args.max_ppm,
        ionised=not args.neutral,
    )
    failed = 0
    for path in paths:
        try:
            result, written = calibrate_to_file(
                path,
                output=args.output,
                output_dir=args.output_dir,
                peaks_csv=not args.no_peaks_csv,
                plots=args.plots,
                calibrants=calibrants,
                spikes=spikes,
                rt_start=args.rt_start,
                rt_stop=args.rt_stop,
                initial=args.initial,
                segments=args.segments,
                config=config,
                block=args.block,
            )
        except Exception as e:
            logger.exception(f"{path}: calibration failed")
            print(f"  CALIBRATION FAILED {path}: {e!r}", file=sys.stderr)
            failed += 1
            continue
        print(result.summary())
        print(f"  {written}")
    return 1 if failed else 0


COMMAND = Command(
    name="calibrate",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
    description=DESCRIPTION,
)
