"""``esper chromatograms``: unit mass chromatograms of h5 files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from esper.cli.common import (
    Command,
    add_block,
    add_files,
    add_ionisation,
    add_output,
    add_rt_range,
    add_segments,
    files,
)

logger = logging.getLogger(__name__)


HELP = "Unit mass chromatograms of h5 files (summed, baseline removed)"
DESCRIPTION = (
    "Read the spectra of h5 files (twutils), sum each scan around each unit "
    "mass, smooth and remove the arPLS baseline (peaxart), and write "
    "<run>.alpinac/<run>.unit_mass_chromatos.parquet (index rt, one column per unit "
    "mass and tic; files with several segments: seg1:69, ...)."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    import peaxart as px

    default_um = px.UnitMassConfig()
    add_files(parser)
    add_rt_range(parser, stop_note="e.g. when the filament failed")
    parser.add_argument(
        "--mass",
        type=float,
        nargs="+",
        help="unit masses (default: those of the spectra, as the legacy pipeline)",
    )
    parser.add_argument(
        "--mass-cal",
        type=Path,
        help="mass calibration file (*_mc.json, or legacy *_mc.txt); default: searched "
        "next to the h5 file",
    )
    add_segments(
        parser,
        "segments read (summed): 1, or 1:3 for 1 and 2. Required for files with "
        "several segments (EI/CI instruments); selects the calibration file",
    )
    add_ionisation(parser)
    parser.add_argument(
        "--mass-axis",
        choices=["drift", "end"],
        default="drift",
        help="m/z windows with the calibration at the rt of each scan (drift), or at the "
        "end of the run for all scans (end: legacy pipeline). Default %(default)s",
    )
    parser.add_argument(
        "--d-m",
        type=float,
        default=default_um.d_m,
        help="half width of the m/z window of each unit mass (default %(default)s)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        help=f"arPLS smoothness (default {default_um.baseline.lam:g}, "
        f"--legacy {px.BaselineConfig.legacy().lam:g})",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="do not remove the baseline"
    )
    parser.add_argument(
        "--no-steps",
        action="store_true",
        help="fit the baseline over the whole run, not separately between the steps of "
        "the acquisition (filament emission target, resonant ejection; from TPS2)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="reproduce the legacy pipeline (unit_mass_chromatos.parquet): --mass-axis "
        "end, --no-steps, lam 1e5, zero padding of the smoothing",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="do not write the raw sums and the baselines (.raw.parquet, "
        ".baseline.parquet next to the chromatograms)",
    )
    add_block(parser)
    add_output(parser)


def run(args: argparse.Namespace) -> int:
    import peaxart as px

    from esper.chromatograms import DEFAULT_STEPS, chromatograms_to_file

    paths = files(args.files)
    if not paths:
        return 1
    if args.output is not None and len(paths) > 1:
        raise SystemExit("--output can only be used with one file (use --output-dir).")
    baseline = {"method": None if args.no_baseline else "arpls"}
    baseline |= {} if args.lam is None else {"lam": args.lam}
    baseline = (
        px.BaselineConfig.legacy(**baseline)
        if args.legacy
        else px.BaselineConfig(**baseline)
    )
    config = px.UnitMassConfig(d_m=args.d_m, baseline=baseline)
    failed = 0
    for path in paths:
        try:
            written = chromatograms_to_file(
                path,
                output=args.output,
                output_dir=args.output_dir,
                masses=args.mass,
                segments=args.segments,
                calibration=args.mass_cal,
                rt_start=args.rt_start,
                rt_stop=args.rt_stop,
                mass_axis="end" if args.legacy else args.mass_axis,
                steps=None if args.legacy or args.no_steps else DEFAULT_STEPS,
                config=config,
                ionisation=args.ionisation,
                keep_raw=not args.no_raw,
                block=args.block,
            )
        except Exception as e:
            logger.exception(f"{path}: chromatograms failed")
            print(f"  CHROMATOGRAMS FAILED {path}: {e!r}", file=sys.stderr)
            failed += 1
            continue
        print(written)
    return 1 if failed else 0


COMMAND = Command(
    name="chromatograms",
    help=HELP,
    add_arguments=add_arguments,
    run=run,
    description=DESCRIPTION,
)
