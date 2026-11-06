"""Command line interface: ``esper gui``, ``esper upload``, ``esper fit``, ``esper targets``,
``esper calibrate``, ``esper plot-calibration`` and ``esper chromatograms``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    from esper.fitting import fit_measurement, fit_pool, summary, to_fit

    if not db.db_path(args.data_dir).is_file():
        print(f"No esper database at {db.db_path(args.data_dir)}", file=sys.stderr)
        return 1
    # create: brings an older schema up to date (fit columns)
    con = db.connect(args.data_dir, create=True)
    failed = 0
    try:
        todo = to_fit(con, force=args.force)
        print(f"measurements to fit: {len(todo)}")
        with fit_pool(args.workers) as pool:
            for mid, name in todo:
                try:
                    table = fit_measurement(
                        con, mid, args.data_dir, workers=1, pool=pool
                    )
                except Exception as e:
                    logger.exception(f"{name}: fit failed")
                    print(f"  FIT FAILED {name}: {e!r}", file=sys.stderr)
                    failed += 1
                    continue
                if table is None:
                    print(f"  {name}: no chromatograms")
                    continue
                print(f"  {name}: {summary(table)}")
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
    n_spectrum = targets.ions.loc[
        targets.ions["spectrum_ref"].notna(), "name"
    ].nunique()
    print(
        f"targets: {len(targets.table)} ({n_ri} with retention index, {n_ions} with expected ions,"
        f" {n_spectrum} of them from their reference spectrum), aliases: {len(targets.aliases)}"
    )
    return 0


def _files(patterns: list[str]) -> list[Path]:
    """The files of the arguments (glob patterns expanded, in order, without duplicates)."""
    import glob

    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            print(f"No file matches {pattern}", file=sys.stderr)
        files += [Path(m) for m in matches if Path(m) not in files]
    return files


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


def _segments(value: str | None) -> tuple[int, int] | None:
    """``1`` -> (1, 2), ``1:3`` -> (1, 3)."""
    if value is None:
        return None
    start, _, stop = value.partition(":")
    return (int(start), int(stop) if stop else int(start) + 1)


def _calibrate(args: argparse.Namespace) -> int:
    from masscal import Calibrants, CalibrationConfig

    from esper.calibrate import calibrate_to_file

    files = _files(args.files)
    if not files:
        return 1
    if args.output is not None and len(files) > 1:
        raise SystemExit("--output can only be used with one file.")
    try:
        spikes = {float(rt): _calibrants(cals) for rt, *cals in args.spike}
    except ValueError:
        raise SystemExit("--spike: expected RT FORMULA [FORMULA ...]") from None
    calibrants = _calibrants(args.calibrant)
    if args.calibrants_file is not None:
        from_file = Calibrants.from_yaml(args.calibrants_file)
        calibrants = from_file if calibrants is None else Calibrants(
            calibrants, from_file.spikes, from_file.initial
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
    for path in files:
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
                segments=_segments(args.segments),
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


def _plot_calibration(args: argparse.Namespace) -> int:
    import twutils
    from masscal.plots import PLOTS, plot_outputs

    files = _files(args.files)
    if not files:
        return 1
    if args.peaks is not None and len(files) > 1:
        raise SystemExit("--peaks can only be used with one file.")
    failed = 0
    for path in files:
        mc_file = path
        if path.suffix == ".h5":
            candidates = [
                twutils.mc_path(path, _segments(args.segments)),
                *twutils.legacy_mc_paths(path, args.ionisation),
            ]
            mc_file = next((c for c in candidates if c.is_file()), candidates[0])
        try:
            pngs = plot_outputs(mc_file, args.peaks, args.plots or PLOTS, show=args.show)
        except Exception as e:
            logger.exception(f"{mc_file}: plots failed")
            print(f"  PLOTS FAILED {mc_file}: {e!r}", file=sys.stderr)
            failed += 1
            continue
        print(mc_file)
        for png in pngs.values():
            print(f"  {png}")
    return 1 if failed else 0


def _chromatograms(args: argparse.Namespace) -> int:
    import peaxart

    from esper.chromatograms import DEFAULT_STEPS, chromatograms_to_file

    files = _files(args.files)
    if not files:
        return 1
    if args.output is not None and len(files) > 1:
        raise SystemExit("--output can only be used with one file (use --output-dir).")
    baseline = {"method": None if args.no_baseline else "arpls"}
    baseline |= {} if args.lam is None else {"lam": args.lam}
    baseline = (
        peaxart.BaselineConfig.legacy(**baseline)
        if args.legacy
        else peaxart.BaselineConfig(**baseline)
    )
    config = peaxart.UnitMassConfig(d_m=args.d_m, baseline=baseline)
    failed = 0
    for path in files:
        try:
            written = chromatograms_to_file(
                path,
                output=args.output,
                output_dir=args.output_dir,
                masses=args.mass,
                segments=_segments(args.segments),
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
        help="Fit the peaks of the measurements not fitted yet, or fitted by an older esper",
    )
    p_fit.add_argument(
        "-f", "--force", action="store_true", help="fit all the measurements again"
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

    from masscal import CalibrationConfig, Mode

    default = CalibrationConfig()
    p_cal = sub.add_parser(
        "calibrate",
        help="Mass calibration of h5 files, following the drift with the retention time",
        description="Read the spectra of h5 files (twutils), fit the mass calibration in rt "
        "windows on calibrant ions (masscal) and write <run>.alpinac/<run>.seg0_mc.json "
        "and the calibrant peaks (.peaks.csv), used by the extraction and esper upload.",
    )
    p_cal.add_argument("files", nargs="+", help="h5 files (glob patterns allowed)")
    p_cal.add_argument(
        "-c",
        "--calibrants-file",
        type=Path,
        help="yaml file of the calibrants (and initial calibration), as "
        "/shared/scripts/tof-processing/config_calib.yml",
    )
    p_cal.add_argument(
        "--calibrant",
        nargs="+",
        metavar="FORMULA",
        help="calibrants present during the run: formulas (CF3, Re+2, [13C]O) or m/z "
        "values (instead of those of --calibrants-file). Default: air, F, Re and PFTBA "
        "ions (masscal.DEFAULT_CALIBRANTS)",
    )
    p_cal.add_argument(
        "--spike",
        nargs="+",
        action="append",
        default=[],
        metavar=("RT", "FORMULA"),
        help="calibrants injected at RT s (repeatable): --spike 1200 CF3 C2F5",
    )
    p_cal.add_argument(
        "--mode",
        default=str(default.mode),
        choices=[m.value for m in Mode],
        help="calibration function (default %(default)s)",
    )
    p_cal.add_argument(
        "--averaging",
        type=float,
        default=default.averaging,
        help="duration of the rt windows, s (default %(default)s)",
    )
    p_cal.add_argument(
        "--spacing",
        type=float,
        default=default.spacing,
        help="interval between the rt windows, s (default %(default)s)",
    )
    p_cal.add_argument("--rt-start", type=float, help="start of the calibrated rt range, s")
    p_cal.add_argument("--rt-stop", type=float, help="end of the calibrated rt range, s")
    p_cal.add_argument(
        "--mz-range",
        type=float,
        default=default.mz_range,
        help="half width of the calibrant domains, m/z (default %(default)s)",
    )
    p_cal.add_argument(
        "--max-ppm",
        type=float,
        default=default.max_ppm,
        help="calibrants with larger residuals are left out (default %(default)s)",
    )
    p_cal.add_argument(
        "--initial",
        type=Path,
        help="calibration file to find the calibrants (default: the h5 calibration)",
    )
    p_cal.add_argument(
        "--neutral",
        action="store_true",
        help="neutral masses of the formulas without charge (legacy alpinac)",
    )
    p_cal.add_argument(
        "--segments",
        help="segments calibrated (summed): 1, or 1:3 for 1 and 2. Required for files "
        "with several segments (EI/CI instruments); names the output (seg1, seg1-2)",
    )
    p_cal.add_argument("-o", "--output", type=Path, help="output file (one h5 file only)")
    p_cal.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <run>.alpinac next to the h5 file)",
    )
    p_cal.add_argument(
        "--no-peaks-csv", action="store_true", help="do not write the calibrant peaks"
    )
    p_cal.add_argument(
        "--plots",
        action="store_true",
        help="also write the plots (.<plot>.png next to the output, needs matplotlib)",
    )
    p_cal.add_argument(
        "--block", type=int, default=600, help="scans read at once (default %(default)s)"
    )
    p_cal.set_defaults(func=_calibrate)

    p_plot = sub.add_parser(
        "plot-calibration",
        help="Plots of mass calibrations written by esper calibrate",
        description="Plot *_mc.json files and their calibrant peaks (.peaks.csv next to "
        "them; without them only the parameters and the drift). The plots are written "
        "next to the calibration file.",
    )
    p_plot.add_argument(
        "files",
        nargs="+",
        help="*_mc.json (or legacy *_mc.txt) files, or h5 files (their calibration file)",
    )
    p_plot.add_argument(
        "--peaks", type=Path, help="calibrant peaks file (one input file only)"
    )
    p_plot.add_argument(
        "--plot",
        nargs="+",
        dest="plots",
        metavar="NAME",
        help="plots to make: parameters drift residuals peak_shape (default all)",
    )
    p_plot.add_argument(
        "--segments", help="segments of the calibration file of the h5 files (1, 1:3)"
    )
    p_plot.add_argument(
        "--ionisation",
        choices=["EI", "CI"],
        default="EI",
        help="legacy calibration file <run>.<EI|CI>_mc.txt, used if there is no json file",
    )
    p_plot.add_argument("--show", action="store_true", help="show the plots")
    p_plot.set_defaults(func=_plot_calibration)

    import peaxart

    default_um = peaxart.UnitMassConfig()
    p_chrom = sub.add_parser(
        "chromatograms",
        help="Unit mass chromatograms of h5 files (summed, baseline removed)",
        description="Read the spectra of h5 files (twutils), sum each scan around each unit "
        "mass, smooth and remove the arPLS baseline (peaxart), and write "
        "<run>.alpinac/<run>.unit_mass_chromatos.parquet (index rt, one column per unit "
        "mass and tic; files with several segments: seg1:69, ...).",
    )
    p_chrom.add_argument("files", nargs="+", help="h5 files (glob patterns allowed)")
    p_chrom.add_argument("--rt-start", type=float, help="start of the rt range, s")
    p_chrom.add_argument(
        "--rt-stop",
        type=float,
        help="end of the rt range, s (e.g. when the filament failed)",
    )
    p_chrom.add_argument(
        "--mass",
        type=float,
        nargs="+",
        help="unit masses (default: those of the spectra, as the legacy pipeline)",
    )
    p_chrom.add_argument(
        "--mass-cal",
        type=Path,
        help="mass calibration file (*_mc.json, or legacy *_mc.txt); default: searched "
        "next to the h5 file",
    )
    p_chrom.add_argument(
        "--segments",
        help="segments read (summed): 1, or 1:3 for 1 and 2. Required for files with "
        "several segments (EI/CI instruments); selects the calibration file",
    )
    p_chrom.add_argument(
        "--ionisation",
        choices=["EI", "CI"],
        default="EI",
        help="legacy calibration file <run>.<EI|CI>_mc.txt, used if there is no json file",
    )
    p_chrom.add_argument(
        "--mass-axis",
        choices=["drift", "end"],
        default="drift",
        help="m/z windows with the calibration at the rt of each scan (drift), or at the "
        "end of the run for all scans (end: legacy pipeline). Default %(default)s",
    )
    p_chrom.add_argument(
        "--d-m",
        type=float,
        default=default_um.d_m,
        help="half width of the m/z window of each unit mass (default %(default)s)",
    )
    p_chrom.add_argument(
        "--lam",
        type=float,
        help=f"arPLS smoothness (default {default_um.baseline.lam:g}, "
        f"--legacy {peaxart.BaselineConfig.legacy().lam:g})",
    )
    p_chrom.add_argument(
        "--no-baseline", action="store_true", help="do not remove the baseline"
    )
    p_chrom.add_argument(
        "--no-steps",
        action="store_true",
        help="fit the baseline over the whole run, not separately between the steps of "
        "the acquisition (filament emission target, resonant ejection; from TPS2)",
    )
    p_chrom.add_argument(
        "--legacy",
        action="store_true",
        help="reproduce the legacy pipeline (unit_mass_chromatos.parquet): --mass-axis "
        "end, --no-steps, lam 1e5, zero padding of the smoothing",
    )
    p_chrom.add_argument(
        "--no-raw",
        action="store_true",
        help="do not write the raw sums and the baselines (.raw.parquet, "
        ".baseline.parquet next to the chromatograms)",
    )
    p_chrom.add_argument(
        "--block", type=int, default=600, help="scans read at once (default %(default)s)"
    )
    p_chrom.add_argument("-o", "--output", type=Path, help="output file (one h5 file only)")
    p_chrom.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <run>.alpinac next to the h5 file)",
    )
    p_chrom.set_defaults(func=_chromatograms)

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
