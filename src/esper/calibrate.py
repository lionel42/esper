"""Mass calibration of TOFWERK h5 files: reads the spectra with twutils, fits with masscal.

``esper calibrate`` (see :func:`calibrate_to_file`):

1. the calibrants (default :data:`masscal.DEFAULT_CALIBRANTS`, or a list, or the yaml
   file of the tof-processing pipeline) and the initial calibration (default: the
   acquisition calibration of the h5 file, ``twutils.h5_calibration``);
2. the rt windows (:func:`masscal.calibration_windows`) and the tof indices of the
   calibrant domains (:func:`masscal.tof_columns`): only those are read;
3. the spectra summed over each window (and around each spike), read once in blocks
   (``twutils.TofFile.sum_spectra``);
4. :func:`masscal.calibrate`, written to ``<run>.alpinac/<run>.seg0_mc.json``
   (``twutils.mc_path``, where the extraction and esper find it) with the calibrant
   peaks (``.peaks.csv``), and optionally the plots (``.<plot>.png``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from os import PathLike
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _initial(initial, calibrants, file):
    from masscal import MassCalibration
    from twutils import h5_calibration

    if initial is None:
        initial = getattr(calibrants, "initial", None)
    if initial is None:
        return h5_calibration(file)
    if isinstance(initial, MassCalibration):
        return initial
    return MassCalibration.from_file(initial)


def calibrate_run(
    file,
    calibrants=None,
    spikes: Mapping[float, Sequence] | None = None,
    rt_start: float | None = None,
    rt_stop: float | None = None,
    initial=None,
    segments: tuple[int, int] | None = None,
    config=None,
    block: int = 600,
    **options,
):
    """Mass calibration of an h5 file, varying with the retention time.

    :param file: h5 file (path or ``twutils.TofFile``).
    :param calibrants: present during the whole run: formulas, m/z or (name, m/z)
        (default :data:`masscal.DEFAULT_CALIBRANTS`); ``[]``: spikes only. Or a
        :class:`masscal.Calibrants` / its yaml file (a ``Path``, or a ``.yml`` /
        ``.yaml`` name: continuous, spikes and initial calibration).
    :param spikes: {rt (s): calibrants} injected at rt, summed over
        ``spike_duration`` s. Added to those of a :class:`masscal.Calibrants`.
    :param rt_start, rt_stop: rt range calibrated (default: the whole run).
    :param initial: calibration used to find the calibrants (a ``MassCalibration`` or
        its file; default: that of the :class:`masscal.Calibrants`, else the h5 attributes).
    :param segments: segment range [start, stop[ calibrated (summed): required for
        files with several segments (EI/CI instruments), default (0, 1).
    :param config: :class:`masscal.CalibrationConfig` (default ``CalibrationConfig()``).
    :param block: number of scans read at once.
    :param options: fields of the config changed, e.g. ``mode="sqrt", averaging=60``.
    :return: :class:`masscal.CalibrationResult`.
    """
    import masscal
    import twutils
    from masscal import Calibrants, CalibrationConfig, Spike

    import esper

    config = replace(config or CalibrationConfig(), **options)  # TypeError: unknown option
    file = file if isinstance(file, twutils.TofFile) else twutils.TofFile(file)
    segments = file.resolve_segments(segments)  # required with several segments
    if isinstance(calibrants, PathLike) or (
        isinstance(calibrants, str) and calibrants.endswith((".yml", ".yaml"))
    ):
        calibrants = Calibrants.from_yaml(calibrants)
    elif isinstance(calibrants, str):  # a single formula
        calibrants = [calibrants]
    initial = _initial(initial, calibrants, file)
    if isinstance(calibrants, Calibrants):
        spikes = {**calibrants.spikes, **(spikes or {})}
        calibrants = calibrants.continuous
    continuous = masscal.resolve_calibrants(calibrants, config.ionised)

    spiked = []
    for rt, cals in (spikes or {}).items():
        scans = file.scan_range(rt - config.spike_duration / 2, rt + config.spike_duration / 2)
        if scans.stop <= scans.start:
            logger.warning("%s: spike at %g s outside the run: left out", file.path.name, rt)
            continue
        spiked.append((float(rt), masscal.resolve_calibrants(cals, config.ionised), scans))
    if not continuous and not spiked:
        raise ValueError(f"No calibrant for {file.path.name}")

    if continuous:
        windows = masscal.calibration_windows(
            file.time_axis, config.averaging, config.spacing, rt_start, rt_stop
        )
        ranges = windows[["scan_start", "scan_stop"]].to_numpy()
    else:
        windows = masscal.spike_windows([s[0] for s in spiked], config.spike_duration)
        ranges = np.zeros((0, 2), dtype=int)
    spike_ranges = np.array([(s.start, s.stop) for _, _, s in spiked], dtype=int).reshape(-1, 2)
    ranges = np.vstack([ranges, spike_ranges]).astype(int)

    mz = [m for _, m in continuous] + [m for _, cals, _ in spiked for _, m in cals]
    columns = masscal.tof_columns(mz, initial, windows["rt"], file.n_points, config.mz_range)
    logger.info(
        "%s: %d calibrants, %d windows, %d spikes, %d tof indices",
        file.path.name, len(continuous), len(windows) if continuous else 0, len(spiked),
        len(columns),
    )
    sums = file.sum_spectra(ranges, columns, segments, block)

    n = len(windows) if continuous else 0
    return masscal.calibrate(
        sums[:n] if continuous else None,
        columns,
        windows["rt"].to_numpy() if continuous else None,
        initial,
        continuous,
        spikes=[Spike(rt, sums[n + k], cals) for k, (rt, cals, _) in enumerate(spiked)],
        rt_ranges=windows[["rt_start", "rt_stop"]].to_numpy() if continuous else None,
        segments=segments,
        config=config,
        name=file.stem,
        metadata={
            "file": str(file.path.absolute()),
            "method": "esper.calibrate (twutils, masscal.calibrate)",
            "versions": {"twutils": twutils.__version__, "esper": esper.__version__},
            "rt_range": [rt_start, rt_stop],
            "block": block,
        },
    )


def default_output(file, segments: tuple[int, int] | None = None,
                   output_dir: str | PathLike | None = None) -> Path:
    """``<run>.alpinac/<run>.seg0_mc.json`` (``twutils.mc_path``), or its name in ``output_dir``."""
    import twutils

    path = twutils.mc_path(file, segments)
    return path if output_dir is None else Path(output_dir) / path.name


def calibrate_to_file(
    file,
    output: str | PathLike | None = None,
    output_dir: str | PathLike | None = None,
    peaks_csv: bool = True,
    plots: bool = False,
    **kwargs,
):
    """:func:`calibrate_run` and write the calibration file (default :func:`default_output`),
    the calibrant peaks (``.peaks.csv``, unless not ``peaks_csv``) and the plots
    (``.<plot>.png``, with ``plots``; needs matplotlib).

    :return: the result and the path of the calibration file.
    """
    result = calibrate_run(file, **kwargs)
    path = Path(output) if output is not None else default_output(
        file, result.calibration.segments, output_dir
    )
    result.write(path, peaks_csv)
    logger.info("Written %s", path)
    if plots:
        result.plot(path)
    return result, path
