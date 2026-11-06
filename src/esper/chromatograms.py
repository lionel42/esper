"""Unit mass chromatograms of a run: read with twutils, summed and baseline removed with peaxart.

``esper chromatograms``. The run is read in blocks (``twutils.TofFile.iter_blocks``), the
spectra summed in the window ``m - d_m < m/z < m + d_m`` of each unit mass
(:func:`peaxart.unit_mass_sums`), then each chromatogram is smoothed and its arPLS
baseline removed (:func:`peaxart.remove_baseline`).

The m/z windows follow the drift of the mass calibration scan by scan
(``mass_axis="drift"``, default), or use the calibration at the end of the run
for all scans (``mass_axis="end"``).

Steps of the acquisition (:class:`peaxart.StepConfig`, default): the background changes
abruptly when the acquisition switches the filament emission target (0.5 <-> 0.05 mA:
all masses x10) or the resonant ejection of the quadrupole (``QUADA_REXAMPL1-4``:
N2 / O2 / CO2 ejected while the air peaks elute). The changes are read from
``TPS2`` (``twutils.TofFile.tps2_changes``) and the cuts of the baseline found by
:func:`peaxart.step_cuts`: all masses at the emission steps, at the ejection switches
the masses whose level drops when the ejection is switched on.

With ``mass_axis="end", steps=None, config=peaxart.UnitMassConfig.legacy()`` (whole
run, lam 1e5, zero padding of the smoothing) the legacy alpinac pipeline is reproduced
(``<run>.unit_mass_chromatos.parquet``, within 1e-6).

    result = esper.chromatograms.unit_mass_chromatograms("241007.0423.air.1.h5")
    result.chromatograms        # DataFrame: index rt (s), columns "2" ... "300" and "tic"
                                # (several segments in the file: "seg1:2" ... "seg1:tic")
    result.raw, result.baseline # the raw sums and the baselines removed (same layout)
    result.write()              # <run>.alpinac/<run>.unit_mass_chromatos.parquet (as the legacy pipeline)
                                # (+ .raw.parquet and .baseline.parquet)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import peaxart
from masscal import MassCalibration
from peaxart import StepConfig
from twutils import TofFile, calibration_for
from twutils.calibration import Ionisation
from twutils.h5 import segments_label

logger = logging.getLogger(__name__)

EMISSION_TARGET = "FILAMENT_EMISSION target"
EJECTION = ("QUADA_REXAMPL1", "QUADA_REXAMPL2", "QUADA_REXAMPL3", "QUADA_REXAMPL4")

MassAxis = Literal["drift", "end"]
"""m/z windows: following the calibration at the rt of each scan, or at the end of the run (legacy)."""
SUFFIX = "unit_mass_chromatos.parquet"
"""``<run>.<SUFFIX>``: the name of the legacy pipeline (the segments are in the column names)."""
PROVENANCE = "esper"
"""Key of the provenance in the ``attrs`` of the chromatograms (and the parquet metadata)."""


@dataclass
class UnitMassChromatograms:
    """Result of :func:`unit_mass_chromatograms`."""

    chromatograms: pd.DataFrame
    """Corrected chromatograms: index rt (s), one column per unit mass (named ``"2"``, ...) and
    ``tic`` (their sum). Provenance in ``chromatograms.attrs["esper"]`` (also in the parquet file, :func:`provenance`)."""
    raw: pd.DataFrame | None
    """The raw sums (before smoothing and baseline removal; ``keep_raw``), same layout, float32."""
    baseline: pd.DataFrame | None
    """The baselines removed (from the smoothed sums; ``keep_raw``), same layout, float32:
    corrected = clip(smoothed raw - baseline, 0)."""
    file: TofFile
    calibration: MassCalibration
    segments: tuple[int, int]
    runtime: float
    steps: dict
    """Cuts of the baseline (:func:`acquisition_cuts`): rts of the emission steps, and per cluster of
    ejection switches the channels, rts and masses cut (also in the provenance)."""

    def default_path(self, output_dir: str | PathLike | None = None) -> Path:
        """``<run>.alpinac/<run>.unit_mass_chromatos.parquet`` (or in ``output_dir``)."""
        directory = self.file.path.parent / f"{self.file.stem}.alpinac" if output_dir is None else Path(output_dir)
        return directory / f"{self.file.stem}.{SUFFIX}"

    def write(self, path: str | PathLike | None = None, output_dir: str | PathLike | None = None) -> Path:
        """Write the chromatograms (parquet), and the raw sums and baselines next to them if kept
        (``.raw.parquet``, ``.baseline.parquet``, see :func:`part_path`)."""
        path = self.default_path(output_dir) if path is None else Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.chromatograms.to_parquet(path)
        for part in ("raw", "baseline"):
            frame = getattr(self, part)
            if frame is not None:
                frame.to_parquet(part_path(path, part))
        return path


Part = Literal["corrected", "raw", "baseline"]


def part_path(path: str | PathLike, part: Part) -> Path:
    """File of the raw sums (``<name>.raw.parquet``) or baselines (``<name>.baseline.parquet``) next to
    a chromatogram file ``<name>.parquet`` (``"corrected"``: the file itself)."""
    path = Path(path)
    if part == "corrected":
        return path
    if part not in ("raw", "baseline"):
        raise ValueError(f"part: 'corrected', 'raw' or 'baseline', not {part!r}")
    return path.with_name(f"{path.name.removesuffix('.parquet')}.{part}.parquet")


def read_chromatograms(path: str | PathLike, part: Part = "corrected") -> pd.DataFrame:
    """Chromatograms written by :meth:`UnitMassChromatograms.write` (``part``: the corrected ones, the raw
    sums or the baselines), or the legacy ``<run>.unit_mass_chromatos.parquet`` (same layout)."""
    return pd.read_parquet(part_path(path, part))


def column_prefix(file: TofFile, segments: tuple[int, int]) -> str:
    """Prefix of the column names: none for files with one segment (``"69"``, ``"tic"``, as the legacy
    pipeline), else the segment range (``"seg1:69"``, ``"seg1-2:tic"``)."""
    return "" if file.n_segments == 1 else f"{segments_label(segments)}:"


def default_masses(file: TofFile, calibration: MassCalibration) -> list[int]:
    """Unit masses of the spectra, as the legacy pipeline: ``round(m/z of the first point)`` to
    ``ceil(m/z of the last point) - 1``, with the calibration at the end of the run."""
    rt_end = float(file.time_axis[-1])
    first, last = (float(v) for v in calibration.mz(rt_end, [0, file.n_points - 1]))
    return list(range(max(1, int(round(first))), math.ceil(last)))


def _drift_bounds(calibration: MassCalibration, rt: np.ndarray, masses: np.ndarray, d_m: float) -> np.ndarray:
    """(n_scans, n_masses, 2) point bounds of ``m - d_m < m/z < m + d_m`` with the calibration at each rt."""
    rt = np.asarray(rt, dtype=float)[:, None]
    low = calibration.tof(rt, masses[None, :] - d_m)  # m/z(i) > m - d_m  <=>  i > low
    high = calibration.tof(rt, masses[None, :] + d_m)  # m/z(i) < m + d_m  <=>  i < high
    return np.stack([np.floor(low).astype(int) + 1, np.ceil(high).astype(int)], axis=-1)


def acquisition_cuts(file: TofFile, sums: np.ndarray, rt: np.ndarray, scans: slice, masses: Sequence[float],
                     config: StepConfig | None = None) -> tuple[list[list[int]], dict]:
    """Scan indices (in ``sums``) where the baseline of each mass is cut, from the changes in ``TPS2``
    (:func:`peaxart.step_cuts`: emission target changes, resonant ejection switches).

    :param sums: (n_scans, n_masses) raw sums of the scans ``scans`` of the file, at ``rt``.
    :return: the cuts of each mass, and a summary {"emission": [rt of the cuts], "ejection": [{"channels",
        "rt" (logged switches, on and off), "masses" (cut)}, one per cluster]}.
    """
    config = config or StepConfig()
    names = ([EMISSION_TARGET] if config.emission else []) + (list(EJECTION) if config.ejection else [])
    if len(sums) < 2 or not names:
        return [[] for _ in range(sums.shape[1])], {"emission": [], "ejection": []}
    changes = file.tps2_changes(names, scans)
    if not changes:
        logger.warning("%s: no change of %s in TPS2: baseline not cut", file.path.name, ", ".join(names))
    start = scans.start or 0
    steps = [scan - start for name, scan, _, _ in changes if name == EMISSION_TARGET]
    switches = [(name, scan - start, old, new) for name, scan, old, new in changes if name in EJECTION]
    cuts, summary = peaxart.step_cuts(sums, rt, steps, switches, [float(m) for m in masses], config)
    return cuts, {"emission": summary["steps"], "ejection": summary["switches"]}


DEFAULT_STEPS = StepConfig()


def unit_mass_chromatograms(
    file: TofFile | str | PathLike,
    masses: Sequence[float] | None = None,
    segments: tuple[int, int] | None = None,
    calibration: MassCalibration | str | PathLike | None = None,
    rt_start: float | None = None,
    rt_stop: float | None = None,
    mass_axis: MassAxis = "drift",
    steps: StepConfig | None = DEFAULT_STEPS,
    config: peaxart.UnitMassConfig | None = None,
    ionisation: Ionisation = "EI",
    keep_raw: bool = True,
    block: int = 600,
) -> UnitMassChromatograms:
    """Unit mass chromatograms of a run (see the module).

    :param masses: unit masses (default :func:`default_masses`).
    :param segments: segment range [start, stop[ read (summed): required for files with several.
    :param calibration: mass calibration, or its file; default ``twutils.calibration_for``.
    :param rt_start, rt_stop: rt range (default the whole run; e.g. ``rt_stop`` = the time the
        filament failed). The baseline depends on the range.
    :param mass_axis: ``"drift"`` (calibration at each scan) or ``"end"`` (legacy).
    :param steps: cut the baseline at the steps of the acquisition (:class:`peaxart.StepConfig`, see the
        module); None: whole run (legacy).
    :param config: peaxart parameters: m/z window and :class:`peaxart.BaselineConfig` (default
        ``peaxart.UnitMassConfig()``; ``.legacy()``: zero padding of the smoothing).
    :param ionisation: name of the legacy calibration files.
    :param keep_raw: also return the raw sums and the baselines (to check the raw data).
    :param block: scans read at once.
    """
    t0 = time.perf_counter()
    file = file if isinstance(file, TofFile) else TofFile(file)
    segments = file.resolve_segments(segments)
    if calibration is None:
        calibration = calibration_for(file, segments, ionisation)
    elif not isinstance(calibration, MassCalibration):
        calibration = calibration_for(file, segments, ionisation, mc_file=calibration)
    config = config or peaxart.UnitMassConfig()
    if mass_axis not in ("drift", "end"):
        raise ValueError(f"mass_axis: 'drift' or 'end', not {mass_axis!r}")
    masses = np.asarray(default_masses(file, calibration) if masses is None else masses, dtype=float)

    scans = file.scan_range(*file.rt_range(rt_start, rt_stop))
    if mass_axis == "end":
        mz = calibration.mz(float(file.time_axis[-1]), np.arange(file.n_points))
        bounds = peaxart.mass_bounds(mz, masses, config.d_m)
        tof_start, tof_stop = int(bounds.min()), int(bounds.max())
    else:
        edges = _drift_bounds(calibration, file.time_axis[scans], masses, config.d_m)  # all scans: cheap
        tof_start, tof_stop = int(max(0, edges.min() - 2)), int(min(file.n_points, edges.max() + 2))

    parts, rts = [], []
    with file.open():
        for _, rt, data in file.iter_blocks(block, tof_start, tof_stop, segments, scans):
            if mass_axis == "end":
                block_bounds = bounds - tof_start
            else:
                block_bounds = _drift_bounds(calibration, rt, masses, config.d_m) - tof_start
            parts.append(peaxart.unit_mass_sums(data, block_bounds))
            rts.append(rt)
    sums = np.concatenate(parts) if parts else np.zeros((0, len(masses)))
    rt = np.concatenate(rts) if rts else np.zeros(0)
    logger.info("%s: %d scans x %d masses summed in %.1f s", file.path.name, len(rt), len(masses),
                time.perf_counter() - t0)

    prefix = column_prefix(file, segments)
    columns = [f"{prefix}{m:g}" for m in masses]
    if config.baseline is None:
        corrected, baseline, cuts, step_summary = sums.copy(), np.zeros_like(sums), None, {}
    else:
        cuts, step_summary = (acquisition_cuts(file, sums, rt, scans, masses, steps) if steps is not None
                              else (None, {}))
        corrected, baseline = peaxart.remove_baseline(sums, config.baseline, return_baseline=True, cuts=cuts)

    def frame(values, dtype=float):
        table = pd.DataFrame(values.astype(dtype), index=pd.Index(rt, name="rt"), columns=columns)
        table[f"{prefix}tic"] = values.sum(axis=1).astype(dtype)
        return table

    chromatograms = frame(corrected)
    runtime = time.perf_counter() - t0
    chromatograms.attrs[PROVENANCE] = json.dumps(_provenance(file, calibration, segments, mass_axis, config,
                                                            (float(rt[0]), float(rt[-1])) if len(rt) else None,
                                                            steps, step_summary))
    logger.info("%s: unit mass chromatograms in %.1f s", file.path.name, runtime)
    return UnitMassChromatograms(chromatograms=chromatograms, raw=frame(sums, np.float32) if keep_raw else None,
                                 baseline=frame(baseline, np.float32) if keep_raw else None, file=file,
                                 calibration=calibration, segments=segments, runtime=runtime, steps=step_summary)


def _provenance(file: TofFile, calibration: MassCalibration, segments, mass_axis, config, rt_range,
                steps, step_summary) -> dict:
    import masscal
    import twutils

    import esper

    return {
        "method": "esper.chromatograms.unit_mass_chromatograms (twutils, peaxart)",
        "file": str(file.path.absolute()),
        "segments": list(segments),
        "calibration": calibration.source,
        "mass_axis": mass_axis,
        "rt_range": rt_range,
        "config": asdict(config),
        "steps": None if steps is None else {"config": asdict(steps), **step_summary},
        "versions": {"esper": esper.__version__, "twutils": twutils.__version__,
                     "peaxart": peaxart.__version__, "masscal": masscal.__version__},
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def provenance(chromatograms: pd.DataFrame) -> dict:
    """How chromatograms were made (empty for the legacy files; also those written by twutils)."""
    text = chromatograms.attrs.get(PROVENANCE) or chromatograms.attrs.get("twutils")
    return json.loads(text) if text else {}


def chromatograms_to_file(file: TofFile | str | PathLike, output: str | PathLike | None = None,
                          output_dir: str | PathLike | None = None, **kwargs) -> Path:
    """:func:`unit_mass_chromatograms` and write them (see :meth:`UnitMassChromatograms.write`)."""
    result = unit_mass_chromatograms(file, **kwargs)
    path = result.write(output, output_dir)
    logger.info("Written %s", path)
    return path


__all__ = [
    "UnitMassChromatograms",
    "acquisition_cuts",
    "chromatograms_to_file",
    "column_prefix",
    "default_masses",
    "part_path",
    "provenance",
    "read_chromatograms",
    "unit_mass_chromatograms",
]
