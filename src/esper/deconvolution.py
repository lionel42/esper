"""Deconvolution step of the processing: from the unit mass chromatograms to the compounds.

The numerics are in :mod:`peaxart.deconvolution`; here the chromatogram table of a run is
turned into the peak / compound part of ``pipeline.nc``.

``legacy``: the extraction of the legacy pipeline (``alpinac.mode_extraction.pipeline.
run_extraction_pipeline``), same dataset: ``peak`` variables (the ``find_peaks`` properties,
``start_ip``, ``stopp_ip``, ``peak_rt``, ``mass_of_peak``, ``flatness``, ``tailing_ratio``,
``compound_id``, ``is_max_of_cmp``) and ``compound`` variables (``peak_rt_mean`` / ``_std`` /
``_count``, ``prominences_sum``, ``start_ip_mean`` / ``_std``, ``stopp_ip_mean`` / ``_std``, ``mai_``).

``fit_dataset``: the peaks of each mass fitted in its chromatogram (:func:`peaxart.fit_chromatogram`:
Pap-Papai peaks by groups of overlapping peaks, starting at the detected rts and at the narrow peaks the
detection missed; peaks added where the data show a coelution), on their own dimension ``fit`` (variables
``fit_*``; ``fit_peak``: the nearest detected peak, -1 if none in the group; ``fit_added``: started from a
peak the detection missed). The ``peak`` / ``compound`` variables are not changed.

    chromatograms = esper.chromatograms.read_chromatograms("run.alpinac/run.unit_mass_chromatos.parquet")
    ds = esper.deconvolution.legacy_dataset(chromatograms)
    ds = esper.deconvolution.fit_dataset(ds, chromatograms)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import pandas as pd
import peaxart as px
import xarray as xr

FIT_CONFIG = px.ChromatogramFitConfig()
"""Default fit of the chromatograms (also of ``esper.fitting``)."""
# dataset variable -> column of px.fit_chromatogram
FIT_VARIABLES = {
    "fit_rt": "rt",
    "fit_rt_u": "rt_u",
    "fit_height": "height",
    "fit_height_u": "height_u",
    "fit_sigma": "rt_sigma",
    "fit_tailing": "rt_tailing",
    "fit_baseline": "baseline",
    "fit_area": "area",
    "fit_area_u": "area_u",
    "fit_sn": "sn",
    "fit_signal_height": "signal_height",
    "fit_signal_prominence": "signal_prominence",
}


def chromatogram_masses(chromatograms: pd.DataFrame) -> list[str]:
    """The columns of the unit masses (``"2"`` ... ``"300"``; not ``tic``)."""
    return [c for c in chromatograms.columns if c != "tic" and not str(c).endswith(":tic")]


def legacy_dataset(chromatograms: pd.DataFrame, rt_start: float = 0.0, rt_stopp: float | None = None,
                   config: px.LegacyDeconvolutionConfig | None = None) -> xr.Dataset:
    """Peaks and compounds of the legacy extraction (:func:`peaxart.legacy_deconvolution`), as the
    legacy ``run_extraction_pipeline`` dataset.

    :param chromatograms: corrected unit mass chromatograms, index rt (s), one column per unit mass
        (``"2"`` ...; ``tic`` left out: the legacy pipeline found no peaks in it).
    :param rt_start, rt_stopp: stored in the attributes (default: 0 and the last rt), as the legacy.
    """
    columns = chromatogram_masses(chromatograms)
    masses = [int(c) if float(c).is_integer() else float(c) for c in columns]
    rt = chromatograms.index.to_numpy(dtype=float)
    compounds, peaks = px.legacy_deconvolution(chromatograms[columns].to_numpy(), rt, masses, config)
    peaks = peaks.rename(columns={"mass": "mass_of_peak"})
    return xr.Dataset(
        {col: ("peak", peaks[col].to_numpy()) for col in peaks.columns}
        | {"_".join(col): ("compound", compounds[col].to_numpy()) for col in compounds.columns},
        coords={"peak": peaks.index.to_numpy(), "compound": compounds.index.to_numpy()},
        attrs={"rt_start": float(rt_start), "rt_stopp": float(rt[-1] if rt_stopp is None else rt_stopp)},
    )




# --- fit of the chromatograms ------------------------------------------------------------


@contextmanager
def fit_pool(workers: int | None = None):
    """Processes for the fits of several chromatograms or runs; None if ``workers`` is 1."""
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        yield None
        return
    with ProcessPoolExecutor(workers) as pool:
        yield pool


def _fit_task(args) -> pd.DataFrame:
    t, y, rt, start, stop, config = args
    return px.fit_chromatogram(t, y, rt, start, stop, config)


def fit_chromatograms(t: np.ndarray, tasks: list[tuple], config: px.ChromatogramFitConfig | None = None,
                      workers: int | None = None, pool: Executor | None = None) -> list[pd.DataFrame]:
    """:func:`peaxart.fit_chromatogram` of several chromatograms of a run, in parallel.

    :param tasks: (y, rt, start, stop) of each chromatogram (its values and its detected peaks).
    :param pool: processes (:func:`fit_pool`), else ``workers`` processes are started (None: the number
        of CPUs, 1: in this process).
    :return: the fitted peaks of each task, in order.
    """
    config = config or FIT_CONFIG
    args = [(t, *task, config) for task in tasks]
    if pool is not None and len(args) > 1:
        return list(pool.map(_fit_task, args))
    if len(args) > 1 and (workers or os.cpu_count() or 1) > 1:
        with fit_pool(min(workers or os.cpu_count() or 1, len(args))) as own:
            return list(own.map(_fit_task, args))
    return [_fit_task(a) for a in args]


def fit_dataset(ds: xr.Dataset, chromatograms: pd.DataFrame, config: px.ChromatogramFitConfig | None = None,
                workers: int | None = None, pool: Executor | None = None) -> xr.Dataset:
    """Add the fit of the detected peaks (``legacy_dataset``) to the dataset, on the dimension ``fit``.

    Variables: ``fit_mass``, ``fit_peak`` (the nearest detected peak of its group, ``peak`` coordinate; -1 if
    the group has only peaks the detection missed), ``fit_added`` (started from a peak the detection missed:
    :func:`peaxart.added_seeds`) and :data:`FIT_VARIABLES` (rt, sigma in s, standard Gaussian convention: the
    areas are true areas); attributes ``fit_config`` (json), ``fit_n_warnings`` and ``fit_n_added_seeds``.
    All the chromatograms are fitted, also those without detected peaks (``tic`` left out).

    :param chromatograms: the chromatograms the peaks were detected in (columns as ``mass_of_peak``).
    """
    config = config or FIT_CONFIG
    peaks = pd.DataFrame({
        "mass": ds["mass_of_peak"].values, "rt": ds["peak_rt"].values, "start": ds["start_ip"].values,
        "stop": ds["stopp_ip"].values,
    }, index=ds["peak"].values)
    t = chromatograms.index.to_numpy(dtype=float)
    by_mass = {f"{mass:g}": group for mass, group in peaks.groupby("mass")}
    empty = peaks.iloc[:0]
    groups = [(column, by_mass.get(str(column), empty)) for column in chromatogram_masses(chromatograms)]
    groups.sort(key=lambda g: -len(g[1]))  # largest first, for the load balance
    tasks = [(chromatograms[column].to_numpy(dtype=float), g["rt"].to_numpy(), g["start"].to_numpy(),
              g["stop"].to_numpy()) for column, g in groups]
    results = fit_chromatograms(t, tasks, config, workers, pool)
    parts = []
    for (column, group), fitted in zip(groups, results, strict=True):
        if len(fitted):
            index = fitted["peak"].to_numpy(dtype=int)
            peak = np.where(index >= 0, group.index.to_numpy()[np.maximum(index, 0)] if len(group) else -1, -1)
            parts.append(fitted.assign(peak=peak, mass=float(column)))
    fitted = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=["peak", "mass", "added", *FIT_VARIABLES.values()]))
    fitted = fitted.sort_values(["mass", "rt"], ignore_index=True)
    out = ds.drop_vars([v for v in ("fit_peak", "fit_mass", "fit_added", *FIT_VARIABLES) if v in ds],
                       errors="ignore")
    if "fit" in out.dims:
        out = out.drop_dims("fit")
    masses = fitted["mass"].to_numpy(dtype=float)
    out = out.assign(
        {"fit_mass": ("fit", masses.astype(np.int64) if np.all(masses == np.round(masses)) else masses),
         "fit_peak": ("fit", fitted["peak"].to_numpy(dtype=np.int64)),
         "fit_added": ("fit", fitted["added"].to_numpy(dtype=bool))}
        | {name: ("fit", fitted[column].to_numpy(dtype=float)) for name, column in FIT_VARIABLES.items()})
    out.attrs["fit_config"] = json.dumps(asdict(config))
    out.attrs["fit_n_warnings"] = int(sum(r.attrs.get("n_warnings", 0) for r in results))
    out.attrs["fit_n_added_seeds"] = int(sum(r.attrs.get("n_added_seeds", 0) for r in results))
    return out


__all__ = ["FIT_CONFIG", "FIT_VARIABLES", "chromatogram_masses", "fit_chromatograms", "fit_dataset", "fit_pool",
           "legacy_dataset"]
