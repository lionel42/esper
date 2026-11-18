"""Chromatographic fit of the peaks found by the extraction (alpinac, or the peak
detection of esper: :mod:`esper.processing`), with peaxart.

The peaks of each mass are fitted in its chromatogram, in each window where it was
measured (:mod:`esper.windows`), with the peak function and the background of the fit
configuration (``FIT_CONFIG``, or the one of the data directory: ``processing: fit:`` of
its esper.yaml, :mod:`esper.processing`; ``peaxart.fit_chromatogram``, standard Gaussian
convention: the areas are true areas), by groups of overlapping peaks. The fit
starts at the rts of the extraction and adds peaks where the data show a coelution
(peaxart's extra-peak search). The fit itself is the one of the processing
(``esper.deconvolution``); here it becomes the peak table of the database.

The fit improves the extraction: the peak table becomes the peaks of the fit.
Each fitted peak goes to the nearest extraction peak of its group:

- given one fitted peak (within its half maximum crossings, or within
  ``MATCH_HWHM`` half widths if none is within them), the extraction peak keeps
  its row (``source`` 'extraction') with the ``fit_*`` columns filled;
- given several, it is split: its row takes the fitted peak nearest to it,
  new rows the others. All get ``source`` 'fit', ``rt_s`` of the fit, ``height``
  and ``prominence`` (scipy's definitions, as the extraction) measured on the
  fitted signal and NULL for the other extraction columns (they describe the
  unsplit peak), and are assigned to the compound that matches them
  (``match_compounds``), or to none (``compound_idx`` NULL);
- given none (rejected by the fit: e.g. a straight background explains the
  data as well), the fit columns stay NULL.

The compounds that gain or lose peaks get their summary columns again. Every
compound gets its largest peak again (``is_max_of_cmp``, and ``main_ion_mass``):
the fit changes them, and alpinac's merged compounds can have several.

The values of the extraction that the fit replaces are kept (``extraction_peak``,
``extraction_compound``), so that a measurement can be fitted again
(``measurement.fit_version`` older than ``FIT_VERSION``, or forced).
"""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import peaxart as px

from esper import db
from esper.deconvolution import FIT_CONFIG, fit_chromatograms, fit_pool  # noqa: F401 (fit_pool: used from here)
from esper.processing import TARGET
from esper.target_index import unit_mass
from esper.windows import is_continuous, measured_windows

logger = logging.getLogger(__name__)

# Increase when the fit changes (2: prominence and height of the fitted signal,
# extraction kept; 3: peaks added by the seeds, groups split at valleys, the peak
# function of FIT_CONFIG (EMG), fit_rt_s / fit_height its apex; 4: the background of each
# group chosen among constant / linear / exponential, fit_baseline_slope)
FIT_VERSION = 4
# Gaussian convention of peaxart's peak shapes, and the half widths of the extraction
# are clipped to this range [s] (peaxart.ChromatogramFitConfig, esper.deconvolution.FIT_CONFIG)
CONVENTION = FIT_CONFIG.convention
# Peak function of the fits stored without fit_shape (FIT_VERSION 1 and 2)
LEGACY_SHAPE = "pap_papai"
HWHM_RANGE_S = FIT_CONFIG.hwhm_range
# A fitted peak outside the half maximum crossings of its extraction peak still takes
# its row within this many half widths (of each side), if no other is within them
MATCH_HWHM = 2.0

SOURCE_EXTRACTION = "extraction"
SOURCE_FIT = "fit"
# The compounds of the peak source (compound.source NULL), not made by the fit
NOT_MADE_BY_THE_FIT = f"source IS NOT '{SOURCE_FIT}'"

# Columns of the peak table filled by the fit, from the columns of peaxart.Fit
FIT_COLUMNS = {
    "fit_rt_s": "rt",
    "fit_rt_u_s": "rt_u",
    "fit_height": "height",
    "fit_height_u": "height_u",
    "fit_sigma_s": "rt_sigma",
    "fit_tailing": "rt_tailing",
    "fit_baseline": "baseline",
    "fit_area": "area",
    "fit_area_u": "area_u",
    "fit_sn": "sn",
    "fit_shape": "shape",
    "fit_shape_rt_s": "shape_rt",
    "fit_shape_height": "shape_height",
    "fit_baseline_slope": "baseline_slope",
    # the override the peak was fitted with (esper.fit_overrides), NaN: the default fit
    "fit_override_id": "override_id",
}
# Columns of fit_mass / fit_peaks
FITTED_COLUMNS = ["peak_idx", *FIT_COLUMNS, "signal_height", "signal_prominence"]


def _given(value) -> bool:
    return isinstance(value, str) or (value is not None and np.isfinite(value))
def peak_baseline(t: np.ndarray, peak) -> np.ndarray:
    """Background under a fitted peak (``fit_*`` columns of a peak row): ``fit_baseline`` at
    the apex, with its slope ``fit_baseline_slope`` (rows fitted before it: constant)."""
    slope = getattr(peak, "fit_baseline_slope", np.nan)
    slope = 0.0 if pd.isna(slope) else slope
    return peak.fit_baseline + slope * (np.asarray(t, dtype=float) - peak.fit_rt_s)


def peak_shape(t: np.ndarray, peak) -> np.ndarray:
    """Signal of a fitted peak above its baseline (``fit_*`` columns of a peak row).

    The fits stored without ``fit_shape`` are Pap-Papai peaks (LEGACY_SHAPE), whose
    parameters are their apex.
    """
    from peaxart.shapes import peak_shape as shape_of

    name = getattr(peak, "fit_shape", None)
    shape = shape_of(name if isinstance(name, str) else LEGACY_SHAPE, CONVENTION)
    location = getattr(peak, "fit_shape_rt_s", None)
    amplitude = getattr(peak, "fit_shape_height", None)
    if not (_given(location) and _given(amplitude)):
        location, amplitude = peak.fit_rt_s, peak.fit_height
    return shape.value(t, amplitude, location, peak.fit_sigma_s, peak.fit_tailing)


def peak_groups(peaks: pd.DataFrame) -> list[tuple[pd.DataFrame, float, float]]:
    """Groups of overlapping peaks of one mass, with the rt range fitted [s]
    (``peaxart.overlapping_groups``)."""
    groups = px.overlapping_groups(
        peaks["rt_s"].to_numpy(dtype=float),
        peaks["start_s"].to_numpy(dtype=float),
        peaks["stop_s"].to_numpy(dtype=float),
        FIT_CONFIG,
    )
    return [(peaks.iloc[members], lo, hi) for members, lo, hi in groups]


def _as_fitted(
    fitted: pd.DataFrame,
    peaks: pd.DataFrame,
    config: px.ChromatogramFitConfig | None = None,
) -> pd.DataFrame:
    """``peaxart.fit_chromatogram`` peaks of one mass as FITTED_COLUMNS (``peaks``: as fitted).

    A fitted peak whose group has only peaks the detection missed (``peak`` -1) goes to the
    nearest extraction peak of the mass: it becomes a new peak (``fitted_peak_table``).
    """
    shape = (config or FIT_CONFIG).peak_shape
    fitted = fitted.assign(shape=fitted.attrs.get("peak_shape", shape), override_id=np.nan)
    out = fitted.rename(
        columns={v: k for k, v in FIT_COLUMNS.items()} | {"peak": "peak_idx"}
    )
    index = fitted["peak"].to_numpy(dtype=int)
    rt = peaks["rt_s"].to_numpy(dtype=float)
    nearest = np.abs(fitted["rt"].to_numpy(dtype=float)[:, None] - rt[None, :]).argmin(
        axis=1
    )
    out["peak_idx"] = peaks.index.to_numpy()[np.where(index >= 0, index, nearest)]
    return out[FITTED_COLUMNS]


def fit_mass(
    t: np.ndarray,
    y: np.ndarray,
    peaks: pd.DataFrame,
    config: px.ChromatogramFitConfig | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fitted peaks of one mass (one window), and the number of warnings of the fits.

    Columns: ``peak_idx`` (the nearest extraction peak), the ``fit_*`` columns and
    ``signal_height`` / ``signal_prominence`` (``peaxart.fit_chromatogram``).
    """
    config = config or FIT_CONFIG
    (fitted,) = fit_chromatograms(t, [_task(y, peaks)], config, workers=1)
    return _as_fitted(fitted, peaks, config), fitted.attrs.get("n_warnings", 0)


def _task(y: np.ndarray, peaks: pd.DataFrame) -> tuple:
    return (
        y,
        peaks["rt_s"].to_numpy(dtype=float),
        peaks["start_s"].to_numpy(dtype=float),
        peaks["stop_s"].to_numpy(dtype=float),
    )


def mass_windows(
    frame: pd.DataFrame, peaks: pd.DataFrame, min_points: int = 1
) -> list[tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """(t, y, peaks) of each window of each mass with peaks (``esper.windows``): the
    chromatogram of a mass where it was measured (at least ``min_points`` values), and
    its peaks there. A peak outside every window (e.g. on a missing value) goes to the
    nearest one; a chromatogram without NaN is one window."""
    t = frame.index.to_numpy(dtype=float)
    continuous = is_continuous(frame)
    out = []
    for mass, group in peaks.groupby("mass"):
        column = f"{mass:g}"
        if column not in frame:
            continue
        y = frame[column].to_numpy(dtype=float)
        windows = measured_windows(y, continuous, min_points)
        if not windows:
            continue
        starts = np.array([t[w[0]] for w in windows])
        stops = np.array([t[w[-1]] for w in windows])
        rt = group["rt_s"].to_numpy(dtype=float)[:, None]
        # 0 inside a window, else the distance to it
        distance = np.maximum(starts[None, :] - rt, 0) + np.maximum(
            rt - stops[None, :], 0
        )
        nearest = distance.argmin(axis=1)
        for k, index in enumerate(windows):
            inside = group[nearest == k]
            if len(inside):
                out.append((t[index], y[index], inside))
    return out


def fit_peaks(
    frame: pd.DataFrame,
    peaks: pd.DataFrame,
    workers: int | None = None,
    pool: Executor | None = None,
    config: px.ChromatogramFitConfig | None = None,
) -> pd.DataFrame:
    """Fitted peaks of all masses (see ``fit_mass``), in each window where the mass was
    measured (``mass_windows``); ``peaks`` indexed by peak_idx.

    ``frame``: the chromatograms (index: rt [s], columns: masses).
    ``pool``: the processes of the fits (``fit_pool``), else ``workers`` processes
    are started (None: the number of CPUs, 1: in this process).
    ``config``: of the fit (None: ``FIT_CONFIG``).
    The number of warnings of the fits is in ``attrs["n_warnings"]``.
    """
    config = config or FIT_CONFIG
    groups = mass_windows(frame, peaks, config.min_points)
    # Largest first, for the load balance
    groups.sort(key=lambda g: -len(g[2]))
    results = fit_chromatograms(
        None,
        [(t, *_task(y, group)) for t, y, group in groups],
        config,
        workers,
        pool,
    )
    n_warnings = sum(r.attrs.get("n_warnings", 0) for r in results)
    parts = [
        _as_fitted(fitted, group, config)
        for (_, _, group), fitted in zip(groups, results, strict=True)
        if not fitted.empty
    ]
    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=FITTED_COLUMNS)
    )
    out.attrs["n_warnings"] = n_warnings
    if n_warnings:
        logger.debug(f"{n_warnings} warnings in the fits")
    return out


# --- compounds of the split peaks ---------------------------------------------------

# A peak matches a compound when its rt is within RT_TOLERANCE_SIGMA of the
# compound's peak width (sigma) of the compound's rt, widened by RT_TOLERANCE_U times
# the rt uncertainty of the peak, and its width within a factor WIDTH_FACTOR of the
# compound's. Measured on 251104.1724.air.1 (peaks not split, against their own
# compound without them): 95 % are within 0.28 sigma (99 %: 0.9, weak or
# misassigned peaks) and within a factor 1.7 of the width; coeluting compounds
# resolved by the fit are ~2 sigma apart.
RT_TOLERANCE_SIGMA = 0.3
RT_TOLERANCE_U = 3.0
WIDTH_FACTOR = 2.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def compound_shapes(reference: pd.DataFrame) -> pd.DataFrame:
    """rt and width (sigma) of each compound, from its fitted peaks that were not split.

    Medians weighted by the heights, indexed by compound_idx: columns ``rt_s``,
    ``sigma_s`` and ``n_peaks``.
    """
    reference = reference[
        reference["compound_idx"].notna()
        & np.isfinite(reference["fit_rt_s"])
        & (reference["fit_height"] > 0)
        & (reference["fit_sigma_s"] > 0)
    ]
    rows = {}
    for compound_idx, group in reference.groupby("compound_idx"):
        weights = group["fit_height"].to_numpy(dtype=float)
        rows[int(compound_idx)] = {
            "rt_s": _weighted_median(group["fit_rt_s"].to_numpy(dtype=float), weights),
            "sigma_s": _weighted_median(
                group["fit_sigma_s"].to_numpy(dtype=float), weights
            ),
            "n_peaks": len(group),
        }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["rt_s", "sigma_s", "n_peaks"]
    )


def match_compounds(peaks: pd.DataFrame, shapes: pd.DataFrame) -> pd.Series:
    """The compound matching each fitted peak best (``compound_shapes``), NA if none matches.

    Cost of a match: squared rt difference in tolerances plus squared log width
    ratio in log(WIDTH_FACTOR); the compounds outside either tolerance do not match.
    """
    out = pd.Series(pd.NA, index=peaks.index, dtype="Int64")
    if shapes.empty or peaks.empty:
        return out
    rt = peaks["fit_rt_s"].to_numpy(dtype=float)[:, None]
    sigma = peaks["fit_sigma_s"].to_numpy(dtype=float)[:, None]
    rt_u = np.nan_to_num(peaks["fit_rt_u_s"].to_numpy(dtype=float), nan=0.0)[:, None]
    ref_rt = shapes["rt_s"].to_numpy(dtype=float)[None, :]
    ref_sigma = shapes["sigma_s"].to_numpy(dtype=float)[None, :]
    tolerance = np.hypot(RT_TOLERANCE_SIGMA * ref_sigma, RT_TOLERANCE_U * rt_u)
    rt_cost = (rt - ref_rt) / tolerance
    with np.errstate(divide="ignore", invalid="ignore"):
        width_cost = np.log(sigma / ref_sigma) / np.log(WIDTH_FACTOR)
    cost = rt_cost**2 + width_cost**2
    cost[~((np.abs(rt_cost) <= 1) & (np.abs(width_cost) <= 1))] = np.inf
    best = cost.argmin(axis=1)
    found = np.isfinite(cost[np.arange(len(best)), best])
    out[found] = shapes.index.to_numpy()[best[found]]
    return out


# --- peak table ---------------------------------------------------------------------

class ExpectedTarget(NamedTuple):
    """A compound named by target search, as the fit sees it: the rt where the target is
    expected and its ions (unit masses), in the run."""

    rt: float
    ions: frozenset[int]


# Columns of the extraction that describe the peak before a split (NULL after it)
EXTRACTION_ONLY_COLUMNS = [
    "width_scans",
    "width_s",
    "width_height",
    "left_base_scan",
    "right_base_scan",
    "start_s",
    "stop_s",
    "flatness",
    "tailing_ratio",
]


def _half_max_interval(
    peak: pd.Series, min_hwhm: float = HWHM_RANGE_S[0]
) -> tuple[float, float]:
    """Half maximum crossings of an extraction peak, at least ``min_hwhm`` from the apex
    (the smallest half width of the fit)."""
    start, stop = peak["start_s"], peak["stop_s"]
    start = peak["rt_s"] - min_hwhm if not np.isfinite(start) else start
    stop = peak["rt_s"] + min_hwhm if not np.isfinite(stop) else stop
    return min(start, peak["rt_s"] - min_hwhm), max(stop, peak["rt_s"] + min_hwhm)


def _taking_row(
    peak: pd.Series, group: pd.DataFrame, min_hwhm: float = HWHM_RANGE_S[0]
) -> pd.Series:
    """The fitted peaks given to an extraction peak that are its pieces (see ``fitted_peak_table``)."""
    start, stop = _half_max_interval(peak, min_hwhm)
    inside = group["fit_rt_s"].between(start, stop)
    if inside.any():
        return inside
    # None within the crossings: the nearest one within MATCH_HWHM half widths
    rt = peak["rt_s"]
    near = group["fit_rt_s"].between(
        rt - MATCH_HWHM * (rt - start), rt + MATCH_HWHM * (stop - rt)
    )
    if near.any():
        nearest = (group.loc[near, "fit_rt_s"] - rt).abs().idxmin()
        inside[nearest] = True
    return inside


def fitted_peak_table(
    peaks: pd.DataFrame,
    fitted: pd.DataFrame,
    config: px.ChromatogramFitConfig | None = None,
    targets: dict[int, ExpectedTarget] | None = None,
    next_peak_idx: int | None = None,
) -> pd.DataFrame:
    """The peak table of a measurement with the fit (see the module docstring).

    ``peaks``: the extraction peaks, indexed by peak_idx; ``fitted``: ``fit_peaks``
    (with ``config``, None: ``FIT_CONFIG``).
    The fitted peaks given to an extraction peak (the nearest) split it when
    several are within its half maximum crossings; the others are new peaks. The pieces
    go to the compound whose shape they match (``match_compounds``); the one keeping
    the row of the extraction peak, if it matches none, stays in its compound when that
    one has no shape (all its peaks split) - else that compound would lose all its peaks.

    ``targets``: the compounds named by target search (by compound_idx). The split
    peaks of their target's ions are judged by the target: the piece nearest the
    expected rt keeps the row and stays in the compound (the identification gave it
    that ion), whatever shape it matches; the other pieces of that peak go to other
    compounds. The pieces matching no compound are left without one (``new_compounds``
    makes compounds of them). ``next_peak_idx``: of the first peak added (None: after the
    last of ``peaks``).
    Returns all the peaks, indexed by peak_idx (new peaks after the last one), with
    ``is_max_of_cmp`` of all the compounds again.
    """
    min_hwhm = (config or FIT_CONFIG).hwhm_range[0]
    targets = targets or {}

    def target_of(peak_idx) -> ExpectedTarget | None:
        """The target judging an extraction peak: of its compound, on one of its ions."""
        compound = peaks.at[peak_idx, "compound_idx"]
        target = targets.get(int(compound)) if pd.notna(compound) else None
        if target is None or unit_mass(peaks.at[peak_idx, "mass"]) not in target.ions:
            return None
        return target

    table = peaks.copy()
    table["source"] = SOURCE_EXTRACTION
    for column in FIT_COLUMNS:
        table[column] = None if column == "fit_shape" else np.nan
    table["compound_idx"] = table["compound_idx"].astype("Int64")

    matched, to_match = [], []  # (peak_idx, fitted row)
    for peak_idx, group in fitted.groupby("peak_idx"):
        peak = peaks.loc[peak_idx]
        inside = _taking_row(peak, group, min_hwhm)
        # nearest the extracted apex, or the expected rt of a target
        target = target_of(peak_idx)
        reference = peak["rt_s"] if target is None else target.rt
        distance = (group["fit_rt_s"] - reference).abs()
        if inside.sum() == 1:
            matched.append((peak_idx, group[inside].iloc[0]))
        elif inside.sum() > 1:
            nearest = distance[inside].idxmin()
            to_match.append((peak_idx, group.loc[nearest]))
            to_match += [
                (None, row) for i, row in group[inside].iterrows() if i != nearest
            ]
        to_match += [(None, row) for _, row in group[~inside].iterrows()]

    for peak_idx, row in matched:
        for column in FIT_COLUMNS:
            table.at[peak_idx, column] = row[column]

    # Compounds of the split and new peaks, from the peaks fitted as extracted
    shapes = compound_shapes(table.loc[[i for i, _ in matched]])
    pieces = pd.DataFrame([row for _, row in to_match], columns=fitted.columns)
    pieces.index = range(len(pieces))
    compounds = match_compounds(pieces, shapes)
    for k, (peak_idx, row) in enumerate(to_match):
        source = peaks.at[row["peak_idx"], "compound_idx"]  # of the split peak
        target = target_of(row["peak_idx"])
        if target is not None:
            # A target's ion: the piece keeping the row is the target's (nearest its
            # expected rt), the other pieces are not
            if peak_idx is not None:
                compounds.iloc[k] = int(source)
            elif pd.notna(compounds.iloc[k]) and compounds.iloc[k] == int(source):
                compounds.iloc[k] = pd.NA
        elif (
            peak_idx is not None
            and pd.isna(compounds.iloc[k])
            and pd.notna(source)
            and int(source) not in shapes.index
        ):
            # Matching no compound, the piece keeping the row stays in its compound
            # when that one has no shape (all its peaks split): else it would lose all
            # its peaks. Matching another compound, it goes there (e.g. a peak of two
            # compounds the detection did not separate, given to each)
            compounds.iloc[k] = int(source)

    if next_peak_idx is not None:  # e.g. the peaks of some compounds only
        next_idx = next_peak_idx
    else:
        next_idx = int(peaks.index.max()) + 1 if len(peaks) else 0
    new_rows = []
    for k, (peak_idx, row) in enumerate(to_match):
        values = {column: row[column] for column in FIT_COLUMNS}
        values.update(
            source=SOURCE_FIT,
            compound_idx=compounds.iloc[k],
            rt_s=row["fit_rt_s"],
            height=row["signal_height"],
            prominence=row["signal_prominence"],
            is_max_of_cmp=0,
        )
        if peak_idx is not None:
            for column, value in values.items():
                table.at[peak_idx, column] = value
            table.loc[peak_idx, EXTRACTION_ONLY_COLUMNS] = np.nan
        else:
            mass = peaks.at[row["peak_idx"], "mass"]
            new_rows.append({"peak_idx": next_idx, "mass": mass, **values})
            next_idx += 1
    if new_rows:
        new = pd.DataFrame(new_rows).set_index("peak_idx")
        new["compound_idx"] = new["compound_idx"].astype("Int64")
        if "measurement_id" in table:
            new["measurement_id"] = table["measurement_id"].iloc[0]
        table = pd.concat([table, new.reindex(columns=table.columns)])
    _set_max_of_cmp(table, table["compound_idx"].dropna().unique())
    return table


def new_compounds(table: pd.DataFrame, first_idx: int) -> list[int]:
    """Compounds of the pieces of the fit matching no compound (in ``table``, changed),
    numbered from ``first_idx``; returns their compound_idx.

    In rt order, a piece joins the compound of the first piece of its group when within
    the rt tolerance of ``match_compounds`` of it (``RT_TOLERANCE_SIGMA`` of the larger
    width; from the first piece: a group spans no more) and on another mass (one peak
    per mass); else it starts a compound.
    """
    alone = table[
        (table["source"] == SOURCE_FIT)
        & table["compound_idx"].isna()
        & np.isfinite(table["fit_rt_s"].astype(float))
    ].sort_values("fit_rt_s")
    made: list[int] = []
    first, masses = None, set()
    for peak_idx, row in alone.iterrows():
        tolerance = (
            0.0
            if first is None
            else RT_TOLERANCE_SIGMA
            * np.nanmax([row["fit_sigma_s"], first["fit_sigma_s"], 0.0])
        )
        if (
            first is None
            or row["fit_rt_s"] - first["fit_rt_s"] > tolerance
            or row["mass"] in masses
        ):
            made.append(first_idx + len(made))
            first, masses = row, set()
        table.at[peak_idx, "compound_idx"] = made[-1]
        masses.add(row["mass"])
    return made


def changed_compounds(peaks: pd.DataFrame, table: pd.DataFrame) -> set[int]:
    """Compounds that gained, lost or had split peaks with the fit (``peaks``: as extracted)."""
    old = peaks["compound_idx"]
    new = table["compound_idx"].reindex(peaks.index)
    moved = new.ne(old).fillna(True).astype(bool)  # NA: given to no compound
    return set(old[moved].dropna().astype(int)) | set(
        table.loc[table["source"] == SOURCE_FIT, "compound_idx"].dropna().astype(int)
    )


def _set_max_of_cmp(table: pd.DataFrame, compound_idxs) -> None:
    """``is_max_of_cmp`` of the peaks of these compounds: their most prominent peak (as alpinac)."""
    peaks = table[table["compound_idx"].isin(list(compound_idxs))]
    table.loc[peaks.index, "is_max_of_cmp"] = 0
    largest = peaks.groupby("compound_idx")["prominence"].idxmax().dropna()
    table.loc[largest.to_numpy(), "is_max_of_cmp"] = 1


# Summary columns of the compounds from their peaks (as alpinac), kept in extraction_compound
COMPOUND_SUMMARY_COLUMNS = [
    "n_peaks",
    "prominence_sum",
    "rt_mean_s",
    "rt_std_s",
    "main_ion_mass",
]


def compound_summaries(
    table: pd.DataFrame, current: pd.DataFrame, compound_idxs
) -> pd.DataFrame:
    """``COMPOUND_SUMMARY_COLUMNS`` of compounds from the peak table, as alpinac computes them.

    ``current``: the summaries now (indexed by compound_idx); a compound left without
    peaks keeps its rts and its main ion.
    """
    out = current.reindex(sorted(compound_idxs))[COMPOUND_SUMMARY_COLUMNS].copy()
    out["n_peaks"] = 0
    out["prominence_sum"] = 0.0
    peaks = table[table["compound_idx"].isin(out.index)]
    grouped = peaks.groupby("compound_idx")
    has = grouped.size().index.astype(int)
    out.loc[has, "n_peaks"] = grouped.size().to_numpy()
    out.loc[has, "prominence_sum"] = grouped["prominence"].sum().to_numpy()
    out.loc[has, "rt_mean_s"] = grouped["rt_s"].mean().to_numpy()
    out.loc[has, "rt_std_s"] = grouped["rt_s"].std().to_numpy()  # ddof 1, as pandas
    main = peaks[peaks["is_max_of_cmp"] == 1].drop_duplicates("compound_idx")
    out.loc[main["compound_idx"].astype(int).to_numpy(), "main_ion_mass"] = main[
        "mass"
    ].to_numpy()
    return out


def summary(table: pd.DataFrame) -> str:
    """One line on the fit of a measurement (``fit_measurement``)."""
    text = (
        f"{int(table['fit_rt_s'].notna().sum()):4} of {len(table):4} peaks fitted,"
        f" {int((table['source'] == SOURCE_FIT).sum()):3} split or added by the fit"
    )
    n_warnings = table.attrs.get("n_warnings")
    return f"{text}, {n_warnings} warnings" if n_warnings else text


# --- measurements -------------------------------------------------------------------


def fit_version(con: sqlite3.Connection, measurement_id: int) -> int | None:
    """FIT_VERSION of the fit of a measurement, None if not fitted."""
    row = con.execute(
        "SELECT fit_version FROM measurement WHERE measurement_id = ?",
        (int(measurement_id),),
    ).fetchone()
    return None if row is None else row[0]


def is_fitted(con: sqlite3.Connection, measurement_id: int) -> bool:
    return fit_version(con, measurement_id) is not None


# Peak sources (measurement.peak_source) whose peaks are fitted: the extraction of
# alpinac, the peak detection of esper (esper.processing.PEAK_SOURCE)
FITTED_PEAK_SOURCES = ("alpinac", "esper")


def to_fit(con: sqlite3.Connection, force: bool = False) -> list[tuple[int, str]]:
    """(measurement_id, name) of the measurements not fitted, or fitted by an older
    FIT_VERSION (all of them if ``force``), by acquisition time. Only the measurements
    whose peak source is in FITTED_PEAK_SOURCES (not e.g. the GCwerks chromatograms)."""
    sources = ", ".join(f"'{s}'" for s in FITTED_PEAK_SOURCES)
    where = f"WHERE peak_source IN ({sources})"
    if not force:
        where += " AND (fit_version IS NULL OR fit_version < ?)"
    return con.execute(
        f"SELECT measurement_id, name FROM measurement {where} ORDER BY acq_datetime, name",
        () if force else (FIT_VERSION,),
    ).fetchall()


def extraction(
    con: sqlite3.Connection, measurement_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The peaks as extracted (indexed by peak_idx) and the summaries of the compounds
    as extracted (indexed by compound_idx), restored from the fit if any.

    Raises if the measurement was fitted without keeping the extraction (FIT_VERSION 1).
    """
    mid = int(measurement_id)
    columns = _extraction_columns(con)
    peaks = pd.read_sql(
        "SELECT * FROM peak WHERE measurement_id = ?",
        con,
        params=(mid,),
        index_col="peak_idx",
    )
    compounds = _read_compounds(con, mid)  # not those made by the fit: made again
    version = fit_version(con, mid)
    if version is None:
        return peaks[columns], compounds
    if version < 2 and (peaks["source"] == SOURCE_FIT).any():
        raise ValueError(
            f"measurement {mid} was fitted without keeping the extraction,"
            " import it again (esper upload --force)"
        )
    kept = pd.read_sql(
        "SELECT * FROM extraction_peak WHERE measurement_id = ?",
        con,
        params=(mid,),
        index_col="peak_idx",
    )
    kept_compounds = pd.read_sql(
        f"SELECT compound_idx, {', '.join(COMPOUND_SUMMARY_COLUMNS)}"
        " FROM extraction_compound WHERE measurement_id = ?",
        con,
        params=(mid,),
        index_col="compound_idx",
    )
    extracted = peaks[
        (peaks["source"] == SOURCE_EXTRACTION) & ~peaks.index.isin(kept.index)
    ]
    # Not with an empty frame: its columns (object) would make all of them object
    parts = [extracted[columns]] + ([kept[columns]] if len(kept) else [])
    peaks = pd.concat(parts).sort_index()
    compounds.update(kept_compounds)
    return peaks, compounds


def _extraction_columns(con: sqlite3.Connection) -> list[str]:
    """The columns of a peak as extracted (extraction_peak), but its keys."""
    return [
        c
        for c in db.table_columns(con, "extraction_peak")
        if c not in ("measurement_id", "peak_idx")
    ]


def _read_compounds(
    con: sqlite3.Connection, mid: int, idxs: list[int] | None = None
) -> pd.DataFrame:
    """COMPOUND_SUMMARY_COLUMNS of the compounds of a measurement (indexed by
    compound_idx): of ``idxs``, or all those not made by the fit (None)."""
    if idxs is None:
        where, params = NOT_MADE_BY_THE_FIT, ()
    else:
        where, params = f"compound_idx IN ({', '.join('?' * len(idxs))})", tuple(idxs)
    return pd.read_sql(
        f"SELECT compound_idx, {', '.join(COMPOUND_SUMMARY_COLUMNS)} FROM compound"
        f" WHERE measurement_id = ? AND {where}",
        con,
        params=(mid, *params),
        index_col="compound_idx",
    )


def _same(a, b):
    """Element-wise equality of two series or frames, NaN equal to NaN.

    A value and NA are not equal (``==`` gives NA with the nullable dtypes, e.g.
    a compound_idx set to NA by the fit, which ``all`` would skip as true).
    """
    equal = (a == b).fillna(False).astype(bool)
    return equal | (a.isna() & b.isna())


def _changed_rows(before: pd.DataFrame, after: pd.DataFrame) -> pd.Index:
    """Rows of ``before`` whose values are not the same in ``after`` (NaN equal to NaN)."""
    after = after.loc[before.index, before.columns]
    return before.index[~_same(after, before).all(axis=1).to_numpy()]


@dataclass
class FitChanges:
    """What the fit of a measurement writes besides its peak table (``fit_changes``)."""

    # The peaks as extracted whose values the fit changes (-> extraction_peak)
    extraction_peaks: pd.DataFrame
    # The summaries as extracted of the compounds the fit changes (-> extraction_compound)
    extraction_compounds: pd.DataFrame
    # COMPOUND_SUMMARY_COLUMNS to write in the compound table, indexed by compound_idx
    summaries: pd.DataFrame


def fit_changes(
    peaks: pd.DataFrame,
    compounds: pd.DataFrame,
    table: pd.DataFrame,
    earlier=(),
) -> FitChanges:
    """The changes of the fit of a measurement, from the extraction to the peak table.

    ``peaks``, ``compounds``: as extracted (``extraction``); ``table``: the peaks with
    the fit (``fitted_peak_table``); ``earlier``: the compounds changed by an earlier
    fit (in extraction_compound).

    - the compounds that gained, lost or had split peaks (``changed_compounds``) get
      all their summary columns again (``compound_summaries``);
    - the compounds whose peaks did not change but whose largest peak did get only
      their main ion again;
    - the compounds changed by an earlier fit and not by this one get their values
      as extracted back.
    """
    changed = changed_compounds(peaks, table)
    new_main = compound_summaries(table, compounds, compounds.index)["main_ion_mass"]
    same_main = _same(new_main, compounds["main_ion_mass"])
    main_changed = sorted(set(compounds.index[~same_main.to_numpy()]) - changed)
    restored = sorted(
        set(compounds.index[compounds.index.isin(list(earlier))])
        - changed
        - set(main_changed)
    )
    summaries = pd.concat(
        [
            compound_summaries(table, compounds, changed),
            compounds.loc[main_changed].assign(
                main_ion_mass=new_main.loc[main_changed]
            ),
            compounds.loc[restored],
        ]
    )
    return FitChanges(
        extraction_peaks=peaks.loc[_changed_rows(peaks, table)],
        extraction_compounds=compounds.loc[sorted(changed | set(main_changed))],
        summaries=summaries,
    )


def _store_fit(
    con: sqlite3.Connection,
    mid: int,
    table: pd.DataFrame,
    changes: FitChanges,
    made: pd.DataFrame | None = None,
) -> None:
    """Replace the peaks of a measurement with the fit, and the compounds made by an
    earlier fit with ``made`` (indexed by compound_idx), in a single transaction."""
    with con:
        for name in ("peak", "extraction_peak", "extraction_compound"):
            con.execute(f"DELETE FROM {name} WHERE measurement_id = ?", (mid,))
        con.execute(
            "DELETE FROM compound WHERE measurement_id = ? AND source = ?",
            (mid, SOURCE_FIT),
        )
        if made is not None and len(made):
            db.insert(con, "compound", made.assign(measurement_id=mid).reset_index())
        _insert_fit(con, mid, table, changes)
        con.execute(
            "UPDATE measurement SET fit_version = ?, fitted_at = ?"
            " WHERE measurement_id = ?",
            (FIT_VERSION, datetime.now().isoformat(timespec="seconds"), mid),
        )
        db.update_counts(con, mid)


def _insert_fit(
    con: sqlite3.Connection, mid: int, table: pd.DataFrame, changes: FitChanges
) -> None:
    """Insert the peaks of a fit and what it changed of the extraction, and write the
    summaries of its compounds (in the caller's transaction)."""
    for name, df in (
        ("peak", table),
        ("extraction_peak", changes.extraction_peaks),
        ("extraction_compound", changes.extraction_compounds),
    ):
        db.insert(con, name, df.assign(measurement_id=mid).reset_index())
    summaries = changes.summaries[COMPOUND_SUMMARY_COLUMNS]
    assignments = ", ".join(f"{c} = ?" for c in COMPOUND_SUMMARY_COLUMNS)
    db_rows = summaries.astype(object).where(summaries.notna(), None)
    con.executemany(
        f"UPDATE compound SET {assignments} WHERE measurement_id = ? AND compound_idx = ?",
        [
            (*row, mid, int(idx))
            for idx, row in zip(db_rows.index, db_rows.itertuples(index=False, name=None))
        ],
    )


def fit_compounds(
    con: sqlite3.Connection,
    measurement_id: int,
    data_dir: Path,
    compound_idxs,
    workers: int | None = None,
    pool: Executor | None = None,
    config: px.ChromatogramFitConfig | None = None,
    target_search=None,
) -> pd.DataFrame | None:
    """Fit the peaks of some compounds of a measurement only, not fitted yet (e.g. found
    again alone: ``Processor.process_compounds``); the other peaks are kept as they are.

    As ``fit_measurement``, but the pieces only go to these compounds: those matching
    none of them are dropped (kept as extracted in extraction_peak: a fit of the whole
    measurement finds them again). The measurement is marked fitted if it has no other
    peaks. Single transaction. Returns their peak table, None without chromatograms.
    """
    mid = int(measurement_id)
    idxs = [int(i) for i in compound_idxs]
    inputs = _fit_inputs(con, mid)
    if inputs[0] is None:
        return None
    marks = ", ".join("?" * len(idxs))
    peaks = pd.read_sql(
        f"SELECT * FROM peak WHERE measurement_id = ? AND compound_idx IN ({marks})",
        con,
        params=(mid, *idxs),
        index_col="peak_idx",
    )[_extraction_columns(con)]
    compounds = _read_compounds(con, mid, idxs)
    names = _compound_names(con, mid).reindex(idxs)
    first_new = db.next_peak_idx(con, mid)  # after all the peaks, not only these
    fitted = _fitted_table(
        con,
        mid,
        data_dir,
        inputs,
        peaks,
        names,
        workers,
        pool,
        config,
        target_search,
        first_new,
    )
    changes = fit_changes(peaks, compounds, fitted)
    table = fitted[fitted["compound_idx"].notna()].copy()
    table.attrs = dict(fitted.attrs)
    with con:
        con.execute(
            f"DELETE FROM peak WHERE measurement_id = ? AND compound_idx IN ({marks})",
            (mid, *idxs),
        )
        _insert_fit(con, mid, table, changes)
        (others,) = con.execute(
            "SELECT count(*) FROM peak WHERE measurement_id = ?"
            f" AND (compound_idx IS NULL OR compound_idx NOT IN ({marks}))",
            (mid, *idxs),
        ).fetchone()
        if not others:  # its only peaks: the measurement is fitted
            con.execute(
                "UPDATE measurement SET fit_version = coalesce(fit_version, ?),"
                " fitted_at = coalesce(fitted_at, ?) WHERE measurement_id = ?",
                (FIT_VERSION, datetime.now().isoformat(timespec="seconds"), mid),
            )
        db.update_counts(con, mid)
    return table


def fit_measurement(
    con: sqlite3.Connection,
    measurement_id: int,
    data_dir: Path,
    workers: int | None = None,
    pool: Executor | None = None,
    config: px.ChromatogramFitConfig | None = None,
    target_search=None,
) -> pd.DataFrame | None:
    """Fit the peaks of a measurement (again: from the extraction) and store the result.

    Single transaction. Returns the new peak table (``attrs["n_warnings"]``: warnings of
    the fits), None if the measurement has no chromatograms. ``workers``, ``pool``,
    ``config``: see ``fit_peaks``. ``target_search``: of the data directory
    (``esper.processing.identification.TargetSearch``), for the runs it identified:
    their targets judge the split peaks of their ions (``fitted_peak_table``) and name
    the compounds made by the fit; None: neither.

    The pieces matching no compound are compounds of their own (``new_compounds``,
    ``compound.source`` 'fit': made again by each fit).
    """
    mid = int(measurement_id)
    inputs = _fit_inputs(con, mid)
    if inputs[0] is None:
        return None
    _, method, time = inputs
    if method != TARGET:
        target_search = None
    peaks, compounds = extraction(con, mid)
    names = _compound_names(con, mid)
    table = _fitted_table(
        con, mid, data_dir, inputs, peaks, names, workers, pool, config, target_search
    )

    earlier = [
        row[0]
        for row in con.execute(
            "SELECT compound_idx FROM extraction_compound WHERE measurement_id = ?",
            (mid,),
        )
    ]
    changes = fit_changes(peaks, compounds, table, earlier)
    # the pieces matching no compound: compounds of their own, named by target search
    # (which may split them: their compounds numbered after)
    first_idx = int(compounds.index.max()) + 1 if len(compounds) else 0
    made = new_compounds(table, first_idx)
    found = {}
    if target_search is not None and made:
        found = target_search.name(table, made, names, time)
    made = sorted(set(table["compound_idx"].dropna().astype(int)) - set(compounds.index))
    _set_max_of_cmp(table, made)
    made_rows = (
        compound_summaries(table, pd.DataFrame(columns=COMPOUND_SUMMARY_COLUMNS), made)
        .rename_axis("compound_idx")
        .assign(
            compound_name=pd.Series(found, dtype=object).reindex(made),
            valid_candidate=[int(i in found) for i in made],
            is_ref_ri=0,
            source=SOURCE_FIT,
        )
    )
    _store_fit(con, mid, table, changes, made_rows)
    return table


def _fitted_table(
    con: sqlite3.Connection,
    mid: int,
    data_dir: Path,
    inputs: tuple,
    peaks: pd.DataFrame,
    names: pd.Series,
    workers: int | None,
    pool: Executor | None,
    config: px.ChromatogramFitConfig | None,
    target_search,
    next_peak_idx: int | None = None,
) -> pd.DataFrame:
    """The peak table of the fit of ``peaks`` of a measurement (``fitted_peak_table``):
    fitted in its chromatograms, with the fit overrides that apply to it, and its
    targets (of ``names``) if it was identified by target search (``inputs``:
    ``_fit_inputs``). ``attrs["n_warnings"]``: warnings of the fits."""
    chromatogram_file, method, time = inputs
    frame = pd.read_parquet(Path(data_dir) / chromatogram_file)
    fitted = fit_peaks(frame, peaks, workers, pool, config)
    fitted = _with_overrides(con, mid, fitted, frame, peaks, config, time)
    targets = None
    if target_search is not None and method == TARGET:
        targets = target_search.expected(names, time)
    table = fitted_peak_table(peaks, fitted, config, targets, next_peak_idx)
    table.attrs["n_warnings"] = fitted.attrs.get("n_warnings", 0)
    return table


def _with_overrides(
    con: sqlite3.Connection,
    mid: int,
    fitted: pd.DataFrame,
    frame: pd.DataFrame,
    peaks: pd.DataFrame,
    config: px.ChromatogramFitConfig | None,
    time,
) -> pd.DataFrame:
    """The fit of a run with the fit overrides that apply to it (esper.fit_overrides)."""
    from esper import fit_overrides
    from esper.target_index import TargetIndex

    overrides = fit_overrides.read_overrides(con)
    if not overrides:
        return fitted
    (sample_type,) = con.execute(
        "SELECT sample_type FROM measurement WHERE measurement_id = ?", (mid,)
    ).fetchone()
    applying = fit_overrides.run_overrides(overrides, time, sample_type)
    if not applying:
        return fitted
    targets = TargetIndex.from_db(con)
    located = []
    for override in applying:
        reference = fit_overrides.reference_rt(con, mid, override.target, targets, time)
        if reference is None:
            logger.info(
                f"measurement {mid}: no rt of {override.target}, override"
                f" {override.override_id} not applied"
            )
            continue
        located.append((override, reference))
    return fit_overrides.apply_overrides(
        fitted, frame, peaks, located, config or FIT_CONFIG
    )


def _fit_inputs(con: sqlite3.Connection, mid: int) -> tuple:
    """(chromatogram_file, identification, acq_datetime) of a measurement."""
    has_method = "identification" in db.table_columns(con, "measurement")
    method = "identification" if has_method else "NULL"
    return con.execute(
        f"SELECT chromatogram_file, {method}, acq_datetime FROM measurement"
        " WHERE measurement_id = ?",
        (mid,),
    ).fetchone()


def _compound_names(con: sqlite3.Connection, mid: int) -> pd.Series:
    """compound_name of the compounds of a measurement (not made by the fit)."""
    return pd.read_sql(
        f"SELECT compound_idx, compound_name FROM compound WHERE measurement_id = ?"
        f" AND {NOT_MADE_BY_THE_FIT}",
        con,
        params=(mid,),
        index_col="compound_idx",
    )["compound_name"]
