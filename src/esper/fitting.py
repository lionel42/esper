"""Chromatographic fit of the peaks found by the extraction (alpinac), with peaxart.

The peaks of each mass are fitted in its unit mass chromatogram with Pap-Papai
peaks and a constant baseline (``peaxart.Fit``, standard Gaussian convention:
the areas are true areas), by groups of overlapping peaks. The fit starts at the
rts of the extraction and adds peaks where the data show a coelution (peaxart's
extra-peak search).

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
import os
import sqlite3
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from esper import db

logger = logging.getLogger(__name__)

# Increase when the fit changes (2: prominence and height of the fitted signal,
# extraction kept)
FIT_VERSION = 2
# Gaussian convention of peaxart's peak shapes
CONVENTION = "standard"
# Iteration limit of each fit: the few fits that do not converge (weak, unresolved
# peaks) would otherwise take most of the time
MAX_ITERATIONS = 100
# Extent of a peak for the groups of overlapping peaks, in half widths at half
# maximum (of each side) from the apex, longer on the tail side
FRONT_HWHM = 6.0
TAIL_HWHM = 8.0
# Half widths of the extraction are clipped to this range [s]
HWHM_RANGE_S = (0.3, 10.0)
# A fitted peak outside the half maximum crossings of its extraction peak still takes
# its row within this many half widths (of each side), if no other is within them
MATCH_HWHM = 2.0

SOURCE_EXTRACTION = "extraction"
SOURCE_FIT = "fit"

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
}
# Columns of fit_mass / fit_peaks
FITTED_COLUMNS = ["peak_idx", *FIT_COLUMNS, "signal_height", "signal_prominence"]


def fit_method():
    from peaxart import Fit
    from peaxart.config import ExtractionConfig

    return Fit(
        ExtractionConfig(
            convention=CONVENTION,
            legacy_numerics=False,
            rt_max_iterations=MAX_ITERATIONS,
        )
    )


def peak_shape(t: np.ndarray, peak) -> np.ndarray:
    """Signal of a fitted peak above its baseline (``fit_*`` columns of a peak row)."""
    from peaxart.shapes import chromato

    return chromato(
        t,
        peak.fit_height,
        peak.fit_rt_s,
        peak.fit_sigma_s,
        peak.fit_tailing,
        CONVENTION,
    )


def peak_groups(peaks: pd.DataFrame) -> list[tuple[pd.DataFrame, float, float]]:
    """Groups of overlapping peaks of one mass, with the rt range fitted [s]."""
    peaks = peaks.sort_values("rt_s")
    rt = peaks["rt_s"].to_numpy(dtype=float)
    front = np.clip(rt - peaks["start_s"].to_numpy(dtype=float), *HWHM_RANGE_S)
    tail = np.clip(peaks["stop_s"].to_numpy(dtype=float) - rt, *HWHM_RANGE_S)
    # A half width not given by the extraction (NaN after the clip): the smallest
    front, tail = (
        np.nan_to_num(front, nan=HWHM_RANGE_S[0]),
        np.nan_to_num(tail, nan=HWHM_RANGE_S[0]),
    )
    lo, hi = rt - FRONT_HWHM * front, rt + TAIL_HWHM * tail
    # Connected intervals, swept by their start (a broad peak can start before the
    # narrow ones eluting before it)
    order = np.argsort(lo, kind="stable")
    groups = []
    members, end = [], -np.inf
    for i in order:
        if members and lo[i] >= end:
            groups.append(members)
            members = []
        members.append(i)
        end = max(end, hi[i]) if len(members) > 1 else hi[i]
    if members:
        groups.append(members)
    return [
        (peaks.iloc[sorted(g)], float(lo[g].min()), float(hi[g].max())) for g in groups
    ]


def signal_peaks(t: np.ndarray, fitted: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Height and prominence of fitted peaks of one group, on the fitted signal.

    The fitted signal is the baseline plus all the fitted peaks of the group; the
    height is its value at the apex, the prominence is scipy's (``find_peaks``, as
    the extraction). A shoulder (apex not a local maximum) has none by scipy's
    definition: it gets its prominence alone, its fitted height above the baseline.
    """
    from scipy.signal import peak_prominences

    signal = fitted["fit_baseline"].iloc[0] + sum(
        np.nan_to_num(peak_shape(t, p)) for p in fitted.itertuples()
    )
    rt = fitted["fit_rt_s"].to_numpy(dtype=float)
    nearest = np.abs(t[None, :] - rt[:, None]).argmin(axis=1)
    # The apex falls between samples: the highest of the nearest one and its neighbours
    apex = np.array(
        [max(i - 1, 0) + int(np.argmax(signal[max(i - 1, 0) : i + 2])) for i in nearest]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PeakPropertyWarning: a prominence is 0
        prominence = peak_prominences(signal, apex)[0]
    shoulder = prominence <= 0
    prominence[shoulder] = fitted["fit_height"].to_numpy(dtype=float)[shoulder]
    return signal[apex], prominence


def fit_mass(
    t: np.ndarray, y: np.ndarray, peaks: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Fitted peaks of one mass, and the number of warnings of the fits.

    Columns: ``peak_idx`` (the nearest extraction peak), the ``fit_*`` columns and
    ``signal_height`` / ``signal_prominence`` (``signal_peaks``).
    """
    method = fit_method()
    rows = []
    n_warnings = 0
    for group, lo, hi in peak_groups(peaks):
        inside = (t >= lo) & (t <= hi)
        if inside.sum() < 10:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = method.integrate(t[inside], y[inside], rt=group["rt_s"].to_numpy())
        n_warnings += len(caught)
        if fitted.empty:
            continue
        rt = group["rt_s"].to_numpy(dtype=float)
        nearest = np.abs(fitted["rt"].to_numpy()[:, None] - rt[None, :]).argmin(axis=1)
        fitted = fitted.rename(columns={v: k for k, v in FIT_COLUMNS.items()})
        fitted["peak_idx"] = group.index.to_numpy()[nearest]
        fitted["signal_height"], fitted["signal_prominence"] = signal_peaks(
            t[inside], fitted
        )
        rows.append(fitted[FITTED_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=FITTED_COLUMNS), n_warnings
    return pd.concat(rows, ignore_index=True), n_warnings


def _fit_mass_task(args) -> tuple[pd.DataFrame, int]:
    return fit_mass(*args)


@contextmanager
def fit_pool(workers: int | None = None):
    """Processes for the fits of several measurements (``fit_peaks``); None if ``workers`` is 1."""
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        yield None
        return
    with ProcessPoolExecutor(workers) as pool:
        yield pool


def fit_peaks(
    frame: pd.DataFrame,
    peaks: pd.DataFrame,
    workers: int | None = None,
    pool: Executor | None = None,
) -> pd.DataFrame:
    """Fitted peaks of all masses (see ``fit_mass``); ``peaks`` indexed by peak_idx.

    ``frame``: the unit mass chromatograms (index: rt [s], columns: masses).
    ``pool``: the processes of the fits (``fit_pool``), else ``workers`` processes
    are started (None: the number of CPUs, 1: in this process).
    The number of warnings of the fits is in ``attrs["n_warnings"]``.
    """
    t = frame.index.to_numpy(dtype=float)
    tasks = [
        (t, frame[str(mass)].to_numpy(dtype=float), group)
        for mass, group in peaks.groupby("mass")
        if str(mass) in frame
    ]
    # Largest first, for the load balance
    tasks.sort(key=lambda task: -len(task[2]))

    def run(pool):
        if pool is None or len(tasks) < 2:
            return [_fit_mass_task(task) for task in tasks]
        return list(pool.map(_fit_mass_task, tasks))

    if pool is None and len(tasks) > 1:
        with fit_pool(min(workers or os.cpu_count() or 1, len(tasks))) as own:
            results = run(own)
    else:
        results = run(pool)
    n_warnings = sum(n for _, n in results)
    results = [r for r, _ in results if not r.empty]
    out = (
        pd.concat(results, ignore_index=True)
        if results
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


def _half_max_interval(peak: pd.Series) -> tuple[float, float]:
    """Half maximum crossings of an extraction peak, at least HWHM_RANGE_S[0] from the apex."""
    start, stop = peak["start_s"], peak["stop_s"]
    start = peak["rt_s"] - HWHM_RANGE_S[0] if not np.isfinite(start) else start
    stop = peak["rt_s"] + HWHM_RANGE_S[0] if not np.isfinite(stop) else stop
    return min(start, peak["rt_s"] - HWHM_RANGE_S[0]), max(
        stop, peak["rt_s"] + HWHM_RANGE_S[0]
    )


def _taking_row(peak: pd.Series, group: pd.DataFrame) -> pd.Series:
    """The fitted peaks given to an extraction peak that are its pieces (see ``fitted_peak_table``)."""
    start, stop = _half_max_interval(peak)
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


def fitted_peak_table(peaks: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """The peak table of a measurement with the fit (see the module docstring).

    ``peaks``: the extraction peaks, indexed by peak_idx; ``fitted``: ``fit_peaks``.
    The fitted peaks given to an extraction peak (the nearest) split it when
    several are within its half maximum crossings; the others are new peaks.
    Returns all the peaks, indexed by peak_idx (new peaks after the last one), with
    ``is_max_of_cmp`` of all the compounds again.
    """
    table = peaks.copy()
    table["source"] = SOURCE_EXTRACTION
    for column in FIT_COLUMNS:
        table[column] = np.nan
    table["compound_idx"] = table["compound_idx"].astype("Int64")

    matched, to_match = [], []  # (peak_idx, fitted row)
    for peak_idx, group in fitted.groupby("peak_idx"):
        peak = peaks.loc[peak_idx]
        inside = _taking_row(peak, group)
        distance = (group["fit_rt_s"] - peak["rt_s"]).abs()
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
        f"{int(table['fit_rt_s'].notna().sum())} of {len(table)} peaks fitted,"
        f" {int((table['source'] == SOURCE_FIT).sum())} split or added by the fit"
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


def to_fit(con: sqlite3.Connection, force: bool = False) -> list[tuple[int, str]]:
    """(measurement_id, name) of the measurements not fitted, or fitted by an older
    FIT_VERSION (all of them if ``force``), by acquisition time."""
    where = "" if force else "WHERE fit_version IS NULL OR fit_version < ?"
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
    columns = [
        c for c in db.table_columns(con, "extraction_peak") if c != "measurement_id"
    ]
    peaks = pd.read_sql(
        "SELECT * FROM peak WHERE measurement_id = ?",
        con,
        params=(mid,),
        index_col="peak_idx",
    )
    compounds = pd.read_sql(
        f"SELECT compound_idx, {', '.join(COMPOUND_SUMMARY_COLUMNS)} FROM compound"
        " WHERE measurement_id = ?",
        con,
        params=(mid,),
        index_col="compound_idx",
    )
    version = fit_version(con, mid)
    if version is None:
        return peaks[[c for c in columns if c != "peak_idx"]], compounds
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
    columns = [c for c in columns if c != "peak_idx"]
    peaks = pd.concat([extracted[columns], kept[columns]]).sort_index()
    compounds.update(kept_compounds)
    return peaks, compounds


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
    con: sqlite3.Connection, mid: int, table: pd.DataFrame, changes: FitChanges
) -> None:
    """Replace the peaks of a measurement with the fit (single transaction)."""
    summaries = changes.summaries[COMPOUND_SUMMARY_COLUMNS]
    assignments = ", ".join(f"{c} = ?" for c in COMPOUND_SUMMARY_COLUMNS)
    db_rows = summaries.astype(object).where(summaries.notna(), None)
    with con:
        for name in ("peak", "extraction_peak", "extraction_compound"):
            con.execute(f"DELETE FROM {name} WHERE measurement_id = ?", (mid,))
        for name, df in (
            ("peak", table),
            ("extraction_peak", changes.extraction_peaks),
            ("extraction_compound", changes.extraction_compounds),
        ):
            db.insert(con, name, df.assign(measurement_id=mid).reset_index())
        con.executemany(
            f"UPDATE compound SET {assignments} WHERE measurement_id = ? AND compound_idx = ?",
            [
                (*row, mid, int(idx))
                for idx, row in zip(
                    db_rows.index, db_rows.itertuples(index=False, name=None)
                )
            ],
        )
        con.execute(
            "UPDATE measurement SET n_peaks = ?, fit_version = ?, fitted_at = ?"
            " WHERE measurement_id = ?",
            (
                len(table),
                FIT_VERSION,
                datetime.now().isoformat(timespec="seconds"),
                mid,
            ),
        )


def fit_measurement(
    con: sqlite3.Connection,
    measurement_id: int,
    data_dir: Path,
    workers: int | None = None,
    pool: Executor | None = None,
) -> pd.DataFrame | None:
    """Fit the peaks of a measurement (again: from the extraction) and store the result.

    Single transaction. Returns the new peak table (``attrs["n_warnings"]``: warnings of
    the fits), None if the measurement has no chromatograms. ``workers``, ``pool``: see
    ``fit_peaks``.
    """
    mid = int(measurement_id)
    (chromatogram_file,) = con.execute(
        "SELECT chromatogram_file FROM measurement WHERE measurement_id = ?", (mid,)
    ).fetchone()
    if chromatogram_file is None:
        return None
    peaks, compounds = extraction(con, mid)
    frame = pd.read_parquet(Path(data_dir) / chromatogram_file)
    fitted = fit_peaks(frame, peaks, workers, pool)
    table = fitted_peak_table(peaks, fitted)
    table.attrs["n_warnings"] = fitted.attrs.get("n_warnings", 0)

    earlier = [
        row[0]
        for row in con.execute(
            "SELECT compound_idx FROM extraction_compound WHERE measurement_id = ?",
            (mid,),
        )
    ]
    _store_fit(con, mid, table, fit_changes(peaks, compounds, table, earlier))
    return table
