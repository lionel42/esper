"""Chromatographic fit of the peaks found by the extraction (alpinac), with peaxart.

The peaks of each mass are fitted in its unit mass chromatogram with Pap-Papai
peaks and a constant baseline (``peaxart.Fit``, standard Gaussian convention:
the areas are true areas), by groups of overlapping peaks. The fit starts at the
rts of the extraction and adds peaks where the data show a coelution (peaxart's
extra-peak search).

Each fitted peak goes to the nearest extraction peak of its group:

- given one fitted peak, the extraction peak keeps its row (``source``
  'extraction') with the ``fit_*`` columns filled;
- given several, it is split: its row takes the fitted peak nearest to it,
  new rows the others. All get ``source`` 'fit', ``rt_s`` and ``height`` /
  ``prominence`` of the fit and NULL for the other extraction columns (they
  describe the unsplit peak), and are assigned to the compound that matches them
  (``match_compounds``), or to none (``compound_idx`` NULL);
- given none (rejected by the fit: e.g. a straight background explains the
  data as well), the fit columns stay NULL.

A measurement is fitted once (``measurement.fit_version``), after its import: to
fit it again, import it again (``esper upload --force``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from esper import db

logger = logging.getLogger(__name__)

# Increase when the fit changes
FIT_VERSION = 1
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
    groups = []
    first, end = 0, hi[0] if len(rt) else 0.0
    for i in range(1, len(rt) + 1):
        if i == len(rt) or lo[i] >= end:
            groups.append((peaks.iloc[first:i], float(lo[first:i].min()), float(end)))
            if i < len(rt):
                first, end = i, hi[i]
        else:
            end = max(end, hi[i])
    return groups


def fit_mass(t: np.ndarray, y: np.ndarray, peaks: pd.DataFrame) -> pd.DataFrame:
    """Fitted peaks of one mass: ``peak_idx`` (the nearest extraction peak) and the ``fit_*`` columns."""
    method = fit_method()
    rows = []
    for group, lo, hi in peak_groups(peaks):
        inside = (t >= lo) & (t <= hi)
        if inside.sum() < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = method.integrate(t[inside], y[inside], rt=group["rt_s"].to_numpy())
        if fitted.empty:
            continue
        rt = group["rt_s"].to_numpy(dtype=float)
        nearest = np.abs(fitted["rt"].to_numpy()[:, None] - rt[None, :]).argmin(axis=1)
        fitted = fitted.rename(columns={v: k for k, v in FIT_COLUMNS.items()})
        fitted["peak_idx"] = group.index.to_numpy()[nearest]
        rows.append(fitted[["peak_idx", *FIT_COLUMNS]])
    if not rows:
        return pd.DataFrame(columns=["peak_idx", *FIT_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def _fit_mass_task(args) -> pd.DataFrame:
    return fit_mass(*args)


def fit_peaks(
    frame: pd.DataFrame, peaks: pd.DataFrame, workers: int | None = None
) -> pd.DataFrame:
    """Fitted peaks of all masses (see ``fit_mass``); ``peaks`` indexed by peak_idx.

    ``frame``: the unit mass chromatograms (index: rt [s], columns: masses).
    ``workers``: processes (None: the number of CPUs, 1: in this process).
    """
    t = frame.index.to_numpy(dtype=float)
    tasks = [
        (t, frame[str(mass)].to_numpy(dtype=float), group)
        for mass, group in peaks.groupby("mass")
        if str(mass) in frame
    ]
    workers = workers or os.cpu_count() or 1
    if workers == 1 or len(tasks) < 2:
        results = [_fit_mass_task(task) for task in tasks]
    else:
        # Largest first, for the load balance
        tasks.sort(key=lambda task: -len(task[2]))
        with ProcessPoolExecutor(min(workers, len(tasks))) as pool:
            results = list(pool.map(_fit_mass_task, tasks))
    results = [r for r in results if not r.empty]
    if not results:
        return pd.DataFrame(columns=["peak_idx", *FIT_COLUMNS])
    return pd.concat(results, ignore_index=True)


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


def fitted_peak_table(peaks: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """The peak table of a measurement with the fit (see the module docstring).

    ``peaks``: the extraction peaks, indexed by peak_idx; ``fitted``: ``fit_peaks``.
    The fitted peaks given to an extraction peak (the nearest) split it when
    several are within its half maximum crossings; the others are new peaks.
    Returns all the peaks, indexed by peak_idx (new peaks after the last one).
    """
    table = peaks.copy()
    table["source"] = SOURCE_EXTRACTION
    for column in FIT_COLUMNS:
        table[column] = np.nan
    table["compound_idx"] = table["compound_idx"].astype("Int64")

    matched, to_match = [], []  # (peak_idx, fitted row)
    for peak_idx, group in fitted.groupby("peak_idx"):
        peak = peaks.loc[peak_idx]
        start, stop = _half_max_interval(peak)
        inside = group["fit_rt_s"].between(start, stop)
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
            height=row["fit_baseline"] + row["fit_height"],
            prominence=row["fit_height"],
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
    return table


def _compound_summaries(table: pd.DataFrame, compound_idxs) -> list[tuple]:
    """(n_peaks, prominence_sum, rt_mean_s, rt_std_s, compound_idx) of compounds, from
    the peak table; the rts are None for a compound left without peaks."""
    rows = []
    for compound_idx in compound_idxs:
        peaks = table[table["compound_idx"] == compound_idx]
        rt = peaks["rt_s"].to_numpy(dtype=float)
        rows.append(
            (
                len(peaks),
                float(peaks["prominence"].sum()),
                float(rt.mean()) if len(rt) else None,
                float(rt.std()) if len(rt) else None,
                int(compound_idx),
            )
        )
    return rows


def is_fitted(con: sqlite3.Connection, measurement_id: int) -> bool:
    row = con.execute(
        "SELECT fit_version FROM measurement WHERE measurement_id = ?",
        (int(measurement_id),),
    ).fetchone()
    return row is not None and row[0] is not None


def fit_measurement(
    con: sqlite3.Connection,
    measurement_id: int,
    data_dir: Path,
    workers: int | None = None,
) -> pd.DataFrame | None:
    """Fit the peaks of a measurement and store the result (single transaction).

    Returns the new peak table, None if the measurement has no chromatograms.
    Raises if the measurement was already fitted (import it again to fit it again).
    """
    mid = int(measurement_id)
    if is_fitted(con, mid):
        raise ValueError(f"measurement {mid} is already fitted, import it again")
    (chromatogram_file,) = con.execute(
        "SELECT chromatogram_file FROM measurement WHERE measurement_id = ?", (mid,)
    ).fetchone()
    if chromatogram_file is None:
        return None
    frame = pd.read_parquet(Path(data_dir) / chromatogram_file)
    peaks = pd.read_sql(
        "SELECT * FROM peak WHERE measurement_id = ?",
        con,
        params=(mid,),
        index_col="peak_idx",
    )
    fitted = fit_peaks(frame, peaks, workers)
    table = fitted_peak_table(peaks, fitted)

    old = peaks["compound_idx"]
    new = table["compound_idx"].reindex(peaks.index)
    moved = new.ne(old).fillna(True).astype(bool)  # NA: given to no compound
    changed = set(old[moved].dropna().astype(int)) | set(
        table.loc[table["source"] == SOURCE_FIT, "compound_idx"].dropna().astype(int)
    )
    summaries = _compound_summaries(table, sorted(changed))
    with con:
        con.execute("DELETE FROM peak WHERE measurement_id = ?", (mid,))
        db.insert(con, "peak", table.assign(measurement_id=mid).reset_index())
        con.executemany(
            "UPDATE compound SET n_peaks = ?, prominence_sum = ?,"
            " rt_mean_s = COALESCE(?, rt_mean_s), rt_std_s = COALESCE(?, rt_std_s)"
            " WHERE measurement_id = ? AND compound_idx = ?",
            [(*row[:4], mid, row[4]) for row in summaries],
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
    return table
