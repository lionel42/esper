"""Integration of the peaks found by alpinac in the unit mass chromatograms.

Window: each peak is integrated from ``N_HWHM_FRONT`` half widths at half
maximum before its apex to ``N_HWHM_TAIL`` after (each side with its own half
width, longer on the tail side because GC peaks tail). The window does not go
beyond the prominence bases of the peak, nor past a valley: a bound that falls
on another peak (not picked by alpinac) is moved back to the valley before it,
and the baseline there is not higher than at the other end (as the drop line
would do if the other peak had been picked).

Overlaps: peaks of the same mass whose windows overlap are split at the lowest
point between their apexes (drop line) and share one baseline, under the whole
group.

Baseline: a straight line from the start to the end of the group, but never
above the signal. Where the signal goes below that line (tail of a large peak,
unpicked peak), the baseline follows the lower convex hull of the signal. The
baseline is drawn on the smoothed signal (``noise.SMOOTH_SCANS``), and only
follows the signal down when it is below the line by more than the noise
(``HULL_NOISE_FACTOR``), so that noise does not pull it down. When the group
is cut by the end of the chromatogram, the baseline is flat at the level of
its other end.

Areas are in counts * s. A negative area is kept as is; it is NaN when it
cannot be computed (no chromatogram for the mass, empty window).

Uncertainty (1 sigma, statistical only, see esper.noise for the noise): the
noise of the signal under the peak, of the baseline levels and of the
position of the splits with the neighbouring peaks, propagated through the
trapezoid sum. Not included: the choice of the window and of the baseline.

The results are stored in the database: per peak (window, baseline at its
ends, area and its uncertainty), per compound (``area_sum`` and ``q1..q5``,
its masses by decreasing area, with uncertainties) and per measurement
(``integration_version`` and the noise parameters).
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from esper import db
from esper.noise import NoiseModel, NoiseParameters, SMOOTH_SCANS, measurement_noise, noise_model, smooth

logger = logging.getLogger(__name__)

# Increase when the method changes: measurements integrated with an older version are redone
INTEGRATION_VERSION = 2

# Integration bounds, in half widths at half maximum of that side from the apex
# (4 ~ 4.7 sigma for a gaussian). Longer after the apex, for tailing peaks.
N_HWHM_FRONT = 4.0
N_HWHM_TAIL = 6.0
# The baseline follows the signal down where it is below the straight line by more
# than this times the noise of the smoothed signal (also for the valleys at the bounds)
HULL_NOISE_FACTOR = 4.0

PEAK_COLUMNS = [name for name, _ in db.ADDED_COLUMNS["peak"]]
COMPOUND_COLUMNS = [name for name, _ in db.ADDED_COLUMNS["compound"]]
# Peak columns needed to integrate
PEAK_INPUT_COLUMNS = ["compound_idx", "mass", "rt_s", "start_s", "stop_s", "left_base_scan", "right_base_scan"]


# --- one chromatogram -------------------------------------------------------------------


def _scans(t: np.ndarray, lo: float, hi: float, inclusive: bool = True) -> np.ndarray:
    """Indices of the scans with lo <= t <= hi (lo < t < hi if not inclusive); t sorted."""
    side_lo, side_hi = ("left", "right") if inclusive else ("right", "left")
    return np.arange(np.searchsorted(t, lo, side_lo), np.searchsorted(t, hi, side_hi))


def _integration_bounds(peaks: pd.DataFrame, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of each peak [s], before the valleys and the splits.

    ``start_s``/``stop_s`` are the half maximum crossings, the peak is extended
    to ``N_HWHM_FRONT`` / ``N_HWHM_TAIL`` half widths of each side but not
    beyond its prominence bases nor the chromatogram.
    """
    rt, start_hm, stop_hm = (peaks[c].to_numpy(dtype=float) for c in ("rt_s", "start_s", "stop_s"))
    last = len(t) - 1
    left_base = t[np.clip(peaks["left_base_scan"].to_numpy(), 0, last)]
    right_base = t[np.clip(peaks["right_base_scan"].to_numpy(), 0, last)]
    start = np.maximum(rt - N_HWHM_FRONT * (rt - start_hm), left_base)
    stop = np.minimum(rt + N_HWHM_TAIL * (stop_hm - rt), right_base)
    return np.clip(start, t[0], t[-1]), np.clip(stop, t[0], t[-1])


def _clip_to_valleys(t, smoothed, noise: NoiseModel, rt, start, stop):
    """Move the bounds back to the lowest point between apex and bound, if the signal rises after it.

    Only when the bound is above that lowest point by more than
    ``HULL_NOISE_FACTOR`` times the noise of the smoothed signal (on a clean
    tail, the lowest point is the bound itself or in the noise).
    Returns start, stop and whether each was moved.
    """
    start, stop = start.copy(), stop.copy()
    moved_start, moved_stop = np.zeros(len(rt), bool), np.zeros(len(rt), bool)
    for k in range(len(rt)):
        for bounds, moved, lo, hi in ((start, moved_start, start[k], rt[k]), (stop, moved_stop, rt[k], stop[k])):
            idx = _scans(t, lo, hi)
            if len(idx) < 2 or np.isnan(smoothed[idx]).all():
                continue
            i = idx[np.nanargmin(smoothed[idx])]
            level = np.interp(bounds[k], t, smoothed)
            if level > smoothed[i] and level > smoothed[i] + HULL_NOISE_FACTOR * noise.mean_sd(smoothed[i]):
                bounds[k] = t[i]
                moved[k] = True
    return start, stop, moved_start, moved_stop


def _valley(t, smoothed, lo: float, hi: float, noise: NoiseModel) -> tuple[float, float]:
    """Time of the lowest point of the smoothed signal in [lo, hi], and its uncertainty [s].

    Refined between scans with a parabola through the lowest scan and its
    neighbours. The uncertainty is the noise of the slope there over the
    curvature (at most half the interval; also when there is no curvature).
    """
    max_sd = (hi - lo) / 2
    idx = _scans(t, lo, hi)
    if not len(idx) or np.isnan(smoothed[idx]).all():
        return 0.5 * (lo + hi), max_sd
    i = idx[np.nanargmin(smoothed[idx])]
    if 0 < i < len(t) - 1:
        s0, s1, s2 = smoothed[i - 1], smoothed[i], smoothed[i + 1]
        curvature = s0 - 2 * s1 + s2
        if curvature > 0:
            # Uniform scans: offset in scans from i
            scan = (t[i + 1] - t[i - 1]) / 2
            offset = 0.5 * (s0 - s2) / curvature
            cut = float(np.clip(t[i] + offset * scan, lo, hi))
            # s2 - s0: difference of two moving means, 2 scans apart
            slope_sd = 2 * math.sqrt(noise.sum_variance(s1)) / SMOOTH_SCANS
            return cut, float(min(0.5 * slope_sd / curvature * scan, max_sd))
    return float(t[i]), max_sd


def _lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of points sorted by x (monotone chain)."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Drop b if it is on or above the segment a -> i
            if (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.array(hull)


@dataclass
class _Baseline:
    """Piecewise linear baseline: its vertices (x, y), and for each vertex the
    time of the smoothed sample its level comes from (two vertices can share
    one: their noise is the same)."""

    x: np.ndarray
    y: np.ndarray
    source: np.ndarray

    def __call__(self, t):
        return np.interp(t, self.x, self.y)

    def source_weights(self, s: float, e: float, tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sources and d area / d source level (minus sign left out) for a peak from s to e
        integrated over the points tt: the integral of the hat functions of their vertices."""
        if len(self.x) == 2:
            # Straight baseline: exact
            w0 = (e - s) * (self.x[1] - (s + e) / 2) / (self.x[1] - self.x[0])
            vertex_weights = np.array([w0, (e - s) - w0])
        else:
            vertex_weights = np.array([
                np.trapezoid(np.interp(tt, self.x, np.eye(len(self.x))[v]), tt) for v in range(len(self.x))
            ])
        sources, vertex_source = np.unique(self.source, return_inverse=True)
        return sources, np.bincount(vertex_source, weights=vertex_weights, minlength=len(sources))


def _baseline(t, smoothed, noise: NoiseModel, start, stop, flat=None, valley_start=False, valley_stop=False) -> _Baseline:
    """Baseline under [start, stop].

    ``flat``: 'start' / 'stop' when that end is cut by the end of the
    chromatogram (the level there is the peak, not the baseline): the baseline
    is flat at the level of the other end.
    ``valley_start`` / ``valley_stop``: that end was moved back to a valley,
    its level is not above the level of the other end.
    Dips below the straight line smaller than ``HULL_NOISE_FACTOR`` times the
    noise of the smoothed signal are ignored.
    """
    b0, b1 = np.interp([start, stop], t, smoothed)
    src0, src1 = start, stop
    if flat == "stop":
        b1, src1 = b0, src0
    elif flat == "start":
        b0, src0 = b1, src1
    if valley_start and b1 < b0:
        b0, src0 = b1, src1
    if valley_stop and b0 < b1:
        b1, src1 = b0, src0
    inner = _scans(t, start, stop, inclusive=False)
    x = np.concatenate([[start], t[inner], [stop]])
    y = np.concatenate([[b0], smoothed[inner], [b1]])
    chord = b0 + (b1 - b0) * (x - start) / (stop - start)
    below = y < chord - HULL_NOISE_FACTOR * noise.mean_sd(y)
    if not below.any():
        return _Baseline(np.array([start, stop]), np.array([b0, b1]), np.array([src0, src1]))
    # Hull of the points clearly below the line (and the ends)
    below[[0, -1]] = True
    source = x.copy()
    source[[0, -1]] = src0, src1
    hull = _lower_hull(x[below], y[below])
    return _Baseline(x[below][hull], y[below][hull], source[below][hull])


def _area_variance(t, smoothed, noise: NoiseModel, baseline: _Baseline, tt, splits) -> float:
    """Variance of the area of a peak integrated over the points tt above the baseline.

    Noise of the signal (trapezoid weights), of the baseline levels (moving
    means) and of the position of the splits with the neighbours (signal above
    the baseline there times the uncertainty of the position). ``splits``:
    (time, uncertainty [s]) of the splits at the bounds of the peak.
    """
    s, e = tt[0], tt[-1]
    dt = np.diff(tt)
    weights = np.concatenate([[0.0], dt]) / 2 + np.concatenate([dt, [0.0]]) / 2
    signal = np.sum(weights**2 * noise.sum_variance(np.interp(tt, t, smoothed)))
    sources, source_weights = baseline.source_weights(s, e, tt)
    levels = np.sum(source_weights**2 * noise.mean_sd(np.interp(sources, t, smoothed)) ** 2)
    split = sum(((np.interp(bound, t, smoothed) - baseline(bound)) * sd) ** 2 for bound, sd in splits if sd > 0)
    return float(signal + levels + split)


def integrate_mass(
    peaks: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    noise: NoiseParameters | None = None,
    baselines: list | None = None,
) -> pd.DataFrame:
    """Integrate the peaks of one mass in its chromatogram (rt ``t`` [s], intensity ``y``).

    Returns ``PEAK_COLUMNS`` with the index of ``peaks``. ``noise``: the noise
    parameters of the measurement (estimated from y alone if not given). The
    baseline of each group of peaks, as (x, y) of its vertices, is appended to
    ``baselines`` if given.
    """
    out = pd.DataFrame(np.nan, index=peaks.index, columns=PEAK_COLUMNS)
    if peaks.empty or len(t) < 2:
        return out
    peaks = peaks.iloc[np.argsort(peaks["rt_s"].to_numpy(dtype=float), kind="stable")]
    rt = peaks["rt_s"].to_numpy(dtype=float)
    smoothed = smooth(y)
    model = noise_model(y, noise, smoothed)
    start, stop = _integration_bounds(peaks, t)
    start, stop, valley_start, valley_stop = _clip_to_valleys(t, smoothed, model, rt, start, stop)

    # Split overlapping neighbours at the valley between them; they form a group
    group = np.zeros(len(peaks), dtype=int)
    split_sd = np.zeros(len(peaks))  # uncertainty of the split after each peak [s]
    for i in range(len(peaks) - 1):
        j = i + 1
        if stop[i] > start[j]:
            cut, split_sd[i] = _valley(t, smoothed, max(rt[i], start[j]), min(rt[j], stop[i]), model)
            stop[i] = start[j] = cut
            group[j] = group[i]
        else:
            group[j] = group[i] + 1

    values = np.full((len(peaks), len(PEAK_COLUMNS)), np.nan)
    for g in np.unique(group):
        members = np.flatnonzero(group == g)
        g_start, g_stop = start[members[0]], stop[members[-1]]
        if not g_stop > g_start:
            continue
        flat = None
        if g_stop >= t[-1] and g_start > t[0]:
            flat = "stop"
        elif g_start <= t[0] and g_stop < t[-1]:
            flat = "start"
        baseline = _baseline(
            t, smoothed, model, g_start, g_stop, flat, valley_start[members[0]], valley_stop[members[-1]]
        )
        if baselines is not None:
            baselines.append((baseline.x, baseline.y))

        for k in members:
            s, e = start[k], stop[k]
            if not e > s:
                continue
            inner = _scans(t, s, e, inclusive=False)
            tt = np.concatenate([[s], t[inner], [e]])
            yy = np.concatenate([[np.interp(s, t, y)], y[inner], [np.interp(e, t, y)]])
            area = float(np.trapezoid(yy - baseline(tt), tt))
            splits = ((s, split_sd[k - 1] if k > 0 else 0.0), (e, split_sd[k]))
            uncertainty = math.sqrt(_area_variance(t, smoothed, model, baseline, tt, splits))
            values[k] = (s, e, baseline(s), baseline(e), area, uncertainty)
    out.loc[peaks.index] = values
    return out


# --- one measurement --------------------------------------------------------------------


def integrate_chromatograms(
    peaks: pd.DataFrame, path: Path, compound_idx: Iterable[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, NoiseParameters]:
    """Integrate the peaks of a measurement in its chromatogram file.

    ``peaks`` are all the peaks of the measurement (``PEAK_INPUT_COLUMNS``,
    indexed by peak_idx). Only the peaks of the compounds ``compound_idx`` are
    integrated if given; the other peaks of their masses are used to split
    overlapping peaks.

    Returns the peak results (``PEAK_COLUMNS``, by peak_idx), the compound
    results (``COMPOUND_COLUMNS``, by compound_idx) and the noise parameters
    of the measurement.
    """
    selected = peaks if compound_idx is None else peaks[peaks["compound_idx"].isin(list(compound_idx))]
    available = set(pq.read_schema(path).names)
    # All the masses with peaks: for the noise parameters
    masses = [int(m) for m in peaks["mass"].unique() if str(int(m)) in available]
    frame = pd.read_parquet(path, columns=[str(m) for m in masses]) if masses else pd.DataFrame()
    noise = measurement_noise(peaks, frame) if masses else NoiseParameters()

    results = []
    selected_masses = [m for m in selected["mass"].unique() if str(int(m)) in frame]
    if selected_masses:
        t = frame.index.to_numpy(dtype=float)
        for mass, group in peaks[peaks["mass"].isin(selected_masses)].groupby("mass"):
            results.append(integrate_mass(group, t, frame[str(int(mass))].to_numpy(dtype=float), noise))
    peak_results = pd.concat(results) if results else pd.DataFrame(columns=PEAK_COLUMNS, dtype=float)
    peak_results = peak_results.reindex(selected.index)
    compounds = compound_areas(selected.assign(area=peak_results["area"], area_unc=peak_results["area_unc"]))
    return peak_results, compounds, noise


def compound_areas(peaks: pd.DataFrame) -> pd.DataFrame:
    """``area_sum`` and the masses of largest area (``q1_mass``, ``q1_area``, ...) of each compound.

    ``peaks`` has the columns compound_idx, mass, area and area_unc. Peaks on
    the same mass are summed, masses without area are left out of the ranking.
    The uncertainties of summed areas are added in quadrature (independent).
    """
    index = pd.Index(peaks["compound_idx"].unique(), name="compound_idx")
    compounds = pd.DataFrame(index=index, columns=COMPOUND_COLUMNS, dtype=float)
    if peaks.empty:
        return compounds
    peaks = peaks.assign(area_var=peaks["area_unc"] ** 2)
    by_compound = peaks.groupby("compound_idx")
    compounds["area_sum"] = by_compound["area"].sum(min_count=1)
    compounds["area_sum_unc"] = np.sqrt(by_compound["area_var"].sum(min_count=1))
    by_mass = peaks.groupby(["compound_idx", "mass"])[["area", "area_var"]].sum(min_count=1)
    by_mass = by_mass.dropna(subset=["area"]).reset_index()
    by_mass = by_mass.sort_values(["compound_idx", "area"], ascending=[True, False])
    by_mass["rank"] = by_mass.groupby("compound_idx").cumcount() + 1
    for i in range(1, db.N_QUALIFIERS + 1):
        q = by_mass[by_mass["rank"] == i].set_index("compound_idx")
        compounds[f"q{i}_mass"] = q["mass"].reindex(compounds.index)
        compounds[f"q{i}_area"] = q["area"].reindex(compounds.index)
        compounds[f"q{i}_area_unc"] = np.sqrt(q["area_var"].reindex(compounds.index))
    return compounds


# --- database ---------------------------------------------------------------------------


@dataclass
class IntegrationSummary:
    integrated: list[str] = field(default_factory=list)
    skipped_integrated: list[str] = field(default_factory=list)  # already integrated (no force)
    no_chromatogram: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False


def _rows(df: pd.DataFrame) -> list[tuple]:
    """Rows (index first) with Python values and None for NaN, for sqlite."""
    return list(df.astype(object).where(df.notna(), None).itertuples(index=True, name=None))


def store_integration(
    con: sqlite3.Connection,
    measurement_id: int,
    peak_results: pd.DataFrame,
    compound_results: pd.DataFrame,
    noise: NoiseParameters,
    complete: bool,
) -> None:
    """Write the results of a measurement; ``complete``: all its peaks were integrated."""
    peak_set = ", ".join(f"{c} = ?" for c in PEAK_COLUMNS)
    compound_set = ", ".join(f"{c} = ?" for c in COMPOUND_COLUMNS)
    measurement = noise.to_columns()
    if complete:
        measurement |= {
            "integration_version": INTEGRATION_VERSION,
            "integrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    with con:
        con.executemany(
            f"UPDATE peak SET {peak_set} WHERE measurement_id = ? AND peak_idx = ?",
            [(*row[1:], measurement_id, row[0]) for row in _rows(peak_results[PEAK_COLUMNS])],
        )
        con.executemany(
            f"UPDATE compound SET {compound_set} WHERE measurement_id = ? AND compound_idx = ?",
            [(*row[1:], measurement_id, row[0]) for row in _rows(compound_results[COMPOUND_COLUMNS])],
        )
        con.execute(
            f"UPDATE measurement SET {', '.join(f'{c} = ?' for c in measurement)} WHERE measurement_id = ?",
            (*measurement.values(), measurement_id),
        )


def integrate_measurements(
    con: sqlite3.Connection,
    data_dir: Path,
    measurement_ids: Iterable[int] | None = None,
    compound_name: str | None = None,
    rt_range: tuple[float, float] | None = None,
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> IntegrationSummary:
    """Integrate the peaks of measurements in the database (``con`` opened with ``create=True``).

    ``measurement_ids``: the measurements to integrate (all if None).
    ``compound_name`` / ``rt_range`` [s]: only the compounds with that name /
    with a mean rt in that range. The measurement is then not marked as
    integrated, so a later complete integration still does it.
    ``force``: also redo the measurements integrated with the current version.
    ``progress(done, total, name)`` is called before each measurement, and the
    work stops when ``cancelled()`` returns True (each measurement is committed).
    """
    data_dir = Path(data_dir)
    summary = IntegrationSummary()
    measurements = pd.read_sql(
        "SELECT measurement_id, name, chromatogram_file, integration_version FROM measurement ORDER BY acq_datetime, name",
        con,
        index_col="measurement_id",
    )
    if measurement_ids is not None:
        measurements = measurements[measurements.index.isin(list(measurement_ids))]

    conditions, params = [], []
    if compound_name is not None:
        conditions.append("compound_name = ?")
        params.append(compound_name)
    if rt_range is not None:
        conditions.append("rt_mean_s BETWEEN ? AND ?")
        params += [float(min(rt_range)), float(max(rt_range))]
    where = " AND ".join(conditions)
    complete = not conditions
    if not complete:
        with_compounds = {r[0] for r in con.execute(f"SELECT DISTINCT measurement_id FROM compound WHERE {where}", params)}
        measurements = measurements[measurements.index.isin(with_compounds)]

    if not force:
        done = measurements["integration_version"] >= INTEGRATION_VERSION
        summary.skipped_integrated = measurements.loc[done, "name"].tolist()
        measurements = measurements[~done]

    for n, (mid, m) in enumerate(measurements.iterrows()):
        if cancelled is not None and cancelled():
            summary.cancelled = True
            break
        if progress is not None:
            progress(n, len(measurements), m["name"])
        path = data_dir / m["chromatogram_file"] if isinstance(m["chromatogram_file"], str) else None
        if path is None or not path.is_file():
            summary.no_chromatogram.append(m["name"])
            continue
        try:
            peaks = pd.read_sql(
                f"SELECT peak_idx, {', '.join(PEAK_INPUT_COLUMNS)} FROM peak WHERE measurement_id = ?",
                con, params=(int(mid),), index_col="peak_idx",
            )
            compound_idx = None
            if not complete:
                compound_idx = [r[0] for r in con.execute(
                    f"SELECT compound_idx FROM compound WHERE measurement_id = ? AND {where}", (int(mid), *params)
                )]
            peak_results, compound_results, noise = integrate_chromatograms(peaks, path, compound_idx)
            store_integration(con, int(mid), peak_results, compound_results, noise, complete)
        except Exception as e:
            logger.exception(f"{m['name']}: integration failed")
            summary.failed.append((m["name"], repr(e)))
            continue
        logger.info(f"{m['name']}: {len(peak_results)} peaks integrated")
        summary.integrated.append(m["name"])
    if progress is not None and not summary.cancelled:
        progress(len(measurements), len(measurements), "")
    return summary
