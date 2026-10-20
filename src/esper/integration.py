"""Integration of the peaks found by alpinac in the unit mass chromatograms.

Window: each peak is integrated from ``N_HWHM_FRONT`` half widths at half
maximum before its apex to ``N_HWHM_TAIL`` after (each side with its own half
width, longer on the tail side because GC peaks tail). The window does not go
beyond the prominence bases of the peak, nor past a valley: an outer bound that
falls on another peak (not picked by alpinac) is moved back to the valley before it,
and the baseline there is not higher than at the other end (as the drop line
would do if the other peak had been picked).

Overlaps: peaks of the same mass whose windows overlap (directly, or through
a broader peak whose window contains them) form a group with one baseline,
from the start of the first window to the end of the last one, unless the
signal comes back to the baseline between two of them (``RESET_FRACTION``):
there the group is split into groups with their own baselines (baseline
reset). In a group, peaks are split at the lowest points between consecutive
apexes (drop line), except a small peak on the tail of a large one (height
above the extrapolated tail below ``SKIM_FRACTION`` of the height of the large
one) when that tail decays exponentially
(``SKIM_TAIL_TOLERANCE``): the tail is extrapolated under the small peak,
which is integrated above it (exponential skim), and the large peak keeps
the tail. A narrow
peak on a broad one keeps only its own window: the tail of the broad peak
after it is given back to the broad peak (which is then integrated in two
pieces; its stored bounds are the outer ones). Only the outer bounds of a
group are moved back to a valley.

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
When the noise is unknown (too little data), the uncertainty is NaN, and the
baseline is the straight line (no hull, no valleys at the outer bounds: both
compare the signal with the noise).

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
INTEGRATION_VERSION = 3

# Integration bounds, in half widths at half maximum of that side from the apex
# (4 ~ 4.7 sigma for a gaussian). Longer after the apex, for tailing peaks.
N_HWHM_FRONT = 4.0
N_HWHM_TAIL = 6.0
# The baseline follows the signal down where it is below the straight line by more
# than this times the noise of the smoothed signal (also for the valleys at the bounds)
HULL_NOISE_FACTOR = 4.0
# Overlapping peaks get separate baselines (baseline reset) when the valley between them
# comes back to the baseline: within this fraction of the smaller peak height (or the noise)
RESET_FRACTION = 0.1
# A small peak on the tail of a large one (its height above the tail extrapolated under it
# below this fraction of the height of the large one) is integrated above that tail
# (exponential skim) ...
SKIM_FRACTION = 0.2
# ... when the tail decays exponentially: the curvature of its log (a parabola fitted to it)
# deviates from a line by at most this much (a gaussian side does: a shoulder there is
# left to the drop line)
SKIM_TAIL_TOLERANCE = 0.06
# Part of the way from the apex of the large peak to the valley where its tail is fitted
SKIM_TAIL_FIT = (0.25, 0.75)
# The tail under a rider decays at most this many times slower than over the tail window of
# the large peak (beyond: not a tail)
SKIM_MAX_DECAY = 2.0

PEAK_COLUMNS = [name for name, _ in db.ADDED_COLUMNS["peak"]]
COMPOUND_COLUMNS = [name for name, _ in db.ADDED_COLUMNS["compound"]]
# Peak columns needed to integrate
PEAK_INPUT_COLUMNS = ["compound_idx", "mass", "rt_s", "start_s", "stop_s", "left_base_scan", "right_base_scan"]


# --- one chromatogram -------------------------------------------------------------------


def _scans(t: np.ndarray, lo: float, hi: float, inclusive: bool = True) -> np.ndarray:
    """Indices of the scans with lo <= t <= hi (lo < t < hi if not inclusive); t sorted."""
    side_lo, side_hi = ("left", "right") if inclusive else ("right", "left")
    return np.arange(np.searchsorted(t, lo, side_lo), np.searchsorted(t, hi, side_hi))


def _points(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Times and values from lo to hi: the scans between, and y interpolated at the bounds."""
    inner = _scans(t, lo, hi, inclusive=False)
    tt = np.concatenate([[lo], t[inner], [hi]])
    return tt, np.concatenate([[np.interp(lo, t, y)], y[inner], [np.interp(hi, t, y)]])


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


def _clip_bound(t, smoothed, noise: NoiseModel, apex: float, bound: float) -> tuple[float, bool]:
    """Move an outer bound of a group back to the lowest point between the apex and it,
    if the signal rises after that point (another peak, not picked by alpinac).

    Only when the bound is above that lowest point by more than
    ``HULL_NOISE_FACTOR`` times the noise of the smoothed signal (on a clean
    tail, the lowest point is the bound itself or in the noise).
    Returns the bound and whether it was moved.
    """
    idx = _scans(t, min(apex, bound), max(apex, bound))
    if len(idx) < 2 or np.isnan(smoothed[idx]).all():
        return bound, False
    i = idx[np.nanargmin(smoothed[idx])]
    level = np.interp(bound, t, smoothed)
    if level > smoothed[i] and level > smoothed[i] + HULL_NOISE_FACTOR * noise.mean_sd(smoothed[i]):
        return float(t[i]), True
    return bound, False


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


def _area_variance(t, smoothed, noise: NoiseModel, baseline: _Baseline, pieces) -> float:
    """Variance of the area of a peak above the baseline, integrated over its pieces.

    Noise of the signal (trapezoid weights), of the baseline levels (moving
    means) and of the position of the splits with the neighbours (signal above
    the baseline there times the uncertainty of the position). ``pieces``:
    (points tt, uncertainty [s] of the start, of the end) of each piece.
    """
    signal = 0.0
    sources, source_weights = None, None
    split = 0.0
    for tt, start_sd, end_sd in pieces:
        dt = np.diff(tt)
        weights = np.concatenate([[0.0], dt]) / 2 + np.concatenate([dt, [0.0]]) / 2
        signal += np.sum(weights**2 * noise.sum_variance(np.interp(tt, t, smoothed)))
        piece_sources, piece_weights = baseline.source_weights(tt[0], tt[-1], tt)
        # Same sources for all the pieces (the vertices of the baseline): weights add up
        source_weights = piece_weights if source_weights is None else source_weights + piece_weights
        sources = piece_sources
        for bound, sd in ((tt[0], start_sd), (tt[-1], end_sd)):
            if sd > 0:
                split += ((np.interp(bound, t, smoothed) - baseline(bound)) * sd) ** 2
    levels = np.sum(source_weights**2 * noise.mean_sd(np.interp(sources, t, smoothed)) ** 2)
    return float(signal + levels + split)


def _groups(start: np.ndarray, stop: np.ndarray) -> list[list[int]]:
    """Peaks (sorted by apex) whose windows overlap, directly or through a broader one."""
    groups: list[list[int]] = []
    envelope_stop = -np.inf
    for k in range(len(start)):
        if groups and start[k] < envelope_stop:
            groups[-1].append(k)
        else:
            groups.append([k])
        envelope_stop = max(envelope_stop, stop[k]) if len(groups[-1]) > 1 else stop[k]
    return groups


def _resets(t, smoothed, noise: NoiseModel, members, rt, start, stop) -> list[tuple[int, float]]:
    """Where a group of overlapping windows is split into groups with their own baselines.

    Between consecutive apexes, at the valley, when the valley comes back to the
    baseline (taken as the lower of the levels at the start of the first window
    and at the end of the second): within ``RESET_FRACTION`` of the height of
    the smaller peak above it, or within the noise. Otherwise the peaks overlap
    for real and share the baseline (drop line). Returns (position in members
    of the first peak after the reset, time of the valley).
    """
    resets = []
    for n, (a, b) in enumerate(zip(members[:-1], members[1:])):
        valley, _ = _valley(t, smoothed, rt[a], rt[b], noise)
        level = np.interp(valley, t, smoothed)
        base = min(np.interp(start[a], t, smoothed), np.interp(stop[b], t, smoothed))
        height = min(np.interp(rt[a], t, smoothed), np.interp(rt[b], t, smoothed)) - base
        tolerance = max(RESET_FRACTION * height, HULL_NOISE_FACTOR * noise.mean_sd(level))
        if level - base <= tolerance:
            resets.append((n + 1, valley))
    return resets


@dataclass
class _Rider:
    """A small peak on the tail of a large one, from the valley to its end, and the tail
    of the large one under it: exponential from the signal at the valley to the signal at
    the end (exponential skim), above the baseline."""

    parent: int
    valley: float
    valley_sd: float
    end: float
    level: float  # of the tail above the baseline at the valley
    rate: float  # decay of the tail [1/s]

    def under(self, tt: np.ndarray, baseline: _Baseline) -> np.ndarray:
        """What is under the rider: the baseline and the tail above it."""
        return baseline(tt) + self.level * np.exp(-self.rate * (tt - self.valley))


def _exponential_tail(
    t, smoothed, noise: NoiseModel, baseline: _Baseline, apex: float, valley: float
) -> tuple[float, float] | None:
    """Exponential fitted to the tail of a peak, between its apex and a valley after it
    (signal above the baseline, on ``SKIM_TAIL_FIT`` of the way): (level at the valley,
    decay rate [1/s]).

    None if it is not a tail there: its log bends down (a parabola fitted to it deviates
    from the fitted line by more than ``SKIM_TAIL_TOLERANCE``), as the side of a gaussian
    does; a tail decays exponentially, or slower and slower (log bending up). The fits are weighted by the signal over its noise, so that the noise
    of the log where the signal is low does not count as curvature.
    """
    idx = _scans(t, apex + SKIM_TAIL_FIT[0] * (valley - apex), apex + SKIM_TAIL_FIT[1] * (valley - apex))
    height = smoothed[idx] - baseline(t[idx])
    if len(idx) < 5 or not (height > 0).all():
        return None
    x, log_height = t[idx] - valley, np.log(height)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = height / noise.mean_sd(smoothed[idx])
    if not np.isfinite(weights).all():
        weights = None
    line = np.polyfit(x, log_height, 1, w=weights)
    parabola = np.polyfit(x, log_height, 2, w=weights)
    curvature = np.abs(np.polyval(parabola, x) - np.polyval(line, x)).max()
    # A gaussian side bends down (log concave); a tail bends up (slowing down) or not at all
    if not line[0] < 0 or (parabola[0] < 0 and curvature > SKIM_TAIL_TOLERANCE):
        return None
    return float(np.exp(line[1])), float(-line[0])


def _riders(t, smoothed, noise: NoiseModel, baseline: _Baseline, members, rt, start, stop) -> dict[int, _Rider]:
    """Small peaks on the tail of the previous, much larger peak of their group, when that
    tail decays exponentially (see ``SKIM_FRACTION``, ``_exponential_tail``).

    The tail under the small peak is the exponential through the signal at the valley and
    at the end of the small peak: it follows the slow part of a tail, which a fit next
    to the apex would miss (the fit only decides whether the side is a tail). The small
    peak must be on the tail: its window starts after the apex of the large one (not a
    broad peak with a narrow one on its front), and the tail must decay like a tail: its
    decay length is at most the tail window of the large peak. The large peak is not
    itself a rider.
    """
    riders: dict[int, _Rider] = {}
    for n, (a, b) in enumerate(zip(members[:-1], members[1:])):
        if a in riders or not start[b] > rt[a]:
            continue
        valley, valley_sd = _valley(t, smoothed, rt[a], rt[b], noise)
        if _exponential_tail(t, smoothed, noise, baseline, rt[a], valley) is None:
            continue
        # The rider ends at the end of its window, or before at the valley before the next peak
        end = stop[b]
        if n + 2 < len(members):
            end = min(end, _valley(t, smoothed, rt[b], rt[members[n + 2]], noise)[0])
        level, level_end = np.interp([valley, end], t, smoothed) - baseline(np.array([valley, end]))
        if not (end > valley and level > 0):
            continue
        # Exponential through the signal above the baseline at the valley and at the end (a
        # tiny level at the end when the signal is back on the baseline there)
        rate = math.log(level / max(level_end, 1e-3 * level)) / (end - valley)
        if not rate > 0 or 1 / rate > SKIM_MAX_DECAY * (stop[a] - rt[a]):
            continue
        rider = _Rider(a, valley, valley_sd, end, level, rate)
        level_a, level_b = np.interp([rt[a], rt[b]], t, smoothed)
        above_tail = level_b - rider.under(np.array([rt[b]]), baseline)[0]
        if above_tail < SKIM_FRACTION * (level_a - baseline(rt[a])):
            riders[b] = rider
    return riders


def _pieces(members, rt, start, stop, cuts, cut_sds, g_start, g_stop) -> dict[int, list]:
    """The parts of the group [g_start, g_stop] integrated for each member: (start, end,
    uncertainty of the start, of the end).

    The group is split at the valleys between consecutive apexes (``cuts``,
    with their uncertainty). A member keeps its part inside its own window;
    what is outside (beyond a narrow peak on a broad one) goes to the broadest
    other member whose window covers it, or stays with the member if none.
    """
    bounds = [g_start, *cuts, g_stop]
    sds = [0.0, *cut_sds, 0.0]
    pieces: dict[int, list] = {k: [] for k in members}
    for n, k in enumerate(members):
        a, b, sd_a, sd_b = bounds[n], bounds[n + 1], sds[n], sds[n + 1]
        own_a, own_b = max(a, start[k]), min(b, stop[k])
        if not own_b > own_a:
            pieces[k].append((a, b, sd_a, sd_b))
            continue
        own = [own_a, own_b, sd_a if own_a == a else 0.0, sd_b if own_b == b else 0.0]
        for lo, hi, sd_lo, sd_hi, side in ((a, own_a, sd_a, 0.0, 0), (own_b, b, 0.0, sd_b, 1)):
            if not hi > lo:
                continue
            middle = (lo + hi) / 2
            hosts = [h for h in members if h != k and start[h] <= middle <= stop[h]]
            if hosts:
                host = max(hosts, key=lambda h: stop[h] - start[h])
                pieces[host].append((lo, hi, sd_lo, sd_hi))
            elif side == 0:
                own[0], own[2] = lo, sd_lo
            else:
                own[1], own[3] = hi, sd_hi
        pieces[k].append(tuple(own))
    # Join the touching pieces of a member (no split between them)
    joined = {}
    for k, parts in pieces.items():
        joined[k] = []
        for part in sorted(parts):
            if joined[k] and joined[k][-1][1] == part[0]:
                lo, _, sd_lo, _ = joined[k][-1]
                joined[k][-1] = (lo, part[1], sd_lo, part[3])
            else:
                joined[k].append(part)
    return joined


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

    values = np.full((len(peaks), len(PEAK_COLUMNS)), np.nan)
    groups = []  # (members, start, stop, start moved back to a valley, stop moved back)
    for overlapping in _groups(start, stop):
        # Outer bounds, back to a valley if another (unpicked) peak starts there
        g_start, valley_start = _clip_bound(t, smoothed, model, rt[overlapping[0]], start[overlapping].min())
        g_stop, valley_stop = _clip_bound(t, smoothed, model, rt[overlapping[-1]], stop[overlapping].max())
        # Separate baselines where the signal comes back to the baseline between two peaks
        bounds = [(0, g_start, valley_start)] + [(n, v, False) for n, v in _resets(t, smoothed, model, overlapping, rt, start, stop)]
        ends = [(v, False) for _, v, _ in bounds[1:]] + [(g_stop, valley_stop)]
        for (n, lo, lo_valley), (hi, hi_valley), n_next in zip(bounds, ends, [b[0] for b in bounds[1:]] + [len(overlapping)]):
            groups.append((overlapping[n:n_next], lo, hi, lo_valley, hi_valley))

    for members, g_start, g_stop, valley_start, valley_stop in groups:
        if not g_stop > g_start:
            continue
        flat = None
        if g_stop >= t[-1] and g_start > t[0]:
            flat = "stop"
        elif g_start <= t[0] and g_stop < t[-1]:
            flat = "start"
        baseline = _baseline(t, smoothed, model, g_start, g_stop, flat, valley_start, valley_stop)
        if baselines is not None:
            baselines.append((baseline.x, baseline.y))
        # Small peaks on the tail of a large one: exponential skim, below
        riders = _riders(t, smoothed, model, baseline, members, rt, start, stop)
        members = [k for k in members if k not in riders]
        # Split at the valleys between consecutive apexes
        cuts, cut_sds = [], []
        for a, b in zip(members[:-1], members[1:]):
            cut, sd = _valley(t, smoothed, rt[a], rt[b], model)
            cuts.append(min(max(cut, g_start), g_stop))
            cut_sds.append(sd)

        # Riders: above the tail of the large peak, extrapolated under them
        spans = {k: (r.valley, min(r.end, g_stop)) for k, r in riders.items() if min(r.end, g_stop) > r.valley}

        def above_tail(lo, hi):
            """Signal of the riders above the tail, between lo and hi: not in the other peaks."""
            total = 0.0
            for k, (r_lo, r_hi) in spans.items():
                a, b = max(lo, r_lo), min(hi, r_hi)
                if b > a:
                    tt, yy = _points(t, y, a, b)
                    total += float(np.trapezoid(yy - riders[k].under(tt, baseline), tt))
            return total

        for k, parts in _pieces(members, rt, start, stop, cuts, cut_sds, g_start, g_stop).items():
            area, pieces = 0.0, []
            for lo, hi, sd_lo, sd_hi in parts:
                if not hi > lo:
                    continue
                tt, yy = _points(t, y, lo, hi)
                area += float(np.trapezoid(yy - baseline(tt), tt)) - above_tail(lo, hi)
                pieces.append((tt, sd_lo, sd_hi))
            if not pieces:
                continue
            s, e = pieces[0][0][0], pieces[-1][0][-1]
            uncertainty = math.sqrt(_area_variance(t, smoothed, model, baseline, pieces))
            values[k] = (s, e, baseline(s), baseline(e), area, uncertainty)

        for k, (lo, hi) in spans.items():
            tt, yy = _points(t, y, lo, hi)
            under = riders[k].under(tt, baseline)
            area = float(np.trapezoid(yy - under, tt))
            uncertainty = math.sqrt(_area_variance(t, smoothed, model, baseline, [(tt, riders[k].valley_sd, 0.0)]))
            values[k] = (lo, hi, under[0], under[-1], area, uncertainty)
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
    The uncertainties of summed areas are added in quadrature (independent);
    a sum is unknown (NaN) if one of its areas has an unknown uncertainty.
    """
    index = pd.Index(peaks["compound_idx"].unique(), name="compound_idx")
    compounds = pd.DataFrame(index=index, columns=COMPOUND_COLUMNS, dtype=float)
    if peaks.empty:
        return compounds
    # Variance: 0 for peaks without area (not in the sums), NaN when the uncertainty is unknown
    area_var = peaks["area_unc"].pow(2).where(peaks["area"].notna(), 0.0)
    peaks = peaks.assign(area_var=area_var, unknown=area_var.isna())
    by_compound = peaks.groupby("compound_idx")
    compounds["area_sum"] = by_compound["area"].sum(min_count=1)
    compounds["area_sum_unc"] = np.sqrt(by_compound["area_var"].sum().where(~by_compound["unknown"].any()))
    by_mass = peaks.groupby(["compound_idx", "mass"]).agg(
        area=("area", "sum"), area_var=("area_var", "sum"), unknown=("unknown", "any"), n=("area", "count")
    )
    by_mass = by_mass[by_mass["n"] > 0]
    by_mass["area_var"] = by_mass["area_var"].where(~by_mass["unknown"])
    by_mass = by_mass.reset_index()
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
