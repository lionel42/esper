"""Fit overrides: other fit parameters for the peaks of a target on one mass.

The default fit (:mod:`esper.fitting`) fits all the peaks of a run with the fit of the
data directory. An override, made in the GUI (right click on a compound, *Refit peaks*),
fits again a window of one chromatogram around a target with its own parameters: the
peak function and backgrounds, the start positions of the peaks (fitted, within a
tolerance, or fixed) and the number of peaks (at least / at most). Overrides are rows of
the ``fit_override`` table (not the esper.yaml: there can be one per target and mass),
assigned to a target and a mass and applied to the runs:

- acquired from ``valid_from`` (NULL: from the first run) up to ``valid_to`` included
  (NULL: no end, also the runs imported later),
- of the ``sample_types`` (NULL: all),

each time such a run is fitted (``fit_measurement``: ``esper process``, ``esper upload``,
the GUI). Their peaks replace those of the default fit in the window, and keep the
override they come from (``peak.fit_override_id``).

The window and the positions are relative to the *reference rt* of the target in each
run (``reference_rt``), so that they follow the retention time drift: the rt where the
target is expected (GCwerks peakid), else the rt of the compound identified as the
target, as extracted. A run without a reference rt for the target is fitted as usual.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import peaxart as px

from esper.windows import cuts_of, is_continuous, measured_windows

logger = logging.getLogger(__name__)

# A fixed position: the fitted apex within this of it [s] (the fit needs a range)
FIXED_TOLERANCE_S = 1e-3
# How far the fit moves the peaks from their start positions by default [s]: the
# positions are approximate starts
DEFAULT_TOLERANCE_S = 2.0
# Confidence level of peaxart's significance test of the peak heights that keeps them all
# (its Student t factor is 0)
KEEP_ALL_CONF_INT = 0.5
# Bounds of the width (sigma of the peak function) of new overrides [s]
DEFAULT_SIGMA_RANGE_S = (0.3, 10.0)
# Extra start positions for a minimum number of peaks: at most this many times the
# minimum tries
MAX_ADD_TRIES = 3


@dataclass(frozen=True)
class OverrideParams:
    """The fit parameters of an override (the others: of the data directory's fit)."""

    peak_shape: str = "emg"
    backgrounds: tuple[str, ...] = ("constant", "linear", "exponential")
    # The width (sigma) of the peaks between these [s], starting at the half width of
    # the highest peak of the window. None: as the default fit, ``sigma_bounds`` times
    # peaxart's expected sigma (its rt model, ``ExtractionConfig.rt_sigma_at``: 2.5 s
    # before 1000 s, 0.38 s at 1000 s growing to 0.6 s at 1250 s: too narrow for the
    # broad peaks of a GC-MS)
    sigma_range_s: tuple[float, float] | None = DEFAULT_SIGMA_RANGE_S
    sigma_bounds: tuple[float, float] = (0.2, 5.0)
    max_iterations: int = 100
    # Start positions of the peaks, relative to the reference rt [s]; empty: the peaks
    # of the extraction in the window
    positions: tuple[float, ...] = ()
    # The fitted peaks stay within this of their start [s]; 0: fixed; None: peaxart's
    # default (3 scans). It bounds the location of the peak function: the apex of a
    # tailing peak (EMG) can still move after it, by up to its tail offset
    position_tolerance: float | None = DEFAULT_TOLERANCE_S
    # At least this many peaks (0: no minimum): positions are added where the residual
    # is largest. With positions or a minimum, no peak is removed by peaxart's tests
    # (against a line, apex covered, height significant)
    min_peaks: int = 0
    # At most this many peaks (0: with positions, exactly them; else the starts and up
    # to 2 more where the data show a coelution, as the default fit)
    max_peaks: int = 0

    @classmethod
    def from_config(cls, config: px.ChromatogramFitConfig) -> OverrideParams:
        background = config.background
        backgrounds = (
            (str(background),)
            if isinstance(background, str)
            else tuple(str(b) for b in background)
        )
        return cls(
            peak_shape=str(config.peak_shape),
            backgrounds=backgrounds,
            sigma_bounds=tuple(config.sigma_bounds),
            max_iterations=config.max_iterations,
        )

    @property
    def keeps_all_peaks(self) -> bool:
        """No peak removed by the per-peak tests: positions given, or a minimum."""
        return bool(self.positions) or self.min_peaks > 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> OverrideParams:
        data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for key in ("backgrounds", "sigma_bounds", "positions"):
            if key in known:
                known[key] = tuple(known[key])
        if known.get("sigma_range_s") is not None:
            known["sigma_range_s"] = tuple(known["sigma_range_s"])
        elif "sigma_range_s" not in known:
            known["sigma_range_s"] = None  # saved before: relative bounds
        return cls(**known)

    def extraction_config(
        self,
        base: px.ChromatogramFitConfig,
        span: float,
        n_starts: int,
        significance: bool = True,
        start_sigma: float | None = None,
    ):
        """The ``peaxart.ExtractionConfig`` of the fit of a window of ``span`` s with
        ``n_starts`` start positions (``base``: the fit of the data directory).
        ``significance``: peaxart removes the peaks whose height is not significant
        (False: kept, its S/N then meaningless). ``start_sigma``: of the peaks, with
        ``sigma_range_s`` [s]."""
        from peaxart.config import ExtractionConfig

        tolerance = self.position_tolerance
        if tolerance is not None and tolerance <= 0:
            tolerance = FIXED_TOLERANCE_S
        extra = {}
        if self.max_peaks > 0:
            extra["rt_extra_peaks"] = max(0, self.max_peaks - n_starts)
        elif self.positions:
            extra["rt_extra_peaks"] = 0
        if self.keeps_all_peaks:
            extra.update(
                peak_linear_ratio=None,
                peak_min_coverage=None,
                ratio_to_linear_fit=np.inf,
            )
        if not significance:
            extra["conf_int"] = KEEP_ALL_CONF_INT
        sigma = {"rt_sigma_bounds": tuple(self.sigma_bounds)}
        if self.sigma_range_s is not None and start_sigma is not None:
            lo, hi = self.sigma_range_s
            # bounds: start x rt_sigma_bounds, the lower at least floor[0], the upper
            # at least floor[1] (peaxart.rtfit.fit_chromato)
            sigma = {
                "rt_sigma": start_sigma,
                "rt_sigma_bounds": (lo / start_sigma, hi / start_sigma),
                "rt_sigma_floor": (lo, lo),
            }
        return ExtractionConfig(
            convention=base.convention,
            legacy_numerics=False,
            rt_peak_shape=self.peak_shape,
            rt_background=self.backgrounds or ("constant",),
            rt_background_ratio=base.background_ratio,
            rt_background_noise=base.background_noise,
            rt_max_iterations=self.max_iterations,
            rt_baseline_max_factor=base.baseline_max_factor,
            rt_tailing_max=base.tailing_max,
            # one window: the whole range
            rt_window=span + 10.0,
            rt_position_bound=tolerance,
            **sigma,
            **extra,
        )


@dataclass(frozen=True)
class FitOverride:
    """Fit parameters of the peaks of a target on one mass, in a window around it."""

    target: str
    mass: float
    # Window fitted, relative to the reference rt of the target [s]
    rt_from: float
    rt_to: float
    params: OverrideParams = field(default_factory=OverrideParams)
    valid_from: str | None = None  # ISO, as measurement.acq_datetime; None: always
    valid_to: str | None = None  # included; None: no end (also future runs)
    sample_types: tuple[str, ...] | None = None  # None: all
    override_id: int | None = None
    created_at: str | None = None
    note: str | None = None

    def applies(self, acq_datetime, sample_type) -> bool:
        """Whether the override applies to a run acquired at ``acq_datetime`` (ISO or
        Timestamp; None: unknown, only without dates) of ``sample_type``."""
        if self.sample_types is not None and sample_type not in self.sample_types:
            return False
        if self.valid_from is None and self.valid_to is None:
            return True
        if acq_datetime is None or pd.isna(acq_datetime):
            return False
        time = pd.Timestamp(acq_datetime)
        if self.valid_from is not None and time < pd.Timestamp(self.valid_from):
            return False
        return self.valid_to is None or time <= pd.Timestamp(self.valid_to)

    def window(self, reference: float) -> tuple[float, float]:
        return reference + self.rt_from, reference + self.rt_to

    def starts(self, reference: float) -> np.ndarray:
        return reference + np.asarray(self.params.positions, dtype=float)


# --- storage ------------------------------------------------------------------------


def _row(override: FitOverride) -> dict:
    return {
        "target": override.target,
        "mass": float(override.mass),
        "rt_from_s": float(override.rt_from),
        "rt_to_s": float(override.rt_to),
        "valid_from": override.valid_from,
        "valid_to": override.valid_to,
        "sample_types": None
        if override.sample_types is None
        else json.dumps(list(override.sample_types)),
        "params": override.params.to_json(),
        "created_at": override.created_at
        or datetime.now().isoformat(timespec="seconds"),
        "note": override.note,
    }


def save_override(con: sqlite3.Connection, override: FitOverride) -> int:
    """Insert the override, or replace the one of its ``override_id``; returns its id.
    Commits."""
    row = _row(override)
    with con:
        if override.override_id is None:
            columns = ", ".join(row)
            cursor = con.execute(
                f"INSERT INTO fit_override ({columns}) VALUES"
                f" ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
            return int(cursor.lastrowid)
        assignments = ", ".join(f"{c} = ?" for c in row)
        con.execute(
            f"UPDATE fit_override SET {assignments} WHERE override_id = ?",
            (*row.values(), int(override.override_id)),
        )
        return int(override.override_id)


def delete_override(con: sqlite3.Connection, override_id: int) -> None:
    with con:
        con.execute("DELETE FROM fit_override WHERE override_id = ?", (int(override_id),))


def read_overrides(
    con: sqlite3.Connection, target: str | None = None
) -> list[FitOverride]:
    """The overrides (of a target), latest first (the one applied when several cover a
    run on the same mass)."""
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fit_override'"
    ).fetchone():
        return []
    where, params = ("WHERE target = ?", (target,)) if target is not None else ("", ())
    rows = con.execute(
        "SELECT override_id, target, mass, rt_from_s, rt_to_s, valid_from, valid_to,"
        f" sample_types, params, created_at, note FROM fit_override {where}"
        " ORDER BY override_id DESC",
        params,
    ).fetchall()
    return [
        FitOverride(
            target=target_,
            mass=mass,
            rt_from=rt_from,
            rt_to=rt_to,
            params=OverrideParams.from_json(params_),
            valid_from=valid_from,
            valid_to=valid_to,
            sample_types=None if types is None else tuple(json.loads(types)),
            override_id=override_id,
            created_at=created_at,
            note=note,
        )
        for (
            override_id,
            target_,
            mass,
            rt_from,
            rt_to,
            valid_from,
            valid_to,
            types,
            params_,
            created_at,
            note,
        ) in rows
    ]


def run_overrides(
    overrides: list[FitOverride], acq_datetime, sample_type
) -> list[FitOverride]:
    """The overrides applying to a run, one per (target, mass): the latest."""
    out, seen = [], set()
    for override in overrides:  # latest first
        key = (override.target, float(override.mass))
        if key not in seen and override.applies(acq_datetime, sample_type):
            seen.add(key)
            out.append(override)
    return out


def runs_covered(
    con: sqlite3.Connection, overrides: list[FitOverride]
) -> list[tuple[int, str]]:
    """(measurement_id, name) of the fitted runs (``FITTED_PEAK_SOURCES``) that one of
    the overrides applies to, by acquisition time."""
    from esper.fitting import FITTED_PEAK_SOURCES

    if not overrides:
        return []
    sources = ", ".join(f"'{s}'" for s in FITTED_PEAK_SOURCES)
    runs = con.execute(
        "SELECT measurement_id, name, acq_datetime, sample_type FROM measurement"
        f" WHERE peak_source IN ({sources}) ORDER BY acq_datetime, name"
    ).fetchall()
    return [
        (mid, name)
        for mid, name, time, sample_type in runs
        if any(o.applies(time, sample_type) for o in overrides)
    ]


# --- reference rt -------------------------------------------------------------------


def reference_rt(
    con: sqlite3.Connection, measurement_id: int, target: str, targets, time=None
) -> float | None:
    """The rt the window of an override of ``target`` is relative to, in a run: where
    the target is expected (``targets``: the ``TargetIndex``; GCwerks peakid, at
    ``time``), else the rt of the compound identified as the target (by hand or by the
    processing; as extracted, before the fit), None if neither."""
    if targets.has_rts(target):
        found = targets.rt_at(target, time)
        if found is not None and np.isfinite(found[0]):
            return float(found[0])
    has_kept = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'extraction_compound'"
    ).fetchone()
    kept = (
        "LEFT JOIN extraction_compound e USING (measurement_id, compound_idx)"
        if has_kept
        else ""
    )
    rt = "COALESCE(e.rt_mean_s, c.rt_mean_s)" if has_kept else "c.rt_mean_s"
    rows = con.execute(
        f"SELECT COALESCE(i.name, c.compound_name), {rt}, c.n_peaks"
        " FROM compound c"
        " LEFT JOIN compound_identification i USING (measurement_id, compound_idx)"
        f" {kept} WHERE c.measurement_id = ? AND (c.source IS NULL OR c.source != 'fit')"
        " AND (i.measurement_id IS NULL OR i.name IS NOT NULL)",
        (int(measurement_id),),
    ).fetchall()
    found = [
        (n_peaks or 0, rt_)
        for name, rt_, n_peaks in rows
        if isinstance(name, str)
        and targets.target_of(name) == target
        and rt_ is not None
        and np.isfinite(rt_)
    ]
    if not found:
        return None
    return float(max(found)[1])  # the compound with the most peaks


# --- fit ----------------------------------------------------------------------------


def _model(t: np.ndarray, fitted: pd.DataFrame, shape) -> np.ndarray:
    """Background and peaks of a ``peaxart.Fit`` result."""
    from peaxart.chromatogram import peak_background

    signal = peak_background(t, fitted)
    for p in fitted.itertuples():
        signal = signal + np.nan_to_num(
            shape.value(t, p.shape_height, p.shape_rt, p.rt_sigma, p.rt_tailing)
        )
    return signal


def _signal_to_noise(
    t: np.ndarray, y: np.ndarray, fitted: pd.DataFrame, shape
) -> np.ndarray:
    """S/N of the peaks of a fit as peaxart computes it (``rtfit.fit_chromato``, at its
    default confidence level): the amplitude of each peak over the limit of detection
    2 t u, u the weighted residual of the fit."""
    from peaxart.chromatogram import peak_background
    from peaxart.config import ExtractionConfig
    from scipy.stats import t as student_t

    if fitted.empty:
        return np.array([])
    model = _model(t, fitted, shape)
    peaks = model - peak_background(t, fitted)
    resid = y - model
    n = len(t)
    w = 1.0 / np.clip(peaks, 1, None) ** 2
    u = np.sqrt(np.sum(w * resid**2) / np.sum(w) * n / (n - 1))
    dt = float(np.median(np.diff(t)))
    points = float(np.min(fitted["rt_sigma"].to_numpy(dtype=float) * 3 / dt))
    t_value = student_t.ppf(ExtractionConfig().conf_int, max(points - 1, 1))
    lod = max(1.0, 2.0 * t_value * u)
    return fitted["shape_height"].to_numpy(dtype=float) / lod


def half_width_sigma(t: np.ndarray, y: np.ndarray) -> float:
    """sigma of a Gaussian of the width at half maximum of the highest peak of a window
    (above its 5 % quantile) [s]."""
    base = float(np.quantile(y, 0.05))
    top = int(np.argmax(y))
    half = base + (y[top] - base) / 2
    above = y >= half
    left = top
    while left > 0 and above[left - 1]:
        left -= 1
    right = top
    while right < len(y) - 1 and above[right + 1]:
        right += 1
    return float(t[right] - t[left]) / 2.3548


def _largest_residual(
    t: np.ndarray, y: np.ndarray, fitted: pd.DataFrame, shape, starts=()
) -> float | None:
    """rt of the largest (smoothed) residual of a fit (of the signal without one), at
    least 3 scans from the ``starts`` (a start there would be the same peak); None if
    there is no such point."""
    if fitted.empty:
        resid = y - np.quantile(y, 0.05)
    else:
        resid = y - _model(t, fitted, shape)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    sigma = float(fitted["rt_sigma"].median()) if not fitted.empty else 1.0
    width = min(len(t), max(3, int(round(sigma / dt))))
    smooth = np.convolve(resid, np.ones(width) / width, mode="same")
    starts = np.asarray(list(starts), dtype=float)
    if len(starts):
        near = np.abs(t[:, None] - starts[None, :]).min(axis=1) < 3 * dt
        smooth = np.where(near, -np.inf, smooth)
    if not np.isfinite(smooth).any():
        return None
    return float(t[int(np.argmax(smooth))])


# Columns of ``fit_window`` from what peaxart's Fit.integrate drops (``_recording``)
DIAGNOSTIC_COLUMNS = [
    "background",  # the background kept for the window (several tried: the best)
    "n_points",  # points of the window fitted
    "linear_ratio",  # residual std with the peak / with a line (small: a real peak)
    "coverage",  # fraction of the scans within 1 sigma of the apex measured
    "iterations",  # of the least-squares fit kept (analytic Jacobian: 1 evaluation each)
    "aborted",  # stopped by the iteration limit
    "converged",  # the least-squares fit reports success
]


@contextmanager
def _recording():
    """What peaxart's ``Fit.integrate`` computes and drops, recorded while it runs: the
    fits of each window (``rtfit.fit_series``: their background, points and per-peak
    tests) and the least-squares fits (``leastsq.fit``: iterations, aborted). Wraps
    those functions of peaxart for the duration; nothing recorded (not an error) if
    they are not there (another peaxart)."""
    import peaxart.chromatogram as chromatogram
    import peaxart.rtfit as rtfit

    record = {"series": [], "chromato": [], "n_least_squares": 0, "iterations": 0, "aborted": 0}
    names = ("fit", "fit_chromato")
    if not all(hasattr(rtfit, n) for n in names) or not hasattr(chromatogram, "fit_series"):
        yield record
        return
    original_fit, original_chromato = rtfit.fit, rtfit.fit_chromato
    original_series = chromatogram.fit_series
    calls: list = []

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        calls.append(result)
        record["n_least_squares"] += 1
        record["iterations"] += int(getattr(result, "nfev", 0))
        record["aborted"] += int(bool(getattr(result, "aborted", False)))
        return result

    def fit_chromato(*args, **kwargs):
        start = len(calls)
        peaks = original_chromato(*args, **kwargs)
        if peaks is not None and len(calls) > start:
            record["chromato"].append((peaks, calls[-1]))  # its last least squares
        return peaks

    def fit_series(*args, **kwargs):
        out = original_series(*args, **kwargs)
        record["series"] = out
        return out

    rtfit.fit, rtfit.fit_chromato, chromatogram.fit_series = fit, fit_chromato, fit_series
    try:
        yield record
    finally:
        rtfit.fit, rtfit.fit_chromato = original_fit, original_chromato
        chromatogram.fit_series = original_series


def _diagnostics(fitted: pd.DataFrame, record: dict) -> pd.DataFrame:
    """``fitted`` (``Fit.integrate``) with the ``DIAGNOSTIC_COLUMNS`` of its peaks (NaN
    where not recorded), and the totals of the least-squares fits in ``attrs``."""
    out = fitted.copy()
    for column in DIAGNOSTIC_COLUMNS:
        out[column] = None if column == "background" else np.nan
    results = {id(peaks): result for peaks, result in record["chromato"]}
    for f, k in record["series"]:
        rows = np.flatnonzero(np.isclose(out["rt"].to_numpy(dtype=float), f.apex_rt[k]))
        if not len(rows):
            continue
        background = getattr(f, "background", None)
        result = results.get(id(f))
        values = {
            "background": "constant" if background is None else str(background.name),
            "n_points": f.n_points,
            "linear_ratio": float(f.linear_ratio[k]),
            "coverage": float(f.coverage[k]),
        }
        if result is not None:
            values.update(
                iterations=int(result.nfev),
                aborted=float(bool(result.aborted)),
                converged=float(bool(result.success)),
            )
        for column, value in values.items():
            out.loc[out.index[rows[0]], column] = value
    out.attrs.update(
        n_least_squares=record["n_least_squares"],
        iterations_total=record["iterations"],
        n_aborted=record["aborted"],
    )
    return out


def fit_window(
    t: np.ndarray,
    y: np.ndarray,
    starts,
    params: OverrideParams,
    base: px.ChromatogramFitConfig,
) -> pd.DataFrame:
    """Fit the peaks of a window of a chromatogram with the parameters of an override.

    ``t``, ``y``: the window (finite values); ``starts``: start positions [s] (empty:
    one at the maximum). Returns the ``peaxart.Fit`` peaks, with ``shape``,
    ``signal_height`` and ``signal_prominence`` (as ``peaxart.fit_chromatogram``), and
    the ``DIAGNOSTIC_COLUMNS`` of the fit kept; in ``attrs``: ``starts`` (the start
    positions, with those added for ``min_peaks``) and the totals of its least-squares
    fits (``n_least_squares``, with the trials of other backgrounds and extra peaks,
    ``iterations_total``, ``n_aborted``).
    """
    from peaxart.chromatogram import Fit
    from peaxart.deconvolution import fitted_signal_peaks
    from peaxart.shapes import peak_shape

    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    columns = [
        *Fit.columns, "shape", "signal_height", "signal_prominence", *DIAGNOSTIC_COLUMNS
    ]
    if len(t) < 5:
        return pd.DataFrame(columns=columns)
    shape = peak_shape(params.peak_shape, base.convention)
    starts = sorted(float(s) for s in starts if t[0] <= s <= t[-1])
    if not starts:
        starts = [float(t[int(np.argmax(y))])]
    if params.max_peaks > 0 and len(starts) > params.max_peaks:
        # the highest ones
        heights = np.interp(starts, t, y)
        keep = np.sort(np.argsort(heights)[::-1][: params.max_peaks])
        starts = [starts[i] for i in keep]

    start_sigma = None
    if params.sigma_range_s is not None:
        start_sigma = float(np.clip(half_width_sigma(t, y), *params.sigma_range_s))

    def fit_with(starts, significance: bool) -> pd.DataFrame:
        config = params.extraction_config(
            base, t[-1] - t[0], len(starts), significance, start_sigma
        )
        sigma = None if start_sigma is None else np.full(len(starts), start_sigma)
        with _recording() as record:
            fitted = Fit(config).integrate(t, y, rt=np.asarray(starts), sigma=sigma)
        return _diagnostics(fitted, record)

    def fit(starts) -> pd.DataFrame:
        """The fit; keeping all the peaks (positions, minimum): those whose height
        is not significant too, their S/N from the noise of the fit without them."""
        fitted = fit_with(starts, True)
        if not params.keeps_all_peaks or len(fitted) >= len(starts):
            return fitted
        kept = fit_with(starts, False)
        out = kept.assign(sn=_signal_to_noise(t, y, kept, shape), lod=np.nan)
        out.attrs = dict(kept.attrs)
        return out

    fitted = fit(starts)
    tries = 0
    while len(fitted) < params.min_peaks and tries < MAX_ADD_TRIES * params.min_peaks:
        tries += 1
        added = _largest_residual(t, y, fitted, shape, starts)
        if added is None:
            break
        starts = sorted([*starts, added])
        fitted = fit(starts)
    attrs = {**fitted.attrs, "starts": list(starts), "start_sigma": start_sigma}
    fitted = fitted.assign(shape=str(params.peak_shape))
    if fitted.empty:
        out = fitted.reindex(columns=columns)
    else:
        height, prominence = fitted_signal_peaks(t, fitted, shape)
        out = fitted.assign(signal_height=height, signal_prominence=prominence)
    out.attrs = attrs
    return out


def chromatogram_window(
    frame: pd.DataFrame, mass, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """(t, y) of the chromatogram of ``mass`` in [lo, hi], where it was measured (the
    longest measured part)."""
    column = f"{mass:g}"
    if column not in frame:
        return np.array([]), np.array([])
    t = frame.index.to_numpy(dtype=float)
    y = frame[column].to_numpy(dtype=float)
    inside = (t >= lo) & (t <= hi)
    # outside the range: as not measured
    windows = measured_windows(
        np.where(inside, y, np.nan), is_continuous(frame), cuts=cuts_of(frame, column)
    )
    if not windows:
        return np.array([]), np.array([])
    index = max(windows, key=len)
    return t[index], y[index]


def override_fit(
    frame: pd.DataFrame,
    peaks: pd.DataFrame,
    override: FitOverride,
    reference: float,
    base: px.ChromatogramFitConfig,
    diagnostics: bool = False,
) -> pd.DataFrame:
    """The fit of an override in a run, as ``esper.fitting.FITTED_COLUMNS`` (each
    fitted peak given to the nearest extraction peak of the mass, ``fit_override_id``
    the override's); ``peaks``: the extraction peaks of the run (indexed by peak_idx).
    Empty if the mass has no chromatogram or no peak. ``diagnostics``: with the
    ``DIAGNOSTIC_COLUMNS`` and the ``attrs`` of ``fit_window`` (not stored)."""
    from esper.fitting import FIT_COLUMNS, FITTED_COLUMNS

    lo, hi = override.window(reference)
    on_mass = peaks[peaks["mass"] == override.mass]
    t, y = chromatogram_window(frame, override.mass, lo, hi)
    if on_mass.empty or len(t) == 0:
        return pd.DataFrame(columns=FITTED_COLUMNS)
    starts = (
        override.starts(reference)
        if override.params.positions
        else on_mass.loc[on_mass["rt_s"].between(lo, hi), "rt_s"].to_numpy()
    )
    fitted = fit_window(t, y, starts, override.params, base)
    if fitted.empty:
        return pd.DataFrame(columns=FITTED_COLUMNS)
    rt = on_mass["rt_s"].to_numpy(dtype=float)
    nearest = np.abs(fitted["rt"].to_numpy(dtype=float)[:, None] - rt[None, :]).argmin(
        axis=1
    )
    out = fitted.assign(override_id=override.override_id).rename(
        columns={v: k for k, v in FIT_COLUMNS.items()}
    )
    out["peak_idx"] = on_mass.index.to_numpy()[nearest]
    if diagnostics:
        result = out[[*FITTED_COLUMNS, *DIAGNOSTIC_COLUMNS]]
        result.attrs = dict(fitted.attrs)
        return result
    return out[FITTED_COLUMNS]


def apply_overrides(
    fitted: pd.DataFrame,
    frame: pd.DataFrame,
    peaks: pd.DataFrame,
    overrides: list[tuple[FitOverride, float]],
    base: px.ChromatogramFitConfig,
) -> pd.DataFrame:
    """The default fit of a run (``esper.fitting.fit_peaks``) with the peaks in the
    window of each override replaced by its fit (``overrides``: with their reference
    rt in the run)."""
    parts = [fitted]
    for override, reference in overrides:
        lo, hi = override.window(reference)
        try:
            own = override_fit(frame, peaks, override, reference, base)
        except Exception:
            logger.exception(
                f"override {override.override_id} ({override.target}, m/z"
                f" {override.mass:g}): fit failed, the default fit kept"
            )
            continue
        masses = peaks["mass"].reindex(parts[0]["peak_idx"]).to_numpy()
        replaced = (masses == override.mass) & parts[0]["fit_rt_s"].between(
            lo, hi
        ).to_numpy()
        parts[0] = parts[0][~replaced]
        parts.append(own)
    parts = [p for p in parts if len(p)]
    if len(parts) <= 1:
        return parts[0] if parts else fitted.iloc[:0]
    out = pd.concat(parts, ignore_index=True)
    out.attrs = dict(fitted.attrs)
    return out



def refit_runs(
    data_dir,
    runs: list[tuple[int, str]],
    progress=None,
    workers: int | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Fit the runs again (``esper.fitting.fit_measurement``: with the overrides that
    apply to them), with the fit and target search of the data directory.

    ``progress(i, name) -> bool``: called before each run, False stops. Returns the
    names of the runs fitted and the (name, error) of those that failed.
    """
    from esper import db
    from esper.fitting import fit_measurement, fit_pool
    from esper.processing.config import read_processing_config
    from esper.processing.identification import TargetSearch
    from esper.target_index import TargetIndex

    processing = read_processing_config(data_dir)
    done, failed = [], []
    con = db.connect(data_dir, create=True)
    try:
        target_search = TargetSearch(TargetIndex.from_db(con), processing.target)
        with fit_pool(workers) as pool:
            for i, (mid, name) in enumerate(runs):
                if progress is not None and progress(i, name) is False:
                    break
                try:
                    fit_measurement(
                        con,
                        mid,
                        data_dir,
                        workers=1,
                        pool=pool,
                        config=processing.fit,
                        target_search=target_search,
                    )
                except Exception as e:
                    logger.exception(f"{name}: fit failed")
                    failed.append((name, repr(e)))
                    continue
                done.append(name)
    finally:
        con.close()
    return done, failed
