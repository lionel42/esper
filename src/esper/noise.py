"""Noise of the unit mass chromatograms, for the integration and its uncertainties.

The variance of one scan grows linearly with the signal (counting noise):
``floor + gain * signal``, with a floor per chromatogram and a gain common to
the masses of a measurement. Neighbouring scans are correlated (by ~0.7-0.85
at one scan on the TOF data), so the noise of a sum of n scans is ``factor *
n * variance`` rather than ``n * variance`` (factor ~3).

The parameters common to a measurement (``NoiseParameters``) are estimated
once from the quiet parts of some of its chromatograms and stored with the
measurement; the noise of each chromatogram (``NoiseModel``) adds its floor.
When there is too little data to estimate the noise, it is unknown (NaN), and
so are the uncertainties computed with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

# Moving mean used for the baselines and to find the quiet parts (11 scans, ~1.8 s)
SMOOTH_SCANS = 11

# The noise is taken as uncorrelated between scans this far apart, correlated up to
# NOISE_CORRELATED_LAGS scans
NOISE_MAX_LAG = 8
NOISE_CORRELATED_LAGS = 4
# Signal levels from which the counting noise dominates the floor [counts]
GAIN_MIN_LEVEL = 10.0
# Least number of quiet scans for a chromatogram to be used; if no chromatogram has
# that many, the ones with at least NOISE_MIN_QUIET_FALLBACK (below: unknown noise)
NOISE_MIN_QUIET = 500
NOISE_MIN_QUIET_FALLBACK = 50
# Most chromatograms used for the parameters of a measurement (medians over them)
NOISE_MAX_MASSES = 60
# Fraction of the largest second differences (peaks) left out of the gain estimate
NOISE_TRIM = 0.02
# Mean square of a gaussian without its NOISE_TRIM largest absolute values, relative to all
_Z = NormalDist().inv_cdf(1 - NOISE_TRIM / 2)
_TRIM_CORRECTION = 1 - _Z * math.exp(-(_Z**2) / 2) * math.sqrt(2 / math.pi) / (1 - NOISE_TRIM)

# Measurement columns of the NoiseParameters fields
NOISE_COLUMNS = {"gain": "noise_gain", "factor": "noise_factor", "d2_ratio": "noise_d2_ratio"}


@dataclass
class NoiseParameters:
    """Noise parameters common to the chromatograms of a measurement.

    ``gain``: variance per scan per count of signal (counting noise).
    ``factor``: variance of a sum of scans over the sum of their variances,
    ``1 + 2 * sum of the correlations``.
    ``d2_ratio``: variance of second differences over the variance of one scan
    (6 for uncorrelated scans), to get the variance from second differences.
    """

    gain: float = 0.0
    factor: float = 1.0
    d2_ratio: float = 6.0

    @classmethod
    def unknown(cls) -> NoiseParameters:
        return cls(gain=np.nan, factor=np.nan, d2_ratio=np.nan)

    @property
    def is_known(self) -> bool:
        return bool(np.isfinite([self.gain, self.factor, self.d2_ratio]).all())

    def to_columns(self) -> dict:
        """As measurement columns."""
        return {column: getattr(self, name) for name, column in NOISE_COLUMNS.items()}

    @classmethod
    def from_columns(cls, row) -> NoiseParameters | None:
        """From measurement columns (a mapping), None if not stored."""
        values = {name: row.get(column) for name, column in NOISE_COLUMNS.items()}
        if any(v is None or not np.isfinite(v) for v in values.values()):
            return None
        return cls(**{k: float(v) for k, v in values.items()})


@dataclass
class NoiseModel:
    """Noise of one chromatogram: variance per scan ``floor + gain * level``.

    ``factor``: see NoiseParameters; the variance of a sum of n scans is
    ``factor * n * variance`` (for n larger than the correlation length).
    """

    floor: float = 0.0
    gain: float = 0.0
    factor: float = 1.0

    def variance(self, level) -> np.ndarray:
        """Variance of one scan at that signal level [counts^2]."""
        return self.floor + self.gain * np.clip(level, 0, None)

    def sum_variance(self, level) -> np.ndarray:
        """Variance per scan in a sum of scans at that level (correlations included)."""
        return self.factor * self.variance(level)

    def mean_sd(self, level, n: int = SMOOTH_SCANS) -> np.ndarray:
        """Standard deviation of the mean of n scans at that level."""
        return np.sqrt(self.sum_variance(level) / n)


def smooth(y: np.ndarray) -> np.ndarray:
    """Moving mean over ``SMOOTH_SCANS`` scans (of each column of a 2D array)."""
    frame = pd.DataFrame(y) if np.ndim(y) == 2 else pd.Series(y)
    return frame.rolling(SMOOTH_SCANS, center=True, min_periods=1).mean().to_numpy()


def _second_differences(y: np.ndarray) -> np.ndarray:
    return y[:-2] - 2 * y[1:-1] + y[2:]


def _mean_square_columns(x: np.ndarray) -> np.ndarray:
    """Mean square of each column of x, NaN ignored."""
    with np.errstate(all="ignore"):
        return np.nanmean(x**2, axis=0)


def _trimmed_variance(x: np.ndarray) -> float:
    """Mean square of x without its largest values, corrected to a gaussian's variance."""
    x = np.abs(x[np.isfinite(x)])
    if len(x) < 10:
        return np.nan
    x = x[x <= np.quantile(x, 1 - NOISE_TRIM)]
    return float(np.mean(x**2) / _TRIM_CORRECTION)


def _quiet(smoothed: np.ndarray) -> np.ndarray:
    """Scans away from the peaks (per column of a 2D array).

    Peaks: smoothed signal above its median by more than 5 robust standard
    deviations, widened by the smoothing window and the noise lags on each side.
    Selecting on the noise itself (e.g. below the median) would bias the estimates.
    """
    one_column = smoothed.ndim == 1
    smoothed = smoothed[:, None] if one_column else smoothed
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(smoothed, axis=0)
        spread = 1.4826 * np.nanmedian(np.abs(smoothed - median), axis=0)
        peak = ~(smoothed <= median + 5 * spread)
    if peak.any():
        widen = SMOOTH_SCANS // 2 + NOISE_MAX_LAG
        peak = pd.DataFrame(peak.astype(float)).rolling(2 * widen + 1, center=True, min_periods=1).max().to_numpy() > 0
    quiet = ~peak
    return quiet[:, 0] if one_column else quiet


def noise_parameters(chromatograms: np.ndarray) -> NoiseParameters:
    """Estimate the noise parameters from chromatograms (scans x masses, or one chromatogram).

    In the quiet parts of each chromatogram (see _quiet), the differences
    between scans ``NOISE_MAX_LAG`` apart give the variance of one scan
    (uncorrelated), those at shorter lags the correlations, hence the factor.
    The gain comes from the variance of second differences (little affected
    by the shape of the peaks) in bins of signal level, above
    ``GAIN_MIN_LEVEL``. Each estimate is the median over the chromatograms.
    """
    y = np.asarray(chromatograms, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if len(y) < NOISE_MIN_QUIET_FALLBACK + NOISE_MAX_LAG or y.shape[1] == 0:
        return NoiseParameters.unknown()
    smoothed = smooth(y)
    quiet = _quiet(smoothed)

    lag_var = np.array([
        _mean_square_columns(np.where(quiet[k:] & quiet[:-k], y[k:] - y[:-k], np.nan)) / 2
        for k in range(1, NOISE_MAX_LAG + 1)
    ])
    quiet3 = quiet[:-2] & quiet[1:-1] & quiet[2:]
    d2_var = _mean_square_columns(np.where(quiet3, _second_differences(y), np.nan))
    variance = lag_var[-1]
    with np.errstate(all="ignore"):
        usable = np.isfinite(variance) & np.isfinite(d2_var)
        ok = usable & (quiet3.sum(axis=0) >= NOISE_MIN_QUIET)
        if not ok.any():
            ok = usable & (quiet3.sum(axis=0) >= NOISE_MIN_QUIET_FALLBACK)
        if not ok.any():
            return NoiseParameters.unknown()
        if not (variance[ok] > 0).any():
            # Quiet parts without any noise (synthetic data): known to be noiseless
            return NoiseParameters()
        ok &= variance > 0
        correlations = 1 - lag_var[:NOISE_CORRELATED_LAGS, ok] / variance[ok]
        factor = float(np.clip(np.median(1 + 2 * correlations.sum(axis=0)), 1, 1 + 2 * NOISE_CORRELATED_LAGS))
        d2_ratio = float(np.median(d2_var[ok] / variance[ok]))

    # Gain: variance / level in bins of level of each chromatogram (peak tops left out)
    ratios = []
    d2 = _second_differences(y)
    level = smoothed[1:-1]
    with np.errstate(all="ignore"):
        # Only the chromatograms reaching GAIN_MIN_LEVEL can have bins above it
        reaching = np.nanquantile(level, 0.99, axis=0) >= GAIN_MIN_LEVEL
    for j in np.flatnonzero(ok & reaching):
        col_level, col_d2 = level[:, j], d2[:, j]
        finite = np.isfinite(col_level) & np.isfinite(col_d2)
        col_level, col_d2 = col_level[finite], col_d2[finite]
        keep = col_level <= np.quantile(col_level, 0.99)
        col_level, col_d2 = col_level[keep], col_d2[keep]
        for b in np.array_split(np.argsort(col_level, kind="stable"), 10):
            bin_level = np.median(col_level[b]) if len(b) >= 50 else np.nan
            if bin_level >= GAIN_MIN_LEVEL:
                ratios.append(_trimmed_variance(col_d2[b]) / d2_ratio / bin_level)
    ratios = np.array(ratios)
    ratios = ratios[np.isfinite(ratios)]
    gain = float(np.median(ratios)) if len(ratios) else 0.0
    return NoiseParameters(gain=gain, factor=factor, d2_ratio=d2_ratio)


def measurement_noise(peaks: pd.DataFrame, frame: pd.DataFrame) -> NoiseParameters:
    """Noise parameters of a measurement, from the chromatograms of the masses with peaks.

    ``frame``: chromatograms (columns: masses as str), with at least those
    masses. At most ``NOISE_MAX_MASSES``, evenly spread over the sorted
    masses: the same masses are always used, so the result does not depend on
    what is integrated (nor on whether the GUI or the integration computes it).
    """
    columns = [str(int(m)) for m in sorted(peaks["mass"].unique()) if str(int(m)) in frame]
    if len(columns) > NOISE_MAX_MASSES:
        columns = [columns[i] for i in np.linspace(0, len(columns) - 1, NOISE_MAX_MASSES).round().astype(int)]
    return noise_parameters(frame[columns].to_numpy(dtype=float))


def noise_model(
    y: np.ndarray, parameters: NoiseParameters | None = None, smoothed: np.ndarray | None = None
) -> NoiseModel:
    """Noise of one chromatogram: the measurement's parameters and its own floor.

    The floor (variance at zero signal) is what the gain does not explain of
    the variance in the quiet parts, or of the whole chromatogram (largest
    second differences, the peaks, left out) if it has too few quiet scans.
    ``parameters`` are estimated from y alone if not given (less reliable: the
    gain needs a range of levels). ``smoothed``: smooth(y), if already computed.
    The noise is unknown (NaN) if the parameters are, or if y is too short.
    """
    y = np.asarray(y, dtype=float)
    if parameters is None:
        parameters = noise_parameters(y)
    unknown = NoiseModel(floor=np.nan, gain=np.nan, factor=np.nan)
    if not parameters.is_known or len(y) < NOISE_MIN_QUIET_FALLBACK + 2:
        return unknown
    if smoothed is None:
        smoothed = smooth(y)
    quiet = _quiet(smoothed)
    quiet3 = quiet[:-2] & quiet[1:-1] & quiet[2:]
    d2 = _second_differences(y)
    if quiet3.sum() >= NOISE_MIN_QUIET:
        variance = float(np.mean(d2[quiet3] ** 2))
        level = float(np.nanmedian(smoothed[quiet]))
    else:
        variance = _trimmed_variance(d2)
        level = float(np.nanmedian(smoothed))
    if not np.isfinite(variance):
        return unknown
    floor = variance / parameters.d2_ratio - parameters.gain * max(level, 0.0)
    return NoiseModel(floor=max(floor, 0.0), gain=parameters.gain, factor=parameters.factor)
