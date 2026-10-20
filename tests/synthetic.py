"""Synthetic chromatograms whose areas and noise are known, for the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import exponnorm

from esper.noise import NoiseParameters, noise_parameters

DT = 0.1666  # s per scan, as the TOF data
T = np.arange(0, 600, DT)
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def gaussian(rt: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((T - rt) / sigma) ** 2)


def gaussian_area(sigma: float, height: float) -> float:
    return height * sigma * np.sqrt(2 * np.pi)


def peaks_of(*params, mass: int = 44) -> pd.DataFrame:
    """Peaks as alpinac gives them, for gaussians (rt, sigma, height); bases at the ends."""
    rows = []
    for k, (rt, sigma, _) in enumerate(params):
        hwhm = sigma * FWHM_PER_SIGMA / 2
        rows.append({
            "compound_idx": k, "mass": mass, "rt_s": rt,
            "start_s": rt - hwhm, "stop_s": rt + hwhm,
            "left_base_scan": 0, "right_base_scan": len(T) - 1,
        })
    return pd.DataFrame(rows, index=pd.Index(range(len(rows)), name="peak_idx"))


def emg(rt0: float, sigma: float, tau: float, area: float) -> np.ndarray:
    """Exponentially modified gaussian: a gaussian (sigma) with an exponential tail (tau), of known area."""
    return area * exponnorm.pdf(T, tau / sigma, loc=rt0, scale=sigma)


def measured_peak(y: np.ndarray, compound_idx: int = 0, lo: float = -np.inf, hi: float = np.inf) -> dict:
    """Peak as alpinac reports it, measured on the curve: apex in [lo, hi] and its half maximum crossings.

    ``y`` is the signal without background (the half maximum is relative to the peak base).
    """
    window = np.flatnonzero((T >= lo) & (T <= hi))
    i = window[np.argmax(y[window])]
    half = y[i] / 2
    left = right = i
    while left > 0 and y[left] > half:
        left -= 1
    while right < len(T) - 1 and y[right] > half:
        right += 1
    return {
        "compound_idx": compound_idx, "mass": 44, "rt_s": T[i], "start_s": T[left], "stop_s": T[right],
        "left_base_scan": 0, "right_base_scan": len(T) - 1,
    }


# --- noise ------------------------------------------------------------------------------

# Noise like the TOF chromatograms: counting noise (variance floor + gain * signal)
# smoothed over 3 scans, so neighbouring scans are correlated (0.67, 0.17)
NOISE_KERNEL = np.array([0.25, 0.5, 0.25])
NOISE_FLOOR, NOISE_GAIN = 4.0, 1.0
# Expected noise per scan, after the smoothing
SCAN_VARIANCE = float(np.sum(NOISE_KERNEL**2))  # 0.375
FACTOR = 1 + 2 * (4 / 6 + 1 / 6)  # 1 + 2 * sum of the correlations
BACKGROUNDS = [0.5, 5, 50, 500, 5000]  # other masses of the "measurement", for the gain


def noisy(signal: np.ndarray, rng) -> np.ndarray:
    white = rng.normal(0, 1, len(signal)) * np.sqrt(NOISE_FLOOR + NOISE_GAIN * np.clip(signal, 0, None))
    return signal + np.convolve(white, NOISE_KERNEL, mode="same")


def pooled_noise(y: np.ndarray, rng) -> NoiseParameters:
    """Noise parameters of a "measurement" made of y and background-only masses."""
    others = [noisy(np.full(len(T), level), rng) for level in BACKGROUNDS]
    return noise_parameters(np.column_stack([y] + others))
