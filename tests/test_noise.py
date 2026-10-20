"""Noise estimates and uncertainties of the areas, on synthetic correlated counting noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
from synthetic import (
    FACTOR, NOISE_FLOOR, NOISE_GAIN, SCAN_VARIANCE, T, emg, gaussian, measured_peak, noisy, peaks_of, pooled_noise,
)

from esper.integration import integrate_mass
from esper.noise import noise_model


def test_noise_parameters_are_recovered():
    rng = np.random.default_rng(2)
    y = noisy(gaussian(300, 2.0, 1000) + 100, rng)
    model = noise_model(y, pooled_noise(y, rng))
    assert np.isclose(model.factor, FACTOR, rtol=0.1)
    assert np.isclose(model.gain, NOISE_GAIN * SCAN_VARIANCE, rtol=0.15)
    assert np.isclose(model.variance(100), (NOISE_FLOOR + NOISE_GAIN * 100) * SCAN_VARIANCE, rtol=0.15)


def test_uncertainty_matches_the_scatter_of_the_areas():
    # Over repeated noise, the standard deviation of the areas is close to their uncertainty
    rng = np.random.default_rng(3)
    cases = {
        "small (5 sd of the noise)": (gaussian(300, 2.0, 15) + 100, peaks_of((300, 2.0, 15))),
        "large (counting noise)": (gaussian(300, 2.0, 1e5) + 100, peaks_of((300, 2.0, 1e5))),
        "pair (split position)": (
            gaussian(300, 2.0, 500) + gaussian(306, 2.0, 500) + 100, peaks_of((300, 2.0, 500), (306, 2.0, 500))
        ),
        "tailing": (emg(300, 1.5, 6, 3000) + 20, pd.DataFrame([measured_peak(emg(300, 1.5, 6, 3000))])),
    }
    for name, (signal, peaks) in cases.items():
        areas, uncertainties = [], []
        for _ in range(150):
            y = noisy(signal, rng)
            res = integrate_mass(peaks, T, y, pooled_noise(y, rng))
            areas.append(res["area"].to_numpy())
            uncertainties.append(res["area_unc"].to_numpy())
        ratio = np.std(areas, axis=0) / np.mean(uncertainties, axis=0)
        assert np.all((ratio > 0.75) & (ratio < 1.25)), (name, ratio)


def test_too_little_data_gives_an_unknown_uncertainty_not_zero():
    rng = np.random.default_rng(5)
    short = T[:40]  # 40 scans: too few to estimate the noise
    y = noisy(gaussian(T[20], 0.5, 1000)[:40] + 100, rng)
    peaks = peaks_of((T[20], 0.5, 1000)).assign(right_base_scan=39)
    res = integrate_mass(peaks, short, y)
    assert np.isfinite(res.loc[0, "area"])
    assert np.isnan(res.loc[0, "area_unc"])


def test_short_chromatogram_still_has_an_uncertainty():
    # 300 scans: fewer than NOISE_MIN_QUIET quiet ones, the fallback is used
    rng = np.random.default_rng(6)
    t = T[:300]
    y = noisy(gaussian(t[150], 1.0, 1000)[:300] + 100, rng)
    res = integrate_mass(peaks_of((t[150], 1.0, 1000)).assign(right_base_scan=299), t, y)
    assert res.loc[0, "area_unc"] > 0

