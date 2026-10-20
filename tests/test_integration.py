"""Integration on synthetic chromatograms whose areas are known."""

from __future__ import annotations

import numpy as np
import pandas as pd
from synthetic import DT, T, emg, gaussian, gaussian_area, measured_peak, peaks_of

from esper.integration import compound_areas, integrate_chromatograms, integrate_mass


def test_single_peak_on_sloped_baseline():
    y = gaussian(300, 2.0, 1000) + 50 + 0.2 * T
    res = integrate_mass(peaks_of((300, 2.0, 1000)), T, y)
    assert np.isclose(res.loc[0, "area"], gaussian_area(2.0, 1000), rtol=0.01)
    # Baseline ends on the background
    assert np.isclose(res.loc[0, "base_start"], 50 + 0.2 * res.loc[0, "int_start_s"], rtol=0.01)


def test_overlapping_peaks_are_split_not_double_counted():
    y = gaussian(300, 2.0, 1000) + gaussian(306, 2.0, 1000) + 20
    res = integrate_mass(peaks_of((300, 2.0, 1000), (306, 2.0, 1000)), T, y)
    total = 2 * gaussian_area(2.0, 1000)
    assert np.isclose(res["area"].sum(), total, rtol=0.01)
    assert np.isclose(res.loc[0, "area"], res.loc[1, "area"], rtol=0.005)
    # The split is at the valley (between scans), shared by both peaks
    assert res.loc[0, "int_stop_s"] == res.loc[1, "int_start_s"]
    assert abs(res.loc[0, "int_stop_s"] - 303) < 0.1 * DT


def test_small_peak_on_the_tail_of_a_large_one():
    y = gaussian(300, 2.0, 10000) + gaussian(308, 2.0, 500)
    res = integrate_mass(peaks_of((300, 2.0, 10000), (308, 2.0, 500)), T, y)
    assert np.isclose(res["area"].sum(), gaussian_area(2.0, 10000) + gaussian_area(2.0, 500), rtol=0.01)
    # Drop line: the front of the small peak, before the valley, is counted in the large one
    small = gaussian_area(2.0, 500)
    assert 0.8 * small < res.loc[1, "area"] < small


def test_peak_cut_by_the_end_of_the_chromatogram():
    rt = T[-1] - 1.0
    res = integrate_mass(peaks_of((rt, 2.0, 1000)), T, gaussian(rt, 2.0, 1000))
    assert res.loc[0, "int_stop_s"] <= T[-1]
    assert 0 < res.loc[0, "area"] < gaussian_area(2.0, 1000)


def test_noise_only_gives_a_small_area():
    areas = [
        integrate_mass(peaks_of((300, 2.0, 0)), T, 100 + np.random.default_rng(seed).normal(0, 5, len(T))).loc[0, "area"]
        for seed in range(100)
    ]
    # Small compared with a peak 5 sigma of the noise high
    assert abs(np.mean(areas)) < 10
    assert np.std(areas) < 0.3 * gaussian_area(2.0, 25)


def test_unsorted_peaks_keep_their_index():
    y = gaussian(300, 2.0, 1000) + gaussian(400, 2.0, 2000)
    peaks = peaks_of((400, 2.0, 2000), (300, 2.0, 1000))
    res = integrate_mass(peaks, T, y)
    assert np.isclose(res.loc[0, "area"], gaussian_area(2.0, 2000), rtol=0.01)
    assert np.isclose(res.loc[1, "area"], gaussian_area(2.0, 1000), rtol=0.01)


def test_tailing_peaks():
    # tau / sigma: area within this relative error (the end of a long tail is lost)
    for ratio, rtol in [(1, 0.01), (2, 0.01), (4, 0.025), (8, 0.04)]:
        signal = emg(300, 1.5, 1.5 * ratio, 1000)
        y = signal + 50
        res = integrate_mass(pd.DataFrame([measured_peak(signal)]), T, y)
        assert np.isclose(res.loc[0, "area"], 1000, rtol=rtol), (ratio, res.loc[0, "area"])
        # The window is longer on the tail side
        rt = T[np.argmax(y)]
        assert res.loc[0, "int_stop_s"] - rt > rt - res.loc[0, "int_start_s"]


def test_overlapping_tailing_peaks_sum():
    y = emg(300, 1.5, 3, 1000) + emg(308, 1.5, 3, 10000)
    peaks = pd.DataFrame([measured_peak(y, 0, 290, 303), measured_peak(y, 1, 305, 315)])
    res = integrate_mass(peaks, T, y)
    assert np.isclose(res["area"].sum(), 11000, rtol=0.01)
    # The front of the large peak is not given to the small one
    assert 0.6 * 1000 < res.loc[0, "area"] < 1000


def test_drop_line_gives_the_tail_of_a_large_peak_to_the_next_one():
    # Known limit of the drop line: a small peak on the tail of a large one gets the
    # tail under it (a tangent skim would not). Documented here so a change is noticed.
    signal = emg(300, 1.5, 3, 10000) + emg(310, 1.5, 3, 1000)
    y = signal + 30
    peaks = pd.DataFrame([measured_peak(signal, 0, 290, 305), measured_peak(signal, 1, 307, 315)])
    res = integrate_mass(peaks, T, y)
    assert np.isclose(res["area"].sum(), 11000, rtol=0.01)
    assert 1.5 * 1000 < res.loc[1, "area"] < 2.5 * 1000


def test_peak_not_picked_at_the_window_start_does_not_lift_the_baseline():
    # As m/z 41 near 1590 s: the window of the picked peak starts on a peak alpinac did not pick
    signal = emg(295.5, 0.8, 0.8, 1000) + emg(300, 0.8, 0.8, 1000)
    res = integrate_mass(pd.DataFrame([measured_peak(signal, 0, 298, 310)]), T, signal + 10)
    # Without the valley, the window would start at ~295.9 s, on the other apex
    between = (T > 296) & (T < 300)
    valley = T[between][np.argmin(signal[between])]
    assert abs(res.loc[0, "int_start_s"] - valley) < 0.5
    assert np.isclose(res.loc[0, "area"], 1000, rtol=0.1)


def test_small_peaks_before_a_large_tailing_one_stay_positive():
    # As m/z 91 near 2200 s: a broad bump and a small peak grouped with a huge tailing peak
    small = emg(220, 1.0, 1.0, 1000)
    bump = emg(260, 8.0, 2.0, 20000)
    large = emg(300, 1.5, 6, 2_000_000)
    signal = small + bump + large
    peaks = pd.DataFrame([measured_peak(signal, 0, 210, 230), measured_peak(signal, 1, 240, 280), measured_peak(signal, 2, 290, 310)])
    peaks["left_base_scan"] = 0
    res = integrate_mass(peaks, T, signal + 10)
    assert (res["area"] > 0).all()
    assert np.isclose(res.loc[0, "area"], 1000, rtol=0.1)
    assert np.isclose(res.loc[2, "area"], 2_000_000, rtol=0.03)


def test_compound_areas_rank_masses():
    peaks = pd.DataFrame({
        "compound_idx": [7, 7, 7, 7, 9],
        "mass": [44, 45, 46, 44, 30],
        "area": [10.0, 50.0, np.nan, 5.0, 3.0],
        "area_unc": [3.0, 2.0, np.nan, 4.0, 1.0],
    })
    res = compound_areas(peaks)
    assert res.loc[7, "area_sum"] == 65.0
    assert np.isclose(res.loc[7, "area_sum_unc"], np.sqrt(9 + 4 + 16))  # in quadrature
    assert (res.loc[7, "q1_mass"], res.loc[7, "q1_area"], res.loc[7, "q1_area_unc"]) == (45, 50.0, 2.0)
    assert (res.loc[7, "q2_mass"], res.loc[7, "q2_area"], res.loc[7, "q2_area_unc"]) == (44, 15.0, 5.0)  # summed per mass
    assert np.isnan(res.loc[7, "q3_area"])  # mass 46 has no area
    assert (res.loc[9, "q1_mass"], res.loc[9, "area_sum"]) == (30, 3.0)


def test_integrate_chromatograms_from_parquet(tmp_path):
    frame = pd.DataFrame(
        {"44": gaussian(300, 2.0, 1000) + gaussian(306, 2.0, 1000), "45": gaussian(300, 2.0, 400)},
        index=pd.Index(T, name="rt"),
    )
    path = tmp_path / "chromatos.parquet"
    frame.to_parquet(path)
    peaks = pd.concat([peaks_of((300, 2.0, 1000), (306, 2.0, 1000)), peaks_of((300, 2.0, 400), mass=45), peaks_of((300, 2.0, 1), mass=99)], ignore_index=True)
    peaks.index.name = "peak_idx"

    # Only compound 1: its peak on 44 is still split from compound 0's peak
    peak_res, cmp_res, _ = integrate_chromatograms(peaks, path, compound_idx=[1])
    assert list(peak_res.index) == [1]
    assert np.isclose(peak_res.loc[1, "area"], gaussian_area(2.0, 1000), rtol=0.01)
    assert list(cmp_res.index) == [1]

    peak_res, cmp_res, _ = integrate_chromatograms(peaks, path)
    assert np.isnan(peak_res.loc[3, "area"])  # no chromatogram for mass 99
    assert cmp_res.loc[0, "q1_mass"] == 44 and cmp_res.loc[0, "q2_mass"] == 45
