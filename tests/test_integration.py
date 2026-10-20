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
    # A shoulder on the side of a gaussian (not an exponential tail): drop line, the front
    # of the small peak, before the valley, is counted in the large one
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
    # Half widths at half prominence: the window of the first peak starts a bit early
    assert np.isclose(res["area"].sum(), 11000, rtol=0.02)
    # The front of the large peak is not given to the small one
    assert 0.6 * 1000 < res.loc[0, "area"] < 1000


def test_small_peak_on_an_exponential_tail_is_skimmed():
    # As CFC-12 on m/z 85: a small peak on the (exponential) tail of a large one. The drop
    # line would give it the tail under it (+40 % here); the tail is extrapolated under it
    p_, r_ = emg(300, 0.9, 1.5, 10000), emg(307.5, 0.8, 0.3, 400)
    y = p_ + r_ + 30
    peaks = pd.DataFrame([measured_peak(p_ + r_, 0, 295, 303), measured_peak(p_ + r_, 1, 305.5, 310)])
    res = integrate_mass(peaks, T, y)
    # A real tail is only exponential far from the apex: slightly under
    assert 0.75 * 400 < res.loc[1, "area"] < 1.05 * 400
    # The large peak keeps its tail: the total is unchanged
    assert np.isclose(res["area"].sum(), 10400, rtol=0.01)
    assert np.isclose(res.loc[0, "area"], 10000, rtol=0.01)


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


def test_compound_areas_with_an_unknown_uncertainty():
    peaks = pd.DataFrame({
        "compound_idx": [7, 7, 7],
        "mass": [44, 45, 46],
        "area": [10.0, 50.0, np.nan],
        "area_unc": [np.nan, 2.0, np.nan],
    })
    res = compound_areas(peaks)
    assert res.loc[7, "area_sum"] == 60.0
    assert np.isnan(res.loc[7, "area_sum_unc"])  # not 2: the uncertainty of mass 44 is unknown
    assert (res.loc[7, "q1_mass"], res.loc[7, "q1_area_unc"]) == (45, 2.0)
    assert res.loc[7, "q2_mass"] == 44 and np.isnan(res.loc[7, "q2_area_unc"])


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


def test_overlapping_peaks_stay_together():
    # As m/z 31 near 1470 s: the stop of the first window is moved back to the valley, which
    # is before the start of the second window. The peaks must still share their split
    # (with its uncertainty) and baseline, not become two groups with their own baselines.
    def peak(k, rt, front, tail):
        return {
            "compound_idx": k, "mass": 31, "rt_s": rt, "start_s": rt - front, "stop_s": rt + tail,
            "left_base_scan": 0, "right_base_scan": len(T) - 1,
        }

    peaks = pd.DataFrame([peak(0, 300.0, 0.8, 1.2), peak(1, 306.5, 0.8, 1.0)])
    y = gaussian(300, 0.85, 29000) + gaussian(306.5, 0.75, 15000) + 2 * T
    res = integrate_mass(peaks, T, y)
    assert res.loc[0, "int_stop_s"] == res.loc[1, "int_start_s"]
    assert res.loc[0, "base_stop"] == res.loc[1, "base_start"]
    assert np.allclose(res["area"], [gaussian_area(0.85, 29000), gaussian_area(0.75, 15000)], rtol=0.015)


def test_narrow_peak_on_a_broad_one():
    # On the flank of the broad peak: the narrow window [~285.3, ~297.1] is inside the broad
    # one [~242, ~336], and the signal does not come back to the baseline between them
    broad, narrow = (280, 8.0, 1000), (290, 1.0, 2000)
    y = gaussian(*broad) + gaussian(*narrow) + 20
    res = integrate_mass(peaks_of(broad, narrow), T, y)
    # Nothing lost: the group ends at the end of the broad window, not of the narrow one
    total = gaussian_area(8.0, 1000) + gaussian_area(1.0, 2000)
    assert np.isclose(res["area"].sum(), total, rtol=0.01)
    assert res.loc[0, "int_stop_s"] > 330
    # The narrow peak keeps its own window, the broad one its tail after it
    assert res.loc[0, "int_start_s"] < res.loc[1, "int_start_s"] < res.loc[1, "int_stop_s"] < res.loc[0, "int_stop_s"]
    assert res.loc[1, "int_stop_s"] < 297.6
    # The narrow peak gets the broad signal under its window (drop line), not the broad tail
    s, e = res.loc[1, "int_start_s"], res.loc[1, "int_stop_s"]
    tt = np.concatenate([[s], T[(T > s) & (T < e)], [e]])
    in_window = np.trapezoid(np.interp(tt, T, gaussian(*broad) + gaussian(*narrow)), tt)
    assert np.isclose(res.loc[1, "area"], in_window, rtol=0.002)
    assert in_window > gaussian_area(1.0, 2000)


def test_baseline_reset_between_separated_peaks():
    # A small peak after a large one whose (long, tail side) window reaches it, while the
    # signal is back on the background in between: separate baselines (as m/z 111 near
    # 2305 s), the small peak's baseline stays on the background
    large, small = (280, 2.0, 5000), (300, 1.5, 100)
    y = gaussian(*large) + gaussian(*small) + 300
    res = integrate_mass(peaks_of(large, small), T, y)
    assert res.loc[0, "int_stop_s"] <= res.loc[1, "int_start_s"]
    assert np.isclose(res.loc[1, "base_start"], 300, atol=2) and np.isclose(res.loc[1, "base_stop"], 300, atol=2)
    assert np.isclose(res.loc[1, "area"], gaussian_area(1.5, 100), rtol=0.03)
