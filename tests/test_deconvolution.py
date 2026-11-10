"""The deconvolution step: the legacy peak / compound dataset of pipeline.nc."""

from __future__ import annotations

import numpy as np
import pandas as pd

from esper.deconvolution import legacy_dataset

PEAK_VARIABLES = ["peak_heights", "prominences", "left_bases", "right_bases", "widths", "width_heights",
                  "start_ip", "stopp_ip", "peak_rt", "mass_of_peak", "flatness", "tailing_ratio", "compound_id",
                  "is_max_of_cmp"]
COMPOUND_VARIABLES = ["peak_rt_mean", "peak_rt_std", "peak_rt_count", "prominences_sum", "start_ip_mean",
                      "start_ip_std", "stopp_ip_mean", "stopp_ip_std", "mai_"]


def test_legacy_dataset():
    rt = np.arange(1, 2001) * 0.17
    peak = 1000 * np.exp(-0.5 * ((rt - 150) / 1.5) ** 2)
    chromatograms = pd.DataFrame({"69": peak, "70": 0.3 * peak, "100": np.roll(peak, 300)},
                                 index=pd.Index(rt, name="rt"))
    chromatograms["tic"] = chromatograms.sum(axis=1)  # left out
    ds = legacy_dataset(chromatograms)
    assert [v for v in ds.data_vars if ds[v].dims == ("peak",)] == PEAK_VARIABLES
    assert [v for v in ds.data_vars if ds[v].dims == ("compound",)] == COMPOUND_VARIABLES
    assert ds["mass_of_peak"].values.tolist() == [69, 70, 100] and ds["mass_of_peak"].dtype.kind == "i"
    assert ds["compound_id"].values.tolist() == [0, 0, 1] and ds["mai_"].values.tolist() == [69, 100]
    assert ds.attrs == {"rt_start": 0.0, "rt_stopp": rt[-1]}
    assert legacy_dataset(chromatograms, rt_start=10.0, rt_stopp=300.0).attrs == {"rt_start": 10.0, "rt_stopp": 300.0}


def test_fit_dataset():
    """The fit on its own dimension, linked to the detected peaks; the peak / compound part unchanged."""
    import json

    from esper.deconvolution import FIT_CONFIG, FIT_VARIABLES, fit_dataset

    shape = FIT_CONFIG.shape()  # the peaks have the shape of the fit
    apex_rt, apex_height = shape.apex(1.0, 0.0, 1.4, 0.3)

    def peak(t, height, rt):  # of apex height at rt
        return shape.value(t, height / apex_height, rt - apex_rt, 1.4, 0.3)

    rt = np.arange(1700.0, 1800.0, 1 / 6)
    coelution = peak(rt, 1000.0, 1740.0) + peak(rt, 600.0, 1745.0)
    alone = peak(rt, 500.0, 1770.0)
    chromatograms = pd.DataFrame({"69": 5 + coelution, "70": 5 + alone}, index=pd.Index(rt, name="rt"))
    ds = legacy_dataset(chromatograms)
    fitted = fit_dataset(ds, chromatograms, workers=1)
    assert fitted["peak"].equals(ds["peak"]) and fitted["compound_id"].equals(ds["compound_id"])
    assert set(fitted.data_vars) - set(ds.data_vars) == {"fit_peak", "fit_mass", "fit_added", *FIT_VARIABLES}
    assert all(fitted[v].dims == ("fit",) for v in ("fit_peak", "fit_mass", "fit_added", *FIT_VARIABLES))
    np.testing.assert_array_equal(fitted["fit_mass"].values, ds["mass_of_peak"].values[fitted["fit_peak"].values])
    # linked to detected peaks of the same mass
    mass = ds["mass_of_peak"].values[fitted["fit_peak"].values]
    np.testing.assert_allclose(fitted["fit_rt"].values[mass == 69], [1740, 1745], atol=0.1)
    np.testing.assert_allclose(fitted["fit_rt"].values[mass == 70], [1770], atol=0.1)
    np.testing.assert_allclose(fitted["fit_height"].values, [1000, 600, 500], rtol=0.01)
    # the apex, and the parameters of the peak function
    np.testing.assert_allclose(fitted["fit_shape_rt"].values, np.array([1740, 1745, 1770]) - apex_rt, atol=0.1)
    config = json.loads(fitted.attrs["fit_config"])
    assert config["convention"] == "standard" and config["peak_shape"] == FIT_CONFIG.peak_shape
    # fitted again: replaced, not added
    assert fit_dataset(fitted, chromatograms, workers=1).sizes["fit"] == 3
