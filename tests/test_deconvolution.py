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
