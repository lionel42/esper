"""Deconvolution step of the processing: from the unit mass chromatograms to the compounds.

The numerics are in :mod:`peaxart.deconvolution`; here the chromatogram table of a run is
turned into the peak / compound part of ``pipeline.nc``.

``legacy``: the extraction of the legacy pipeline (``alpinac.mode_extraction.pipeline.
run_extraction_pipeline``), same dataset: ``peak`` variables (the ``find_peaks`` properties,
``start_ip``, ``stopp_ip``, ``peak_rt``, ``mass_of_peak``, ``flatness``, ``tailing_ratio``,
``compound_id``, ``is_max_of_cmp``) and ``compound`` variables (``peak_rt_mean`` / ``_std`` /
``_count``, ``prominences_sum``, ``start_ip_mean`` / ``_std``, ``stopp_ip_mean`` / ``_std``, ``mai_``).

    chromatograms = esper.chromatograms.read_chromatograms("run.alpinac/run.unit_mass_chromatos.parquet")
    ds = esper.deconvolution.legacy_dataset(chromatograms)
"""

from __future__ import annotations

import pandas as pd
import peaxart
import xarray as xr


def chromatogram_masses(chromatograms: pd.DataFrame) -> list[str]:
    """The columns of the unit masses (``"2"`` ... ``"300"``; not ``tic``)."""
    return [c for c in chromatograms.columns if c != "tic" and not str(c).endswith(":tic")]


def legacy_dataset(chromatograms: pd.DataFrame, rt_start: float = 0.0, rt_stopp: float | None = None,
                   config: peaxart.LegacyDeconvolutionConfig | None = None) -> xr.Dataset:
    """Peaks and compounds of the legacy extraction (:func:`peaxart.legacy_deconvolution`), as the
    legacy ``run_extraction_pipeline`` dataset.

    :param chromatograms: corrected unit mass chromatograms, index rt (s), one column per unit mass
        (``"2"`` ...; ``tic`` left out: the legacy pipeline found no peaks in it).
    :param rt_start, rt_stopp: stored in the attributes (default: 0 and the last rt), as the legacy.
    """
    columns = chromatogram_masses(chromatograms)
    masses = [int(c) if float(c).is_integer() else float(c) for c in columns]
    rt = chromatograms.index.to_numpy(dtype=float)
    compounds, peaks = peaxart.legacy_deconvolution(chromatograms[columns].to_numpy(), rt, masses, config)
    peaks = peaks.rename(columns={"mass": "mass_of_peak"})
    return xr.Dataset(
        {col: ("peak", peaks[col].to_numpy()) for col in peaks.columns}
        | {"_".join(col): ("compound", compounds[col].to_numpy()) for col in compounds.columns},
        coords={"peak": peaks.index.to_numpy(), "compound": compounds.index.to_numpy()},
        attrs={"rt_start": float(rt_start), "rt_stopp": float(rt[-1] if rt_stopp is None else rt_stopp)},
    )


__all__ = ["chromatogram_masses", "legacy_dataset"]
