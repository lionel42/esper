"""Compare the steps of the new processing chain with the legacy pipeline on the test case.

    python scripts/compare_legacy.py [deconvolution] [--runs 251104.1724.air.1 ...] [--root DIR]

``deconvolution`` (step 4a of the refactoring plan): on the legacy unit mass chromatograms of each
run, the legacy functions (``halosearch/alpinac/mode_extraction/peaks.py``, loaded from their file:
no legacy alpinac import) and ``esper.deconvolution.legacy_dataset`` must give the same peaks and
compounds, to the bit. Also checked: the peak table against the final legacy ``pipeline.nc`` (the
compounds there are changed by the cleaning step: only their number is reported), and the rts of the
chromatograms against the time axis of the h5 file (the legacy code used the latter).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/coli/Data/alpinac_cases/20261014_reprocessing_for_massbank/h5_files")
LEGACY_PEAKS = Path("/home/coli/gitlabs/halosearch/alpinac/mode_extraction/peaks.py")
STEP2_PEAK_VARIABLES = ["peak_heights", "prominences", "left_bases", "right_bases", "widths", "width_heights",
                        "start_ip", "stopp_ip", "peak_rt", "mass_of_peak", "flatness", "tailing_ratio"]


def _legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_peaks", LEGACY_PEAKS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_peaks"] = module
    spec.loader.exec_module(module)
    return module


def legacy_reference(chromatograms: pd.DataFrame, legacy) -> xr.Dataset:
    """The legacy ``run_extraction_pipeline`` on chromatograms computed in the run (no ``tic``,
    integer masses), with the rts of the chromatograms."""
    time_axis = chromatograms.index.to_numpy(dtype=float)
    chromato_dict = {int(c): chromatograms[c].to_numpy() for c in chromatograms.columns if c != "tic"}
    df_peaks = legacy.extract_peaks_from_chromatograms(
        chromato_dict, find_peaks_kwargs={"width": 9, "height": 40, "prominence": 120},
        time_bin_to_time=lambda x: np.interp(x, np.arange(len(time_axis)), time_axis))
    df_compounds, df_peaks = legacy.group_peaks_into_compounds(df_peaks)
    df_peaks = df_peaks.rename(columns={"mass": "mass_of_peak"})
    return xr.Dataset(
        {col: (("peak"), df_peaks[col]) for col in df_peaks.columns}
        | {"_".join(col): (("compound"), df_compounds[col]) for col in df_compounds.columns},
        coords={"peak": df_peaks.index, "compound": df_compounds.index.values},
        attrs={"rt_start": 0.0, "rt_stopp": float(time_axis[-1])},
    )


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a.shape == b.shape and np.array_equal(a, b)


def compare_deconvolution(run: str, root: Path, legacy) -> list[str]:
    from esper.deconvolution import legacy_dataset

    folder = root / f"{run}.alpinac"
    chromatograms = pd.read_parquet(folder / f"{run}.unit_mass_chromatos.parquet")
    new = legacy_dataset(chromatograms)
    ref = legacy_reference(chromatograms, legacy)
    problems = []
    if set(new.data_vars) != set(ref.data_vars):
        problems.append(f"variables differ: {set(new.data_vars) ^ set(ref.data_vars)}")
    for v in sorted(set(new.data_vars) & set(ref.data_vars)):
        if not _same(new[v].values, ref[v].values):
            problems.append(f"{v} differs")
    for c in ("peak", "compound"):
        if not _same(new[c].values, ref[c].values):
            problems.append(f"coordinate {c} differs")
    if new.attrs != ref.attrs:
        problems.append(f"attrs {new.attrs} != {ref.attrs}")
    line = f"{run}: {new.sizes['peak']} peaks, {new.sizes['compound']} compounds"

    pipeline = folder / "pipeline.nc"
    if pipeline.is_file():
        with xr.open_dataset(pipeline) as final:
            same = new.sizes["peak"] == final.sizes["peak"] and all(
                _same(new[v].values, final[v].values) for v in STEP2_PEAK_VARIABLES)
            line += (f"; pipeline.nc: peaks {'identical' if same else 'DIFFERENT'}, "
                     f"{final.sizes['compound']} compounds after the cleaning")
            if not same:
                problems.append("peaks differ from pipeline.nc")
    h5 = root / f"{run}.h5"
    if h5.is_file():
        import twutils

        try:
            t = twutils.TofFile(h5).time_axis
            rt = chromatograms.index.to_numpy(dtype=float)
            line += "; rt = h5 time axis" if len(t) == len(rt) and np.allclose(t, rt, rtol=0, atol=1e-9) else (
                f"; rt != h5 time axis ({len(rt)} vs {len(t)} scans)")
        except Exception as e:  # noqa: BLE001 - report and go on
            line += f"; h5 not readable: {e!r}"
    print(("OK   " if not problems else "FAIL ") + line)
    for p in problems:
        print(f"     {p}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("steps", nargs="*", default=["deconvolution"], help="steps to compare: deconvolution")
    parser.add_argument("--root", type=Path, default=ROOT, help="folder of the test case (h5 + .alpinac folders)")
    parser.add_argument("--runs", nargs="+", help="run names (default: all with legacy chromatograms)")
    args = parser.parse_args(argv)
    unknown = set(args.steps) - {"deconvolution"}
    if unknown:
        parser.error(f"unknown steps {sorted(unknown)}")
    runs = args.runs or sorted(p.name.removesuffix(".unit_mass_chromatos.parquet")
                               for p in args.root.glob("*.alpinac/*.unit_mass_chromatos.parquet"))
    failed = 0
    if "deconvolution" in args.steps:
        legacy = _legacy_module()
        print(f"== deconvolution (legacy method), {len(runs)} runs")
        for run in runs:
            failed += bool(compare_deconvolution(run, args.root, legacy))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
