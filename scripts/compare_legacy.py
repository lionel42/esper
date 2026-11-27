"""Compare the steps of the new processing chain with the legacy pipeline on the test case.

    python scripts/compare_legacy.py [deconvolution] [detection --data-dir DIR] [--runs ...] [--root DIR]

``deconvolution`` (step 4a of the refactoring plan): on the legacy unit mass chromatograms of each
run, the legacy functions (``halosearch/alpinac/mode_extraction/peaks.py``, loaded from their file:
no legacy alpinac import) and ``esper.deconvolution.legacy_dataset`` must give the same peaks and
compounds, to the bit. Also checked: the peak table against the final legacy ``pipeline.nc`` (the
compounds there are changed by the cleaning step: only their number is reported), and the rts of the
chromatograms against the time axis of the h5 file (the legacy code used the latter).

``detection`` (step 4 of the plan): the peaks that ``esper process`` stored in a data directory
(``--data-dir``: the test case uploaded with a method and the legacy detection parameters, see
``v2/esper_legacy/esper.yaml``) against the peaks of the legacy ``pipeline.nc`` (same values, the
prominence bases as times), and its compounds against the legacy grouping (the functions of
``peaks.py``: the compounds of ``pipeline.nc`` are changed by the cleaning step).
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


# esper peak column -> legacy pipeline.nc peak variable (the bases: times of the scans)
DETECTION_COLUMNS = {
    "mass": "mass_of_peak",
    "rt_s": "peak_rt",
    "height": "peak_heights",
    "prominence": "prominences",
    "width_height": "width_heights",
    "start_s": "start_ip",
    "stop_s": "stopp_ip",
    "flatness": "flatness",
    "tailing_ratio": "tailing_ratio",
    "left_base_s": "left_bases",
    "right_base_s": "right_bases",
}


def _peak_groups(compound: np.ndarray, keys: list[tuple]) -> set[frozenset]:
    """The peaks (their keys) of each compound, as a set of groups."""
    groups: dict = {}
    for c, key in zip(compound, keys):
        groups.setdefault(c, set()).add(key)
    return {frozenset(g) for g in groups.values()}


def compare_detection(run: str, root: Path, data_dir: Path, legacy) -> list[str]:
    import sqlite3

    from esper import db

    folder = root / f"{run}.alpinac"
    con = sqlite3.connect(f"file:{db.db_path(data_dir)}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT measurement_id, peak_source, n_peaks FROM measurement WHERE name = ?", (run,)
        ).fetchone()
        if row is None:
            print(f"FAIL {run}: not in {data_dir}")
            return ["not imported"]
        mid = row[0]
        peaks = pd.read_sql("SELECT * FROM peak WHERE measurement_id = ?", con, params=(mid,))
    finally:
        con.close()
    with xr.open_dataset(folder / "pipeline.nc") as final:
        final = final.load()
    chromatograms = pd.read_parquet(folder / f"{run}.unit_mass_chromatos.parquet")
    t = chromatograms.index.to_numpy(dtype=float)
    ref = pd.DataFrame({column: final[var].values for column, var in DETECTION_COLUMNS.items()})
    for side in ("left", "right"):
        ref[f"{side}_base_s"] = t[ref[f"{side}_base_s"].to_numpy(dtype=int)]
    problems = []
    if row[1] != "esper":
        problems.append(f"peak_source {row[1]!r}, not found by esper")
    order = ["mass", "rt_s"]
    new = peaks[list(DETECTION_COLUMNS)].sort_values(order, ignore_index=True)
    ref = ref.sort_values(order, ignore_index=True)
    line = f"{run}: {len(new)} peaks (legacy {len(ref)})"
    if len(new) != len(ref):
        problems.append(f"{len(new)} peaks, legacy {len(ref)}")
    else:
        for column in DETECTION_COLUMNS:
            a, b = new[column].to_numpy(dtype=float), ref[column].to_numpy(dtype=float)
            if not np.array_equal(a, b, equal_nan=True):
                diff = np.nanmax(np.abs(a - b))
                problems.append(f"{column} differs (max {diff:.3g})")
    # compounds: the same peaks together as the legacy grouping (before the cleaning)
    grouped = legacy_reference(chromatograms, legacy)
    ref_keys = list(zip(grouped["mass_of_peak"].values.astype(int), grouped["peak_rt"].values))
    new_keys = list(zip(peaks["mass"].astype(int), peaks["rt_s"]))
    ref_groups = _peak_groups(grouped["compound_id"].values, ref_keys)
    new_groups = _peak_groups(peaks["compound_idx"].to_numpy(), new_keys)
    line += f", {len(new_groups)} compounds (legacy grouping {len(ref_groups)})"
    if new_groups != ref_groups:
        problems.append(f"compounds differ: {len(new_groups ^ ref_groups)} groups not in both")
    main_new = set(zip(peaks.loc[peaks["is_max_of_cmp"] == 1, "mass"].astype(int),
                       peaks.loc[peaks["is_max_of_cmp"] == 1, "rt_s"]))
    is_max = grouped["is_max_of_cmp"].values.astype(bool)
    main_ref = {k for k, m in zip(ref_keys, is_max) if m}
    if main_new != main_ref:
        problems.append(f"main peaks of the compounds differ ({len(main_new ^ main_ref)})")
    print(("OK   " if not problems else "FAIL ") + line)
    for p in problems:
        print(f"     {p}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("steps", nargs="*", default=["deconvolution"],
                        help="steps to compare: deconvolution, detection")
    parser.add_argument("--root", type=Path, default=ROOT, help="folder of the test case (h5 + .alpinac folders)")
    parser.add_argument("--runs", nargs="+", help="run names (default: all with legacy chromatograms)")
    parser.add_argument("--data-dir", type=Path,
                        help="detection: the esper data directory of the test case (processed)")
    args = parser.parse_args(argv)
    unknown = set(args.steps) - {"deconvolution", "detection"}
    if unknown:
        parser.error(f"unknown steps {sorted(unknown)}")
    if "detection" in args.steps and args.data_dir is None:
        parser.error("detection: give the processed data directory (--data-dir)")
    runs = args.runs or sorted(p.name.removesuffix(".unit_mass_chromatos.parquet")
                               for p in args.root.glob("*.alpinac/*.unit_mass_chromatos.parquet"))
    failed = 0
    if "deconvolution" in args.steps:
        legacy = _legacy_module()
        print(f"== deconvolution (legacy method), {len(runs)} runs")
        for run in runs:
            failed += bool(compare_deconvolution(run, args.root, legacy))
    if "detection" in args.steps:
        legacy = _legacy_module()
        print(f"== detection of esper process ({args.data_dir}), {len(runs)} runs")
        for run in runs:
            failed += bool(compare_detection(run, args.root, args.data_dir, legacy))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
