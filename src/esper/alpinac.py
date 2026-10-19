"""Read the outputs of the alpinac extraction pipeline (a ``<measurement>.alpinac`` folder)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ALPINAC_SUFFIX = ".alpinac"
PIPELINE_FILENAME = "pipeline.nc"
# Older alpinac versions saved dated pipelines, e.g. pipeline_20260915_0555.nc
DATED_PIPELINE_GLOB = "pipeline_*.nc"

# e.g. 241007.0423.air.1
_NAME_RE = re.compile(r"^(?P<date>\d{6})\.(?P<time>\d{4})\.(?P<type>[^.]+)\.(?P<number>\d+)$")

_RANKS = {1: "first", 2: "second", 3: "third"}

# pipeline.nc variable -> sql column
_PEAK_COLUMNS = {
    "compound_id": "compound_idx",
    "mass_of_peak": "mass",
    "peak_rt": "rt_s",
    "peak_heights": "height",
    "prominences": "prominence",
    "widths": "width_scans",
    "width_heights": "width_height",
    "left_bases": "left_base_scan",
    "right_bases": "right_base_scan",
    "start_ip": "start_s",
    "stopp_ip": "stop_s",
    "flatness": "flatness",
    "tailing_ratio": "tailing_ratio",
    "is_max_of_cmp": "is_max_of_cmp",
}
_COMPOUND_COLUMNS = {
    "peak_rt_mean": "rt_mean_s",
    "peak_rt_std": "rt_std_s",
    "peak_rt_count": "n_peaks",
    "prominences_sum": "prominence_sum",
    "start_ip_mean": "start_mean_s",
    "start_ip_std": "start_std_s",
    "stopp_ip_mean": "stop_mean_s",
    "stopp_ip_std": "stop_std_s",
    "mai_": "main_ion_mass",
    "retention_index": "retention_index",
    "is_ref_ri": "is_ref_ri",
    "valid_candidate": "valid_candidate",
    "compound_name": "compound_name",
}

# Header lines of the *_mc.txt file -> sql column
_MC_HEADER = {
    "Number of points, mass cal. vs RT": "n_rt",
    "Number of points, mass cal. u vs mass": "n_mass",
    "Mass cal. mode": "mode",
    "Peak shape, sigma_slope vs TOF index": "sigma_tofidx_slope",
    "Peak shape, sigma_b vs TOF index": "sigma_tofidx_b",
    "Peak shape, sigma_slope vs mass": "sigma_m_slope",
    "Peak shape, sigma_b vs mass": "sigma_m_b",
    "alpha, for tofidx Pseudo-Voigt fit": "alpha_pseudo_voigt",
    "s_per_bin": "s_per_bin",
}


def parse_measurement_name(name: str) -> dict:
    """Split 'YYMMDD.HHMM.<type>.<number>' into its parts (None if not matching)."""
    m = _NAME_RE.match(name)
    if m is None:
        return {"acq_datetime": None, "sample_type": None, "sample_number": None}
    dt = datetime.strptime(m["date"] + m["time"], "%y%m%d%H%M")
    return {
        "acq_datetime": dt.isoformat(),
        "sample_type": m["type"],
        "sample_number": int(m["number"]),
    }


@dataclass
class MassCalibration:
    ionization: str
    params: dict
    rt: pd.DataFrame  # rt_idx, p1..pN
    masses: pd.DataFrame  # exact_mass, u_ppm


def read_mass_cal(path: Path) -> MassCalibration:
    """Read a ``<name>.<ionization>_mc.txt`` file written by ``alpinac.io_tools.export_mc``."""
    lines = [l.split("\t") for l in path.read_text().splitlines() if l.strip()]

    header = {}
    for values in lines[: len(_MC_HEADER)]:
        key = _MC_HEADER[values[1].strip()]
        header[key] = float(values[0])
    n_rt, n_mass = int(header.pop("n_rt")), int(header.pop("n_mass"))
    header["mode"] = int(header["mode"])

    body = lines[len(_MC_HEADER) :]
    if len(body) != n_rt + n_mass:
        raise ValueError(f"{path}: expected {n_rt} + {n_mass} lines, got {len(body)}")

    rt_rows = [[float(v) for v in values] for values in body[:n_rt]]
    n_params = len(rt_rows[0]) - 1 if rt_rows else 0
    rt = pd.DataFrame(rt_rows, columns=["rt_idx"] + [f"p{i}" for i in range(1, n_params + 1)])
    rt["rt_idx"] = rt["rt_idx"].astype(int)
    rt["rt_s"] = rt["rt_idx"] * header["s_per_bin"]
    header["n_params"] = n_params

    masses = pd.DataFrame(
        [[float(v) for v in values] for values in body[n_rt:]],
        columns=["exact_mass", "u_ppm"],
    )

    ionization = path.name.removesuffix("_mc.txt").rsplit(".", 1)[-1]
    return MassCalibration(ionization=ionization, params=header, rt=rt, masses=masses)


def _str_or_none(values: np.ndarray) -> list:
    return [None if v is None or str(v) == "" else str(v) for v in values]


@dataclass
class AlpinacResult:
    """The content of one .alpinac folder, as tables ready for the database."""

    name: str
    folder: Path
    attrs: dict
    peaks: pd.DataFrame
    compounds: pd.DataFrame
    candidates: pd.DataFrame
    mass_cal: MassCalibration | None
    chromatogram_file: Path | None
    warnings: list[str] = field(default_factory=list)

    @property
    def pipeline_file(self) -> Path:
        return find_pipeline(self.folder)


def find_pipeline(folder: Path) -> Path | None:
    """The pipeline file of a .alpinac folder, as alpinac loads it.

    ``pipeline.nc``, otherwise the latest dated ``pipeline_<YYYYMMDD_HHMM>.nc``
    (the last one by name).
    """
    folder = Path(folder)
    pipeline = folder / PIPELINE_FILENAME
    if pipeline.is_file():
        return pipeline
    dated = sorted(p for p in folder.glob(DATED_PIPELINE_GLOB) if p.is_file())
    return dated[-1] if dated else None


def is_complete(folder: Path) -> bool:
    return find_pipeline(folder) is not None


def read_alpinac_folder(folder: Path) -> AlpinacResult:
    folder = Path(folder)
    name = folder.name.removesuffix(ALPINAC_SUFFIX)
    warnings = []

    with xr.open_dataset(find_pipeline(folder)) as ds:
        ds = ds.load()

    peaks = pd.DataFrame({"peak_idx": ds["peak"].values.astype(int)})
    for var, col in _PEAK_COLUMNS.items():
        peaks[col] = ds[var].values
    peaks["is_max_of_cmp"] = peaks["is_max_of_cmp"].astype(bool).astype(int)

    compounds = pd.DataFrame({"compound_idx": ds["compound"].values.astype(int)})
    for var, col in _COMPOUND_COLUMNS.items():
        compounds[col] = ds[var].values
    for col in ("is_ref_ri", "valid_candidate"):
        compounds[col] = compounds[col].astype(bool).astype(int)
    compounds["compound_name"] = _str_or_none(compounds["compound_name"].values)

    bc_spectra = _str_or_none(ds["bc_spectra"].values)
    candidates = []
    for rank, prefix in _RANKS.items():
        spectra = _str_or_none(ds[f"{prefix}_spectra"].values)
        candidates.append(
            pd.DataFrame(
                {
                    "compound_idx": compounds["compound_idx"],
                    "rank": rank,
                    "candidate_name": _str_or_none(ds[f"{prefix}_candidate"].values),
                    "spectrum_ref": spectra,
                    "library": [s.split("_", 1)[0] if s else None for s in spectra],
                    "score": ds[f"{prefix}_score"].values,
                    "ri": ds[f"{prefix}_ri"].values,
                    "ri_diff": ds[f"{prefix}_ri_diff"].values,
                    "score_combined": ds[f"{prefix}_score_combined"].values,
                    "is_best": [int(s is not None and s == bc) for s, bc in zip(spectra, bc_spectra)],
                }
            )
        )
    candidates = pd.concat(candidates, ignore_index=True)
    candidates = candidates[candidates["spectrum_ref"].notna() | candidates["candidate_name"].notna()]
    # Only one best candidate per compound (the highest ranked if the same spectrum appears twice)
    best = candidates[candidates["is_best"] == 1].sort_values("rank")
    duplicated = best.duplicated("compound_idx")
    candidates.loc[best.index[duplicated], "is_best"] = 0

    missing = set(peaks["compound_idx"]) - set(compounds["compound_idx"])
    if missing:
        warnings.append(f"{len(missing)} compound ids of peaks are not in the compound table")

    mc_files = sorted(folder.glob("*_mc.txt"))
    mass_cal = None
    if mc_files:
        mass_cal = read_mass_cal(mc_files[0])
        if len(mc_files) > 1:
            warnings.append(f"Several mass calibration files, using {mc_files[0].name}")
    else:
        warnings.append("No mass calibration file")
    if mass_cal is not None:
        s_per_bin = mass_cal.params["s_per_bin"]
        peaks["width_s"] = peaks["width_scans"] * s_per_bin
    else:
        peaks["width_s"] = np.nan

    chromatos = folder / f"{name}.unit_mass_chromatos.parquet"
    if not chromatos.is_file():
        warnings.append("No unit mass chromatograms")
        chromatos = None

    attrs = {
        "rt_start_s": float(ds.attrs.get("rt_start", np.nan)),
        "rt_stop_s": float(ds.attrs.get("rt_stopp", np.nan)),
        "cleaning": ds.attrs.get("cleaning"),
    }

    return AlpinacResult(
        name=name,
        folder=folder,
        attrs=attrs,
        peaks=peaks,
        compounds=compounds,
        candidates=candidates,
        mass_cal=mass_cal,
        chromatogram_file=chromatos,
        warnings=warnings,
    )
