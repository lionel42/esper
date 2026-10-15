"""Lazy access to an esper data directory for the GUI.

Only the measurement table is read upfront. Compounds, peaks and chromatograms
are queried when a measurement is shown and kept in small caches.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from esper import db

TIC = None  # mass value standing for the total ion chromatogram


class EsperData:
    def __init__(self, data_dir: Path, chromatogram_cache_size: int = 64) -> None:
        self.data_dir = Path(data_dir)
        self.con = db.connect(self.data_dir, readonly=True)
        self._measurements: pd.DataFrame | None = None
        self._chromatograms: OrderedDict = OrderedDict()
        self._chromatogram_cache_size = chromatogram_cache_size

    def close(self) -> None:
        self.con.close()

    def _query(self, sql: str, params=(), **kwargs) -> pd.DataFrame:
        return pd.read_sql(sql, self.con, params=params, **kwargs)

    @property
    def measurements(self) -> pd.DataFrame:
        """All measurements, indexed by measurement_id, in acquisition order."""
        if self._measurements is None:
            df = self._query(
                """
                SELECT m.*, mc.mode AS mass_cal_mode, mc.s_per_bin
                FROM measurement m LEFT JOIN mass_cal mc USING (measurement_id)
                ORDER BY m.acq_datetime, m.name
                """,
                index_col="measurement_id",
            )
            df["acq_datetime"] = pd.to_datetime(df["acq_datetime"])
            self._measurements = df
        return self._measurements

    def sample_types(self) -> list[str]:
        return sorted(self.measurements["sample_type"].dropna().unique())

    def compound_names(self, valid_only: bool) -> pd.Series:
        """Number of occurrences of each compound name."""
        where = "AND valid_candidate = 1" if valid_only else ""
        df = self._query(
            f"""
            SELECT compound_name, COUNT(*) AS n FROM compound
            WHERE compound_name IS NOT NULL {where}
            GROUP BY compound_name ORDER BY compound_name COLLATE NOCASE
            """
        )
        return df.set_index("compound_name")["n"]

    @lru_cache(maxsize=32)
    def compounds(self, measurement_id: int) -> pd.DataFrame:
        """Compounds of a measurement with their best candidate, sorted by rt."""
        return self._query(
            """
            SELECT c.*,
                cc.spectrum_ref AS best_spectrum,
                cc.score AS best_score,
                cc.score_combined AS best_score_combined
            FROM compound c
            LEFT JOIN compound_candidate cc
                ON cc.measurement_id = c.measurement_id
                AND cc.compound_idx = c.compound_idx
                AND cc.is_best = 1
            WHERE c.measurement_id = ?
            ORDER BY c.rt_mean_s
            """,
            (int(measurement_id),),
            index_col="compound_idx",
        )

    @lru_cache(maxsize=32)
    def peaks(self, measurement_id: int) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM peak WHERE measurement_id = ? ORDER BY rt_s",
            (int(measurement_id),),
            index_col="peak_idx",
        )

    def compound_peaks(self, measurement_id: int, compound_idx: int) -> pd.DataFrame:
        peaks = self.peaks(measurement_id)
        return peaks[peaks["compound_idx"] == compound_idx]

    def main_ion(self, measurement_id: int, compound_idx: int) -> int | None:
        """Mass of the most prominent peak of the compound."""
        peaks = self.compound_peaks(measurement_id, compound_idx)
        if peaks.empty:
            return None
        return int(peaks.loc[peaks["prominence"].idxmax(), "mass"])

    def find_compound(
        self, measurement_id: int, name: str | None, near_rt: float | None = None
    ) -> int | None:
        """Compound of the measurement with that name.

        Valid candidates are preferred, then the one closest to ``near_rt``,
        or the most intense one.
        """
        if name is None:
            return None
        compounds = self.compounds(measurement_id)
        matching = compounds[compounds["compound_name"] == name]
        if matching.empty:
            return None
        valid = matching[matching["valid_candidate"] == 1]
        if not valid.empty:
            matching = valid
        if near_rt is not None:
            return int((matching["rt_mean_s"] - near_rt).abs().idxmin())
        return int(matching["prominence_sum"].idxmax())

    def results(self, name: str, valid_only: bool) -> pd.DataFrame:
        """All occurrences of a compound in the measurements, with peak summaries."""
        where = "AND c.valid_candidate = 1" if valid_only else ""
        df = self._query(
            f"""
            SELECT
                c.measurement_id, c.compound_idx, c.rt_mean_s, c.retention_index,
                c.valid_candidate,
                COUNT(p.peak_idx) AS n_peaks,
                SUM(p.prominence) AS prominence_sum,
                MAX(p.height) AS max_height,
                MAX(p.prominence) AS max_prominence,
                cc.score AS best_score,
                cc.score_combined AS best_score_combined
            FROM compound c
            JOIN peak p
                ON p.measurement_id = c.measurement_id AND p.compound_idx = c.compound_idx
            LEFT JOIN compound_candidate cc
                ON cc.measurement_id = c.measurement_id
                AND cc.compound_idx = c.compound_idx
                AND cc.is_best = 1
            WHERE c.compound_name = ? {where}
            GROUP BY c.measurement_id, c.compound_idx
            """,
            (name,),
        )
        meas = self.measurements[["name", "acq_datetime", "sample_type", "sample_number"]]
        df = df.join(meas, on="measurement_id")
        df["measurement_pos"] = self.measurements.index.get_indexer(df["measurement_id"])
        return df

    def search(self, masses: list[int], window: str, lo: float, hi: float) -> pd.DataFrame:
        """Peaks of the given masses within an rt [s] or retention index window.

        The retention index of a peak is the one of its compound.
        """
        column = {"rt": "p.rt_s", "ri": "c.retention_index"}[window]
        placeholders = ", ".join("?" * len(masses))
        df = self._query(
            f"""
            SELECT
                p.measurement_id, p.peak_idx, p.compound_idx, p.mass, p.rt_s,
                p.height, p.prominence, p.width_s,
                c.retention_index, c.compound_name, c.valid_candidate
            FROM peak p
            JOIN compound c
                ON c.measurement_id = p.measurement_id AND c.compound_idx = p.compound_idx
            WHERE p.mass IN ({placeholders}) AND {column} BETWEEN ? AND ?
            """,
            (*[int(m) for m in masses], float(lo), float(hi)),
        )
        meas = self.measurements[["name", "acq_datetime", "sample_type", "sample_number"]]
        df = df.join(meas, on="measurement_id")
        df["measurement_pos"] = self.measurements.index.get_indexer(df["measurement_id"])
        return df

    def _chromatogram_path(self, measurement_id: int) -> Path | None:
        file = self.measurements.loc[measurement_id, "chromatogram_file"]
        if not isinstance(file, str):
            return None
        return self.data_dir / file

    @lru_cache(maxsize=None)
    def chromatogram_columns(self, measurement_id: int) -> frozenset[str]:
        path = self._chromatogram_path(measurement_id)
        if path is None or not path.is_file():
            return frozenset()
        return frozenset(pq.read_schema(path).names)

    @lru_cache(maxsize=2)
    def chromatogram_frame(self, measurement_id: int) -> pd.DataFrame | None:
        """All unit mass chromatograms of a measurement (whole parquet file)."""
        path = self._chromatogram_path(measurement_id)
        if path is None or not path.is_file():
            return None
        return pd.read_parquet(path)

    def chromatogram(self, measurement_id: int, mass: int | None) -> pd.Series | None:
        """Unit mass chromatogram (index: rt [s]); ``mass=None`` gives the TIC.

        Only the requested column of the parquet file is read.
        """
        column = "tic" if mass is TIC else str(mass)
        key = (measurement_id, column)
        if key in self._chromatograms:
            self._chromatograms.move_to_end(key)
            return self._chromatograms[key]
        if column not in self.chromatogram_columns(measurement_id):
            return None

        series = pd.read_parquet(self._chromatogram_path(measurement_id), columns=[column])[column]
        self._chromatograms[key] = series
        if len(self._chromatograms) > self._chromatogram_cache_size:
            self._chromatograms.popitem(last=False)
        return series
