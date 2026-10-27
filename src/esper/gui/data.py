"""Lazy access to an esper data directory for the GUI.

Only the measurement table is read upfront. Compounds, peaks and chromatograms
are queried when a measurement is shown and kept in small caches.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from datetime import datetime
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from esper import db
from esper.integration import integration_bounds, peak_area
from esper.libraries import Spectrum
from esper.targets import interp_extrapolate

TIC = None  # mass value standing for the total ion chromatogram

# Temporary view of the compounds with their identification edited by hand, if any
COMPOUND_VIEW = "compound_named"
# Columns of the best candidate kept with an identification edited by hand (added to the table later)
IDENTIFICATION_SCORES = {
    "spectrum_ref": "TEXT",
    "score": "REAL",
    "score_combined": "REAL",
}
# The best candidate of a compound: the one identified by hand, else alpinac's (cc: compound_candidate)
BEST_CANDIDATE = {
    "best_spectrum": "spectrum_ref",
    "best_score": "score",
    "best_score_combined": "score_combined",
}
BEST_CANDIDATE_SELECT = ", ".join(
    f"CASE WHEN c.identified_at IS NULL THEN cc.{col} ELSE c.identified_{col} END AS {best}"
    for best, col in BEST_CANDIDATE.items()
)


def _now() -> str:
    """Time of an edit by hand."""
    return datetime.now().isoformat(timespec="seconds")


class EsperData:
    def __init__(self, data_dir: Path, chromatogram_cache_size: int = 64) -> None:
        self.data_dir = Path(data_dir)
        self.con = db.connect(self.data_dir, readonly=True)
        self._create_compound_view()
        self._measurements: pd.DataFrame | None = None
        self._chromatograms: OrderedDict = OrderedDict()
        self._chromatogram_cache_size = chromatogram_cache_size

    def close(self) -> None:
        self.con.close()

    def _query(self, sql: str, params=(), **kwargs) -> pd.DataFrame:
        return pd.read_sql(sql, self.con, params=params, **kwargs)

    def _has_table(self, name: str) -> bool:
        sql = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?"
        return self.con.execute(sql, (name,)).fetchone() is not None

    def _create_compound_view(self) -> None:
        """The compounds, with ``compound_name`` and ``valid_candidate`` edited by hand if they were.

        alpinac's values are kept in ``alpinac_name`` and ``alpinac_valid``;
        ``identified_at`` is when the identification was edited (None: by alpinac).
        A compound identified by hand is a valid candidate, unless marked as not identified.
        ``identified_spectrum_ref``, ``identified_score`` and ``identified_score_combined``
        are the best candidate of the name given by hand (see ``BEST_CANDIDATE_SELECT``).
        """
        self.con.execute(f"DROP VIEW IF EXISTS temp.{COMPOUND_VIEW}")
        alpinac = "c.compound_name AS alpinac_name, c.valid_candidate AS alpinac_valid"
        not_edited = ", ".join(
            ["NULL AS identified_at"]
            + [f"NULL AS identified_{col}" for col in IDENTIFICATION_SCORES]
        )
        if not self._has_table("compound_identification"):
            # Database from before the identifications could be edited
            sql = f"SELECT c.*, {alpinac}, {not_edited} FROM compound c"
        else:
            # Table from before the scores were kept with the identification: none is
            kept = {
                row[1]
                for row in self.con.execute(
                    "PRAGMA table_info(compound_identification)"
                )
            }
            scores = ", ".join(
                f"{'ci.' + col if col in kept else 'NULL'} AS identified_{col}"
                for col in IDENTIFICATION_SCORES
            )
            edited = {
                "compound_name": "ci.name",
                "valid_candidate": "ci.name IS NOT NULL",
            }
            # All the columns of the compound table, whatever the schema version
            columns = [
                row[1] for row in self.con.execute("PRAGMA table_info(compound)")
            ]
            select = ", ".join(
                f'{edited[col]} AS "{col}"' if col in edited else f'c."{col}"'
                for col in columns
            )
            # Two parts, so that a search by name in the compounds not edited uses the index on compound_name
            sql = f"""
                SELECT c.*, {alpinac}, {not_edited}
                FROM compound c
                WHERE NOT EXISTS (
                    SELECT 1 FROM compound_identification ci
                    WHERE ci.measurement_id = c.measurement_id AND ci.compound_idx = c.compound_idx
                )
                UNION ALL
                SELECT {select}, {alpinac}, ci.edited_at AS identified_at, {scores}
                FROM compound_identification ci
                CROSS JOIN compound c ON c.measurement_id = ci.measurement_id AND c.compound_idx = ci.compound_idx
            """
        self.con.execute(f"CREATE TEMP VIEW {COMPOUND_VIEW} AS {sql}")

    # --- editing ----------------------------------------------------------------------

    def _write(self, *statements: tuple[str, tuple]) -> None:
        """Write to the database in one transaction, with its own connection (the one of the GUI is read-only).

        ``statements`` are (sql, params) pairs.
        """
        con = db.connect(self.data_dir)
        try:
            # Database from before the identifications could be edited: creates the missing tables
            con.executescript(db.SCHEMA)
            # and columns
            kept = {
                row[1]
                for row in con.execute("PRAGMA table_info(compound_identification)")
            }
            for col, kind in IDENTIFICATION_SCORES.items():
                if col not in kept:
                    con.execute(
                        f"ALTER TABLE compound_identification ADD COLUMN {col} {kind}"
                    )
            for sql, params in statements:
                con.execute(sql, params)
            con.commit()
        finally:
            con.close()
        self._create_compound_view()
        for cached in (
            self.compounds,
            self.peaks,
            self._candidates,
            self.compound_spectrum,
            self.ri_anchors,
            self.targets_found,
        ):
            cached.cache_clear()

    def set_identification(
        self,
        measurement_id: int,
        compound_idx: int,
        name: str | None,
        spectrum_ref: str | None = None,
        score: float | None = None,
        score_combined: float | None = None,
    ) -> None:
        """Identify a compound by hand as ``name`` (a target, alias or any name); None: not identified.

        ``spectrum_ref``, ``score`` and ``score_combined`` describe the library spectrum
        of the name matching the compound best, as for alpinac's candidates (None: none).
        """
        self._write(
            (
                "INSERT OR REPLACE INTO compound_identification"
                " (measurement_id, compound_idx, name, edited_at, spectrum_ref, score, score_combined)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    int(measurement_id),
                    int(compound_idx),
                    name,
                    _now(),
                    spectrum_ref,
                    None if score is None else float(score),
                    None if score_combined is None else float(score_combined),
                ),
            )
        )

    def reset_identification(self, measurement_id: int, compound_idx: int) -> None:
        """Back to alpinac's identification of the compound."""
        self._write(
            (
                "DELETE FROM compound_identification WHERE measurement_id = ? AND compound_idx = ?",
                (int(measurement_id), int(compound_idx)),
            )
        )

    def merge_compounds(self, measurement_id: int, compound_idxs) -> int:
        """Merge compounds of a measurement into the one with the lowest index, for good.

        Their peaks are given to that main compound and they are deleted (with
        their candidates and identification). The main compound keeps its
        identification; its peak summary (n_peaks, prominence_sum, rt) is
        recomputed. Returns the index of the main compound.
        """
        main, *others = sorted({int(i) for i in compound_idxs})
        mid = int(measurement_id)
        peaks = self.compound_peaks(mid, (main, *others))
        rt = peaks["rt_s"].to_numpy(dtype=float)
        ri = float(self.expected_ris(mid, [rt.mean()])[0])
        placeholders = ", ".join("?" * len(others))
        self._write(
            (
                f"UPDATE peak SET compound_idx = ? WHERE measurement_id = ? AND compound_idx IN ({placeholders})",
                (main, mid, *others),
            ),
            (
                "UPDATE compound SET n_peaks = ?, prominence_sum = ?, rt_mean_s = ?, rt_std_s = ?,"
                " retention_index = COALESCE(?, retention_index)"
                " WHERE measurement_id = ? AND compound_idx = ?",
                (
                    len(peaks),
                    float(peaks["prominence"].sum()),
                    float(rt.mean()),
                    float(rt.std()),
                    ri if np.isfinite(ri) else None,
                    mid,
                    main,
                ),
            ),
            (
                f"DELETE FROM compound WHERE measurement_id = ? AND compound_idx IN ({placeholders})",
                (mid, *others),
            ),
            (
                "UPDATE measurement SET n_compounds = n_compounds - ? WHERE measurement_id = ?",
                (len(others), mid),
            ),
        )
        return main

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

    def candidate_names(self, valid_only: bool) -> pd.Series:
        """Number of compounds of all measurements with each main name (aliases resolved)."""
        where = "AND valid_candidate = 1" if valid_only else ""
        # Counted in the compound table (fast with its index), then corrected for the few edited compounds
        df = self._query(
            f"""
            SELECT compound_name, COUNT(*) AS n FROM compound
            WHERE compound_name IS NOT NULL {where}
            GROUP BY compound_name
            """
        )
        if self._has_table("compound_identification"):
            # Minus alpinac's names, plus the names given by hand (always valid)
            edited = self._query(
                f"""
                SELECT c.compound_name, -COUNT(*) AS n
                FROM compound_identification ci
                CROSS JOIN compound c ON c.measurement_id = ci.measurement_id AND c.compound_idx = ci.compound_idx
                WHERE c.compound_name IS NOT NULL {where.replace("valid_candidate", "c.valid_candidate")}
                GROUP BY c.compound_name
                UNION ALL
                SELECT name, COUNT(*) FROM compound_identification WHERE name IS NOT NULL GROUP BY name
                """
            )
            df = pd.concat([df, edited])
            df = df.groupby("compound_name", as_index=False)["n"].sum()
            df = df[df["n"] > 0]
        counts = df.groupby(df["compound_name"].map(self.main_name))["n"].sum()
        return counts.sort_index(key=lambda names: names.str.lower())

    @lru_cache(maxsize=32)
    def compounds(self, measurement_id: int) -> pd.DataFrame:
        """Compounds detected in a measurement with their best candidate, sorted by rt.

        ``compound_name`` is the name of alpinac's best candidate (or the one given
        by hand, see ``set_identification``), which may be an alias of a target;
        ``main_name`` is that target, or the candidate name. ``best_*`` describe
        that candidate (its spectrum and scores).
        """
        df = self._query(
            f"""
            SELECT c.*, {BEST_CANDIDATE_SELECT}
            FROM {COMPOUND_VIEW} c
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
        df["main_name"] = df["compound_name"].map(self.main_name)
        return df

    @lru_cache(maxsize=32)
    def peaks(self, measurement_id: int) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM peak WHERE measurement_id = ? ORDER BY rt_s",
            (int(measurement_id),),
            index_col="peak_idx",
        )

    def compound_peaks(
        self, measurement_id: int, compound_idx: int | tuple[int, ...]
    ) -> pd.DataFrame:
        """The peaks of a compound, or of several compounds together."""
        peaks = self.peaks(measurement_id)
        idxs = compound_idx if isinstance(compound_idx, tuple) else (compound_idx,)
        return peaks[peaks["compound_idx"].isin(idxs)]

    @lru_cache(maxsize=64)
    def compound_spectrum(
        self, measurement_id: int, compound_idx: int | tuple[int, ...], intensity: str
    ) -> Spectrum:
        """Spectrum of a compound (or of several together): unit masses and 'prominence', 'height' or 'area' of its peaks.

        Peaks on the same unit mass are summed. The area is integrated in the
        chromatograms (reads the chromatogram of every mass of the compound).
        """
        peaks = self.compound_peaks(measurement_id, compound_idx)
        if intensity != "area":
            return Spectrum.from_pairs(peaks["mass"], peaks[intensity])
        masses, areas = [], []
        for mass, group in peaks.groupby("mass"):
            series = self.chromatogram(measurement_id, int(mass))
            if series is None:
                continue
            t = series.index.to_numpy(dtype=float)
            y = series.to_numpy(dtype=float)
            start, stop = integration_bounds(group, t)
            for a, b in zip(start, stop):
                masses.append(mass)
                areas.append(max(peak_area(t, y, a, b), 0.0))
        return Spectrum.from_pairs(masses, np.nan_to_num(areas))

    @lru_cache(maxsize=32)
    def _candidates(self, measurement_id: int) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM compound_candidate WHERE measurement_id = ? ORDER BY compound_idx, rank",
            (int(measurement_id),),
        )

    def candidates(self, measurement_id: int, compound_idx: int) -> pd.DataFrame:
        """The library candidates of a compound found by alpinac, by rank."""
        df = self._candidates(measurement_id)
        return df[df["compound_idx"] == compound_idx]

    def candidate_of(self, measurement_id: int, compound_idx: int) -> str | None:
        """Main name of alpinac's best candidate for a compound, None if it has none."""
        name = self.compounds(measurement_id).loc[compound_idx, "main_name"]
        return name if isinstance(name, str) else None

    def main_ion(self, measurement_id: int, compound_idx: int) -> int | None:
        """Mass of the most prominent peak of the compound."""
        peaks = self.compound_peaks(measurement_id, compound_idx)
        if peaks.empty:
            return None
        return int(peaks.loc[peaks["prominence"].idxmax(), "mass"])

    def find_compound(
        self, measurement_id: int, name: str | None, near_rt: float | None = None
    ) -> int | None:
        """Compound of the measurement with that main name (its candidate or one of its aliases).

        Valid candidates are preferred, then the one closest to ``near_rt``,
        or the most intense one.
        """
        if name is None:
            return None
        compounds = self.compounds(measurement_id)
        matching = compounds[compounds["main_name"] == name]
        if matching.empty:
            return None
        valid = matching[matching["valid_candidate"] == 1]
        if not valid.empty:
            matching = valid
        if near_rt is not None:
            return int((matching["rt_mean_s"] - near_rt).abs().idxmin())
        return int(matching["prominence_sum"].idxmax())

    # --- targets ----------------------------------------------------------------------

    def _query_or_empty(self, sql: str, columns: list[str], **kwargs) -> pd.DataFrame:
        """Query a table that may not exist (database without imported targets)."""
        try:
            return self._query(sql, **kwargs)
        except (pd.errors.DatabaseError, sqlite3.Error):
            return pd.DataFrame(columns=columns)

    @cached_property
    def targets(self) -> pd.DataFrame:
        """Target compounds, indexed by name, with their expected unit masses (``masses``)."""
        targets = self._query_or_empty(
            "SELECT * FROM target", ["name", "rt_ref_s", "ri"]
        ).set_index("name")
        ions = self.target_ions_all
        masses = ions.groupby("name")["mass"].agg(
            lambda m: list(dict.fromkeys(int(x) for x in m))
        )
        targets["masses"] = masses.reindex(targets.index)
        targets["masses"] = targets["masses"].apply(
            lambda m: m if isinstance(m, list) else []
        )
        return targets

    @cached_property
    def target_ions_all(self) -> pd.DataFrame:
        return self._query_or_empty(
            "SELECT * FROM target_ion ORDER BY name, rank",
            ["name", "rank", "ion", "exact_mass", "mass"],
        )

    @cached_property
    def _target_by_name(self) -> dict[str, str]:
        """Target name or alias -> target name."""
        aliases = self._query_or_empty(
            "SELECT alias, name FROM target_alias", ["alias", "name"]
        )
        mapping = dict(zip(aliases["alias"], aliases["name"]))
        mapping.update({name: name for name in self.targets.index})
        return mapping

    def target_of(self, name: str | None) -> str | None:
        """The target that a candidate name (a target name or alias) stands for."""
        return self._target_by_name.get(name) if name is not None else None

    def target_names(self, target: str) -> list[str]:
        """The target name and its aliases."""
        return [n for n, t in self._target_by_name.items() if t == target]

    def main_name(self, name: str | None) -> str | None:
        """The target that a candidate name stands for (aliases resolved), else the name itself."""
        if not isinstance(name, str):
            return None
        return self._target_by_name.get(name, name)

    def names_of(self, main_name: str) -> list[str]:
        """The candidate names with that main name: a target and its aliases, or the name itself."""
        return self.target_names(main_name) or [main_name]

    def target_masses(self, target: str) -> list[int]:
        """Expected unit masses of a target, main ion first."""
        if target not in self.targets.index:
            return []
        return self.targets.at[target, "masses"]

    @lru_cache(maxsize=32)
    def ri_anchors(self, measurement_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(retention index, rt) of the reference compounds of a measurement, by increasing rt."""
        compounds = self.compounds(measurement_id)
        refs = compounds[compounds["is_ref_ri"] == 1].dropna(subset=["retention_index"])
        refs = refs.sort_values("rt_mean_s")
        # Keep a strictly increasing retention index
        keep = refs["retention_index"] > refs["retention_index"].cummax().shift(
            fill_value=-np.inf
        )
        refs = refs[keep]
        return refs["retention_index"].to_numpy(dtype=float), refs[
            "rt_mean_s"
        ].to_numpy(dtype=float)

    def expected_rts(self, measurement_id: int, ri) -> np.ndarray:
        """rt expected in a measurement for retention indexes (NaN if not possible)."""
        ri = np.asarray(ri, dtype=float)
        anchors_ri, anchors_rt = self.ri_anchors(measurement_id)
        if len(anchors_ri) < 2:
            return np.full(ri.shape, np.nan)
        return interp_extrapolate(ri, anchors_ri, anchors_rt)

    def expected_ris(self, measurement_id: int, rt) -> np.ndarray:
        """Retention indexes of rts in a measurement (NaN if not possible)."""
        rt = np.asarray(rt, dtype=float)
        anchors_ri, anchors_rt = self.ri_anchors(measurement_id)
        if len(anchors_ri) < 2:
            return np.full(rt.shape, np.nan)
        return interp_extrapolate(rt, anchors_rt, anchors_ri)

    def expected_rt(self, measurement_id: int, target: str) -> float | None:
        """rt where the target should elute in the measurement, from its retention index."""
        if target not in self.targets.index:
            return None
        rt = float(
            self.expected_rts(measurement_id, [self.targets.at[target, "ri"]])[0]
        )
        return rt if np.isfinite(rt) else None

    def find_target(self, measurement_id: int, target: str) -> int | None:
        """Compound of the measurement identified as the target (or one of its aliases)."""
        return self.find_compound(
            measurement_id, target, self.expected_rt(measurement_id, target)
        )

    @lru_cache(maxsize=32)
    def targets_found(self, measurement_id: int) -> frozenset[str]:
        """Targets that have a compound in the measurement."""
        names = self.compounds(measurement_id)["compound_name"].dropna().unique()
        return frozenset(t for t in (self.target_of(n) for n in names) if t is not None)

    # --- across measurements ------------------------------------------------------------

    def results(self, name: str, valid_only: bool) -> pd.DataFrame:
        """All compounds with that main name (a target found under any of its aliases), with peak summaries."""
        where = "AND c.valid_candidate = 1" if valid_only else ""
        candidates = self.names_of(name)
        placeholders = ", ".join("?" * len(candidates))
        df = self._query(
            f"""
            SELECT
                c.measurement_id, c.compound_idx, c.rt_mean_s, c.retention_index,
                c.compound_name, c.valid_candidate,
                COUNT(p.peak_idx) AS n_peaks,
                SUM(p.prominence) AS prominence_sum,
                MAX(p.height) AS max_height,
                MAX(p.prominence) AS max_prominence,
                {BEST_CANDIDATE_SELECT}
            FROM {COMPOUND_VIEW} c
            JOIN peak p
                ON p.measurement_id = c.measurement_id AND p.compound_idx = c.compound_idx
            LEFT JOIN compound_candidate cc
                ON cc.measurement_id = c.measurement_id
                AND cc.compound_idx = c.compound_idx
                AND cc.is_best = 1
            WHERE c.compound_name IN ({placeholders}) {where}
            GROUP BY c.measurement_id, c.compound_idx
            """,
            tuple(candidates),
        )
        meas = self.measurements[
            ["name", "acq_datetime", "sample_type", "sample_number"]
        ]
        df = df.join(meas, on="measurement_id")
        df["measurement_pos"] = self.measurements.index.get_indexer(
            df["measurement_id"]
        )
        return df

    def search(
        self, masses: list[int], window: str, lo: float, hi: float
    ) -> pd.DataFrame:
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
        # Not joined with COMPOUND_VIEW, which SQLite would read whole: the names given by hand afterwards
        if self._has_table("compound_identification") and len(df):
            edited = self._query(
                "SELECT measurement_id, compound_idx, name FROM compound_identification"
            ).set_index(["measurement_id", "compound_idx"])["name"]
            keys = pd.MultiIndex.from_frame(df[["measurement_id", "compound_idx"]])
            hit = keys.isin(edited.index)
            names = edited.reindex(keys[hit])
            df.loc[hit, "compound_name"] = names.to_numpy()
            df.loc[hit, "valid_candidate"] = names.notna().astype(int).to_numpy()
        df["main_name"] = df["compound_name"].map(self.main_name)
        meas = self.measurements[
            ["name", "acq_datetime", "sample_type", "sample_number"]
        ]
        df = df.join(meas, on="measurement_id")
        df["measurement_pos"] = self.measurements.index.get_indexer(
            df["measurement_id"]
        )
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

        series = pd.read_parquet(
            self._chromatogram_path(measurement_id), columns=[column]
        )[column]
        self._chromatograms[key] = series
        if len(self._chromatograms) > self._chromatogram_cache_size:
            self._chromatograms.popitem(last=False)
        return series
