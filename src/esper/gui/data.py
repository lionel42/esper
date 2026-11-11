"""Lazy access to an esper data directory for the GUI.

Only the measurement table is read upfront. Compounds, peaks and chromatograms
are queried when a measurement is shown and kept in small caches.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from functools import cached_property, lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from esper import db
from esper.fitting import FIT_COLUMNS, SOURCE_EXTRACTION
from esper.gui.chromatogram_store import (
    ChromatogramStore,
    column_mass,
    mass_column,
    masses_of_unit,
    unit_masses,
)
from esper.gui.target_index import TargetIndex
from esper.integration import has_ips, integration_bounds, peak_area
from esper.libraries import Spectrum
from esper.targets import interp_extrapolate

TIC = None  # mass value standing for the total ion chromatogram

# Temporary view of the compounds with their identification edited by hand, if any
COMPOUND_VIEW = "compound_named"
# Columns of the best candidate kept with an identification edited by hand (added to the
# table later: db.ADDED_COLUMNS)
IDENTIFICATION_SCORES = ["spectrum_ref", "score", "score_combined"]
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
# The compounds (c, COMPOUND_VIEW) with alpinac's best candidate (cc), for BEST_CANDIDATE_SELECT
COMPOUNDS_WITH_BEST = f"""
    {COMPOUND_VIEW} c
    LEFT JOIN compound_candidate cc
        ON cc.measurement_id = c.measurement_id
        AND cc.compound_idx = c.compound_idx
        AND cc.is_best = 1
"""
# Columns of the measurements added to the rows of the queries across measurements
MEASUREMENT_COLUMNS = ["name", "acq_datetime", "sample_type", "sample_number"]


class _cached_method:
    """``functools.lru_cache`` of a method, one cache per instance.

    ``lru_cache`` on a method is shared by all the instances and keeps them alive (with
    their data frames) after the data directory is closed; its ``cache_clear`` clears the
    caches of all of them. The cache is made at the first access, stored on the instance.
    """

    def __init__(self, maxsize: int | None) -> None:
        self.maxsize = maxsize

    def __call__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        return self

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        cached = lru_cache(maxsize=self.maxsize)(partial(self.func, obj))
        obj.__dict__[self.name] = cached  # found before this (non-data) descriptor
        return cached


def _now() -> str:
    """Time of an edit by hand."""
    return datetime.now().isoformat(timespec="seconds")


class EsperData:
    def __init__(self, data_dir: Path, chromatogram_cache_size: int = 64) -> None:
        self.data_dir = Path(data_dir)
        self.con = db.connect(self.data_dir, readonly=True)
        self._create_compound_view()
        self._measurements: pd.DataFrame | None = None
        self.chromatograms = ChromatogramStore(
            self._chromatogram_path, chromatogram_cache_size
        )

    def close(self) -> None:
        self.con.close()
        self.invalidate()
        self.chromatograms.clear()

    def invalidate(self) -> None:
        """Forget what was read from the database (after a write): the measurements and
        the caches of the methods (the chromatograms are kept)."""
        self._measurements = None
        for name, attr in vars(type(self)).items():
            if isinstance(attr, _cached_method):
                self.__dict__.pop(name, None)

    def _query(self, sql: str, params=(), **kwargs) -> pd.DataFrame:
        return pd.read_sql(sql, self.con, params=params, **kwargs)

    def _numeric(self, df: pd.DataFrame, table: str, *extra: str) -> pd.DataFrame:
        """The columns of ``df`` declared INTEGER or REAL in ``table`` (and ``extra``) as numbers.

        A column all NULL (e.g. no retention index in a measurement) is read as objects.
        """
        numeric = [
            name
            for name, kind in self._table_info(table)
            if kind.split(" ")[0].upper() in ("INTEGER", "REAL")
        ]
        columns = [c for c in (*numeric, *extra) if c in df.columns]
        df[columns] = df[columns].apply(pd.to_numeric)
        return df

    @_cached_method(maxsize=None)
    def _table_info(self, table: str) -> tuple[tuple[str, str], ...]:
        """(name, declared type) of the columns of a table (none if there is no table)."""
        return tuple(
            (row[1], row[2]) for row in self.con.execute(f"PRAGMA table_info({table})")
        )

    def _table_columns(self, table: str) -> list[str]:
        return [name for name, _ in self._table_info(table)]

    def _has_table(self, name: str) -> bool:
        return bool(self._table_info(name))

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
            kept = set(self._table_columns("compound_identification"))
            scores = ", ".join(
                f"{'ci.' + col if col in kept else 'NULL'} AS identified_{col}"
                for col in IDENTIFICATION_SCORES
            )
            edited = {
                "compound_name": "ci.name",
                "valid_candidate": "ci.name IS NOT NULL",
            }
            # All the columns of the compound table, whatever the schema version
            columns = self._table_columns("compound")
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
        # create: a database from before the identifications could be edited gets the
        # missing tables and columns
        con = db.connect(self.data_dir, create=True)
        try:
            for sql, params in statements:
                con.execute(sql, params)
            con.commit()
        finally:
            con.close()
        self.invalidate()
        self._create_compound_view()

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
            df = self._numeric(df, "measurement", "mass_cal_mode", "s_per_bin")
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
        counts = df.groupby(df["compound_name"].map(self.target_index.main_name))[
            "n"
        ].sum()
        return counts.sort_index(key=lambda names: names.str.lower())

    @_cached_method(maxsize=32)
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
            FROM {COMPOUNDS_WITH_BEST}
            WHERE c.measurement_id = ?
            ORDER BY c.rt_mean_s
            """,
            (int(measurement_id),),
            index_col="compound_idx",
        )
        df = self._numeric(
            df,
            "compound",
            "alpinac_valid",
            "best_score",
            "best_score_combined",
            "identified_score",
            "identified_score_combined",
        )
        df["main_name"] = df["compound_name"].map(self.target_index.main_name)
        return df

    @_cached_method(maxsize=32)
    def peaks(self, measurement_id: int) -> pd.DataFrame:
        """Peaks of a measurement, with the fit columns (NaN if not fitted).

        ``compound_idx`` is NA for a peak of the fit that matches no compound.
        """
        df = self._query(
            "SELECT * FROM peak WHERE measurement_id = ? ORDER BY rt_s",
            (int(measurement_id),),
            index_col="peak_idx",
        )
        df["compound_idx"] = df["compound_idx"].astype("Int64")
        # Database from before the fit
        if "source" not in df:
            df["source"] = SOURCE_EXTRACTION
        for column in FIT_COLUMNS:
            if column not in df:
                df[column] = np.nan
        # A column all NULL (e.g. the extraction columns when the fit split every peak)
        # is read as objects
        numeric = [
            c for c in df.columns if c not in ("source", "compound_idx", "fit_shape")
        ]
        df[numeric] = df[numeric].apply(pd.to_numeric)
        return df

    def is_fitted(self, measurement_id: int) -> bool:
        """The peaks of the measurement were fitted (esper.fitting)."""
        measurements = self.measurements
        return "fit_version" in measurements and pd.notna(
            measurements.at[measurement_id, "fit_version"]
        )

    def compound_peaks(
        self, measurement_id: int, compound_idx: int | tuple[int, ...]
    ) -> pd.DataFrame:
        """The peaks of a compound, or of several compounds together."""
        peaks = self.peaks(measurement_id)
        idxs = compound_idx if isinstance(compound_idx, tuple) else (compound_idx,)
        return peaks[peaks["compound_idx"].isin(idxs)]

    def compound_masses(
        self, measurement_id: int, compound_idx: int | tuple[int, ...]
    ) -> list[int]:
        """Masses of a compound (or of several together), by decreasing prominence of
        their largest peak."""
        peaks = self.compound_peaks(measurement_id, compound_idx)
        peaks = peaks.sort_values("prominence", ascending=False)
        return [column_mass(f"{m:g}") for m in peaks["mass"].drop_duplicates()]

    @_cached_method(maxsize=64)
    def compound_spectrum(
        self, measurement_id: int, compound_idx: int | tuple[int, ...], intensity: str
    ) -> Spectrum:
        """Spectrum of a compound (or of several together): unit masses and 'prominence', 'height', 'area' or 'fit_area' of its peaks.

        Peaks on the same unit mass are summed. The area is integrated in the
        chromatograms between the bounds from the ips (reads the chromatogram of
        every mass of the compound; peaks split by the fit have none and are left
        out); 'fit_area' is the area of the fitted peaks (unfitted peaks left out).
        """
        peaks = self.compound_peaks(measurement_id, compound_idx)
        if intensity == "fit_area":
            fitted = peaks[peaks["fit_area"].notna()]
            return Spectrum.from_pairs(fitted["mass"], fitted["fit_area"])
        if intensity != "area":
            return Spectrum.from_pairs(peaks["mass"], peaks[intensity])
        peaks = peaks[has_ips(peaks)]
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

    @_cached_method(maxsize=32)
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
        """Main ion of a compound: of its target (``target_main_ion``) if it has one, else
        the mass of its most prominent peak."""
        name = self.compounds(measurement_id).at[compound_idx, "main_name"]
        main = self.target_index.target_main_ion(name)
        if main is not None:
            return main
        peaks = self.compound_peaks(measurement_id, compound_idx)
        if peaks.empty:
            return None
        return int(peaks.loc[peaks["prominence"].idxmax(), "mass"])

    def main_ions(self, measurement_id: int) -> pd.Series:
        """``main_ion`` of all the compounds of a measurement (Int64, NA without peaks)."""
        compounds = self.compounds(measurement_id)
        main = (
            compounds["main_name"]
            .map(self.target_index.target_main_ion)
            .astype("Int64")
        )
        peaks = self.peaks(measurement_id).dropna(subset=["compound_idx"])
        largest = peaks.loc[
            peaks.groupby("compound_idx")["prominence"].idxmax().dropna()
        ]
        largest = largest.set_index("compound_idx")["mass"].astype("Int64")
        return main.fillna(largest.reindex(compounds.index))

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
    def target_index(self) -> TargetIndex:
        """The targets and their aliases (empty if none were imported)."""
        targets = self._query_or_empty("SELECT * FROM target", ["name", "rt_ref_s", "ri"])
        if self._has_table("target"):  # numbers also when all NULL (GCwerks: no RI)
            targets = self._numeric(targets, "target")
        return TargetIndex(
            targets.set_index("name"),
            self._query_or_empty(
                "SELECT * FROM target_ion ORDER BY name, rank",
                ["name", "rank", "ion", "exact_mass", "mass", "spectrum_ref"],
            ),
            self._query_or_empty(
                "SELECT alias, name FROM target_alias", ["alias", "name"]
            ),
            self._query_or_empty(
                "SELECT * FROM target_rt", ["name", "valid_from", "rt_s", "window_s"]
            ),
        )

    @property
    def targets(self) -> pd.DataFrame:
        """Target compounds, indexed by name, with their expected unit masses (``masses``)."""
        return self.target_index.table

    @_cached_method(maxsize=32)
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

    def measurement_time(self, measurement_id: int | None):
        """When the run was made (from its name), to find the version of the targets that
        applies to it (``target_rt``, ``target_ion``); None: unknown."""
        if measurement_id is None or measurement_id not in self.measurements.index:
            return None
        time = self.measurements.at[measurement_id, "acq_datetime"]
        return None if pd.isna(time) else time

    def target_masses(self, measurement_id: int | None, target: str | None) -> list[int]:
        """Expected unit masses of a target in a measurement (the version of its time; the
        latest without measurement), main ion first."""
        return self.target_index.masses_at(target, self.measurement_time(measurement_id))

    def target_window(self, measurement_id: int, target: str) -> tuple[float, float] | None:
        """(start, stop) of the identification window of the target in the measurement
        (GCwerks peakid), None if it has none."""
        found = self.target_index.rt_at(target, self.measurement_time(measurement_id))
        if found is None or not np.isfinite(found[1]):
            return None
        rt, window = found
        return rt - window / 2, rt + window / 2

    def expected_rt(self, measurement_id: int, target: str) -> float | None:
        """rt where the target should elute in the measurement: given for the time of the
        measurement (GCwerks peakid), else from its retention index."""
        if target not in self.targets.index:
            return None
        if self.target_index.has_rts(target):
            found = self.target_index.rt_at(target, self.measurement_time(measurement_id))
            return None if found is None else found[0]
        rt = float(
            self.expected_rts(measurement_id, [self.targets.at[target, "ri"]])[0]
        )
        return rt if np.isfinite(rt) else None

    def expected_target_rts(self, measurement_id: int) -> np.ndarray:
        """``expected_rt`` of all the targets (``targets`` order), NaN if unknown."""
        targets = self.targets
        rts = self.expected_rts(measurement_id, targets["ri"].to_numpy(dtype=float))
        time = self.measurement_time(measurement_id)
        for i, target in enumerate(targets.index):
            if self.target_index.has_rts(target):
                found = self.target_index.rt_at(target, time)
                rts[i] = np.nan if found is None else found[0]
        return rts

    def find_target(self, measurement_id: int, target: str) -> int | None:
        """Compound of the measurement identified as the target (or one of its aliases)."""
        return self.find_compound(
            measurement_id, target, self.expected_rt(measurement_id, target)
        )

    @_cached_method(maxsize=32)
    def targets_found(self, measurement_id: int) -> frozenset[str]:
        """Targets that have a compound in the measurement."""
        names = self.compounds(measurement_id)["compound_name"].dropna().unique()
        return frozenset(
            t for t in (self.target_index.target_of(n) for n in names) if t is not None
        )

    # --- across measurements ------------------------------------------------------------

    def _with_measurement(self, df: pd.DataFrame, how: str = "left") -> pd.DataFrame:
        """Rows with a measurement_id, with the ``MEASUREMENT_COLUMNS`` of their
        measurement and its position in the measurements (``measurement_pos``)."""
        df = df.join(
            self.measurements[MEASUREMENT_COLUMNS], on="measurement_id", how=how
        )
        df["measurement_pos"] = self.measurements.index.get_indexer(
            df["measurement_id"]
        )
        return df

    def results(self, name: str, valid_only: bool) -> pd.DataFrame:
        """All compounds with that main name (a target found under any of its aliases), with peak summaries.

        See ``_peak_summaries``; the main ion is the target's (``target_main_ion``).
        """
        where = "AND c.valid_candidate = 1" if valid_only else ""
        candidates = self.target_index.names_of(name)
        placeholders = ", ".join("?" * len(candidates))
        df = self._query(
            f"""
            SELECT
                c.measurement_id, c.compound_idx, c.rt_mean_s, c.retention_index,
                c.compound_name, c.valid_candidate,
                {BEST_CANDIDATE_SELECT}
            FROM {COMPOUNDS_WITH_BEST}
            WHERE c.compound_name IN ({placeholders}) {where}
            """,
            tuple(candidates),
        )
        df = df.merge(
            self._peak_summaries(
                candidates, where, self.target_index.target_main_ion(name)
            ),
            on=["measurement_id", "compound_idx"],
        )
        return self._with_measurement(df)

    @cached_property
    def mass_cal_points(self) -> pd.DataFrame:
        """Points of the mass calibration of all measurements (``mass_cal_rt``, see
        ``esper.mass_cal``), with their measurement and calibration mode, in time order.

        ``time``: absolute time of the point (datetime, UTC); for points stored by an older esper,
        from the measurement name (to the minute) + rt. ``ok`` 0: rejected rt window.
        """
        columns = set(self._table_columns("mass_cal_rt"))
        optional = ["time", "ok", "rt_start_s", "rt_stop_s", "n_calibrants", "n_used"]
        optional += ["rms_ppm", "max_ppm"]
        select = ", ".join(
            f"r.{c}" if c in columns else f"NULL AS {c}" for c in optional
        )
        meas_columns = set(self._table_columns("measurement"))
        start = "m.acq_start" if "acq_start" in meas_columns else "NULL"
        df = self._query(
            f"""
            SELECT
                r.measurement_id, r.rt_s, r.p1, r.p2, r.p3, r.p4, r.p5, {select},
                {start} AS acq_start, mc.mode, mc.n_params,
                mc.sigma_tofidx_slope, mc.sigma_tofidx_b, mc.alpha_pseudo_voigt
            FROM mass_cal_rt r
            JOIN mass_cal mc USING (measurement_id)
            JOIN measurement m USING (measurement_id)
            """
        )
        df = self._with_measurement(df, how="inner")
        start = pd.to_datetime(df["acq_start"]).fillna(df["acq_datetime"])
        computed = start + pd.to_timedelta(df["rt_s"], unit="s")
        df["time"] = pd.to_datetime(df["time"]).fillna(computed)
        df["ok"] = df["ok"].fillna(1).astype(bool)
        return df.sort_values(["time", "rt_s"], ignore_index=True)

    @_cached_method(maxsize=16)
    def mass_cal_run(self, measurement_id: int):
        """Mass calibration of a measurement as a ``masscal.MassCalibration`` (the points used),
        and its calibrant peaks (``mass_cal_peak``, empty if not stored); (None, peaks) without
        calibration."""
        from masscal import MassCalibration, Mode

        points = self.mass_cal_points
        points = points[(points["measurement_id"] == measurement_id) & points["ok"]]
        peaks = pd.DataFrame()
        if self._has_table("mass_cal_peak"):
            peaks = self._query(
                "SELECT * FROM mass_cal_peak WHERE measurement_id = ? ORDER BY mz_exact, rt_s",
                (measurement_id,),
            )
            peaks["used"] = peaks["used"].astype(bool)
        if points.empty:
            return None, peaks
        n = points["n_params"].iloc[0]
        if pd.isna(n):
            n = points[["p1", "p2", "p3", "p4", "p5"]].notna().sum(axis=1).max()
        n = int(n)
        cal = MassCalibration(
            mode=Mode.from_tofwerk(
                int(points["mode"].iloc[0])
            ),  # stored: TOFWERK number
            rt=points["rt_s"].to_numpy(dtype=float),
            params=points[[f"p{k + 1}" for k in range(n)]].to_numpy(dtype=float),
        )
        return cal, peaks

    def _peak_summaries(
        self, candidates: list[str], where: str, main_ion: int | None
    ) -> pd.DataFrame:
        """Peaks of the compounds of ``results`` (the compounds without peaks left out).

        ``n_peaks``, ``prominence_sum``, ``fit_area_sum`` and ``n_fitted`` (the fitted
        peaks), and the ``main_*`` columns of the main peak: the most prominent peak on
        the ``main_ion`` (NaN if the compound has none), or the most prominent peak if
        no main ion is given; ``main_fit_area_rel_u`` is its relative area uncertainty.
        """
        main_columns = ["mass", "height", "prominence", *FIT_COLUMNS]
        keys = ["measurement_id", "compound_idx"]
        fitted_db = "fit_height" in self._table_columns("peak")
        selected = main_columns if fitted_db else main_columns[:3]
        placeholders = ", ".join("?" * len(candidates))
        peaks = self._query(
            f"""
            SELECT p.measurement_id, p.compound_idx, {", ".join(f"p.{c}" for c in selected)}
            FROM {COMPOUND_VIEW} c
            JOIN peak p
                ON p.measurement_id = c.measurement_id AND p.compound_idx = c.compound_idx
            WHERE c.compound_name IN ({placeholders}) {where}
            """,
            tuple(candidates),
        ).reindex(columns=keys + main_columns)
        grouped = peaks.groupby(keys)
        summary = pd.DataFrame(
            {
                "n_peaks": grouped.size(),
                "prominence_sum": grouped["prominence"].sum(),
                "fit_area_sum": grouped["fit_area"].sum(min_count=1),
                "n_fitted": grouped["fit_area"].count(),
            }
        )
        candidates_main = (
            peaks if main_ion is None else peaks[peaks["mass"] == main_ion]
        )
        main = (
            candidates_main.sort_values("prominence", ascending=False)
            .drop_duplicates(keys)
            .set_index(keys)[main_columns]
            .add_prefix("main_")
        )
        df = summary.join(main).reset_index()
        df["main_fit_area_rel_u"] = df["main_fit_area_u"] / df["main_fit_area"]
        return df

    def search(
        self, masses: list[int], window: str, lo: float, hi: float
    ) -> pd.DataFrame:
        """Peaks of the given masses within an rt [s] or retention index window.

        The retention index of a peak is the one of its compound (``compound_idx`` NA:
        a peak of the fit without compound, found by rt only).
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
            LEFT JOIN compound c
                ON c.measurement_id = p.measurement_id AND c.compound_idx = p.compound_idx
            WHERE p.mass IN ({placeholders}) AND {column} BETWEEN ? AND ?
            """,
            (*[int(m) for m in masses], float(lo), float(hi)),
        )
        df["compound_idx"] = df["compound_idx"].astype("Int64")
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
        df["main_name"] = df["compound_name"].map(self.target_index.main_name)
        return self._with_measurement(df)

    def _chromatogram_path(self, measurement_id: int) -> Path | None:
        file = self.measurements.loc[measurement_id, "chromatogram_file"]
        if not isinstance(file, str):
            return None
        return self.data_dir / file

    def chromatogram_columns(self, measurement_id: int) -> frozenset[str]:
        return self.chromatograms.columns(measurement_id)

    def chromatogram_masses(self, measurement_id: int) -> list[int | float]:
        """The masses of the chromatograms of a measurement, as measured (GCwerks: ``69.5``,
        the scans ``69.2`` ...), by increasing mass."""
        columns = self.chromatograms.columns(measurement_id)
        return sorted(m for m in map(column_mass, columns) if m is not None)

    def unit_masses(self, measurement_id: int) -> list[int]:
        """The unit masses of the chromatograms of a measurement (``masses_of_unit``)."""
        return unit_masses(self.chromatogram_masses(measurement_id))

    def measured_masses(self, measurement_id: int, mass: int | float | None) -> list:
        """The chromatograms a mass stands for: a unit mass, the masses measured within
        +- 0.5 (``masses_of_unit``); a mass as measured, itself; the TIC, itself."""
        if mass is TIC:
            return [TIC]
        masses = self.chromatogram_masses(measurement_id)
        if mass in masses and not float(mass).is_integer():
            return [mass]
        return masses_of_unit(masses, mass)

    def chromatogram_frame(self, measurement_id: int) -> pd.DataFrame | None:
        """All unit mass chromatograms of a measurement (whole parquet file)."""
        return self.chromatograms.frame(measurement_id)

    def chromatogram(self, measurement_id: int, mass: int | None) -> pd.Series | None:
        """Unit mass chromatogram (index: rt [s]); ``mass=None`` gives the TIC.

        Only the requested column of the parquet file is read.
        """
        return self.chromatograms.series(
            measurement_id, "tic" if mass is TIC else mass_column(mass)
        )

    def chromatogram_drawn(self, measurement_id: int, mass: int | None) -> pd.Series | None:
        """``chromatogram`` to draw: one NaN in each gap where the mass was not measured
        (``chromatogram_store.with_breaks``)."""
        return self.chromatograms.drawn(
            measurement_id, "tic" if mass is TIC else mass_column(mass)
        )
