"""Retention indices: the references of a data directory and the RI of the compounds of a run.

The references (table ``ri_reference``, ``esper ri-refs``) are compounds of known RI that
elute in every run: Kr, Xe, the n-alkanes, ... Each one is a target (its ``rt_ref_s`` in
the reference run gives the RIs of the targets: ``target_ris``); ``names``: the other
candidate names that are it (library spectra: ``C4H10-Butane`` for n-butane; the aliases
of the target count too). The references of the same names (Xe-A and Xe-B: the two peaks
of Xe; mp- and o-xylene) have an ``occurrence``.

In a run (``find_references``, as the legacy ``calculate_retention_index``): the
references by increasing RI, each the compound after the previous one whose best
candidate is one of its names with a score above ``RetentionIndexConfig.min_score``: the
first such compound for a reference with an occurrence, else the largest. The RI of the
compounds is then interpolated in rt between the references found, extrapolated beyond
the first and last (``rt_to_ri``; the legacy clamped).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from esper import db

logger = logging.getLogger(__name__)

REFERENCE_COLUMNS = ["name", "ri", "names", "occurrence", "enabled", "note"]
# Separator of ri_reference.names
SEPARATOR = ";"
FOUND_COLUMNS = ["name", "compound_idx", "rt_s", "score", "ri"]

# The legacy reference list (rt_refs_v2.csv: compound, ri) has no names: those of its
# hard-coded aliases (halosearch rt_index_finder.aliases) and double calibrants
LEGACY_NAMES = {
    "Kr": ["Kr+2only"],
    "ethane": ["ethane28nudged"],
    "propane": ["C3H8-Propane"],
    "n-butane": ["butane", "C4H10-Butane"],
    "n-pentane": ["pentane", "C5H12-Pentane"],
    "n-hexane": ["hexane"],
    "n-heptane": ["heptane"],
    "ethylbenzene": ["ethylbenz"],
}
_XE = ["Xe", "Xe-A", "Xe-B"]
_XYLENES = [
    "xylene",
    "m-xylene",
    "p-xylene",
    "mp-xylene",
    "o-xylene",
    "xylene-A",
    "xylene-B",
]
LEGACY_OCCURRENCES = {
    "Xe-A": (1, _XE),
    "Xe-B": (2, _XE),
    "mp-xylene": (1, _XYLENES),
    "o-xylene": (2, _XYLENES),
}


@dataclass(frozen=True)
class RetentionIndexConfig:
    """The references of a run (``processing: ri:`` of the esper.yaml)."""

    #: spectral score of the best candidate of a compound to make it a reference
    #: (legacy 0.95)
    min_score: float = 0.95


# --- the references ---------------------------------------------------------------------


def default_references() -> Path:
    """The seed of the references (the legacy ``rt_refs_v2.csv``, with its names)."""
    return Path(__file__).parent / "resources" / "ri_references.csv"


def split_names(names) -> list[str]:
    """The names of ``ri_reference.names`` (``SEPARATOR``)."""
    if not isinstance(names, str):
        return []
    return [n.strip() for n in names.split(SEPARATOR) if n.strip()]


def _normalised(refs: pd.DataFrame) -> pd.DataFrame:
    refs = refs.reindex(columns=REFERENCE_COLUMNS).copy()
    refs["name"] = refs["name"].astype(str).str.strip()
    refs["ri"] = pd.to_numeric(refs["ri"], errors="raise").astype(float)
    refs["names"] = refs["names"].map(
        lambda v: f"{SEPARATOR} ".join(split_names(v)) or None
    )
    refs["occurrence"] = pd.to_numeric(refs["occurrence"]).astype("Int64")
    refs["enabled"] = refs["enabled"].fillna(1).astype(int)
    refs["note"] = refs["note"].where(refs["note"].notna(), None)
    if refs["name"].duplicated().any():
        raise ValueError(
            f"references given twice: {sorted(set(refs.loc[refs['name'].duplicated(), 'name']))}"
        )
    return refs.sort_values("ri", kind="stable", ignore_index=True)


def read_reference_csv(path: Path) -> pd.DataFrame:
    """References of a CSV file (``REFERENCE_COLUMNS``, ``#`` comments), or of the legacy
    format (``compound,ri``: its commented lines left out, ``LEGACY_NAMES`` and
    ``LEGACY_OCCURRENCES`` added)."""
    refs = pd.read_csv(path, comment="#", dtype={"names": str, "note": str})
    if "compound" in refs.columns and "name" not in refs.columns:
        refs = refs.rename(columns={"compound": "name"})
        occurrence = {n: o for n, (o, _) in LEGACY_OCCURRENCES.items()}
        names = {
            n: [m for m in ns if m != n] for n, (_, ns) in LEGACY_OCCURRENCES.items()
        }
        names |= LEGACY_NAMES
        refs["names"] = refs["name"].map(
            lambda n: f"{SEPARATOR} ".join(names.get(n, []))
        )
        refs["occurrence"] = refs["name"].map(occurrence)
    missing = {"name", "ri"} - set(refs.columns)
    if missing:
        raise ValueError(f"{path}: no column {', '.join(sorted(missing))}")
    return _normalised(refs)


def write_reference_csv(refs: pd.DataFrame, path: Path) -> None:
    """The references as a CSV file (``read_reference_csv`` reads it again)."""
    _normalised(refs).to_csv(path, index=False)


def read_references(con: sqlite3.Connection, enabled: bool = True) -> pd.DataFrame:
    """The references of the database by increasing RI (only the enabled ones if
    ``enabled``); empty without the table."""
    if not db.table_columns(con, "ri_reference"):
        return pd.DataFrame(columns=REFERENCE_COLUMNS)
    refs = pd.read_sql(
        "SELECT * FROM ri_reference"
        + (" WHERE enabled = 1" if enabled else "")
        + " ORDER BY ri, name",
        con,
    )
    return _normalised(refs) if len(refs) else pd.DataFrame(columns=REFERENCE_COLUMNS)


def store_references(con: sqlite3.Connection, refs: pd.DataFrame) -> None:
    """Replace the references of the database, and the RIs of the targets
    (``update_target_ris``). Single transaction."""
    refs = _normalised(refs)
    with con:
        con.execute("DELETE FROM ri_reference")
        db.insert(
            con,
            "ri_reference",
            refs.astype(object).where(refs.notna(), None),
        )
        update_target_ris(con, commit=False)


def target_ris(rt_ref: pd.Series, refs: pd.DataFrame) -> pd.Series:
    """The RI of each target from its rt in the reference run (``rt_ref``: rt_ref_s by
    target name; NaN, 0: unknown), between the enabled references that are targets with
    an rt (extrapolated beyond), rounded to 0.01; NaN without two of them."""
    rt = rt_ref.where(rt_ref > 0)
    used = refs[(refs["enabled"] == 1) & refs["name"].isin(rt.dropna().index)]
    anchors = pd.DataFrame(
        {"rt": rt.loc[used["name"]].to_numpy(), "ri": used["ri"].to_numpy()}
    ).sort_values("rt")
    if len(anchors) < 2:
        return pd.Series(np.nan, index=rt_ref.index)
    from esper.targets import interp_extrapolate

    ri = interp_extrapolate(rt, anchors["rt"], anchors["ri"])
    return pd.Series(ri, index=rt_ref.index).where(rt.notna()).round(2)


def update_target_ris(con: sqlite3.Connection, commit: bool = True) -> int:
    """Set the RI of the targets of the database from its references (``target_ris``);
    left as they are without references. Returns the number of targets with an RI."""
    refs = read_references(con)
    if refs.empty:
        return 0
    rt_ref = pd.read_sql("SELECT name, rt_ref_s FROM target", con, index_col="name")[
        "rt_ref_s"
    ].astype(float)
    ri = target_ris(rt_ref, refs)
    rows = [(None if pd.isna(v) else float(v), name) for name, v in ri.items()]
    con.executemany("UPDATE target SET ri = ? WHERE name = ?", rows)
    if commit:
        con.commit()
    return int(ri.notna().sum())


# --- in a run ---------------------------------------------------------------------------


def reference_names(refs: pd.DataFrame, targets=None) -> dict[str, set[str]]:
    """The candidate names that are each reference: its name, its ``names`` and the
    aliases of the targets among them (``targets``: an ``esper.target_index.TargetIndex``,
    None: no aliases)."""
    out = {}
    for ref in refs.itertuples():
        names = {ref.name, *split_names(ref.names)}
        if targets is not None:
            for name in list(names):
                target = targets.target_of(name)
                if target is not None:
                    names |= set(targets.target_names(target))
        out[ref.name] = names
    return out


def find_references(
    compounds: pd.DataFrame,
    best: pd.DataFrame,
    refs: pd.DataFrame,
    targets=None,
    config: RetentionIndexConfig | None = None,
) -> pd.DataFrame:
    """The references found in a run (see the module).

    :param compounds: indexed by compound_idx, with ``rt_mean_s`` and ``prominence_sum``.
    :param best: the best candidate of the compounds: ``name`` and ``score``, indexed by
        compound_idx (target search: the target found, score 1).
    :param refs: the enabled references (``read_references``).
    :return: ``FOUND_COLUMNS``, by increasing rt.
    """
    config = config or RetentionIndexConfig()
    names = reference_names(refs, targets)
    best = best[best["score"] > config.min_score].join(
        compounds[["rt_mean_s", "prominence_sum"]], how="inner"
    )
    rows = []
    previous = -np.inf
    for ref in refs.sort_values("ri", kind="stable").itertuples():
        candidates = best[
            best["name"].isin(names[ref.name]) & (best["rt_mean_s"] > previous)
        ]
        if candidates.empty:
            continue
        if pd.notna(ref.occurrence):
            idx = candidates["rt_mean_s"].idxmin()
        else:
            idx = candidates["prominence_sum"].idxmax()
        rt = float(candidates.at[idx, "rt_mean_s"])
        rows.append(
            (ref.name, int(idx), rt, float(candidates.at[idx, "score"]), ref.ri)
        )
        previous = rt
    return pd.DataFrame(rows, columns=FOUND_COLUMNS)


def rt_to_ri(found: pd.DataFrame) -> Callable[[float | np.ndarray], np.ndarray] | None:
    """The rt -> RI function of a run from its references found (``FOUND_COLUMNS``):
    interpolated, extrapolated beyond the first and last; None without two of them."""
    if len(found) < 2:
        return None
    from esper.targets import interp_extrapolate

    anchors = found.sort_values("rt_s")
    rt, ri = anchors["rt_s"].to_numpy(dtype=float), anchors["ri"].to_numpy(dtype=float)
    return lambda x: interp_extrapolate(x, rt, ri)


def run_references(con: sqlite3.Connection, measurement_id: int) -> pd.DataFrame:
    """The references found in a run (``measurement_ri_reference``) with their current RI
    (``FOUND_COLUMNS``; those no longer references left out)."""
    if not db.table_columns(con, "measurement_ri_reference"):
        return pd.DataFrame(columns=FOUND_COLUMNS)
    return pd.read_sql(
        "SELECT m.name, m.compound_idx, m.rt_s, m.score, r.ri"
        " FROM measurement_ri_reference m JOIN ri_reference r USING (name)"
        " WHERE m.measurement_id = ? ORDER BY m.rt_s",
        con,
        params=(int(measurement_id),),
    )


def store_run_references(
    con: sqlite3.Connection, measurement_id: int, found: pd.DataFrame
) -> None:
    """Replace the references found in a run (in the caller's transaction)."""
    mid = int(measurement_id)
    con.execute("DELETE FROM measurement_ri_reference WHERE measurement_id = ?", (mid,))
    db.insert(
        con,
        "measurement_ri_reference",
        found[["name", "compound_idx", "rt_s", "score"]].assign(measurement_id=mid),
    )


def update_compound_ris(con: sqlite3.Connection, measurement_id: int) -> bool:
    """The RI of the compounds of a run from its references (e.g. after the fit moved
    them); False (left as they are) without two references. In the caller's
    transaction."""
    ri_of = rt_to_ri(run_references(con, measurement_id))
    if ri_of is None:
        return False
    compounds = pd.read_sql(
        "SELECT compound_idx, rt_mean_s FROM compound WHERE measurement_id = ?",
        con,
        params=(int(measurement_id),),
    )
    ri = ri_of(compounds["rt_mean_s"].to_numpy(dtype=float))
    con.executemany(
        "UPDATE compound SET retention_index = ? WHERE measurement_id = ? AND compound_idx = ?",
        [
            (
                None if not np.isfinite(v) else round(float(v), 2),
                int(measurement_id),
                int(i),
            )
            for v, i in zip(ri, compounds["compound_idx"], strict=True)
        ],
    )
    return True


__all__ = [
    "FOUND_COLUMNS",
    "REFERENCE_COLUMNS",
    "RetentionIndexConfig",
    "default_references",
    "find_references",
    "read_reference_csv",
    "read_references",
    "reference_names",
    "rt_to_ri",
    "run_references",
    "store_references",
    "store_run_references",
    "target_ris",
    "update_compound_ris",
    "update_target_ris",
    "write_reference_csv",
]
