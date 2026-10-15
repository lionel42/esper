"""Target compounds: the substances we look for in the measurements.

The targets are imported into the esper database (``esper targets``), the GUI
only reads them from there. The source format is handled by a loader chosen
from the file suffix (``LOADERS``); to support another source, write a
function returning a ``Targets`` and register it.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns of Targets.table (index: name)
TARGET_COLUMNS = [
    "rt_ref_s", "ri", "formula", "chemical_name", "cas", "cmp_class", "pubchem_cid", "inchi", "smiles",
]
# Columns of Targets.ions
ION_COLUMNS = ["name", "rank", "ion", "exact_mass", "mass"]


@dataclass
class Targets:
    """Target compounds, whatever their source.

    - ``table``: one row per target, indexed by name, with ``TARGET_COLUMNS``
    - ``ions``: expected ions (``ION_COLUMNS``), rank 0 is the main ion
    - ``aliases``: other name -> target name
    """

    table: pd.DataFrame
    ions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ION_COLUMNS))
    aliases: dict[str, str] = field(default_factory=dict)

    def check(self) -> None:
        missing = set(TARGET_COLUMNS) - set(self.table.columns)
        if missing:
            raise ValueError(f"Missing target columns: {sorted(missing)}")
        if self.table.index.duplicated().any():
            raise ValueError(f"Duplicated targets: {self.table.index[self.table.index.duplicated()].tolist()}")
        unknown = set(self.ions["name"]) - set(self.table.index)
        if unknown:
            raise ValueError(f"Ions of unknown targets: {sorted(unknown)[:10]}")


def interp_extrapolate(x, xp, fp) -> np.ndarray:
    """Linear interpolation, extrapolated linearly beyond the first and last points.

    ``np.interp`` would clamp to the end values instead.
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    y = np.interp(x, xp, fp)
    if len(xp) >= 2:
        low, high = x < xp[0], x > xp[-1]
        y = np.where(low, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
        y = np.where(high, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y


# --- Excel (substances.xlsx) -------------------------------------------------------

EXCEL_SHEET = "all"
EXCEL_NAME_COLUMN = "Compound name (Gcwerks)"
# (ion formula column, exact mass column), by rank
EXCEL_ION_COLUMNS = [
    ("Q0 Ion", "Q0 Mass"),
    ("Q1 Ion", "Q1 Mass"),
    ("Q2 Ion", "Q2 Mass"),
    ("Q3 Ion", "Q3 Mass"),
    ("Q4 Ion", "Q4 Mass"),
    ("Q5 Ion", "Q5 Mass"),
    (None, "Q6"),
    (None, "Q7"),
    (None, "Q8"),
]


def default_ri_references() -> Path:
    """Retention index references of alpinac."""
    import alpinac.mode_extraction

    return Path(alpinac.mode_extraction.__file__).parent / "rt_refs_v2.csv"


def load_excel(path: Path, ri_references: Path | None = None) -> Targets:
    """Read the substances Excel file as alpinac does (``load_excel_compounds``).

    The retention index is computed from the rt of the reference run with the
    alpinac reference list, but extrapolated beyond the first/last reference
    instead of clamped. An rt of 0 means unknown.
    """
    from alpinac.mode_extraction.read_from_excel import load_excel_compounds

    ri_references = Path(ri_references) if ri_references else default_ri_references()
    # alpinac prints its warnings (duplicated names, ...): send them to the log
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        df, aliases = load_excel_compounds(Path(path), ri_references)
    for line in printed.getvalue().splitlines():
        logger.debug(f"alpinac: {line}")

    rt = df["rt"].where(df["rt"] > 0)
    refs = pd.read_csv(ri_references, comment="#")
    refs = refs[refs["compound"].isin(df.index)]
    refs = refs.assign(rt=df.loc[refs["compound"], "rt"].values).dropna(subset=["rt"]).sort_values("rt")
    ri = pd.Series(interp_extrapolate(rt, refs["rt"], refs["ri"]), index=df.index).where(rt.notna())

    table = pd.DataFrame(
        {
            "rt_ref_s": rt,
            "ri": ri.round(2),
            "formula": df["formula"].str.strip(),
            "chemical_name": df["chemical_name"],
            "cas": df["cas"],
            "cmp_class": df["cmp_class"],
            "pubchem_cid": df["pubchem_cid"].where(df["pubchem_cid"] > 0),
            "inchi": df["inchi"],
            "smiles": df["smiles"],
        },
        index=df.index.rename("name"),
    ).replace("", None)

    # Expected ions, from the same rows as alpinac (first occurrence of a name)
    raw = pd.read_excel(path, sheet_name=EXCEL_SHEET)
    raw = raw.dropna(subset=[EXCEL_NAME_COLUMN]).drop_duplicates(EXCEL_NAME_COLUMN)
    raw = raw[raw[EXCEL_NAME_COLUMN].isin(table.index)]
    ions = []
    for rank, (ion_column, mass_column) in enumerate(EXCEL_ION_COLUMNS):
        exact = pd.to_numeric(raw[mass_column], errors="coerce")
        valid = exact > 0
        if not valid.any():
            continue
        if ion_column:
            ion = [str(v).strip() if isinstance(v, str) and v.strip() else None for v in raw.loc[valid, ion_column]]
        else:
            ion = [None] * int(valid.sum())
        ions.append(
            pd.DataFrame(
                {"name": raw.loc[valid, EXCEL_NAME_COLUMN], "rank": rank, "ion": pd.Series(ion, dtype=object, index=raw.index[valid]), "exact_mass": exact[valid]}
            )
        )
    ions = pd.concat(ions, ignore_index=True) if ions else pd.DataFrame(columns=ION_COLUMNS)
    ions["mass"] = ions["exact_mass"].round().astype(int)

    aliases = {alias: name for alias, name in aliases.items() if name in table.index}
    return Targets(table=table, ions=ions[ION_COLUMNS], aliases=aliases)


# Loaders by file suffix
LOADERS: dict[str, Callable[..., Targets]] = {
    ".xlsx": load_excel,
    ".xls": load_excel,
}


def load_targets(source: Path, **options) -> Targets:
    source = Path(source)
    loader = LOADERS.get(source.suffix.lower())
    if loader is None:
        raise ValueError(f"No target loader for {source.suffix!r} files (known: {sorted(LOADERS)})")
    targets = loader(source, **options)
    targets.check()
    return targets


def store_targets(con: sqlite3.Connection, targets: Targets, source: str) -> None:
    """Replace all the targets of the database (single transaction)."""
    table = targets.table.reset_index()
    with con:
        con.execute("DELETE FROM target_alias")
        con.execute("DELETE FROM target_ion")
        con.execute("DELETE FROM target")
        _insert(con, "target", table[["name"] + TARGET_COLUMNS])
        _insert(con, "target_ion", targets.ions[ION_COLUMNS])
        _insert(con, "target_alias", pd.DataFrame(list(targets.aliases.items()), columns=["alias", "name"]))
        for key, value in {
            "targets_source": source,
            "targets_imported_at": datetime.now().isoformat(timespec="seconds"),
        }.items():
            con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def _insert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.astype(object).where(df.notna(), None)
    columns = ", ".join(df.columns)
    placeholders = ", ".join("?" * len(df.columns))
    con.executemany(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        df.itertuples(index=False, name=None),
    )
