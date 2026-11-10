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

from esper import db
from esper.libraries import READERS, Library, default_libraries

logger = logging.getLogger(__name__)

# Columns of Targets.table (index: name)
TARGET_COLUMNS = [
    "rt_ref_s",
    "ri",
    "formula",
    "chemical_name",
    "cas",
    "cmp_class",
    "pubchem_cid",
    "inchi",
    "smiles",
]
# Columns of Targets.ions (spectrum_ref: the reference spectrum the ion comes from,
# None if from the target list; valid_from: from when the ions apply, None: always)
ION_COLUMNS = ["name", "valid_from", "rank", "ion", "exact_mass", "mass", "spectrum_ref"]
# Columns of Targets.rts: expected rt from valid_from on (rt_s None: not identified)
RT_COLUMNS = ["name", "valid_from", "rt_s", "window_s"]


class TargetError(ValueError):
    """The targets of a source cannot be read (e.g. names that match no target)."""
# Ions of a target without ions in the target list: the largest masses of its
# reference spectrum (as many as the ions of the Excel list)
N_SPECTRUM_IONS = 8


@dataclass
class Targets:
    """Target compounds, whatever their source.

    - ``table``: one row per target, indexed by name, with ``TARGET_COLUMNS``
    - ``ions``: expected ions (``ION_COLUMNS``), rank 0 is the main ion (the
      quantification ion)
    - ``aliases``: other name -> target name
    - ``rts``: expected retention times by version (``RT_COLUMNS``), for the
      instruments that give them (GCwerks); empty for the target lists, whose
      targets are found from their retention index

    The versions (``valid_from``) are ISO times, as ``measurement.acq_datetime``.
    """

    table: pd.DataFrame
    ions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=ION_COLUMNS)
    )
    aliases: dict[str, str] = field(default_factory=dict)
    rts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RT_COLUMNS))

    def check(self) -> None:
        missing = set(TARGET_COLUMNS) - set(self.table.columns)
        if missing:
            raise ValueError(f"Missing target columns: {sorted(missing)}")
        if self.table.index.duplicated().any():
            raise ValueError(
                f"Duplicated targets: {self.table.index[self.table.index.duplicated()].tolist()}"
            )
        for what, df in (("Ions", self.ions), ("Retention times", self.rts)):
            unknown = set(df["name"]) - set(self.table.index)
            if unknown:
                raise ValueError(f"{what} of unknown targets: {sorted(unknown)[:10]}")


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
        y = np.where(
            high, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y
        )
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
    refs = (
        refs.assign(rt=df.loc[refs["compound"], "rt"].values)
        .dropna(subset=["rt"])
        .sort_values("rt")
    )
    ri = pd.Series(
        interp_extrapolate(rt, refs["rt"], refs["ri"]), index=df.index
    ).where(rt.notna())

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
            ion = [
                str(v).strip() if isinstance(v, str) and v.strip() else None
                for v in raw.loc[valid, ion_column]
            ]
        else:
            ion = [None] * int(valid.sum())
        ions.append(
            pd.DataFrame(
                {
                    "name": raw.loc[valid, EXCEL_NAME_COLUMN],
                    "rank": rank,
                    "ion": pd.Series(ion, dtype=object, index=raw.index[valid]),
                    "exact_mass": exact[valid],
                }
            )
        )
    ions = (
        pd.concat(ions, ignore_index=True)
        if ions
        else pd.DataFrame(columns=ION_COLUMNS)
    )
    ions["mass"] = ions["exact_mass"].round().astype(int)
    ions["spectrum_ref"] = None
    ions["valid_from"] = None

    aliases = {alias: name for alias, name in aliases.items() if name in table.index}
    return Targets(table=table, ions=ions[ION_COLUMNS], aliases=aliases)


# Loaders by file suffix
LOADERS: dict[str, Callable[..., Targets]] = {
    ".xlsx": load_excel,
    ".xls": load_excel,
}


def load_targets(
    source: Path, libraries: list[Library] | None = None, **options
) -> Targets:
    """Targets of a target list, with the ions of their reference spectrum if the list
    gives none (``spectrum_ions``; ``libraries``: the default ones, [] for none)."""
    source = Path(source)
    loader = LOADERS.get(source.suffix.lower())
    if loader is None:
        raise ValueError(
            f"No target loader for {source.suffix!r} files (known: {sorted(LOADERS)})"
        )
    targets = loader(source, **options)
    targets.check()
    ions = spectrum_ions(
        targets, default_libraries() if libraries is None else libraries
    )
    if not ions.empty:
        # (all-None columns left out of the concatenation, they are reindexed)
        targets.ions = pd.concat(
            [df.dropna(axis=1, how="all") for df in (targets.ions, ions)],
            ignore_index=True,
        ).reindex(columns=ION_COLUMNS)
    return targets


def spectrum_ions(targets: Targets, libraries: list[Library]) -> pd.DataFrame:
    """Ions (``ION_COLUMNS``) of the targets without ions, from their reference spectrum.

    The spectrum is the first found in the enabled libraries (by priority) under the
    target name, then under its aliases; its ``N_SPECTRUM_IONS`` most intense masses,
    by decreasing intensity (rank 0: the main ion).
    """
    names = {name: [name] for name in targets.table.index}
    for alias, name in targets.aliases.items():
        names[name].append(alias)
    with_ions = set(targets.ions["name"])
    rows = []
    for name in targets.table.index:
        if name in with_ions:
            continue
        found = _reference_spectrum(
            names[name], [lib for lib in libraries if lib.enabled]
        )
        if found is None:
            continue
        spectrum_ref, spectrum = found
        order = np.argsort(-spectrum.intensities, kind="stable")[:N_SPECTRUM_IONS]
        rows += [
            (name, None, rank, None, None, int(spectrum.mz[i]), spectrum_ref)
            for rank, i in enumerate(order)
        ]
    return pd.DataFrame(rows, columns=ION_COLUMNS)


def _reference_spectrum(names: list[str], libraries: list[Library]):
    """(spectrum_ref, spectrum) of the first spectrum of one of the names, None if none."""
    for library in libraries:
        for name in names:
            for suffix, read in READERS.items():
                file = Path(library.path) / f"{name}{suffix}"
                if not file.is_file():
                    continue
                try:
                    spectrum = read(file)
                except Exception as e:
                    logger.warning(f"Skipping unreadable spectrum {file}: {e}")
                    continue
                if len(spectrum):
                    return f"{library.family}_{name}", spectrum
    return None


def store_targets(con: sqlite3.Connection, targets: Targets, source: str) -> None:
    """Replace all the targets of the database (single transaction)."""
    table = targets.table.reset_index()
    with con:
        for name in ("target_alias", "target_ion", "target_rt", "target"):
            con.execute(f"DELETE FROM {name}")
        db.insert(con, "target", table[["name"] + TARGET_COLUMNS])
        db.insert(con, "target_ion", targets.ions[ION_COLUMNS])
        db.insert(con, "target_rt", targets.rts[RT_COLUMNS])
        db.insert(
            con,
            "target_alias",
            pd.DataFrame(list(targets.aliases.items()), columns=["alias", "name"]),
        )
        for key, value in {
            "targets_source": source,
            "targets_imported_at": datetime.now().isoformat(timespec="seconds"),
        }.items():
            con.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
            )
