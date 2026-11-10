"""The targets of a GCwerks directory: the compounds it identifies.

- The names are those of ``config/peak.list`` (one target per active line).
- Where each one elutes and its ions come from the peak identification,
  ``integrator/<channel>/peakid/<version>`` (:mod:`gcwerks.io.peakid`): its retention
  time, window and target and qualifier ions, from the date of the version on.

A peakid name is the target of the same name, else the target the ``aliases`` of the
esper.yaml give it (``null``: not a target, e.g. the instrument gases ``O2``), else the
target it is an `openghg-defs <https://github.com/openghg/openghg_defs>`_ name of
(e.g. ``CFC12`` -> ``CFC-12``, :func:`gcwerks.species.species_aliases`). A name that is
none of these raises a :class:`~esper.targets.TargetError`.

Only the changes are kept: a row of ``target_rt`` when the rt or window of a target
changes (rt NULL when a version no longer identifies it), the ions of ``target_ion``
when they change.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from esper.targets import (
    ION_COLUMNS,
    RT_COLUMNS,
    TARGET_COLUMNS,
    TargetError,
    Targets,
    UnresolvedNamesError,
)

logger = logging.getLogger(__name__)

ALIASES_KEY = "aliases"  # of the esper.yaml: peakid name -> peak.list name or null
_ION_COLUMNS = ["target_ion", "qualifier_ion_1", "qualifier_ion_2"]


def unit_mass(mass: float) -> int:
    """Unit mass of a peakid ion. From 2007-12 to 2009-07 the masses were set 0.5 higher
    (e.g. 67.5 for 67), so a half mass is the mass below it."""
    return int(math.floor(mass + 0.25))


def _iso(time: pd.Timestamp) -> str:
    return time.isoformat(timespec="seconds")


def check_aliases(aliases, names: list[str]) -> dict[str, str | None]:
    """The ``aliases`` of the esper.yaml, checked: each a peak.list name or null."""
    if aliases is None:
        return {}
    if not isinstance(aliases, dict):
        raise TargetError(
            f"{ALIASES_KEY} of esper.yaml: expected 'peakid name: peak.list name'"
        )
    unknown = {k: v for k, v in aliases.items() if v is not None and v not in names}
    if unknown:
        raise TargetError(
            f"{ALIASES_KEY} of esper.yaml: not peak.list names: "
            + ", ".join(f"{k}: {v}" for k, v in unknown.items())
        )
    return {str(k): v for k, v in aliases.items()}


def target_names(peak_list: pd.DataFrame) -> list[str]:
    """The targets: the names of peak.list, in its order, without repetition."""
    names = peak_list["name"].tolist()
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        logger.warning(f"peak.list: names repeated, kept once: {', '.join(repeated)}")
    return list(dict.fromkeys(names))


class NameResolver:
    """peakid name -> target name (None: not a target), see the module."""

    def __init__(self, names: list[str], aliases: dict[str, str | None]) -> None:
        from gcwerks.species import species_aliases

        self.names = set(names)
        self.aliases = aliases
        try:
            self.species = species_aliases(names)
        except ValueError as e:  # two targets are names of the same species
            raise TargetError(f"peak.list: {e}") from e

    def __call__(self, name: str) -> str | None:
        if name in self.names:
            return name
        if name in self.aliases:
            return self.aliases[name]
        return self.species.get(name.lower(), _UNRESOLVED)


_UNRESOLVED = object()


def resolve_names(peakid: pd.DataFrame, resolver: NameResolver) -> pd.DataFrame:
    """The peakid lines of the targets (``target`` column), the first one of a target in
    each version; raises for the names that match no target."""
    resolved = {name: resolver(name) for name in peakid["name"].unique()}
    unresolved = sorted(n for n, t in resolved.items() if t is _UNRESOLVED)
    if unresolved:
        raise UnresolvedNamesError(
            f"{len(unresolved)} peakid names are not peak.list names, nor openghg-defs names"
            f" of one: {', '.join(unresolved)}. Map each to its peak.list name, or to null"
            f" if it is not a target, in the {ALIASES_KEY} of the esper.yaml"
            " (e.g. 'aliases: {CH2F2: HFC-32, O2: null}'; esper upload --add-aliases"
            " adds them as null)",
            unresolved,
            ALIASES_KEY,
        )
    lines = peakid.assign(target=peakid["name"].map(resolved))
    lines = lines[lines["target"].notna()]
    repeated = lines.duplicated(["valid_from", "target"])
    if repeated.any():
        versions = lines[repeated].groupby("target")["file"].nunique()
        logger.warning(
            "peakid: targets identified twice in a version, the first line kept: "
            + ", ".join(f"{t} ({n} versions)" for t, n in versions.items())
        )
    return lines[~repeated]


def _versions(lines: pd.DataFrame, starts: list[pd.Timestamp]) -> tuple[list, list]:
    """(rt rows, ion rows) of the changes, from the lines of the targets by version."""
    rt_rows, ion_rows = [], []
    by_target = {t: g.set_index("valid_from") for t, g in lines.groupby("target")}
    for target, g in by_target.items():
        rt_state = ions_state = None
        for start in starts:
            if start in g.index:
                line = g.loc[start]
                rt = (float(line["rt"]), float(line["window"]))
                ions = tuple(
                    dict.fromkeys(
                        unit_mass(m) for m in line[_ION_COLUMNS] if np.isfinite(m)
                    )
                )
            else:
                rt, ions = (None, None), ions_state  # no longer identified; ions kept
            if rt != rt_state and not (rt_state is None and rt == (None, None)):
                rt_rows.append((target, _iso(start), *rt))
                rt_state = rt
            if ions is not None and ions != ions_state:
                ion_rows += [
                    (target, _iso(start), rank, None, None, mass, None)
                    for rank, mass in enumerate(ions)
                ]
                ions_state = ions
    return rt_rows, ion_rows


def read_targets(root: Path, aliases=None, channel: str = "channel0") -> Targets:
    """The targets of the GCwerks directory ``root`` (see the module).

    ``aliases``: of the esper.yaml (peakid name -> peak.list name or None).
    """
    from gcwerks.io.peakid import read_peakid_versions
    from gcwerks.io.peaks import read_peak_list
    from gcwerks.paths import GCWerksPaths

    paths = GCWerksPaths(root)
    if not paths.peak_list.is_file():
        raise TargetError(f"No peak list {paths.peak_list}: the names of the targets")
    names = target_names(read_peak_list(paths.peak_list))
    resolver = NameResolver(names, check_aliases(aliases, names))
    peakid_dir = paths.peakid_dir(channel)
    peakid = read_peakid_versions(peakid_dir) if peakid_dir.is_dir() else None
    if peakid is None or peakid.empty:
        logger.warning(f"No peak identification in {peakid_dir}: targets without rt")
        lines, starts = pd.DataFrame(columns=["target"]), []
    else:
        lines = resolve_names(peakid, resolver)
        starts = sorted(peakid["valid_from"].unique())
    rt_rows, ion_rows = _versions(lines, starts)
    rts = pd.DataFrame(rt_rows, columns=RT_COLUMNS)
    ions = pd.DataFrame(ion_rows, columns=ION_COLUMNS)

    table = pd.DataFrame(index=pd.Index(names, name="name"), columns=TARGET_COLUMNS)
    latest = rts.dropna(subset=["rt_s"]).groupby("name")["rt_s"].last()
    table["rt_ref_s"] = latest.reindex(table.index)  # the latest identification
    without = [n for n in names if n not in set(rts["name"])]
    if without:
        logger.info(
            f"{len(without)} of the {len(names)} peak.list targets are not in the peak"
            f" identification: {', '.join(without)}"
        )
    return Targets(table=table, ions=ions, rts=rts)
