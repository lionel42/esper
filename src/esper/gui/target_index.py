"""The target compounds and the names they go by, for the GUI."""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

# valid_from of the rows that always apply (target lists: valid_from NULL)
_ALWAYS = pd.Timestamp.min


def _valid_from(values: pd.Series) -> pd.Series:
    times = pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")
    return times.fillna(_ALWAYS)


class TargetIndex:
    """Targets (``target`` table) with their expected ions and aliases.

    Resolves the names of the candidates (a target name, one of its aliases, or any
    other name) to targets. The ions (``target_ion``) and the retention times
    (``target_rt``, GCwerks) can be versioned: what applies to a run is the latest
    ``valid_from`` before it (``*_at`` methods); without a time, the latest version.
    """

    def __init__(
        self,
        targets: pd.DataFrame,
        ions: pd.DataFrame,
        aliases: pd.DataFrame,
        rts: pd.DataFrame | None = None,
    ) -> None:
        """``targets`` indexed by name; ``ions`` (name, rank, mass, optional valid_from);
        ``aliases`` with the columns alias and name; ``rts`` (name, valid_from, rt_s,
        window_s), None: none."""
        ions = ions.assign(
            valid_from=_valid_from(ions["valid_from"])
            if "valid_from" in ions
            else _ALWAYS
        )
        # target -> ([valid_from], [masses, main ion first]), by valid_from
        self._ion_versions: dict[str, tuple[list, list]] = {}
        for name, g in ions.sort_values(["valid_from", "rank"]).groupby("name"):
            starts, masses = [], []
            for start, v in g.groupby("valid_from", sort=True):
                starts.append(start)
                masses.append(list(dict.fromkeys(int(x) for x in v["mass"])))
            self._ion_versions[name] = (starts, masses)
        # target -> ([valid_from], [(rt_s, window_s)]), by valid_from
        self._rt_versions: dict[str, tuple[list, list]] = {}
        if rts is not None and len(rts):
            rts = rts.assign(valid_from=_valid_from(rts["valid_from"]))
            for name, g in rts.sort_values("valid_from").groupby("name"):
                self._rt_versions[name] = (
                    g["valid_from"].tolist(),
                    list(zip(g["rt_s"].astype(float), g["window_s"].astype(float))),
                )

        table = targets.copy()
        table["masses"] = [self.masses_at(name) for name in table.index]
        # Targets, indexed by name, with their expected unit masses (``masses``, main ion
        # first, of the latest version)
        self.table = table
        self.ions = ions
        # Target name or alias -> target name
        self._by_name = dict(zip(aliases["alias"], aliases["name"]))
        self._by_name.update({name: name for name in table.index})

    @staticmethod
    def _at(versions: tuple[list, list] | None, time):
        """The value of the version in effect at ``time`` (None: the latest), None if none."""
        if versions is None:
            return None
        starts, values = versions
        if time is None or pd.isna(time):
            return values[-1]
        i = bisect.bisect_right(starts, pd.Timestamp(time)) - 1
        return values[i] if i >= 0 else None

    def masses_at(self, target: str | None, time=None) -> list[int]:
        """Expected unit masses of a target in a run at ``time``, main ion first."""
        return self._at(self._ion_versions.get(target), time) or []

    def has_rts(self, target: str | None) -> bool:
        """Whether the expected rt of the target is given (versioned), not from its RI."""
        return target in self._rt_versions

    def rt_at(self, target: str | None, time=None) -> tuple[float, float] | None:
        """(rt, window) expected for the target in a run at ``time``; None: not given,
        or not identified then."""
        value = self._at(self._rt_versions.get(target), time)
        if value is None or not np.isfinite(value[0]):
            return None
        return value

    def target_of(self, name: str | None) -> str | None:
        """The target that a candidate name (a target name or alias) stands for."""
        return self._by_name.get(name) if name is not None else None

    def target_names(self, target: str) -> list[str]:
        """The target name and its aliases."""
        return [n for n, t in self._by_name.items() if t == target]

    def main_name(self, name: str | None) -> str | None:
        """The target that a candidate name stands for (aliases resolved), else the name itself."""
        if not isinstance(name, str):
            return None
        return self._by_name.get(name, name)

    def names_of(self, main_name: str) -> list[str]:
        """The candidate names with that main name: a target and its aliases, or the name itself."""
        return self.target_names(main_name) or [main_name]

    def target_masses(self, target: str) -> list[int]:
        """Expected unit masses of a target (latest version), main ion first."""
        if target not in self.table.index:
            return []
        return self.table.at[target, "masses"]

    def target_main_ion(self, name: str | None) -> int | None:
        """Main ion (rank 0 of ``target_ion``) of the target a name stands for, None if none."""
        target = self.target_of(name) if isinstance(name, str) else None
        masses = self.target_masses(target) if target is not None else []
        return masses[0] if masses else None
