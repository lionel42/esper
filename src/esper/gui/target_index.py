"""The target compounds and the names they go by, for the GUI."""

from __future__ import annotations

import pandas as pd


class TargetIndex:
    """Targets (``target`` table) with their expected ions and aliases.

    Resolves the names of the candidates (a target name, one of its aliases, or any
    other name) to targets.
    """

    def __init__(
        self, targets: pd.DataFrame, ions: pd.DataFrame, aliases: pd.DataFrame
    ) -> None:
        """``targets`` indexed by name; ``ions`` (name, rank, mass, ...) by rank;
        ``aliases`` with the columns alias and name."""
        masses = ions.groupby("name")["mass"].agg(
            lambda m: list(dict.fromkeys(int(x) for x in m))
        )
        table = targets.copy()
        table["masses"] = masses.reindex(table.index)
        table["masses"] = table["masses"].apply(
            lambda m: m if isinstance(m, list) else []
        )
        # Targets, indexed by name, with their expected unit masses (``masses``, main ion first)
        self.table = table
        self.ions = ions
        # Target name or alias -> target name
        self._by_name = dict(zip(aliases["alias"], aliases["name"]))
        self._by_name.update({name: name for name in table.index})

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
        """Expected unit masses of a target, main ion first."""
        if target not in self.table.index:
            return []
        return self.table.at[target, "masses"]

    def target_main_ion(self, name: str | None) -> int | None:
        """Main ion (rank 0 of ``target_ion``) of the target a name stands for, None if none."""
        target = self.target_of(name) if isinstance(name, str) else None
        masses = self.target_masses(target) if target is not None else []
        return masses[0] if masses else None
