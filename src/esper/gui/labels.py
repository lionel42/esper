"""Labels of compounds and masses, shared by the panels."""

from __future__ import annotations

from esper.gui.data import TIC

# compound_idx of the peaks of the fit that match no compound
NO_COMPOUND = -1


def compound_label(name, compound_idx) -> str:
    return name if isinstance(name, str) else f"#{compound_idx}"


def combined_label(compound_idxs: tuple[int, ...]) -> str:
    """The other compounds selected with the main one, e.g. ' + #12, #15'."""
    return (
        " + " + ", ".join(f"#{i}" for i in compound_idxs[1:])
        if len(compound_idxs) > 1
        else ""
    )


def peak_compound_label(names, compound_idx: int) -> str:
    """Compound of a peak (``NO_COMPOUND``: none)."""
    if compound_idx == NO_COMPOUND:
        return "no compound"
    return compound_label(names.get(compound_idx), compound_idx)


def mass_label(mass: int | None) -> str:
    return "TIC" if mass is TIC else f"m/z {mass}"
