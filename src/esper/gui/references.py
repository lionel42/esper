"""The reference spectrum compared with the selected compound."""

from __future__ import annotations

from esper.gui.data import EsperData
from esper.gui.library_loader import library_loader
from esper.gui.selection import Selection
from esper.gui.settings import settings_manager


def current_reference(data: EsperData, selection: Selection) -> tuple[str, str] | None:
    """(library, name) of the selected reference, or of alpinac's best candidate of the compound.

    None while the libraries are being read (in the background).
    """
    if selection.reference is not None:
        return selection.reference
    if selection.measurement_id is None or selection.compound_idx is None:
        return None
    if not library_loader().ensure():
        return None
    candidates = data.candidates(selection.measurement_id, selection.compound_idx)
    best = candidates[candidates["is_best"] == 1]
    if best.empty:
        return None
    return settings_manager().libraries.resolve(best["spectrum_ref"].iloc[0])
