"""The reference spectrum compared with the selected compound, and the best one of a name."""

from __future__ import annotations

import numpy as np

from esper.gui.data import EsperData
from esper.gui.library_loader import library_loader
from esper.gui.selection import Selection
from esper.gui.settings import settings_manager
from esper.libraries import combined_score


def current_reference(data: EsperData, selection: Selection) -> tuple[str, str] | None:
    """(library, name) of the selected reference, or of the best candidate of the compound.

    The best candidate is alpinac's, or the one of the identification edited by hand.

    None while the libraries are being read (in the background).
    """
    if selection.reference is not None:
        return selection.reference
    if selection.measurement_id is None or selection.compound_idx is None:
        return None
    if not library_loader().ensure():
        return None
    # alpinac's best candidate, or the spectrum of the identification edited by hand
    best = data.compounds(selection.measurement_id).at[selection.compound_idx, "best_spectrum"]
    return settings_manager().libraries.resolve(best) if isinstance(best, str) else None


def identification_candidate(
    data: EsperData, measurement_id: int, compound_idx: int, name: str
) -> tuple[str | None, float | None, float | None]:
    """(spectrum_ref, score, score_combined) of the best library spectrum of a name for a compound.

    One of alpinac's candidates with that name (or an alias of the same target)
    is taken as it is. Otherwise the spectra of the name and its aliases in the
    enabled libraries are scored against the compound (intensity and processing
    of the settings), and the score is combined with the retention index difference
    as alpinac does. All None if no library has a spectrum of the name.
    Waits for the libraries to be read.
    """
    main_name = data.main_name(name)
    candidates = data.candidates(measurement_id, compound_idx)
    same = candidates[candidates["candidate_name"].map(data.main_name) == main_name]
    if not same.empty:
        best = same.sort_values("rank").iloc[0]
        return best["spectrum_ref"], best["score"], best["score_combined"]

    loader = library_loader()
    if not loader.ensure():
        loader.wait()
    manager = settings_manager()
    libraries = manager.libraries
    spectrum = data.compound_spectrum(measurement_id, compound_idx, manager.settings.spectrum_intensity)
    scored = [
        (libraries.score(spectrum, libraries.spectrum(library, spectrum_name))[0], library, spectrum_name)
        for spectrum_name in data.names_of(main_name)
        for library in libraries.libraries_with(spectrum_name)
    ]
    if not scored:
        return None, None, None
    score, library, spectrum_name = max(scored)
    target = data.target_of(main_name)
    ri = data.targets.at[target, "ri"] if target is not None else np.nan
    ri_diff = data.compounds(measurement_id).at[compound_idx, "retention_index"] - ri
    family = library.split("_", 1)[0]
    return f"{family}_{spectrum_name}", score, combined_score(score, ri_diff)
