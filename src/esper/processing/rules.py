"""Screening rules of the compounds of a run, after their identification (``processing:
rules:`` of the esper.yaml; step 7 of the refactoring plan).

- Noise merge (``noise_merge``, suspect screening; the legacy ``clean_noisy_compounds``):
  each identified compound, by index, takes the unidentified compounds after it whose rt
  is within its range (mean start to mean stop of its peaks) when its spectrum with
  theirs matches its best candidate better and on more masses (one at a time, each
  against the compound as merged so far). The merged compounds are then identified
  again. Saturation, from the data (user 2026-11-27, instead of the legacy list Xe-A,
  CO2-A, H2O): a compound with at least ``saturation_pieces`` such compounds in its
  range is taken as saturated (its large peaks break into pieces) and takes all of
  them, whatever the match gets, as long as it stays identified (score >= the
  ``suspect: min_score``; else the rule above: on v2/esper the count alone also took
  the coeluting compounds of e.g. cumene, which lost its identification). Not ported:
  H2O merging when not identified.
- Unfitted one-peak compounds (``drop_unfitted_single_peaks``): after the fit, the
  compounds of a single peak that the fit rejected are deleted with their peak; not
  those identified (user 2026-11-27: the fit rejects some real peaks, e.g. on the tail
  of a large one) or identified by hand.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RulesConfig:
    #: merge the unidentified compounds into an identified one when they improve its match
    noise_merge: bool = True
    #: an identified compound with at least this many unidentified ones in its range is
    #: saturated: it takes them all (0: none; v2/esper: the legacy saturators have a
    #: median of 15, the others 1, 99th percentile 21)
    saturation_pieces: int = 10
    #: delete the compounds of one peak that the fit rejected (and the peak)
    drop_unfitted_single_peaks: bool = True


def noise_merges(
    peaks: pd.DataFrame,
    compounds: pd.DataFrame,
    candidates: pd.DataFrame,
    libraries,
    saturation_pieces: int = 0,
    min_score: float = 0.0,
) -> dict[int, list[int]]:
    """The merges of the noise merge: identified compound_idx -> the compound_idx merged
    into it (in order). ``saturation_pieces``: see ``RulesConfig`` (0: none saturated);
    ``min_score``: the score a saturated compound keeps with all its pieces.

    ``peaks``: compound_idx, mass and height of the peaks as found; ``compounds``: indexed
    by compound_idx, with rt_mean_s, start_mean_s, stop_mean_s and valid_candidate;
    ``candidates``: with is_best and spectrum_ref; ``libraries``: an
    ``esper.libraries.LibraryCollection``.
    """
    from esper.libraries import Spectrum
    from esper.target_index import unit_mass

    best = (
        candidates[candidates["is_best"] == 1]
        .drop_duplicates("compound_idx")
        .set_index("compound_idx")["spectrum_ref"]
    )
    members = {
        int(i): g
        for i, g in peaks.dropna(subset=["compound_idx"]).groupby("compound_idx")
    }

    def spectrum(idxs: list[int]) -> Spectrum:
        g = pd.concat([members[i] for i in idxs if i in members])
        return Spectrum.from_pairs(g["mass"].map(unit_mass), g["height"])

    compounds = compounds.sort_index()
    valid = compounds["valid_candidate"] == 1
    unidentified = compounds[~valid]
    merged: set[int] = set()
    out = {}
    for idx, row in compounds[valid].iterrows():
        idx = int(idx)
        ref = libraries.resolve(best.get(idx))
        if ref is None or idx not in members:
            continue
        reference = libraries.spectrum(*ref)
        near = unidentified[
            (unidentified.index > idx)
            & unidentified["rt_mean_s"].between(row["start_mean_s"], row["stop_mean_s"])
        ]
        pieces = [
            int(i) for i in near.index if int(i) not in merged and int(i) in members
        ]
        if (
            saturation_pieces
            and len(pieces) >= saturation_pieces
            and libraries.score(spectrum([idx, *pieces]), reference)[0] >= min_score
        ):
            logger.debug(f"compound {idx}: saturated, {len(pieces)} pieces merged")
            merged.update(pieces)
            out[idx] = pieces
            continue
        group = [idx]
        for other in pieces:
            score, matched = libraries.score(spectrum(group), reference)
            with_score, with_matched = libraries.score(
                spectrum(group + [other]), reference
            )
            if with_score > score and with_matched > matched:
                group.append(other)
                merged.add(other)
        if len(group) > 1:
            out[idx] = group[1:]
    return out


def merge(
    peaks: pd.DataFrame, compounds: pd.DataFrame, merges: dict[int, list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The peaks and compounds of a run (as found, not stored yet) with the ``merges``
    (``noise_merges``) done: the peaks go to the main compound, the others are dropped,
    the main ones get their summaries again (``esper.fitting.compound_summaries``; their
    range kept, as the legacy)."""
    from esper.fitting import _set_max_of_cmp, compound_summaries

    into = {other: main for main, others in merges.items() for other in others}
    if not into:
        return peaks, compounds
    peaks = peaks.copy()
    peaks["compound_idx"] = peaks["compound_idx"].map(lambda i: into.get(i, i))
    mains = sorted(merges)
    _set_max_of_cmp(peaks, mains)
    compounds = compounds.drop(index=list(into))
    summaries = compound_summaries(peaks, compounds, mains)
    compounds.loc[mains, summaries.columns] = summaries.to_numpy()
    return peaks, compounds


def drop_unfitted_single_peaks(con: sqlite3.Connection, measurement_id: int) -> int:
    """Delete the compounds of a fitted run made of a single peak that the fit rejected
    (``fit_rt_s`` NULL), with that peak; not those identified (``valid_candidate``) or
    identified by hand. Their retention
    index references (if any) are dropped and the RIs of the compounds set again
    (single transaction). Returns the number of compounds deleted."""
    from esper import db
    from esper.fitting import fit_version
    from esper.retention_index import update_compound_ris

    mid = int(measurement_id)
    if fit_version(con, mid) is None:
        return 0
    idxs = [
        row[0]
        for row in con.execute(
            "SELECT p.compound_idx FROM peak p"
            " JOIN compound c USING (measurement_id, compound_idx)"
            " LEFT JOIN compound_identification ci USING (measurement_id, compound_idx)"
            " WHERE p.measurement_id = ? AND p.compound_idx IS NOT NULL"
            " AND ci.compound_idx IS NULL AND c.valid_candidate IS NOT 1"
            " GROUP BY p.compound_idx HAVING count(*) = 1 AND max(p.fit_rt_s) IS NULL",
            (mid,),
        )
    ]
    if not idxs:
        return 0
    marks = ", ".join("?" * len(idxs))
    with con:
        for table in (
            "peak",
            "extraction_peak",
            "extraction_compound",
            "measurement_ri_reference",
            "compound",  # candidates: cascade
        ):
            con.execute(
                f"DELETE FROM {table} WHERE measurement_id = ?"
                f" AND compound_idx IN ({marks})",
                (mid, *idxs),
            )
        update_compound_ris(con, mid)
        db.update_counts(con, mid)
    return len(idxs)
