"""Identification of the compounds of a run, by the method of the data directory.

Both methods name compounds found by the detection (:mod:`esper.processing.detection`);
the compounds that match nothing are kept, without a name.

Target search (``target``): a target is the compound with a peak on one of its ions
(``target_ion``, as unit masses: :func:`esper.target_index.unit_mass`) within its window
around its expected rt (``target_rt`` at the time of the run, e.g. GCwerks peakid; else
its ``rt_ref_s`` and ``TargetSearchConfig.window_s``). Its main ion is looked at first,
then the others in order; among the peaks of that ion in the window, the nearest to the
expected rt. When targets fall on the same compound (they co-elute, the detection grouped
their peaks), the nearest one keeps it and the peaks of each other one on its own ions
become a compound of their own.

Suspect screening (``suspect``): the spectrum of each compound (the heights of its peaks)
is matched against the spectral libraries; the best candidates are kept
(``compound_candidate``) and the best one names the compound, valid from
``SuspectScreeningConfig.min_score`` on (as alpinac).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from esper.processing.config import SuspectScreeningConfig, TargetSearchConfig
from esper.target_index import TargetIndex, unit_mass

logger = logging.getLogger(__name__)

CANDIDATE_COLUMNS = [
    "compound_idx",
    "rank",
    "candidate_name",
    "spectrum_ref",
    "library",
    "score",
    "ri",
    "ri_diff",
    "score_combined",
    "is_best",
]


# --- compounds from their peaks -------------------------------------------------------


def compound_summaries(peaks: pd.DataFrame, compound_idxs) -> pd.DataFrame:
    """The summary columns of compounds from their peaks (as peaxart's grouping), indexed
    by compound_idx; ``is_max_of_cmp`` of their peaks is set again (in ``peaks``)."""
    rows = {}
    for idx in compound_idxs:
        members = peaks[peaks["compound_idx"] == idx]
        if members.empty:
            continue
        largest = members["prominence"].idxmax()
        peaks.loc[members.index, "is_max_of_cmp"] = 0
        peaks.loc[largest, "is_max_of_cmp"] = 1
        rows[int(idx)] = {
            "rt_mean_s": members["rt_s"].mean(),
            "rt_std_s": members["rt_s"].std(),
            "n_peaks": len(members),
            "prominence_sum": members["prominence"].sum(),
            "start_mean_s": members["start_s"].mean(),
            "start_std_s": members["start_s"].std(),
            "stop_mean_s": members["stop_s"].mean(),
            "stop_std_s": members["stop_s"].std(),
            "main_ion_mass": members.at[largest, "mass"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# --- target search ----------------------------------------------------------------------


def _expected(
    targets: TargetIndex, target: str, time, config: TargetSearchConfig
) -> tuple[float, float] | None:
    """(rt, window) of a target in a run at ``time``, None if it is not expected."""
    if targets.has_rts(target):
        found = targets.rt_at(target, time)  # None: not identified then
        if found is None:
            return None
        rt, window = found
        return rt, window if np.isfinite(window) else config.window_s
    rt = targets.table.at[target, "rt_ref_s"] if "rt_ref_s" in targets.table else None
    if rt is None or not np.isfinite(rt):
        return None
    return float(rt), config.window_s


def search_targets(
    peaks: pd.DataFrame,
    compounds: pd.DataFrame,
    targets: TargetIndex,
    time=None,
    config: TargetSearchConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The peaks and compounds of a run with the targets found named (see the module
    docstring); ``time``: of the run (the version of the targets then)."""
    config = config or TargetSearchConfig()
    peaks, compounds = peaks.copy(), compounds.copy()
    units = peaks["mass"].map(unit_mass)
    rts = peaks["rt_s"].to_numpy(dtype=float)
    matches = []  # (target, compound_idx, distance, ions)
    for target in targets.table.index:
        expected = _expected(targets, target, time, config)
        ions = targets.masses_at(target, time)
        if expected is None or not ions:
            continue
        rt, window = expected
        near = (np.abs(rts - rt) <= window / 2) & peaks["compound_idx"].notna()
        for ion in ions:
            on = peaks[near & (units == ion)]
            if len(on):
                distance = (on["rt_s"] - rt).abs()
                matches.append(
                    (
                        target,
                        int(on.at[distance.idxmin(), "compound_idx"]),
                        float(distance.min()),
                        ions,
                    )
                )
                break

    changed = set()
    next_idx = int(compounds.index.max()) + 1 if len(compounds) else 0
    for compound_idx, group in pd.DataFrame(
        matches, columns=["target", "compound_idx", "distance", "ions"]
    ).groupby("compound_idx"):
        group = group.sort_values("distance")
        keeper = group.iloc[0]
        compounds.loc[compound_idx, ["compound_name", "valid_candidate"]] = (
            keeper["target"],
            1,
        )
        for _, other in group.iloc[1:].iterrows():
            own = (
                (peaks["compound_idx"] == compound_idx)
                & units.isin(other["ions"])
                & ~units.isin(keeper["ions"])
            )
            if not own.any():
                logger.debug(
                    f"{other['target']}: on the compound of {keeper['target']}, no"
                    " ion of its own"
                )
                continue
            peaks.loc[own, "compound_idx"] = next_idx
            compounds.loc[next_idx] = {
                "compound_name": other["target"],
                "valid_candidate": 1,
                "is_ref_ri": 0,
            }
            changed |= {compound_idx, next_idx}
            next_idx += 1
    if changed:
        summaries = compound_summaries(peaks, sorted(changed))
        compounds.loc[summaries.index, summaries.columns] = summaries
    compounds["valid_candidate"] = compounds["valid_candidate"].fillna(0).astype(int)
    compounds["is_ref_ri"] = compounds["is_ref_ri"].fillna(0).astype(int)
    compounds["n_peaks"] = compounds["n_peaks"].astype(int)
    found = compounds["compound_name"].notna().sum()
    logger.debug(f"{found} targets found, {len(compounds) - found} other compounds")
    return peaks, compounds


# --- suspect screening ------------------------------------------------------------------


def screen_suspects(
    peaks: pd.DataFrame,
    compounds: pd.DataFrame,
    libraries,
    config: SuspectScreeningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The compounds named by their best library candidate, and the candidates (see the
    module docstring); ``libraries``: an ``esper.libraries.LibraryCollection``."""
    from esper.libraries import Spectrum, combined_score

    config = config or SuspectScreeningConfig()
    compounds = compounds.copy()
    rows = []
    for compound_idx, members in peaks.dropna(subset=["compound_idx"]).groupby(
        "compound_idx"
    ):
        spectrum = Spectrum.from_pairs(
            members["mass"].map(unit_mass), members["height"]
        )
        found = libraries.best_matches(spectrum, config.n_candidates)
        for rank, match in enumerate(found, start=1):
            rows.append(
                {
                    "compound_idx": int(compound_idx),
                    "rank": rank,
                    "candidate_name": match.name,
                    "spectrum_ref": f"{match.library.split('_', 1)[0]}_{match.name}",
                    "library": match.library.split("_", 1)[0],
                    "score": match.score,
                    "ri": np.nan,
                    "ri_diff": np.nan,
                    "score_combined": combined_score(match.score, None),
                    "is_best": int(rank == 1),
                }
            )
        if found:
            compounds.loc[compound_idx, "compound_name"] = found[0].name
            compounds.loc[compound_idx, "valid_candidate"] = int(
                found[0].score >= config.min_score
            )
    return compounds, pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
