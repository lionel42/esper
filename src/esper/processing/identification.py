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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from esper.fitting import ExpectedTarget
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
    skip=(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The peaks and compounds of a run with the targets found named (see the module
    docstring); ``time``: of the run (the version of the targets then); ``skip``: the
    targets not looked for (e.g. found already)."""
    config = config or TargetSearchConfig()
    peaks, compounds = peaks.copy(), compounds.copy()
    units = peaks["mass"].map(unit_mass)
    rts = peaks["rt_s"].to_numpy(dtype=float)
    matches = []  # (target, compound_idx, distance, ions)
    for target in targets.table.index:
        if target in skip:
            continue
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


@dataclass(frozen=True)
class TargetSearch:
    """The target search of a data directory (its targets and ``target:`` config), for
    the fit of the runs it identified (``esper.fitting.fit_measurement``): built once,
    passed to the fit of each run."""

    targets: TargetIndex
    config: TargetSearchConfig = field(default_factory=TargetSearchConfig)

    def expected(self, names: pd.Series, time=None) -> dict[int, ExpectedTarget]:
        """The compounds named a target (``names``: compound_name by compound_idx) in a
        run at ``time``: where their target is expected, and its ions."""
        out = {}
        for idx, name in names.items():
            target = self.targets.target_of(name) if isinstance(name, str) else None
            found = (
                None
                if target is None
                else _expected(self.targets, target, time, self.config)
            )
            if found is not None:
                ions = frozenset(self.targets.masses_at(target, time))
                out[int(idx)] = ExpectedTarget(found[0], ions)
        return out

    def name(
        self, table: pd.DataFrame, made: list[int], names: pd.Series, time=None
    ) -> dict[int, str]:
        """Target search among the compounds made by the fit (``made``, in the peak
        table ``table``), not of the targets found already (``names``: of the other
        compounds): their names, by compound_idx. ``table`` gets the peaks of the
        targets found co-eluting split into compounds of their own (numbered after)."""
        rows = table["compound_idx"].isin(made)
        peaks = table.loc[rows, ["mass", "rt_s", "compound_idx", "prominence"]].assign(
            start_s=np.nan, stop_s=np.nan
        )
        compounds = pd.DataFrame(
            {"compound_name": None, "valid_candidate": 0, "is_ref_ri": 0},
            index=pd.Index(made, name="compound_idx"),
        ).assign(n_peaks=peaks.groupby("compound_idx").size().reindex(made))
        found = set(names.dropna().map(self.targets.target_of))
        peaks, compounds = search_targets(
            peaks, compounds, self.targets, time, self.config, skip=found
        )
        table.loc[peaks.index, "compound_idx"] = peaks["compound_idx"].astype("Int64")
        named = compounds["compound_name"].dropna()
        return {int(i): name for i, name in named.items()}


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
        # one candidate per name; not those of no mass in common (score 0), as the legacy
        found = [
            match
            for match in libraries.best_matches(
                spectrum, config.n_candidates, distinct=True
            )
            if match.score > 0
        ]
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


# --- retention indices ------------------------------------------------------------------


def retention_indices(
    compounds: pd.DataFrame,
    candidates: pd.DataFrame | None,
    references: pd.DataFrame,
    targets: TargetIndex | None = None,
    ri_config=None,
    suspect: SuspectScreeningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """The retention indices of a run (``esper.retention_index``): its references found,
    the RI of its compounds, and with suspect screening (``candidates``) the RI of the
    candidates and the identification by the combined score (as the legacy steps
    ``RI_CALIBRANTS``, ``BEST_CANDIDATE_RI``, ``VALIDATION``).

    The best candidate of a compound gives its references (its spectral best, score
    > ``ri_config.min_score``; target search: its target); a reference is named as such
    (e.g. Xe-A), identified, ``is_ref_ri``. The RI of a candidate is the one of its
    target (aliases resolved; NaN if none or without rt); the best candidate is then the
    one of the best combined score (``esper.libraries.combined_score``, ties: the
    spectral order), the compound is identified if its spectral score is at least
    ``suspect.min_score`` and its RI within ``suspect.max_ri_diff``. Without two
    references the compounds have no RI, and no compound is identified by suspect
    screening.

    :return: the compounds, the candidates and the references found
        (``esper.retention_index.FOUND_COLUMNS``).
    """
    from esper.libraries import combined_score
    from esper.retention_index import find_references, rt_to_ri

    suspect = suspect or SuspectScreeningConfig()
    compounds = compounds.copy()
    if candidates is not None:
        first = candidates[candidates["rank"] == 1].set_index("compound_idx")
        best = pd.DataFrame({"name": first["candidate_name"], "score": first["score"]})
    else:
        named = compounds["compound_name"].dropna()
        best = pd.DataFrame({"name": named, "score": 1.0})
    found = find_references(compounds, best, references, targets, ri_config)
    ri_of = rt_to_ri(found)
    if ri_of is None:
        logger.warning(
            f"{len(found)} retention index reference(s) found: the compounds have no"
            " retention index"
        )
        compounds["retention_index"] = np.nan
    else:
        compounds["retention_index"] = np.round(
            ri_of(compounds["rt_mean_s"].to_numpy(dtype=float)), 2
        )
    compounds["is_ref_ri"] = 0
    if candidates is not None:
        candidates = candidates.copy()
        target_ri = targets.table["ri"] if targets is not None else pd.Series(dtype=float)

        def ri_of_name(name):
            target = targets.target_of(name) if targets is not None else None
            return target_ri.get(target, np.nan) if target is not None else np.nan

        candidates["ri"] = candidates["candidate_name"].map(ri_of_name).astype(float)
        candidates["ri_diff"] = (
            candidates["compound_idx"].map(compounds["retention_index"]).astype(float)
            - candidates["ri"]
        )
        candidates["score_combined"] = [
            combined_score(score, diff)
            for score, diff in zip(candidates["score"], candidates["ri_diff"], strict=True)
        ]
        # the best combined score of each compound (the first in the spectral order)
        order = candidates.sort_values(
            ["compound_idx", "score_combined", "rank"], ascending=[True, False, True]
        )
        best_rows = order.drop_duplicates("compound_idx")
        candidates["is_best"] = candidates.index.isin(best_rows.index).astype(int)
        chosen = best_rows.set_index("compound_idx")
        compounds.loc[chosen.index, "compound_name"] = chosen["candidate_name"]
        compounds.loc[chosen.index, "valid_candidate"] = (
            (chosen["score"] >= suspect.min_score)
            & (chosen["ri_diff"].abs() < suspect.max_ri_diff)
        ).astype(int)
    for ref in found.itertuples():
        compounds.loc[ref.compound_idx, ["compound_name", "valid_candidate", "is_ref_ri"]] = (
            ref.name,
            1,
            1,
        )
    compounds["valid_candidate"] = compounds["valid_candidate"].fillna(0).astype(int)
    return compounds, candidates, found
