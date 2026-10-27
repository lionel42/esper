"""Current selection shared by the views."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from esper.gui.data import EsperData


class Selection(QObject):
    """What is currently looked at: measurement, compound, mass and rt.

    Two kinds of compounds are distinguished:

    - the *compound*: a group of co-eluting peaks detected in the measurement
      (``compound_idx``). ``candidate`` is the name of alpinac's best library
      candidate for it, used to follow the same compound in other measurements.
      An alias is resolved: the candidate is then the name of its target.
    - the *target*: a substance of the target list (``target``), with its
      expected retention index and ions. It is the target of the candidate, or
      a target chosen in the target table, which may not be found in the
      measurement (then ``compound_idx`` and ``candidate`` are None).

    Several compounds can be selected together (``compound_idxs``, e.g. to merge
    them): they are shown combined, ``compound_idx`` is the main one (lowest index).

    ``label`` is the name shown for the selection: the target, else the candidate.

    ``reference`` is the library spectrum compared with the compound, as
    (library, name); None means the default (alpinac's best candidate). It is
    reset when the compound or the measurement changes.

    ``changed`` is emitted with the set of fields that changed.
    """

    changed = Signal(set)

    def __init__(self, data: EsperData) -> None:
        super().__init__()
        self.data = data
        self.measurement_id: int | None = None
        self.compound_idx: int | None = None
        self.compound_idxs: tuple[int, ...] = ()  # compound_idx and the others selected with it
        self.candidate: str | None = None
        self.mass: int | None = None  # None: TIC
        self.rt: float | None = None
        self.target: str | None = None
        self.reference: tuple[str, str] | None = None

    def _set(self, **values) -> None:
        if "compound_idx" in values and "compound_idxs" not in values:
            compound_idx = values["compound_idx"]
            values["compound_idxs"] = () if compound_idx is None else (compound_idx,)
        if "reference" not in values and {"measurement_id", "compound_idx"} & {
            k for k, v in values.items() if getattr(self, k) != v
        }:
            values["reference"] = None
        changed = {k for k, v in values.items() if getattr(self, k) != v}
        for k in changed:
            setattr(self, k, values[k])
        if changed:
            self.changed.emit(changed)

    @property
    def label(self) -> str | None:
        return self.target if self.target is not None else self.candidate

    def _compound_values(self, measurement_id: int, compound_idx: int) -> dict:
        compound = self.data.compounds(measurement_id).loc[compound_idx]
        return {
            "compound_idx": compound_idx,
            "mass": self.data.main_ion(measurement_id, compound_idx),
            "rt": float(compound["rt_mean_s"]),
        }

    def select_compound(self, measurement_id: int, compound_idx: int) -> None:
        self.select_compounds(measurement_id, [compound_idx])

    def select_compounds(self, measurement_id: int, compound_idxs) -> None:
        """Select compounds together: the one with the lowest index is the main one."""
        idxs = tuple(sorted({int(i) for i in compound_idxs}))
        candidate = self.data.candidate_of(measurement_id, idxs[0])
        self._set(
            measurement_id=measurement_id,
            candidate=candidate,
            target=self.data.target_of(candidate),
            **self._compound_values(measurement_id, idxs[0]),
            compound_idxs=idxs,
        )

    def _target_values(self, measurement_id: int, target: str, keep_mass: bool) -> dict:
        """Compound of the target in the measurement, or its expected rt and main ion."""
        compound_idx = self.data.find_target(measurement_id, target)
        if compound_idx is not None:
            values = self._compound_values(measurement_id, compound_idx)
            values["candidate"] = self.data.candidate_of(measurement_id, compound_idx)
        else:
            masses = self.data.target_masses(target)
            rt = self.data.expected_rt(measurement_id, target)
            values = {
                "compound_idx": None,
                "candidate": None,
                # No retention index: no position, the whole chromatogram is shown
                "rt": rt,
                "mass": masses[0] if masses else self.mass,
            }
        if keep_mass and self.mass is not None:
            del values["mass"]
        return values

    def select_target(self, target: str) -> None:
        """Go to the compound of the target, or to where it is expected."""
        values = {"target": target}
        if self.measurement_id is not None:
            values.update(self._target_values(self.measurement_id, target, keep_mass=False))
        self._set(**values)

    def select_peak(self, measurement_id: int, compound_idx: int, mass: int, rt: float) -> None:
        """Select the compound of a peak, showing the mass and rt of that peak."""
        candidate = self.data.candidate_of(measurement_id, compound_idx)
        self._set(
            measurement_id=measurement_id,
            compound_idx=compound_idx,
            candidate=candidate,
            target=self.data.target_of(candidate),
            mass=mass,
            rt=rt,
        )

    def select_measurement(self, measurement_id: int) -> None:
        """Change measurement, following the same compound (or target) and keeping the mass."""
        if self.target is not None:
            self._set(measurement_id=measurement_id, **self._target_values(measurement_id, self.target, keep_mass=True))
            return
        values = {"measurement_id": measurement_id, "compound_idx": None}
        compound_idx = self.data.find_compound(measurement_id, self.candidate, self.rt)
        if compound_idx is not None:
            compound_values = self._compound_values(measurement_id, compound_idx)
            if self.mass is not None:
                del compound_values["mass"]
            values.update(compound_values)
        self._set(**values)

    def select_candidate(self, candidate: str | None) -> None:
        """Follow the compounds with that candidate name (and its target, if any)."""
        values = {"candidate": candidate, "compound_idx": None, "target": self.data.target_of(candidate)}
        if self.measurement_id is not None:
            compound_idx = self.data.find_compound(self.measurement_id, candidate)
            if compound_idx is not None:
                values.update(self._compound_values(self.measurement_id, compound_idx))
        self._set(**values)

    def identification_edited(self) -> None:
        """The identification of a compound was edited (or compounds merged): the selected compound may have a new name.

        ``changed`` is emitted with ``"identification"``, for what shows the names to reload.
        """
        values = {}
        if self.measurement_id is not None and self.compound_idx is not None:
            candidate = self.data.candidate_of(self.measurement_id, self.compound_idx)
            values = {"candidate": candidate, "target": self.data.target_of(candidate)}
        changed = {k for k, v in values.items() if getattr(self, k) != v}
        for k in changed:
            setattr(self, k, values[k])
        self.changed.emit(changed | {"identification"})

    def select_reference(self, reference: tuple[str, str] | None) -> None:
        self._set(reference=reference)

    def select_mass(self, mass: int | None) -> None:
        self._set(mass=mass)

    def select_rt(self, rt: float | None) -> None:
        self._set(rt=rt)
