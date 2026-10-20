"""Current selection shared by the views."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from esper.gui.data import EsperData


class Selection(QObject):
    """What is currently looked at: measurement, compound, mass and rt.

    ``target`` is the target compound followed, if any: the target of the
    selected compound, or a target chosen in the target table (which may not
    be found in the measurement, then ``compound_idx`` is None).

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
        self.cmp_name: str | None = None
        self.mass: int | None = None  # None: TIC
        self.rt: float | None = None
        self.target: str | None = None
        self.reference: tuple[str, str] | None = None

    def _set(self, **values) -> None:
        if "reference" not in values and {"measurement_id", "compound_idx"} & {
            k for k, v in values.items() if getattr(self, k) != v
        }:
            values["reference"] = None
        changed = {k for k, v in values.items() if getattr(self, k) != v}
        for k in changed:
            setattr(self, k, values[k])
        if changed:
            self.changed.emit(changed)

    def _compound_values(self, measurement_id: int, compound_idx: int) -> dict:
        compound = self.data.compounds(measurement_id).loc[compound_idx]
        return {
            "compound_idx": compound_idx,
            "mass": self.data.main_ion(measurement_id, compound_idx),
            "rt": float(compound["rt_mean_s"]),
        }

    def select_compound(self, measurement_id: int, compound_idx: int) -> None:
        name = self.data.compounds(measurement_id).loc[compound_idx, "compound_name"]
        self._set(
            measurement_id=measurement_id,
            cmp_name=name,
            target=self.data.target_of(name),
            **self._compound_values(measurement_id, compound_idx),
        )

    def _target_values(self, measurement_id: int, target: str, keep_mass: bool) -> dict:
        """Compound of the target in the measurement, or its expected rt and main ion."""
        compound_idx = self.data.find_target(measurement_id, target)
        if compound_idx is not None:
            values = self._compound_values(measurement_id, compound_idx)
            values["cmp_name"] = self.data.compounds(measurement_id).loc[compound_idx, "compound_name"]
        else:
            masses = self.data.target_masses(target)
            rt = self.data.expected_rt(measurement_id, target)
            values = {
                "compound_idx": None,
                "cmp_name": target,
                # No retention index: no position, the whole chromatogram is shown
                "rt": rt,
                "mass": masses[0] if masses else self.mass,
            }
        if keep_mass and self.mass is not None:
            del values["mass"]
        return values

    def select_target(self, target: str) -> None:
        """Go to the compound of the target, or to where it is expected."""
        values = {"target": target, "cmp_name": target}
        if self.measurement_id is not None:
            values.update(self._target_values(self.measurement_id, target, keep_mass=False))
        self._set(**values)

    def select_peak(self, measurement_id: int, compound_idx: int, mass: int, rt: float) -> None:
        """Select the compound of a peak, showing the mass and rt of that peak."""
        name = self.data.compounds(measurement_id).loc[compound_idx, "compound_name"]
        self._set(
            measurement_id=measurement_id,
            compound_idx=compound_idx,
            cmp_name=name,
            target=self.data.target_of(name),
            mass=mass,
            rt=rt,
        )

    def select_measurement(self, measurement_id: int) -> None:
        """Change measurement, following the same compound (or target) and keeping the mass."""
        if self.target is not None:
            self._set(measurement_id=measurement_id, **self._target_values(measurement_id, self.target, keep_mass=True))
            return
        values = {"measurement_id": measurement_id, "compound_idx": None}
        compound_idx = self.data.find_compound(measurement_id, self.cmp_name, self.rt)
        if compound_idx is not None:
            compound_values = self._compound_values(measurement_id, compound_idx)
            if self.mass is not None:
                del compound_values["mass"]
            values.update(compound_values)
        self._set(**values)

    def select_cmp_name(self, name: str | None) -> None:
        values = {"cmp_name": name, "compound_idx": None, "target": self.data.target_of(name)}
        if self.measurement_id is not None:
            compound_idx = self.data.find_compound(self.measurement_id, name)
            if compound_idx is not None:
                values.update(self._compound_values(self.measurement_id, compound_idx))
        self._set(**values)

    def refresh(self) -> None:
        """Tell all the panels to redraw, e.g. after the database changed."""
        self.changed.emit({"measurement_id", "compound_idx", "cmp_name", "mass", "rt", "target", "reference"})

    def select_reference(self, reference: tuple[str, str] | None) -> None:
        self._set(reference=reference)

    def select_mass(self, mass: int | None) -> None:
        self._set(mass=mass)

    def select_rt(self, rt: float | None) -> None:
        self._set(rt=rt)
