"""Current selection shared by the views."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from esper.gui.data import EsperData


class Selection(QObject):
    """What is currently looked at: measurement, compound, mass and rt.

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

    def _set(self, **values) -> None:
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
            **self._compound_values(measurement_id, compound_idx),
        )

    def select_peak(self, measurement_id: int, compound_idx: int, mass: int, rt: float) -> None:
        """Select the compound of a peak, showing the mass and rt of that peak."""
        name = self.data.compounds(measurement_id).loc[compound_idx, "compound_name"]
        self._set(
            measurement_id=measurement_id,
            compound_idx=compound_idx,
            cmp_name=name,
            mass=mass,
            rt=rt,
        )

    def select_measurement(self, measurement_id: int) -> None:
        """Change measurement, following the same compound and keeping the mass."""
        values = {"measurement_id": measurement_id, "compound_idx": None}
        compound_idx = self.data.find_compound(measurement_id, self.cmp_name, self.rt)
        if compound_idx is not None:
            compound_values = self._compound_values(measurement_id, compound_idx)
            if self.mass is not None:
                del compound_values["mass"]
            values.update(compound_values)
        self._set(**values)

    def select_cmp_name(self, name: str | None) -> None:
        values = {"cmp_name": name, "compound_idx": None}
        if self.measurement_id is not None:
            compound_idx = self.data.find_compound(self.measurement_id, name)
            if compound_idx is not None:
                values.update(self._compound_values(self.measurement_id, compound_idx))
        self._set(**values)

    def select_mass(self, mass: int | None) -> None:
        self._set(mass=mass)

    def select_rt(self, rt: float | None) -> None:
        self._set(rt=rt)
