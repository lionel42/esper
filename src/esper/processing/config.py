"""How the peaks of a data directory are processed: the ``processing:`` of its esper.yaml.

::

    processing:
      method: target          # how the compounds are identified: target | suspect
      baseline:               # peaxart.BaselineConfig, before the detection (false: none)
        lam: 1.0e+6
      detection:              # peaxart.LegacyDeconvolutionConfig
        prominence: 200
      fit:                    # peaxart.ChromatogramFitConfig
        peak_shape: emg
      target:                 # target search (TargetSearchConfig)
        window_s: 20
      suspect:                # suspect screening (SuspectScreeningConfig)
        min_score: 0.8

Every key is optional: what is not given has its default (peaxart's, and esper's
``FIT_CONFIG`` for the fit). The method is the data directory's, not the instrument's:
``esper peaks --method`` declares it.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import peaxart as px

from esper import db
from esper.deconvolution import FIT_CONFIG
from esper.processing import METHODS

logger = logging.getLogger(__name__)

PROCESSING_KEY = "processing"


class ProcessingConfigError(ValueError):
    """The ``processing:`` of an esper.yaml is not valid."""


@dataclass(frozen=True)
class TargetSearchConfig:
    # Full width of the identification window of the targets without one (e.g. a
    # target list: around its rt_ref_s), s
    window_s: float = 20.0


@dataclass(frozen=True)
class SuspectScreeningConfig:
    # Library candidates kept for each compound
    n_candidates: int = 3
    # Spectral score of the best candidate from which the compound is identified
    # (valid_candidate); below it, it keeps the name, as not valid
    min_score: float = 0.8


@dataclass(frozen=True)
class ProcessingConfig:
    method: str | None = None
    # None: the baseline is not removed before the detection
    baseline: px.BaselineConfig | None = field(default_factory=px.BaselineConfig)
    detection: px.LegacyDeconvolutionConfig = field(
        default_factory=px.LegacyDeconvolutionConfig
    )
    fit: px.ChromatogramFitConfig = FIT_CONFIG
    target: TargetSearchConfig = field(default_factory=TargetSearchConfig)
    suspect: SuspectScreeningConfig = field(default_factory=SuspectScreeningConfig)


# Sections of ``processing:`` and their defaults
_DEFAULTS = ProcessingConfig()
SECTIONS = ("baseline", "detection", "fit", "target", "suspect")


def _value(default, value, where: str):
    """``value`` of the esper.yaml as the type of ``default``."""
    try:
        if isinstance(default, Enum):
            names = ", ".join(str(e.value) for e in type(default))
            if value not in {e.value for e in type(default)}:
                raise ValueError(f"one of {names}")
            return type(default)(value)
        if isinstance(default, tuple):
            if not isinstance(value, list | tuple):
                raise TypeError("a list")
            kinds = [type(d) for d in default]
            return tuple(
                kinds[min(i, len(kinds) - 1)](v) if kinds else v
                for i, v in enumerate(value)
            )
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise TypeError("true or false")
            return value
        if isinstance(default, int | float):
            # YAML reads 1e7 as a string (1.0e+7 is a number)
            number = float(value) if isinstance(value, str) else value
            if isinstance(number, bool) or not isinstance(number, int | float):
                raise TypeError("a number")
            if isinstance(default, int) and float(number).is_integer():
                return int(number)
            return float(number)
        return value
    except (TypeError, ValueError) as e:
        raise ProcessingConfigError(f"{where}: {value!r} is not valid ({e})") from None


def _section(default, values, where: str):
    """The dataclass ``default`` with the ``values`` given."""
    if not isinstance(values, dict):
        raise ProcessingConfigError(f"{where}: expected keys, not {values!r}")
    fields = {f.name for f in dataclasses.fields(default)}
    unknown = sorted(set(values) - fields)
    if unknown:
        raise ProcessingConfigError(
            f"{where}: unknown {', '.join(unknown)} (known: {', '.join(sorted(fields))})"
        )
    return dataclasses.replace(
        default,
        **{
            k: _value(getattr(default, k), v, f"{where}.{k}") for k, v in values.items()
        },
    )


def processing_config(section: dict | None, where: str = PROCESSING_KEY):
    """The ProcessingConfig of a ``processing:`` mapping (None: all defaults)."""
    section = section or {}
    if not isinstance(section, dict):
        raise ProcessingConfigError(f"{where}: expected keys, not {section!r}")
    unknown = sorted(set(section) - {"method", *SECTIONS})
    if unknown:
        raise ProcessingConfigError(
            f"{where}: unknown {', '.join(unknown)}"
            f" (known: method, {', '.join(SECTIONS)})"
        )
    method = section.get("method")
    if method is not None and method not in METHODS:
        raise ProcessingConfigError(
            f"{where}.method: {method!r}, expected one of {', '.join(METHODS)}"
        )
    values = {"method": method}
    for name in SECTIONS:
        given = section.get(name)
        default = getattr(_DEFAULTS, name)
        if name == "baseline" and given is False:
            values[name] = None
        elif given is not None:
            values[name] = _section(
                default or px.BaselineConfig(), given, f"{where}.{name}"
            )
    config = dataclasses.replace(_DEFAULTS, **values)
    if config.fit.convention != FIT_CONFIG.convention:
        # The stored widths and areas are of this convention (esper.fitting)
        raise ProcessingConfigError(
            f"{where}.fit.convention: only {FIT_CONFIG.convention!r}"
        )
    return config


def read_processing_config(data_dir: Path) -> ProcessingConfig:
    """The processing of a data directory (its esper.yaml; defaults without it)."""
    where = f"{Path(data_dir) / db.CONFIG_FILENAME}: {PROCESSING_KEY}"
    return processing_config(db.read_config(data_dir).get(PROCESSING_KEY), where)


def resolve_method(data_dir: Path, method: str | None = None) -> str | None:
    """The method of a data directory, declared in its esper.yaml if given and not yet.

    ``method``: as asked (``--method``), must be the declared one. None if neither.
    """
    declared = read_processing_config(data_dir).method
    if method is None:
        return declared
    if method not in METHODS:
        raise ProcessingConfigError(
            f"unknown method {method!r}, expected one of {', '.join(METHODS)}"
        )
    if declared is not None:
        if method != declared:
            raise ProcessingConfigError(
                f"{data_dir} uses {declared} (processing: method: of its"
                f" {db.CONFIG_FILENAME}), not {method}: change it there to change it"
            )
        return declared
    try:
        db.add_config_entries(data_dir, PROCESSING_KEY, {"method": method})
    except ValueError as e:  # e.g. processing: written on one line ({...})
        raise ProcessingConfigError(f"{e}; add 'method: {method}' to it") from None
    logger.info(f"{Path(data_dir) / db.CONFIG_FILENAME}: method {method} declared")
    return method
