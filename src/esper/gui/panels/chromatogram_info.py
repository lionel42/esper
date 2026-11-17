"""The details shown at the top left of the chromatogram panel: the run, the selected
compound and target, the peak of the compound at the selected mass (its fit, baseline and
integration) and how the peaks are processed.

Each line is an :class:`InfoItem`; which ones are shown is a setting
(``chromatogram_info``: key -> shown, the items not in it have their ``default``), chosen
in the settings dialog.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd

from esper.fitting import FIT_VERSION, SOURCE_FIT
from esper.gui.data import TIC, EsperData
from esper.gui.labels import combined_label, compound_label
from esper.gui.selection import Selection
from esper.gui.settings import Integration

# Groups of the items, in display order
RUN, COMPOUND, TARGET, PEAK, PROCESSING = (
    "Run",
    "Compound",
    "Target",
    "Peak",
    "Processing",
)
GROUPS = (RUN, COMPOUND, TARGET, PEAK, PROCESSING)
IDENTIFICATIONS = {"target": "target search", "suspect": "suspect screening"}


@dataclass(frozen=True)
class InfoItem:
    key: str  # in the settings
    group: str
    label: str  # in the panel
    description: str  # in the settings dialog
    default: bool = True


INFO_ITEMS: tuple[InfoItem, ...] = (
    InfoItem("sample", RUN, "sample", "Sample type"),
    InfoItem("port", RUN, "port", "Inlet port"),
    InfoItem("acquired", RUN, "acquired", "Start of the acquisition"),
    InfoItem(
        "instrument", RUN, "instrument", "Instrument, peak source and identification"
    ),
    InfoItem("n_peaks", RUN, "peaks", "Number of peaks of the run"),
    InfoItem("n_compounds", RUN, "compounds", "Number of compounds of the run"),
    InfoItem("mass_cal", RUN, "mass cal", "Mass calibration (TOF runs)"),
    InfoItem("fitted", RUN, "fitted", "Version and date of the fit of the run"),
    InfoItem("compound", COMPOUND, "cmp", "Name and index of the compound"),
    InfoItem("cmp_rt", COMPOUND, "rt", "Mean rt of its peaks"),
    InfoItem("cmp_ri", COMPOUND, "RI", "Retention index (if used)"),
    InfoItem("spectrum", COMPOUND, "spectrum", "Best library spectrum (if used)"),
    InfoItem("score", COMPOUND, "score", "Library match score (if used)"),
    InfoItem("valid", COMPOUND, "valid", "Valid candidate"),
    InfoItem("cmp_peaks", COMPOUND, "n peaks", "Number of peaks (masses)", False),
    InfoItem("target", TARGET, "target", "Selected target, whether found"),
    InfoItem("expected_ri", TARGET, "expected RI", "Retention index (if used)"),
    InfoItem("expected_rt", TARGET, "expected rt", "rt expected in this run"),
    InfoItem("expected_ions", TARGET, "expected ions", "Ions, main first"),
    InfoItem(
        "target_window", TARGET, "id window", "Identification window (GCwerks peakid)"
    ),
    InfoItem("peak_rt", PEAK, "peak rt", "Apex rt (of the fit, ± 1σ)"),
    InfoItem(
        "peak_height", PEAK, "height", "Height (of the fit: above the baseline)"
    ),
    InfoItem("peak_area", PEAK, "area", "Area of the fit (± 1σ)"),
    InfoItem("peak_width", PEAK, "FWHM", "Width at half maximum of the extraction"),
    InfoItem("peak_shape", PEAK, "shape", "Fitted peak function and parameters"),
    InfoItem("peak_sn", PEAK, "S/N", "Signal to noise of the fit"),
    InfoItem("peak_baseline", PEAK, "baseline", "Background at the apex, its slope"),
    InfoItem(
        "peak_background",
        PEAK,
        "background",
        "Background of the fit: constant, or linear / exponential (from its slope)",
    ),
    InfoItem("integration", PEAK, "integration", "Integration drawn"),
    InfoItem(
        "peak_override", PEAK, "override", "Fit override the peak was fitted with"
    ),
    InfoItem(
        "peak_source", PEAK, "from", "Extraction peak, or split by the fit", False
    ),
    InfoItem("prominence", PEAK, "prominence", "Prominence (extraction)", False),
    InfoItem("peak_bounds", PEAK, "bounds", "Half maximum crossings", False),
    InfoItem("tailing_ratio", PEAK, "tailing ratio", "Tailing ratio", False),
    InfoItem("flatness", PEAK, "flatness", "Flatness of the apex", False),
    InfoItem(
        "detection_baseline",
        PROCESSING,
        "detection",
        "Peak detection and its baseline removal",
    ),
    InfoItem("fit_config", PROCESSING, "fit", "Peak function and backgrounds of the fit"),
)
INFO_KEYS = {item.key: item for item in INFO_ITEMS}

# Integration drawn: how its area is computed
_INTEGRATION_TEXTS = {
    Integration.FIT: "fit: shape above its background",
    Integration.FIT_SHAPES: "fit: peak shapes",
    Integration.EXTRACTION: "extraction: straight line between bounds",
    Integration.NONE: "none",
}


def shown_items(settings: dict[str, bool]) -> set[str]:
    """Keys of the items shown with the setting ``chromatogram_info``."""
    return {item.key for item in INFO_ITEMS if settings.get(item.key, item.default)}


def _finite(value) -> bool:
    return value is not None and not pd.isna(value) and np.isfinite(value)


def _pm(value, uncertainty, digits: int = 0) -> str:
    """'value ± uncertainty' (without it if unknown)."""
    text = f"{value:.{digits}f}"
    return f"{text} ± {uncertainty:.{digits}f}" if _finite(uncertainty) else text


def _backgrounds(config) -> tuple[str, ...]:
    """The backgrounds the fit chooses among (``ChromatogramFitConfig.background``)."""
    background = config.background
    if isinstance(background, str):
        return (str(background),)
    return tuple(str(b) for b in background)


def shape_text(p) -> str:
    """Width and tail of a fitted peak: EMG (tail: tau / sigma) or Pap-Papai (tail: a)."""
    shape = getattr(p, "fit_shape", None)
    if shape == "emg":
        return f"EMG σ = {p.fit_sigma_s:.2f}, τ/σ = {p.fit_tailing:.2f}"
    return f"σ = {p.fit_sigma_s:.2f}, tail = {p.fit_tailing:.2f}"


def selected_peak(data: EsperData, selection: Selection):
    """The peak of the selected compounds at the selected mass (the highest), None if
    none."""
    sel = selection
    if sel.compound_idx is None or sel.mass is TIC:
        return None
    peaks = data.compound_peaks(sel.measurement_id, sel.compound_idxs)
    peaks = peaks[peaks["mass"] == sel.mass]
    if peaks.empty:
        return None
    return peaks.loc[peaks["height"].idxmax()]


def info_lines(
    data: EsperData,
    selection: Selection,
    integration: Integration,
    shown: set[str],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """(group, [(label, value)]) of the items ``shown`` that apply to the selection, the
    groups without any left out."""
    sel = selection
    processing = data.processing
    values: dict[str, str] = {}
    m = data.measurements.loc[sel.measurement_id]

    # --- run
    values["sample"] = f"{m.sample_type}"
    values["port"] = "–" if pd.isna(m.port) else f"{m.port:.0f}"
    values["acquired"] = (
        "unknown" if pd.isna(m.acq_datetime) else f"{m.acq_datetime:%Y-%m-%d %H:%M}"
    )
    instrument = getattr(m, "instrument", None)
    if isinstance(instrument, str):
        source = getattr(m, "peak_source", None)
        text = f"{instrument}, peaks: {source if isinstance(source, str) else 'none'}"
        identification = getattr(m, "identification", None)
        if isinstance(identification, str):
            text += f", {IDENTIFICATIONS.get(identification, identification)}"
        values["instrument"] = text
    values["n_peaks"] = f"{m.n_peaks}"
    values["n_compounds"] = f"{m.n_compounds}"
    if pd.notna(m.mass_cal_mode):
        values["mass_cal"] = f"mode {m.mass_cal_mode}, {m.s_per_bin:.4f} s/bin"
    version = getattr(m, "fit_version", None)
    if _finite(version):
        text = f"v{version:.0f}"
        if version < FIT_VERSION:
            text += f" (outdated, now v{FIT_VERSION})"
        fitted_at = getattr(m, "fitted_at", None)
        if isinstance(fitted_at, str):
            text += f", {fitted_at[:16].replace('T', ' ')}"
        values["fitted"] = text
    else:
        values["fitted"] = "no"

    # --- compound
    if sel.compound_idx is not None:
        c = data.compounds(sel.measurement_id).loc[sel.compound_idx]
        values["compound"] = (
            f"{compound_label(c.main_name, sel.compound_idx)} (#{sel.compound_idx})"
            f"{combined_label(sel.compound_idxs)}"
        )
        values["cmp_rt"] = (
            f"{c.rt_mean_s:.1f} ± {c.rt_std_s:.2f} s"
            if _finite(c.rt_std_s)
            else f"{c.rt_mean_s:.1f} s"
        )
        if processing.uses_retention_index:
            values["cmp_ri"] = f"{c.retention_index:.1f}"
        if processing.uses_library:  # e.g. target search: no library match
            values["spectrum"] = f"{c.best_spectrum}"
            values["score"] = (
                f"{c.best_score:.3f} (combined {c.best_score_combined:.3f})"
            )
        values["valid"] = "yes" if c.valid_candidate else "no"
        values["cmp_peaks"] = f"{c.n_peaks}"

    # --- target
    if sel.target is not None:
        target = data.targets.loc[sel.target]
        expected_rt = data.expected_rt(sel.measurement_id, sel.target)
        masses = data.target_masses(sel.measurement_id, sel.target)
        values["target"] = sel.target + (
            "" if sel.compound_idx is not None else " (not found)"
        )
        if processing.uses_retention_index:
            values["expected_ri"] = (
                "unknown" if not _finite(target["ri"]) else f"{target['ri']:.1f}"
            )
        values["expected_rt"] = (
            "unknown" if expected_rt is None else f"{expected_rt:.1f} s"
        )
        values["expected_ions"] = ", ".join(map(str, masses)) or "unknown"
        window = data.target_window(sel.measurement_id, sel.target)
        if window is not None:
            values["target_window"] = f"{window[0]:.1f} – {window[1]:.1f} s"

    # --- peak of the compound at the selected mass
    p = selected_peak(data, sel)
    if p is not None:
        values.update(_peak_values(p, integration, processing.fit))

    # --- processing (the esper.yaml now, not necessarily when the run was processed)
    source = getattr(m, "peak_source", None)
    if source == "esper":
        baseline = processing.baseline
        removal = (
            "no baseline removal"
            if baseline is None
            else f"{baseline.method} baseline removed (λ = {baseline.lam:.0e})"
        )
        values["detection_baseline"] = f"esper, {removal}"
    elif isinstance(source, str):
        values["detection_baseline"] = source
    if _finite(version):
        values["fit_config"] = (
            f"{processing.fit.peak_shape} on {' | '.join(_backgrounds(processing.fit))}"
        )

    groups = []
    for group in GROUPS:
        lines = [
            (item.label, values[item.key])
            for item in INFO_ITEMS
            if item.group == group and item.key in shown and item.key in values
        ]
        if lines:
            groups.append((group, lines))
    return groups


def _peak_values(p, integration: Integration, fit_config) -> dict[str, str]:
    values = {}
    fitted = _finite(p.get("fit_rt_s"))
    if fitted:
        values["peak_rt"] = f"{_pm(p.fit_rt_s, p.get('fit_rt_u_s'), 2)} s"
        values["peak_height"] = (
            f"{_pm(p.fit_height, p.get('fit_height_u'))} above the baseline"
            f" (signal {p.height:.0f})"
        )
        values["peak_area"] = _pm(p.fit_area, p.get("fit_area_u"))
        values["peak_shape"] = shape_text(p)
        if _finite(p.get("fit_sn")):
            values["peak_sn"] = f"{p.fit_sn:.0f}"
        slope = p.get("fit_baseline_slope")
        slope = 0.0 if not _finite(slope) else float(slope)
        values["peak_baseline"] = f"{p.fit_baseline:.0f}, slope {slope:+.3g} /s"
        # only the level and slope of the background are stored: constant or not
        candidates = _backgrounds(fit_config)
        if len(candidates) == 1:
            values["peak_background"] = candidates[0]
        else:
            values["peak_background"] = (
                "constant" if slope == 0 else "linear or exponential"
            )
    else:
        values["peak_rt"] = f"{p.rt_s:.2f} s"
        values["peak_height"] = f"{p.height:.0f}"
        values["peak_area"] = "not fitted"
    if _finite(p.get("width_s")):
        values["peak_width"] = f"{p.width_s:.2f} s"
    values["integration"] = _INTEGRATION_TEXTS[integration]
    override = p.get("fit_override_id")
    values["peak_override"] = (
        f"#{int(override)} (right click on the compound: Refit peaks)"
        if _finite(override)
        else "none (fit of the data directory)"
    )
    values["peak_source"] = "split by the fit" if p.get("source") == SOURCE_FIT else (
        "extraction"
    )
    if _finite(p.get("prominence")):
        values["prominence"] = f"{p.prominence:.0f}"
    if _finite(p.get("start_s")) and _finite(p.get("stop_s")):
        values["peak_bounds"] = f"{p.start_s:.1f} – {p.stop_s:.1f} s"
    if _finite(p.get("tailing_ratio")):
        values["tailing_ratio"] = f"{p.tailing_ratio:.2f}"
    if _finite(p.get("flatness")):
        values["flatness"] = f"{p.flatness:.2f}"
    return values


def info_html(groups: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """The lines of ``info_lines``, under the name of their group."""
    parts = []
    for group, lines in groups:
        parts.append(f"<b>{html.escape(group)}</b>")
        parts += [
            f"{html.escape(label)}&nbsp;&nbsp;{html.escape(value)}"
            for label, value in lines
        ]
    return "<br>".join(parts)
