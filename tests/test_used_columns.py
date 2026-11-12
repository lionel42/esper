"""What a data directory uses to identify its compounds, and the columns shown for it."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from esper.gui.panels.compound_table import CompoundTablePanel, shown_columns  # noqa: E402
from esper.processing.config import ProcessingConfigError, processing_config  # noqa: E402


def _keys(section: dict | None) -> list[str]:
    columns = shown_columns(CompoundTablePanel.columns, processing_config(section))
    return [c.key for c in columns]


def test_uses():
    alpinac = processing_config(None)  # no method: alpinac's suspect screening
    assert alpinac.uses_library and alpinac.uses_retention_index
    target = processing_config({"method": "target"})
    assert not target.uses_library and not target.uses_retention_index
    target_ri = processing_config({"method": "target", "retention_index": True})
    assert not target_ri.uses_library and target_ri.uses_retention_index
    suspect = processing_config({"method": "suspect", "retention_index": False})
    assert suspect.uses_library and not suspect.uses_retention_index


def test_retention_index_must_be_a_bool():
    with pytest.raises(ProcessingConfigError, match="retention_index"):
        processing_config({"retention_index": "yes"})


def test_compound_columns():
    every = [c.key for c in CompoundTablePanel.columns]
    assert _keys(None) == every
    assert _keys({"method": "suspect"}) == every
    # Target search: no library match, no RI (nor their combined score)
    target = _keys({"method": "target"})
    assert not {"best_score", "best_spectrum", "retention_index"} & set(target)
    assert "best_score_combined" not in target
    assert {"compound_idx", "rt_mean_s", "name", "valid", "main_ion"} <= set(target)
    # With its RIs: the RI, still no library match
    with_ri = _keys({"method": "target", "retention_index": True})
    assert "retention_index" in with_ri and "best_score" not in with_ri
    assert "best_score_combined" not in with_ri
    # Suspect screening without RI: the library match, no combined score
    no_ri = _keys({"method": "suspect", "retention_index": False})
    assert "best_score" in no_ri and "best_score_combined" not in no_ri
