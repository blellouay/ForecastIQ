import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent_tools.forecasting_tools import ToolError
from src.agents.ml_routing import predict_snapshot
from src.features.m5_lightgbm import (
    FEATURE_NAMES, PIPELINE_ID, feature_provider, make_feature_provider,
    normalize_series_id,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models/registry.json"
MAPPING = ROOT / "models/lightgbm/series_mapping.json"

pytestmark = pytest.mark.integration


def _trained_series():
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    series_id = mapping["training_series_ids"][0]
    panel = pd.read_csv(ROOT / "data/processed/m5_sample_400.csv")
    row = panel.loc[panel.series_id == series_id].iloc[0]
    history = [float(row[f"d_{day}"]) for day in range(1, 1886)]
    return series_id, history


def _full_m5_id(canonical):
    item, store = canonical.split("__", 1)
    return f"{item}_{store}_validation"


@pytest.mark.parametrize("model", ["DirectLightGBM", "HurdleLightGBM"])
def test_real_snapshot_loads_and_predicts(model):
    series_id, history = _trained_series()
    result = predict_snapshot(series_id, history, 1885, 28, model, REGISTRY, feature_provider)
    values = np.asarray(result["forecast"])
    assert values.shape == (28,)
    assert np.isfinite(values).all() and (values >= 0).all()


def test_provider_schema_mapping_and_future_independence():
    series_id, history = _trained_series()
    first = feature_provider(series_id, history, 1885, 28, FEATURE_NAMES)
    mutated_full = history + [999999.0] * 28
    second = feature_provider(series_id, mutated_full[:1885], 1885, 28, FEATURE_NAMES)
    pd.testing.assert_frame_equal(first, second)
    assert str(first["series_code"].dtype) == "category"
    with pytest.raises(ToolError, match="Unknown series_id"):
        feature_provider("FOODS_1_001_CA_1_validation", history, 1885, 28, FEATURE_NAMES)


def test_full_and_canonical_m5_ids_preserve_saved_code():
    canonical, history = _trained_series()
    full = _full_m5_id(canonical)
    assert normalize_series_id(full) == canonical
    canonical_frame = feature_provider(canonical, history, 1885, 28, FEATURE_NAMES)
    full_frame = feature_provider(full, history, 1885, 28, FEATURE_NAMES)
    pd.testing.assert_frame_equal(canonical_frame, full_frame)
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    assert int(canonical_frame["series_code"].iloc[0]) == mapping["series_id_to_code"][canonical]


def test_unknown_canonical_identity_is_not_remapped():
    _, history = _trained_series()
    with pytest.raises(ToolError, match="canonical identity 'FOODS_1_001__CA_1'"):
        feature_provider("FOODS_1_001_CA_1_validation", history, 1885, 28, FEATURE_NAMES)
    with pytest.raises(ToolError, match="canonical identity 'FOODS_1_001__CA_1'"):
        feature_provider("FOODS_1_001__CA_1", history, 1885, 28, FEATURE_NAMES)


def test_conflicting_normalized_mapping_fails_closed(tmp_path):
    canonical, history = _trained_series()
    full = _full_m5_id(canonical)
    conflict = {
        "schema_version": 1,
        "feature_pipeline_id": PIPELINE_ID,
        "training_series_ids": [canonical, full],
        "series_id_to_code": {canonical: 0, full: 1},
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(conflict), encoding="utf-8")
    provider = make_feature_provider(mapping_path=path)
    with pytest.raises(ToolError, match="Conflicting LightGBM mappings"):
        provider(canonical, history, 1885, 28, FEATURE_NAMES)


def test_registry_has_complete_historical_snapshots():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for entry in registry["models"].values():
        assert set(entry["snapshots"]) == {"1801", "1829", "1857", "1885"}
        assert entry["feature_names"] == FEATURE_NAMES
        assert entry["feature_pipeline_id"] == PIPELINE_ID
        for origin, snapshot in entry["snapshots"].items():
            assert snapshot["trained_through_day"] <= int(origin)
            metadata = json.loads(
                (ROOT / "models" / snapshot["metadata_path"]).read_text(encoding="utf-8")
            )
            assert metadata["feature_names"] == FEATURE_NAMES
            assert metadata["feature_pipeline_id"] == PIPELINE_ID
            assert metadata["trained_through_day"] == snapshot["trained_through_day"]
            if entry["kind"] == "direct":
                assert (ROOT / "models" / snapshot["model_path"]).is_file()
            else:
                assert (ROOT / "models" / snapshot["classifier_path"]).is_file()
                assert (ROOT / "models" / snapshot["regressor_path"]).is_file()
