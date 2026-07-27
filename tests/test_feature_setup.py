from __future__ import annotations

import json

import pandas as pd

from marvis.agent.feature_setup import FeatureProposal, infer_meaning_directions
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        backend,
        settings.datasets_dir,
    )
    return backend, registry


def _dictionary(registry, tmp_path, rows):
    path = tmp_path / "字段字典.csv"
    pd.DataFrame(rows, columns=["特征名", "含义"]).to_csv(path, index=False)
    return registry.register_from_upload(
        "task-feature",
        path,
        role="feature_dictionary",
    )


def test_meaning_direction_llm_is_schema_bounded_and_records_provenance(tmp_path):
    backend, registry = _runtime(tmp_path)
    dictionary = _dictionary(
        registry,
        tmp_path,
        [("overdue_count", "历史逾期次数")],
    )

    class FakeLLM:
        profile = {"model_name": "semantic-test-model"}

        def __init__(self):
            self.request = None

        def complete(self, **kwargs):
            self.request = kwargs
            return json.dumps({
                "directions": [{
                    "feature": "overdue_count",
                    "direction": "positive",
                    "confidence": "high",
                    "rationale": "逾期次数越多通常风险越高。",
                }],
            }, ensure_ascii=False)

    client = FakeLLM()
    proposal = FeatureProposal(
        dataset_id="sample",
        dataset_name="sample.parquet",
        target_col="y",
        features=["overdue_count"],
        notes=[],
        metrics=["meaning_consistency"],
        dictionary_id=dictionary.id,
    )

    decisions = infer_meaning_directions(client, backend, registry, proposal)

    assert decisions["overdue_count"] == {
        "business_meaning": "历史逾期次数",
        "expected_direction": "positive",
        "confidence": "high",
        "rationale": "逾期次数越多通常风险越高。",
        "judgement_source": "llm_semantic_direction",
        "model": "semantic-test-model",
        "prompt_name": "feature_meaning_direction",
        "prompt_version": 1,
    }
    schema = client.request["json_schema"]["schema"]
    item_schema = schema["properties"]["directions"]["items"]["properties"]
    assert item_schema["feature"]["enum"] == ["overdue_count"]
    assert set(item_schema["direction"]["enum"]) == {
        "positive",
        "negative",
        "u_shape",
        "uncertain",
    }
    assert client.request["temperature"] == 0.0
    assert "不得生成或猜测任何统计指标" in client.request["user_prompt"]


def test_meaning_direction_without_llm_is_conservatively_uncertain(tmp_path):
    backend, registry = _runtime(tmp_path)
    dictionary = _dictionary(
        registry,
        tmp_path,
        [("safe_balance", "余额越高风险越低")],
    )
    proposal = FeatureProposal(
        dataset_id="sample",
        dataset_name="sample.parquet",
        target_col="y",
        features=["safe_balance"],
        notes=[],
        metrics=["meaning_consistency"],
        dictionary_id=dictionary.id,
    )

    decisions = infer_meaning_directions(None, backend, registry, proposal)

    assert decisions["safe_balance"]["expected_direction"] == "uncertain"
    assert decisions["safe_balance"]["confidence"] == "low"
    assert decisions["safe_balance"]["judgement_source"] == "no_llm_fallback"


def test_joined_meaning_direction_scopes_to_selected_feature_table_columns(tmp_path):
    backend, registry = _runtime(tmp_path)
    feature_path = tmp_path / "vars.csv"
    pd.DataFrame({
        "join_id": [1, 2],
        "bureau_score": [500, 700],
    }).to_csv(feature_path, index=False)
    feature_dataset = registry.register_from_upload(
        "task-feature",
        feature_path,
        role="feature",
    )
    dictionary = _dictionary(
        registry,
        tmp_path,
        [
            ("bureau_score", "征信评分"),
            ("unrelated_column", "不属于所选特征表"),
        ],
    )

    class FakeLLM:
        profile = {"model_name": "semantic-test-model"}

        def complete(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            assert [item["feature"] for item in payload["features"]] == [
                "bureau_score",
            ]
            return json.dumps({
                "directions": [{
                    "feature": "bureau_score",
                    "direction": "negative",
                    "confidence": "medium",
                    "rationale": "评分越高通常风险越低。",
                }],
            }, ensure_ascii=False)

    proposal = FeatureProposal(
        dataset_id="anchor",
        dataset_name="anchor.parquet",
        target_col="y",
        features=[],
        notes=[],
        metrics=["meaning_consistency"],
        template_id="feature_analysis_with_join",
        anchor_id="anchor",
        feature_ids=[feature_dataset.id],
        dictionary_id=dictionary.id,
    )

    decisions = infer_meaning_directions(
        FakeLLM(),
        backend,
        registry,
        proposal,
    )

    assert set(decisions) == {"bureau_score"}
    assert decisions["bureau_score"]["expected_direction"] == "negative"
