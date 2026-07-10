import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.validation.platform_metrics import stress_category_resolution_for_metrics
from marvis.validation.results import FeatureImportanceRow


def _write_artifact(path: Path, feature_categories: dict[str, list[str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "marvis.validation_stress_scores.v2",
                "row_index": [0],
                "feature_categories": feature_categories,
                "unclassified_features": [],
                "source_counts": {
                    "notebook": 1,
                    "dictionary": 0,
                    "unresolved": 0,
                },
                "conflicts": [],
                "categories": [
                    {
                        "category": category,
                        "dropped_features": features,
                        "row_index": [0],
                        "scores": [0.5],
                        "error": None,
                        "status": "completed",
                    }
                    for category, features in feature_categories.items()
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_platform_uses_artifact_mapping_instead_of_rebuilding_raw_dictionary(
    tmp_path: Path,
):
    artifact = tmp_path / "stress_scenario_scores.json"
    _write_artifact(artifact, {"睿智": ["BH_A044_C0580"]})

    resolution = stress_category_resolution_for_metrics(
        feature_importance=[
            FeatureImportanceRow(
                rank=1,
                feature="BH_A044_C0580",
                category="睿智",
                importance=0.8,
            )
        ],
        dictionary=pd.DataFrame({"特征名": ["BH_A044"], "类别": ["睿智"]}),
        feature_col="特征名",
        category_col="类别",
        stress_scores_path=artifact,
    )

    assert resolution.per_category == {"睿智": ["BH_A044_C0580"]}
    assert resolution.unclassified_features == []


def test_platform_rejects_artifact_features_absent_from_model_metadata(tmp_path: Path):
    artifact = tmp_path / "stress_scenario_scores.json"
    _write_artifact(artifact, {"睿智": ["BH_A044_C0580", "BH_A055_C0580"]})

    with pytest.raises(
        ValueError,
        match="stress scenario artifact category mapping does not match model metadata",
    ):
        stress_category_resolution_for_metrics(
            feature_importance=[
                FeatureImportanceRow(
                    rank=1,
                    feature="BH_A044_C0580",
                    category="睿智",
                    importance=0.8,
                )
            ],
            dictionary=pd.DataFrame(
                {"特征名": ["BH_A044"], "类别": ["睿智"]}
            ),
            feature_col="特征名",
            category_col="类别",
            stress_scores_path=artifact,
        )


def test_platform_falls_back_to_sample_columns_when_importance_is_absent():
    resolution = stress_category_resolution_for_metrics(
        feature_importance=[],
        fallback_model_features=["income", "y", "split"],
        dictionary=pd.DataFrame({"特征名": ["income"], "类别": ["内部特征"]}),
        feature_col="特征名",
        category_col="类别",
        stress_scores_path=None,
    )

    assert resolution.per_category == {"内部特征": ["income"]}
    assert resolution.unclassified_features == []
