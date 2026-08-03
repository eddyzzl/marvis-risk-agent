from marvis.agent.turn_handlers import _parse_c1_reply


def _state() -> dict:
    return {
        "files": [
            {
                "dataset_id": "ds-old",
                "name": "vintage_panel.parquet",
                "columns": ["account_id", "bad"],
                "target_candidates": ["bad"],
            },
            {
                "dataset_id": "ds-reg",
                "name": "MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet",
                "columns": ["application_id", "case_weight", "loss_amount_target"],
                "target_candidates": [],
            },
            {
                "dataset_id": "ds-features",
                "name": "bureau_features.parquet",
                "columns": ["application_id", "bureau_score"],
                "target_candidates": [],
            },
        ],
        "anchor_id": "ds-old",
        "feature_ids": ["ds-reg", "ds-features"],
        "target_col": "bad",
    }


def test_c1_natural_language_can_switch_anchor_ignore_rest_and_set_target() -> None:
    assignment = _parse_c1_reply(
        "只使用 MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet 作为样本主表，"
        "目标列设为 loss_amount_target；其余文件全部忽略，不要作为特征表。",
        _state(),
    )

    assert assignment == {
        "anchor_id": "ds-reg",
        "feature_ids": [],
        "target_col": "loss_amount_target",
    }


def test_c1_natural_language_keeps_explicit_feature_when_ignoring_rest() -> None:
    assignment = _parse_c1_reply(
        "MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet 是样本主表，"
        "bureau_features.parquet 作为特征表，其他文件忽略，"
        "目标列 loss_amount_target。",
        _state(),
    )

    assert assignment == {
        "anchor_id": "ds-reg",
        "feature_ids": ["ds-features"],
        "target_col": "loss_amount_target",
    }
