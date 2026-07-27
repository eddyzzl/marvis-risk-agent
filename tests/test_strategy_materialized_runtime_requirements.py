from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.workspace import data_semantic_mapping_hash
from marvis.db_schema import connect
from marvis.packs.strategy import dsl_delivery_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_materialization_tools import (
    run_materialize_strategy_from_pool,
)
from marvis.packs.modeling.evidence import RAW_SCORE_PRODUCT
from marvis.packs.strategy.pool_requirement_resolver import (
    model_score_virtual_field,
)
from marvis.packs.strategy.pool_tools import (
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    run_add_candidate_to_pool,
)
from marvis.packs.strategy.tools import (
    tool_adopt_strategy,
    tool_apply_strategy,
    tool_backtest_strategy,
)
from marvis.packs.strategy.typed_backtest import (
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from tests.test_strategy_pool_materialization_tools import (
    _materialization_input,
)
from tests.test_strategy_pool_scorecard import (
    _add_inputs,
    _real_scorecard,
    _selection,
)


def _synthetic_runtime_requirements(
    *,
    strategy_id: str,
    strategy_spec_hash_value: str,
) -> dict:
    vector_id = "a" * 64
    requirement = {
        "rule_id": "score-cutoff",
        "fragment_id": "scorecard-fragment",
        "requirement": {
            "type": "model_score_vector.v1",
            "virtual_field": model_score_virtual_field(vector_id),
            "score_product": RAW_SCORE_PRODUCT,
            "score_evidence_artifact_id": "b" * 64,
            "score_evidence_artifact_content_hash": "c" * 64,
            "score_vector_artifact_id": vector_id,
            "score_vector_artifact_content_hash": "d" * 64,
        },
    }
    requirements = [requirement]
    requirements_hash = hashlib.sha256(
        json.dumps(
            requirements,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    binding = {
        "materialization_id": "materialization-1",
        "strategy_id": strategy_id,
        "strategy_type": "approval",
        "strategy_version": 1,
        "strategy_spec_hash": strategy_spec_hash_value,
        "pool_id": "pool-1",
        "pool_revision_id": "pool-revision-1",
        "pool_revision": 1,
        "pool_snapshot_hash": "e" * 64,
        "pool_artifact_id": "f" * 64,
        "pool_artifact_content_hash": "1" * 64,
        "selected_design_hash": "2" * 64,
        "source_dataset_ref": {
            "dataset_id": "dataset-1",
            "content_hash": "3" * 64,
        },
        "sample_design_ref": {
            "artifact_id": "4" * 64,
            "artifact_content_hash": "5" * 64,
            "sample_design_id": "sample-1",
            "sample_design_content_hash": "6" * 64,
            "partition": "development",
        },
        "requirement_bindings": {
            "requirements_hash": requirements_hash,
            "requirements": requirements,
            "virtual_fields": [model_score_virtual_field(vector_id)],
        },
    }
    return {
        "schema_version": "strategy.materialized-runtime-requirements.v1",
        "candidate": binding,
        "baseline": None,
    }


def test_typed_backtest_rejects_tampered_runtime_requirement_provenance() -> None:
    spec = {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "reject-high",
                "priority": 1,
                "condition": {
                    "op": "compare",
                    "field": "x",
                    "operator": ">",
                    "value": 0.5,
                },
                "action": {"type": "reject"},
            }
        ],
    }
    effect_hash = strategy_spec_hash(spec)
    provenance = _synthetic_runtime_requirements(
        strategy_id="candidate-1",
        strategy_spec_hash_value=effect_hash,
    )
    result = run_typed_backtest(
        pd.DataFrame({"x": [0.1, 0.9], "bad": [0, 1]}),
        spec,
        target_col="bad",
        strategy_id="candidate-1",
        runtime_requirements=provenance,
    )
    tampered = deepcopy(result.to_dict())
    tampered["normalized_input"]["runtime_requirements"]["candidate"][
        "requirement_bindings"
    ]["requirements_hash"] = "9" * 64

    with pytest.raises(
        strategy_tools.StrategyError,
        match="requirements hash changed",
    ):
        StrategyBacktestResult.from_dict(tampered)


@pytest.mark.slow
def test_materialized_score_requirement_runs_apply_backtest_adopt_and_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _real_scorecard(tmp_path)
    selection = _selection(real)
    added = run_add_candidate_to_pool(
        _add_inputs(
            selection,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        real["fx"]["ctx"],
        real["runtime"],
    )
    materialized = run_materialize_strategy_from_pool(
        _materialization_input(added),
        real["fx"]["ctx"],
        real["runtime"],
    )
    strategy_id = materialized["strategy_ref"]["strategy_id"]
    assert materialized["requirements"]["runtime_requirements_supported"] is True
    assert materialized["requirements"]["blocker_code"] is None

    pool = load_current_strategy_candidate_pool_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        strategy_type="approval",
        expected_pool_revision=added["revision"],
        expected_pool_snapshot_hash=added["snapshot_hash"],
    )
    development = bind_strategy_pool_development_execution(
        real["runtime"],
        pool,
    )

    apply_inputs = {
        "dataset_id": real["fx"]["dataset"].id,
        "strategy_id": strategy_id,
    }
    original_tool_reauth = (
        strategy_tools.require_materialized_strategy_runtime_requirements_on_connection
    )

    def reject_runtime_reauth(*_args, **_kwargs) -> None:
        raise strategy_tools.StrategyError(
            "model score artifact drifted before commit"
        )

    monkeypatch.setattr(
        strategy_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        reject_runtime_reauth,
    )
    with pytest.raises(
        strategy_tools.StrategyError,
        match="score artifact drifted",
    ):
        tool_apply_strategy(apply_inputs, real["fx"]["ctx"])
    with connect(real["fx"]["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE role = 'strategy.applied'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = 'strategy.apply'"
        ).fetchone()[0] == 0
    monkeypatch.setattr(
        strategy_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        original_tool_reauth,
    )

    applied = tool_apply_strategy(apply_inputs, real["fx"]["ctx"])
    requirement_evidence = applied["evidence"]["runtime_requirements"]
    assert requirement_evidence["schema_version"] == (
        "strategy.materialized-runtime-requirements.v1"
    )
    assert requirement_evidence["candidate"]["strategy_id"] == strategy_id
    virtual_fields = requirement_evidence["candidate"]["requirement_bindings"][
        "virtual_fields"
    ]
    derived = real["runtime"].backend.read_frame(
        real["runtime"].registry.resolve_verified_path(
            applied["result_dataset_id"]
        )
    )
    assert set(virtual_fields).isdisjoint(derived.columns)

    backtest_inputs = {
        "dataset_id": real["fx"]["dataset"].id,
        "strategy_id": strategy_id,
        "baseline_strategy_id": strategy_id,
        "target_col": "bad",
        "sample_design_ref": development.sample_design.to_ref_dict(),
        "drop_nan_labels": development.sample_design.drop_nan_labels,
    }
    monkeypatch.setattr(
        strategy_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        reject_runtime_reauth,
    )
    with pytest.raises(
        strategy_tools.StrategyError,
        match="score artifact drifted",
    ):
        tool_backtest_strategy(backtest_inputs, real["fx"]["ctx"])
    with connect(real["fx"]["settings"].db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM backtests").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = 'strategy.backtest'"
        ).fetchone()[0] == 0
    monkeypatch.setattr(
        strategy_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        original_tool_reauth,
    )
    backtest = tool_backtest_strategy(backtest_inputs, real["fx"]["ctx"])
    backtest_requirements = backtest["normalized_input"][
        "runtime_requirements"
    ]
    assert backtest_requirements["candidate"] == requirement_evidence[
        "candidate"
    ]
    assert backtest_requirements["baseline"] == requirement_evidence[
        "candidate"
    ]
    assert backtest["normalized_input"]["baseline_effect_hash"] == (
        backtest["normalized_input"]["strategy_effect_hash"]
    )

    adoption_inputs = {
        "strategy_id": strategy_id,
        "backtest_id": backtest["backtest_id"],
        "adoption_reason": "策略委员会确认评分卡切点与样本证据",
    }
    monkeypatch.setattr(
        strategy_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        reject_runtime_reauth,
    )
    with pytest.raises(
        strategy_tools.StrategyError,
        match="score artifact drifted",
    ):
        tool_adopt_strategy(adoption_inputs, real["fx"]["ctx"])
    with connect(real["fx"]["settings"].db_path) as conn:
        strategy_row = conn.execute(
            "SELECT status, asset_status FROM strategies WHERE id = ?",
            (strategy_id,),
        ).fetchone()
        assert tuple(strategy_row) == ("draft", "draft")
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_artifacts WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_monitoring_plans WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = 'strategy.adopt'"
        ).fetchone()[0] == 0
    monkeypatch.setattr(
        strategy_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        original_tool_reauth,
    )
    adopted = tool_adopt_strategy(adoption_inputs, real["fx"]["ctx"])
    assert adopted["adoption_evidence"]["runtime_requirements"] == (
        backtest_requirements
    )

    workspace = real["fx"]["workspace"]
    delivery = real["runtime"].strategies.get_strategy(strategy_id)
    assert delivery is not None and delivery.spec is not None
    exported = real["runtime"].strategies.get_strategy_meta(strategy_id)
    assert exported is not None
    delivery_inputs = {
            "strategy_ref": {
                "strategy_id": strategy_id,
                "expected_strategy_type": "approval",
                "expected_version": exported["version"],
                "expected_spec_hash": strategy_spec_hash(delivery.spec),
            },
            "dataset_ref": {
                "dataset_id": real["fx"]["dataset"].id,
                "expected_content_hash": real["fx"]["dataset"].content_hash,
            },
            "workspace_ref": {
                "revision": workspace.revision,
                "analysis_generation": workspace.analysis_generation,
                "semantic_mapping_hash": data_semantic_mapping_hash(
                    workspace.semantic_mapping
                ),
                "active_dataset_id": workspace.active_dataset_id,
                "active_dataset_content_hash": (
                    workspace.active_dataset_content_hash
                ),
            },
            "maximum_equivalence_rows": 4096,
        }
    original_delivery_reauth = (
        dsl_delivery_tools.require_materialized_strategy_runtime_requirements_on_connection
    )
    monkeypatch.setattr(
        dsl_delivery_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        reject_runtime_reauth,
    )
    with pytest.raises(
        dsl_delivery_tools.StrategyDeliveryToolError,
        match="score artifact drifted",
    ):
        dsl_delivery_tools.run_export_strategy_delivery(
            delivery_inputs,
            real["fx"]["ctx"],
            real["runtime"],
        )
    with connect(real["fx"]["settings"].db_path) as conn:
        assert conn.execute(
            """
            SELECT COUNT(*) FROM task_artifacts
             WHERE kind LIKE 'strategy_delivery_%'
            """
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (dsl_delivery_tools.DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0
    monkeypatch.setattr(
        dsl_delivery_tools,
        "require_materialized_strategy_runtime_requirements_on_connection",
        original_delivery_reauth,
    )
    delivery_output = dsl_delivery_tools.run_export_strategy_delivery(
        delivery_inputs,
        real["fx"]["ctx"],
        real["runtime"],
    )
    assert delivery_output["runtime_requirements"] == requirement_evidence
    assert delivery_output["not_deployed"] is True

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    outputs = {
        "materialize_strategy_from_pool": materialized,
        "apply_strategy": applied,
        "backtest_strategy": backtest,
        "adopt_strategy": adopted,
        "export_strategy_delivery": delivery_output,
    }
    for tool_name, output in outputs.items():
        tool = next(item for item in manifest.tools if item.name == tool_name)
        validate_against_schema(
            output,
            tool.output_schema,
            label=f"{tool_name} output",
        )
