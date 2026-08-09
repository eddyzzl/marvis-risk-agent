"""End-to-end HTTP test of the data-join entry through the generic PlanDriver.

Drives the real FastAPI endpoints (/agent/start, /agent/messages) for a
task_type='data_join' task whose materials are two joinable tables (an anchor
sample with a label + a feature table keyed by md5(mobile)). No LLM is
configured — this is exactly the no-LLM preview scenario the manual-first build
targets. Asserts the driver pauses at the forced-confirm gate (showing join
diagnostics) and, on confirmation, executes a 1:1-anchored join.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


def _join_dir(root: Path, n: int = 50) -> Path:
    src = root / "join_material"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(src / "sample.parquet")
    pd.DataFrame({
        "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
        "balance": list(range(n)),
    }).to_parquet(src / "features.parquet")
    return src


def _join_dir_with_ambiguous_targets(root: Path, n: int = 50) -> Path:
    src = root / "join_ambiguous_targets"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({
        "mobile": phones,
        "label_sqandzy": [i % 2 for i in range(n)],
        "label_sqandzy_new": [(i + 1) % 2 for i in range(n)],
    }).to_parquet(src / "sample.parquet")
    pd.DataFrame({
        "phone_md5": [hashlib.md5(phone.encode()).hexdigest() for phone in phones],
        "balance": list(range(n)),
    }).to_parquet(src / "features.parquet")
    return src


def _join_dir_with_plain_named_excel(root: Path, n: int = 50) -> Path:
    """Three uploaded data tables, including an xlsx whose name has no role hint."""
    src = _join_dir(root, n=n)
    pd.DataFrame({
        "mobile": [f"138{i:08d}" for i in range(n)],
        "bureau_score": list(range(600, 600 + n)),
    }).to_excel(src / "vars.xlsx", index=False)
    return src


def _join_dir_with_conflicts(root: Path, n: int = 50) -> Path:
    """Like _join_dir but the feature table repeats the first 5 keys with a DIFFERENT
    value — a same-key conflict that makes the join key non-unique, so confirm_join
    leaves the feature awaiting a dedup strategy (the §4 dedup picker scenario)."""
    src = root / "join_conflict"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(src / "sample.parquet")
    md5s = [hashlib.md5(p.encode()).hexdigest() for p in phones]
    pd.DataFrame({
        "phone_md5": md5s + md5s[:5],          # 5 duplicate keys
        "balance": list(range(n)) + [999] * 5,  # ...with a conflicting value
    }).to_parquet(src / "features.parquet")
    return src


def _join_dir_with_two_conflicting_features(root: Path, n: int = 14) -> Path:
    """Two feature tables both require deduplication, matching the live failure."""
    src = root / "join_two_conflicts"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    md5s = [hashlib.md5(phone.encode()).hexdigest() for phone in phones]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(
        src / "sample.parquet"
    )
    for suffix, offset in (("a", 0), ("b", 100)):
        pd.DataFrame({
            "phone_md5": md5s + md5s[:2],
            f"balance_{suffix}": list(range(offset, offset + n)) + [999, 1000],
        }).to_parquet(src / f"features_{suffix}.parquet")
    return src


def _join_dir_with_dictionary(root: Path, n: int = 50) -> Path:
    """Like _join_dir but with a 字典.csv naming the join key columns (GAP-4): the
    C1/C2 gates should annotate `id_no`/`id_no_md5` with their business meaning
    instead of showing only the bare column codes. Reuses _join_dir's exact phone-
    number-shaped key values (the join engine's fingerprint/alignment heuristics are
    tuned to that shape) but under a column name that avoids marvis.redaction.
    _SENSITIVE_KEY_RE (mobile/phone) — that regex matches on the dict *key* (the
    column name) at plan-output persistence, so a mobile/phone-named key column
    would scrub the dictionary map's values wholesale, same as any real sensitive
    field would be. This test is about the dictionary-lookup wiring, not redaction."""
    src = root / "join_with_dictionary"
    src.mkdir(parents=True, exist_ok=True)
    ids = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"id_no": ids, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(src / "sample.parquet")
    pd.DataFrame({
        "id_no_md5": [hashlib.md5(i.encode()).hexdigest() for i in ids],
        "balance": list(range(n)),
    }).to_parquet(src / "features.parquet")
    pd.DataFrame({
        "特征名": ["id_no", "id_no_md5", "bad_flag", "balance"],
        "含义": ["客户标识", "客户标识哈希", "是否坏账", "账户余额"],
    }).to_csv(src / "字典.csv", index=False)
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


_TYPED_GATE_SNAPSHOT_FIELDS = frozenset(
    {
        "expected_plan_status",
        "expected_plan_revision",
        "expected_plan_fingerprint",
        "expected_step_fingerprint",
    }
)


def _typed_gate_action_payload(
    message: dict,
    *,
    action: str,
    content: str,
    adjust_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a browser action from the exact gate snapshot returned by GET."""

    metadata = message.get("metadata") or {}
    snapshot = metadata.get("confirmation_snapshot") or {}
    assert _TYPED_GATE_SNAPSHOT_FIELDS <= set(snapshot), metadata
    assert metadata.get("plan_id") and metadata.get("step_id"), metadata
    payload: dict[str, object] = {
        "content": content,
        "ui_action": action,
        "expected_plan_id": metadata["plan_id"],
        "expected_step_id": metadata["step_id"],
        **snapshot,
        "acceptance_mode": "manual",
    }
    if adjust_params is not None:
        payload["adjust_params"] = adjust_params
    return payload


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _prebind_single_c1_workspace(
    client: TestClient,
    task_id: str,
    c1_state: dict,
) -> None:
    from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
    from marvis.repositories.data_workspace import DataWorkspaceRepository
    from marvis.repositories.datasets import DatasetRepository

    dataset = DatasetRepository(
        client.app.state.settings.db_path
    ).get_dataset(c1_state["anchor_id"])
    assert dataset is not None
    target_col = c1_state["target_col"]
    DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).save_initial_binding(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="overview",
            selected_field=target_col,
            semantic_mapping=DataSemanticMapping(
                target_col=target_col,
                field_roles={target_col: "target"},
                business_names={},
            ),
        ),
        expected_revision=0,
    )


def test_data_join_conversation_end_to_end(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
):
    src = _join_dir(tmp_path)
    resp = client.post("/api/tasks", json={
        "model_name": "拼接测试",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    })
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["id"]

    # turn 0 — start: C1 file-role assignment gate (propose anchor/feature + target)
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    c1 = _last_assistant(msgs)
    assert "样本主表" in c1["content"]
    assert c1["metadata"]["join_c1"]["anchor_id"]  # proposal carried for the form
    assert any(t["title"].startswith("输入文件") for t in c1["metadata"].get("tables", []))

    # turn 1 — confirm C1 roles: build the join plan, pause at the plan-overview gate
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert resp.status_code == 202, resp.text
    role_messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    audit_confirm = next(message for message in role_messages if message["role"] == "user")
    assert audit_confirm["metadata"]["display_in_timeline"] is False
    assert any(
        message["role"] == "assistant" and "收到角色与目标列确认" in message["content"]
        for message in role_messages
    )
    from marvis.repositories.tasks import TaskRepository as CurrentTaskRepository

    persisted_task = CurrentTaskRepository(
        client.app.state.settings.db_path
    ).get_task(task_id)
    assert persisted_task.target_col == "bad_flag"
    overview = _last_assistant(role_messages)
    assert "手动模式请点击「开始执行」" in overview["content"]
    assert "Agent 模式请回复「开始」或「继续」" in overview["content"]

    # turn 2 — 开始: run the plan, pause at the C2 diagnostics gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]
    assert any(t["title"].startswith("拼接诊断") for t in gate["metadata"].get("tables", []))

    # turn 3 — confirm C2: confirm_join + execute_join run, anchor preserved 1:1
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]
    result_dataset = done["metadata"]["result_dataset"]
    assert result_dataset["dataset_id"]
    assert result_dataset["title"] == "拼接结果已生成"
    assert result_dataset["download_label"] == "下载拼接结果"
    parsed_download = urlparse(result_dataset["download_url"])
    assert parsed_download.path == (
        f"/api/tasks/{task_id}/datasets/{result_dataset['dataset_id']}/download"
    )
    bound_query = parse_qs(parsed_download.query)
    assert bound_query == {
        "plan_id": [result_dataset["plan_id"]],
        "step_id": [result_dataset["step_id"]],
        "output_ref": [result_dataset["output_ref"]],
        "expected_content_hash": [result_dataset["content_hash"]],
    }
    download = client.get(result_dataset["download_url"])
    assert download.status_code == 200
    assert download.content
    assert "attachment" in download.headers["content-disposition"]
    partial = client.get(
        result_dataset["download_url"],
        headers={"Range": "bytes=0-31"},
    )
    assert partial.status_code == 206
    assert partial.content == download.content[:32]
    assert partial.headers["accept-ranges"] == "bytes"
    assert partial.headers["content-range"] == (
        f"bytes 0-31/{len(download.content)}"
    )

    # Close the verify-to-open race: replacing the source after registry
    # verification but before the HTTP response opens it must not serve the
    # replacement bytes.
    from marvis.data.registry import DatasetRegistry

    original_resolve = DatasetRegistry.resolve_verified_path
    replaced_paths = []

    def replace_after_registry_verification(registry, dataset_id):
        path = original_resolve(registry, dataset_id)
        path.write_bytes(b"replacement during response open")
        replaced_paths.append(path)
        return path

    monkeypatch.setattr(
        DatasetRegistry,
        "resolve_verified_path",
        replace_after_registry_verification,
    )
    raced_download = client.get(result_dataset["download_url"])
    assert raced_download.status_code == 409
    assert raced_download.content != b"replacement during response open"
    monkeypatch.setattr(
        DatasetRegistry,
        "resolve_verified_path",
        original_resolve,
    )
    assert replaced_paths
    replaced_paths[0].write_bytes(download.content)

    # Compatibility: an old completion message has no download metadata.  A
    # reload recovers it from immutable plan output without changing the audit row.
    from marvis.db import TaskRepository

    TaskRepository(tmp_path / "marvis.sqlite").update_agent_message(
        done["id"],
        content=done["content"],
        metadata={key: value for key, value in done["metadata"].items() if key != "result_dataset"},
    )
    reloaded = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    recovered = reloaded["metadata"]["result_dataset"]
    assert recovered["dataset_id"] == result_dataset["dataset_id"]
    assert recovered["recovered_from_plan"] is True
    assert recovered["title"] == "拼接结果已生成"
    assert recovered["download_label"] == "下载拼接结果"

    # The action is bound to the exact bytes committed by this plan.  Even if
    # both the registry record and file are consistently replaced later, the
    # old completion link must reject rather than silently serve the new bytes.
    replacement = tmp_path / "datasets" / "replacement-result.parquet"
    replacement.write_bytes(b"replacement result bytes")
    replacement_hash = hashlib.sha256(replacement.read_bytes()).hexdigest()
    from marvis.db import connect

    with connect(tmp_path / "marvis.sqlite") as conn:
        conn.execute(
            "UPDATE datasets SET source_path = ?, content_hash = ? WHERE id = ?",
            (
                replacement.relative_to(tmp_path / "datasets").as_posix(),
                replacement_hash,
                result_dataset["dataset_id"],
            ),
        )
    drifted_download = client.get(result_dataset["download_url"])
    assert drifted_download.status_code == 409
    assert "binding changed" in drifted_download.json()["detail"]


def test_data_join_typed_replan_actions_use_live_get_snapshot_and_fail_closed(
    client: TestClient,
    tmp_path: Path,
):
    """JOIN key/exclusion controls reject malformed, stale, and withheld actions."""

    src = _join_dir_with_two_conflicting_features(tmp_path, n=14)
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "拼接 typed UI action",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    ).status_code == 202
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始"},
    ).status_code == 202

    join_gate = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    plan = client.app.state.plan_repo.load_plan(join_gate["metadata"]["plan_id"])
    propose_step = next(
        step for step in plan.steps if step.tool_ref.tool == "propose_join"
    )
    feature_ids = list(propose_step.inputs["feature_ids"])
    assert len(feature_ids) == 2
    key_overrides = {feature_id: ["mobile"] for feature_id in feature_ids}

    before_invalid = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    missing_keys = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_typed_gate_action_payload(
            join_gate,
            action="apply_join_keys",
            content="重新诊断拼接键",
        ),
    )
    assert missing_keys.status_code == 422, missing_keys.text
    unknown_field_payload = _typed_gate_action_payload(
        join_gate,
        action="apply_join_keys",
        content="重新诊断拼接键",
        adjust_params={"key_overrides": key_overrides},
    )
    unknown_field_payload["unknown_top_level"] = True
    unknown_field = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=unknown_field_payload,
    )
    assert unknown_field.status_code == 422, unknown_field.text
    assert client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"] == before_invalid

    apply_keys_payload = _typed_gate_action_payload(
        join_gate,
        action="apply_join_keys",
        content="重新诊断拼接键",
        adjust_params={"key_overrides": key_overrides},
    )
    applied_keys = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=apply_keys_payload,
    )
    assert applied_keys.status_code == 202, applied_keys.text
    revised_join_gate = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    revised_plan = client.app.state.plan_repo.load_plan(
        revised_join_gate["metadata"]["plan_id"]
    )
    revised_propose = next(
        step for step in revised_plan.steps if step.tool_ref.tool == "propose_join"
    )
    assert revised_propose.inputs["key_overrides"] == key_overrides

    after_keys = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    stale_keys = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=apply_keys_payload,
    )
    assert stale_keys.status_code == 409, stale_keys.text
    assert client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"] == after_keys

    unknown_feature = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_typed_gate_action_payload(
            revised_join_gate,
            action="exclude_join_feature",
            content="排除不存在的特征表",
            adjust_params={"exclude_join_feature_id": "not-a-live-feature-id"},
        ),
    )
    assert unknown_feature.status_code == 409, unknown_feature.text
    assert "已不在当前拼接方案" in unknown_feature.json()["detail"]
    assert client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"] == after_keys
    unchanged_unknown_plan = client.app.state.plan_repo.load_plan(
        revised_join_gate["metadata"]["plan_id"]
    )
    unchanged_unknown_propose = next(
        step
        for step in unchanged_unknown_plan.steps
        if step.tool_ref.tool == "propose_join"
    )
    assert unchanged_unknown_propose.inputs["feature_ids"] == feature_ids

    excluded_feature_id = feature_ids[0]
    exclude_params = {"exclude_join_feature_id": excluded_feature_id}
    missing_feature = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_typed_gate_action_payload(
            revised_join_gate,
            action="exclude_join_feature",
            content="排除一张特征表",
        ),
    )
    assert missing_feature.status_code == 422, missing_feature.text

    before_rejection = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    rejected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_typed_gate_action_payload(
            revised_join_gate,
            action="exclude_join_feature",
            content="先别执行",
            adjust_params=exclude_params,
        ),
    )
    assert rejected.status_code == 409, rejected.text
    assert client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"] == before_rejection
    unchanged_plan = client.app.state.plan_repo.load_plan(
        revised_join_gate["metadata"]["plan_id"]
    )
    unchanged_propose = next(
        step for step in unchanged_plan.steps if step.tool_ref.tool == "propose_join"
    )
    assert unchanged_propose.inputs["feature_ids"] == feature_ids

    excluded = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_typed_gate_action_payload(
            revised_join_gate,
            action="exclude_join_feature",
            content=f"排除特征表 {excluded_feature_id}",
            adjust_params=exclude_params,
        ),
    )
    assert excluded.status_code == 202, excluded.text
    final_gate = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    final_plan = client.app.state.plan_repo.load_plan(final_gate["metadata"]["plan_id"])
    final_propose = next(
        step for step in final_plan.steps if step.tool_ref.tool == "propose_join"
    )
    assert final_propose.inputs["feature_ids"] == [feature_ids[1]]


def test_data_join_two_conflicting_features_reaches_diagnostic_gate(
    client: TestClient, tmp_path: Path
):
    src = _join_dir_with_two_conflicting_features(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "双特征冲突拼接",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    }).json()["id"]

    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"}
    )

    assert response.status_code == 202, response.text
    gate = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    assert "'type' object is not subscriptable" not in gate["content"]
    assert "拼接诊断完成" in gate["content"]
    assert len(gate["metadata"]["dedup"]["needs_dedup"]) == 2


def test_data_join_c1_lists_all_three_uploaded_tables_including_plain_named_excel(
    client: TestClient, tmp_path: Path
):
    src = _join_dir_with_plain_named_excel(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "三文件拼接",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    }).json()["id"]

    response = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert response.status_code == 202, response.text
    c1 = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    files = c1["metadata"]["join_c1"]["files"]

    assert len(files) == 3
    names = {item["name"] for item in files}
    assert names == {"sample.parquet", "features.parquet", "vars.xlsx"}


def test_data_join_c2_gate_annotates_key_columns_with_dictionary_meaning(client: TestClient, tmp_path: Path):
    """GAP-4: when the task's materials include a data dictionary (字典.csv), the C2
    diagnostics gate's "匹配键" cell should carry the business meaning next to the
    raw column code, and the dictionary should register as its own dataset."""
    src = _join_dir_with_dictionary(tmp_path)
    resp = client.post("/api/tasks", json={
        "model_name": "拼接字典测试",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    })
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["id"]

    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]
    diagnostics_table = next(
        t for t in gate["metadata"].get("tables", []) if t["title"].startswith("拼接诊断")
    )
    keys_cell = diagnostics_table["rows"][0][diagnostics_table["columns"].index("匹配键")]
    assert "客户标识" in keys_cell

    datasets = client.get(f"/api/tasks/{task_id}/datasets").json()["datasets"]
    assert any(d["role"] == "feature_dictionary" for d in datasets)


@pytest.mark.slow
def test_data_join_dedup_picker_resolves_conflicts(client: TestClient, tmp_path: Path):
    """§4 join dedup picker: a non-unique feature key leaves confirm_join awaiting a
    strategy; the C2 gate surfaces it via metadata.dedup; posting a per-feature
    strategy re-confirms (resolving the conflict) and the join then completes 1:1."""
    src = _join_dir_with_conflicts(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接去重", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    dedup = gate["metadata"].get("dedup")
    assert dedup is not None, gate["metadata"]
    assert dedup["needs_dedup"], dedup
    assert dedup["strategies"] == ["first", "last"]
    feature_id = dedup["needs_dedup"][0]
    # the picker shows the conflict count from the propose-step diagnostics
    feature_payload = dedup["features"][0]
    assert feature_payload["conflict_keys"] >= 1
    # UX-6: conflict_columns and a real conflict-value example are surfaced too, not
    # just the bare count -- the conflicting rows disagree on "balance".
    assert "balance" in feature_payload["conflict_columns"]
    assert feature_payload["examples"], feature_payload
    example = feature_payload["examples"][0]
    assert "key" in example and example["key"]
    assert example["values"].get("balance"), example

    # pick a strategy -> re-confirm; conflict resolved, re-pause at the now-clear gate
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "确认",
            "dedup_strategies": {feature_id: "first"},
            "expected_step_id": gate["metadata"]["step_id"],
        },
    )
    assert resp.status_code == 202, resp.text
    gate2 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert gate2["metadata"].get("kind") == "gate"
    assert not gate2["metadata"].get("dedup")  # no strategy still needed

    # confirm execute -> 1:1 anchored join completes
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]


@pytest.mark.slow
def test_data_join_dedup_text_instruction_resolves_conflicts(client: TestClient, tmp_path: Path):
    """Manual-mode TEXT resolution of a same-key conflict when no §4 picker is wired: the C2
    gate surfaces the conflict + the 「去重 first/last」 hint, and replying with that text
    applies the strategy to every needs_dedup feature so the join completes 1:1. Without this
    a pure-text manual user dead-ends at 'all joins must be confirmed before execute'."""
    src = _join_dir_with_conflicts(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接去重文字", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    # the gate surfaces the conflict + the text-resolution hint, naming the feature by its
    # friendly file name (.parquet), not a raw ds_<hash> id
    assert "同键冲突" in gate["content"] and "去重 first" in gate["content"]
    assert ".parquet" in gate["content"] and "`ds_" not in gate["content"]

    # plain-text dedup choice resolves every conflicting feature, re-pause at the clear gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "去重 first"})
    assert resp.status_code == 202, resp.text
    gate2 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert not gate2["metadata"].get("dedup")  # no strategy still needed

    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"] and "1:1 保持" in done["content"]


def test_data_join_c1_form_assignment_drives_the_join(client: TestClient, tmp_path: Path):
    """The C1 control form posts a structured [C1] assignment; the driver honors it."""
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接表单", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    state = c1["metadata"]["join_c1"]
    anchor = state["anchor_id"]
    features = state["feature_ids"]
    payload = json.dumps({"anchor_id": anchor, "feature_ids": features, "target_col": state["target_col"]})
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"[C1]{payload}", "ui_action": "confirm_roles"},
    )
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "手动模式请点击「开始执行」" in overview["content"]
    assert "Agent 模式请回复「开始」或「继续」" in overview["content"]
    # 开始 → run to the C2 diagnostics gate
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]


def test_data_join_c1_selected_target_is_registered_on_join_result(
    client: TestClient,
    tmp_path: Path,
):
    """The joined dataset retains the exact target selected at the C1 gate."""

    src = _join_dir_with_ambiguous_targets(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接多目标选择",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    state = c1["metadata"]["join_c1"]
    assert state["target_col"] is None

    selected_target = "label_sqandzy_new"
    assignment = json.dumps({
        "anchor_id": state["anchor_id"],
        "feature_ids": state["feature_ids"],
        "target_col": selected_target,
    })
    confirmed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"[C1]{assignment}", "ui_action": "confirm_roles"},
    )
    assert confirmed.status_code == 202, confirmed.text
    started = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始"},
    )
    assert started.status_code == 202, started.text
    executed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )
    assert executed.status_code == 202, executed.text

    done = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    result_id = done["metadata"]["result_dataset"]["dataset_id"]
    result = next(
        dataset
        for dataset in client.get(f"/api/tasks/{task_id}/datasets").json()["datasets"]
        if dataset["id"] == result_id
    )
    assert result["has_target"] is True
    assert result["target_col"] == selected_target


@pytest.mark.parametrize("invalid_target", ["balance", "column_that_does_not_exist"])
def test_data_join_c1_typed_target_must_belong_to_authenticated_anchor(
    client: TestClient,
    tmp_path: Path,
    invalid_target: str,
):
    """A tampered C1 control cannot bind a feature-only or unknown target."""

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接非法目标列",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    }).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202
    messages_before = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    state = _last_assistant(messages_before)["metadata"]["join_c1"]
    assignment = json.dumps({
        "anchor_id": state["anchor_id"],
        "feature_ids": state["feature_ids"],
        "target_col": invalid_target,
    })

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": f"[C1]{assignment}",
            "ui_action": "confirm_roles",
        },
    )

    assert response.status_code == 422, response.text
    assert "目标列" in response.json()["detail"]
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
    assert client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"] == messages_before


@pytest.mark.parametrize(
    "target_phrase",
    [
        "目标列用 balance",
        "把 balance 设为目标列",
        "用 balance 当目标列",
    ],
)
def test_data_join_c1_natural_language_feature_only_target_requests_clarification(
    client: TestClient,
    tmp_path: Path,
    target_phrase: str,
):
    """Free text preserves an invalid declared target long enough to reject it."""

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接自然语言非法目标列",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    }).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "sample.parquet 作为样本主表，features.parquet 作为特征表，"
                f"{target_phrase}"
            )
        },
    )

    assert response.status_code == 202, response.text
    messages = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    clarification = _last_assistant(messages)
    assert "目标列" in clarification["content"]
    assert "样本主表" in clarification["content"] or "样本锚表" in clarification["content"]
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


def test_data_join_c1_explicit_no_target_overrides_registry_inference(
    client: TestClient,
    tmp_path: Path,
):
    """The C1 ``不指定`` choice remains explicit through joined registration."""

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接明确无目标列",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    }).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202
    state = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )["metadata"]["join_c1"]
    assignment = json.dumps({
        "anchor_id": state["anchor_id"],
        "feature_ids": state["feature_ids"],
        "target_col": "",
    })
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"[C1]{assignment}", "ui_action": "confirm_roles"},
    ).status_code == 202
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"}
    ).status_code == 202
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"}
    ).status_code == 202

    done = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    result_id = done["metadata"]["result_dataset"]["dataset_id"]
    result = next(
        dataset
        for dataset in client.get(f"/api/tasks/{task_id}/datasets").json()["datasets"]
        if dataset["id"] == result_id
    )
    assert result["has_target"] is False
    assert result["target_col"] is None
    assert client.get(f"/api/tasks/{task_id}").json()["target_col"] == ""


def test_data_join_single_file_explicit_no_target_clears_all_target_facts(
    client: TestClient,
    tmp_path: Path,
):
    from marvis.db import DatasetRepository

    src = tmp_path / "single_material_without_target"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件明确无目标列",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202
    c1 = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )["metadata"]["join_c1"]
    assignment = json.dumps(
        {
            "anchor_id": c1["anchor_id"],
            "feature_ids": [],
            "target_col": "",
        }
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"[C1]{assignment}", "ui_action": "confirm_roles"},
    )

    assert response.status_code == 202, response.text
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace").json()
    assert workspace["active_dataset_id"] == c1["anchor_id"]
    assert workspace["semantic_mapping"]["target_col"] is None
    anchor = next(
        item
        for item in DatasetRepository(
            client.app.state.settings.db_path
        ).list_datasets(task_id)
        if item.id == c1["anchor_id"]
    )
    assert anchor.has_target is False
    assert anchor.target_col is None
    assert client.get(f"/api/tasks/{task_id}").json()["target_col"] == ""


def test_data_join_c1_form_rejects_duplicate_sample_primary_role(client: TestClient, tmp_path: Path):
    """UX-7: the C1 form must reject a payload that marks two datasets as the
    sample anchor table with a typed, explicit error rather than silently
    dropping the second dataset from the join (join_setup.JoinSetupError)."""
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接双主表", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    state = c1["metadata"]["join_c1"]
    anchor = state["anchor_id"]
    other = state["feature_ids"][0]
    payload = json.dumps({
        "anchor_id": anchor,
        "anchor_ids": [anchor, other],
        "feature_ids": [],
        "target_col": state["target_col"],
    })
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"[C1]{payload}", "ui_action": "confirm_roles"},
    )
    assert resp.status_code == 202, resp.text
    error = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert error["metadata"].get("error") is True
    assert "样本主表只能有一个" in error["content"]
    assert "特征表" in error["content"] or "忽略" in error["content"]

    # the gate must still be the same C1 form (not silently advanced) so the
    # user can correct the roles and resubmit.
    still_open = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert still_open["metadata"].get("error") is True


def test_data_join_single_file_confirms_then_skips(client: TestClient, tmp_path: Path):
    from marvis.db import DatasetRepository

    src = tmp_path / "single_material"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(src / "only.parquet")
    task_id = client.post("/api/tasks", json={
        "model_name": "单文件", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    # C1 still confirms the sample + target even with one file (spec §1 / §3 single-file degenerate)
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert c1["metadata"]["join_c1"]["skip"] is True
    anchor_id = c1["metadata"]["join_c1"]["anchor_id"]
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    skip = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "无需拼接" in skip["content"]
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace").json()
    assert workspace["active_dataset_id"] == anchor_id
    assert len(workspace["active_dataset_content_hash"]) == 64
    assert workspace["revision"] == 1
    assert workspace["analysis_generation"] == 1
    assert workspace["semantic_mapping"]["target_col"] == "bad_flag"
    anchor = next(
        item
        for item in DatasetRepository(
            client.app.state.settings.db_path
        ).list_datasets(task_id)
        if item.id == anchor_id
    )
    assert anchor.has_target is True
    assert anchor.target_col == "bad_flag"
    from marvis.repositories.tasks import TaskRepository

    persisted_task = TaskRepository(
        client.app.state.settings.db_path
    ).get_task(task_id)
    assert persisted_task.target_col == "bad_flag"


def test_data_join_single_file_identical_prebinding_still_records_typed_confirmation(
    client: TestClient,
    tmp_path: Path,
):
    src = tmp_path / "single_material_prebound"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件预绑定确认",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202
    messages = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    c1_state = _last_assistant(messages)["metadata"]["join_c1"]
    _prebind_single_c1_workspace(client, task_id, c1_state)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )

    assert response.status_code == 202, response.text
    messages = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    assert sum(bool(item.get("metadata", {}).get("join_skip")) for item in messages) == 1
    assert sum(
        item.get("metadata", {}).get("ui_action") == "confirm_roles"
        and item["role"] == "user"
        for item in messages
    ) == 1
    assert sum(
        item.get("metadata", {}).get("intent") == "ui_action_ack"
        and item.get("metadata", {}).get("ui_action") == "confirm_roles"
        for item in messages
    ) == 1
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace").json()
    assert workspace["revision"] == 1
    assert workspace["active_dataset_id"] == c1_state["anchor_id"]

    repeated = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert repeated.status_code == 202, repeated.text
    repeated_messages = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    assert sum(
        bool(item.get("metadata", {}).get("join_skip"))
        for item in repeated_messages
    ) == 1
    assert sum(
        item.get("metadata", {}).get("intent") == "ui_action_ack"
        and item.get("metadata", {}).get("ui_action") == "confirm_roles"
        for item in repeated_messages
    ) == 1


def test_data_join_single_file_identical_prebinding_persists_semantic_receipt(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    src = tmp_path / "single_material_semantic_prebound"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )

    class _SemanticClient:
        def complete(self, *_args, **kwargs):
            if kwargs.get("prompt_name") == "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS":
                instruction = json.loads(kwargs["user_prompt"])["instruction"]
                return json.dumps(
                    {
                        "verdict": "authorize",
                        "evidence_quote": instruction,
                        "reason": "用户明确接受当前展示值。",
                        "confidence": "high",
                        "is_question": False,
                        "is_conditional": False,
                        "requests_change": False,
                        "withholds_authorization": False,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "action": "confirm",
                    "params": {},
                    "constraint": "",
                    "reason": "用户明确接受当前展示值。",
                    "confidence": "high",
                    "explicit_authorization": True,
                },
                ensure_ascii=False,
            )

    semantic_client = _SemanticClient()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: semantic_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: semantic_client,
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件语义预绑定确认",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "agent",
        },
    ).json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202
    messages = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    c1_state = _last_assistant(messages)["metadata"]["join_c1"]
    _prebind_single_c1_workspace(client, task_id, c1_state)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "我确认采用当前唯一样本表和 bad_flag 目标列。"},
    )

    assert response.status_code == 202, response.text
    messages = client.get(
        f"/api/tasks/{task_id}/agent/messages"
    ).json()["messages"]
    assert sum(
        item.get("metadata", {}).get("intent") == "c1_semantic_authorization"
        for item in messages
    ) == 1
    assert sum(
        bool(item.get("metadata", {}).get("join_skip")) for item in messages
    ) == 1


def test_data_join_agent_semantic_review_rejects_normalized_feature_byte_swap(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A two-pass C1 receipt never commits after normalized bytes drift."""

    from marvis.db import DatasetRepository

    src = _join_dir(tmp_path, n=12)

    class _SemanticClient:
        normalized_path: Path | None = None
        route_calls = 0
        review_calls = 0

        def complete(self, *_args, **kwargs):
            if kwargs.get("prompt_name") == "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS":
                self.review_calls += 1
                assert self.normalized_path is not None
                frame = pd.read_parquet(self.normalized_path)
                frame["balance"] = list(reversed(frame["balance"].tolist()))
                frame.to_parquet(self.normalized_path)
                instruction = json.loads(kwargs["user_prompt"])["instruction"]
                return json.dumps(
                    {
                        "verdict": "authorize",
                        "evidence_quote": instruction,
                        "reason": "用户明确接受当前展示值。",
                        "confidence": "high",
                        "is_question": False,
                        "is_conditional": False,
                        "requests_change": False,
                        "withholds_authorization": False,
                    },
                    ensure_ascii=False,
                )
            self.route_calls += 1
            return json.dumps(
                {
                    "action": "confirm",
                    "params": {},
                    "constraint": "",
                    "reason": "用户明确接受当前展示值。",
                    "confidence": "high",
                    "explicit_authorization": True,
                },
                ensure_ascii=False,
            )

    semantic_client = _SemanticClient()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: semantic_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: semantic_client,
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "拼接 C1 归一化数据漂移阻断",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "agent",
        },
    ).json()["id"]
    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    c1 = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )["metadata"]["join_c1"]
    feature_id = c1["feature_ids"][0]
    feature = next(
        item
        for item in DatasetRepository(
            client.app.state.settings.db_path
        ).list_datasets(task_id)
        if item.id == feature_id
    )
    semantic_client.normalized_path = (
        client.app.state.settings.datasets_dir / feature.source_path
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "我确认无误，请按推荐的样本主表和特征表继续。"},
    )

    assert response.status_code == 202, response.text
    assert semantic_client.route_calls == 1
    assert semantic_client.review_calls == 1
    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    error = _last_assistant(messages)
    assert error["metadata"].get("error") is True
    assert "数据或 DataWorkspace 已变化" in error["content"]
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
    semantic_receipts = [
        item
        for item in messages
        if item.get("metadata", {}).get("intent") == "c1_semantic_authorization"
    ]
    assert semantic_receipts == []
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace")
    assert workspace.status_code == 200, workspace.text
    assert workspace.json()["revision"] == 0
    assert workspace.json()["active_dataset_id"] is None


@pytest.mark.parametrize(
    ("selected_field", "semantic_mapping"),
    [
        (
            "mobile",
            {
                "target_col": "mobile",
                "field_roles": {"mobile": "target"},
                "business_names": {},
            },
        ),
        (
            "bad_flag",
            {
                "target_col": "bad_flag",
                "field_roles": {"bad_flag": "target"},
                "business_names": {"bad_flag": "人工确认的违约标签"},
            },
        ),
    ],
)
def test_data_join_single_file_repeat_confirmation_rejects_semantic_drift(
    client: TestClient,
    tmp_path: Path,
    selected_field: str,
    semantic_mapping: dict,
):
    src = tmp_path / f"single_semantic_drift_{selected_field}"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件语义漂移",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    initial = client.get(f"/api/tasks/{task_id}/data-workspace").json()

    changed = client.put(
        f"/api/tasks/{task_id}/data-workspace",
        headers={"If-Match": str(initial["revision"])},
        json={
            "active_dataset_id": initial["active_dataset_id"],
            "active_dataset_content_hash": initial[
                "active_dataset_content_hash"
            ],
            "page": "semantics",
            "selected_field": selected_field,
            "semantic_mapping": semantic_mapping,
        },
    )
    assert changed.status_code == 200, changed.text

    repeated = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert repeated.status_code == 202, repeated.text
    error = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    assert error["metadata"].get("error") is True
    assert "语义映射" in error["content"]


def test_data_join_single_file_confirmation_rejects_changed_registered_bytes(
    client: TestClient,
    tmp_path: Path,
):
    src = tmp_path / "single_material_drift"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件漂移",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})

    c1 = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    anchor_id = c1["metadata"]["join_c1"]["anchor_id"]
    dataset = next(
        item
        for item in client.get(f"/api/tasks/{task_id}/datasets").json()["datasets"]
        if item["id"] == anchor_id
    )
    registered_path = (
        client.app.state.settings.datasets_dir / dataset["source_path"]
    )
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [1, 1]}).to_parquet(
        registered_path
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert response.status_code == 202, response.text
    error = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    assert error["metadata"].get("error") is True
    assert "数据或 DataWorkspace 已变化" in error["content"]

    workspace = client.get(f"/api/tasks/{task_id}/data-workspace")
    assert workspace.status_code == 200, workspace.text
    assert workspace.json()["active_dataset_id"] is None
    assert workspace.json()["revision"] == 0


def test_data_join_single_file_race_after_precheck_rolls_back_workspace(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
):
    """A byte swap after the caller's precheck must not create revision 1."""

    from marvis.data.registry import DatasetRegistry

    src = tmp_path / "single_material_toctou"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件认证竞态",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})

    original_resolve = DatasetRegistry.resolve_verified_path
    calls = 0

    def swap_after_first_verification(self, dataset_id):
        nonlocal calls
        path = original_resolve(self, dataset_id)
        calls += 1
        if calls == 1:
            pd.DataFrame(
                {"mobile": ["a", "b"], "bad_flag": [1, 1]}
            ).to_parquet(path)
        return path

    monkeypatch.setattr(
        DatasetRegistry,
        "resolve_verified_path",
        swap_after_first_verification,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert response.status_code == 202, response.text
    error = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    assert error["metadata"].get("error") is True
    assert "数据或 DataWorkspace 已变化" in error["content"]
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace").json()
    assert workspace["active_dataset_id"] is None
    assert workspace["revision"] == 0


def test_data_join_single_file_final_source_swap_binds_authenticated_cas_snapshot(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
):
    """A mutable source swap after the last check cannot change the bound bytes."""

    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository

    src = tmp_path / "single_material_final_source_race"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件最终源竞态",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    repo = DatasetRepository(client.app.state.settings.db_path)
    registered = repo.list_datasets(task_id)[0]
    original_path = (
        client.app.state.settings.datasets_dir / registered.source_path
    ).resolve()
    original_resolve = DatasetRegistry.resolve_verified_path
    calls = 0

    def swap_original_after_final_cas_verification(self, dataset_id):
        nonlocal calls
        path = original_resolve(self, dataset_id)
        calls += 1
        if calls == 2:
            pd.DataFrame(
                {"mobile": ["a", "b"], "bad_flag": [1, 1]}
            ).to_parquet(original_path)
        return path

    monkeypatch.setattr(
        DatasetRegistry,
        "resolve_verified_path",
        swap_original_after_final_cas_verification,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert response.status_code == 202, response.text
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace").json()
    assert workspace["revision"] == 1
    assert workspace["active_dataset_id"] == registered.id
    pinned = repo.get_dataset(registered.id)
    assert pinned is not None
    assert pinned.source_path.startswith(f"_cas/{registered.content_hash}/")
    pinned_path = client.app.state.settings.datasets_dir / pinned.source_path
    assert hashlib.sha256(pinned_path.read_bytes()).hexdigest() == registered.content_hash
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() != registered.content_hash


def test_data_join_single_file_final_cas_overwrite_attempt_rolls_back_workspace(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
):
    """The object verified immediately before commit is not platform-writable."""

    from marvis.data.registry import DatasetRegistry

    src = tmp_path / "single_material_final_cas_race"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(
        src / "only.parquet"
    )
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "单文件最终 CAS 竞态",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "manual",
        },
    ).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    original_resolve = DatasetRegistry.resolve_verified_path
    calls = 0

    def overwrite_after_final_verification(self, dataset_id):
        nonlocal calls
        path = original_resolve(self, dataset_id)
        calls += 1
        if calls == 2:
            pd.DataFrame(
                {"mobile": ["a", "b"], "bad_flag": [1, 1]}
            ).to_parquet(path)
        return path

    monkeypatch.setattr(
        DatasetRegistry,
        "resolve_verified_path",
        overwrite_after_final_verification,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "ui_action": "confirm_roles"},
    )
    assert response.status_code == 202, response.text
    workspace = client.get(f"/api/tasks/{task_id}/data-workspace").json()
    assert workspace["active_dataset_id"] is None
    assert workspace["revision"] == 0


def test_data_join_double_confirm_second_request_gets_409_without_second_turn(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """REL-1: a double-sent confirm (dual tab / retried request) while the first
    driver turn is still executing must be rejected with 409 by the
    idx_jobs_active_task guard *before* it reaches dispatch_plan_driver_turn a
    second time — not raced into a second PlanExecutor.run that could mis-flag
    the in-flight step as a restart orphan (REL-1) and clobber it with a 500."""
    from marvis.agent import validation_app_service as vas

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "并发确认", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    turn_calls: list[str] = []
    real_dispatch = vas.dispatch_plan_driver_turn

    def racing_dispatch(runtime, repo, task, **kwargs):
        turn_calls.append(task.id)
        # Simulate Tab B re-sending the confirmation while Tab A's turn (this
        # call) is still executing: it must be rejected before ever reaching
        # dispatch_plan_driver_turn again.
        second = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "确认"},
        )
        assert second.status_code == 409, second.text
        assert second.json()["detail"] == "该任务正在执行上一步，请等待完成"
        return real_dispatch(runtime, repo, task, **kwargs)

    monkeypatch.setattr(vas, "dispatch_plan_driver_turn", racing_dispatch)

    first = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})

    assert first.status_code == 202, first.text
    # dispatch_plan_driver_turn (the real turn / PlanExecutor.run path) ran
    # exactly once for this confirm — the racing second request never got in.
    assert turn_calls == [task_id]
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]


def test_data_join_active_driver_job_reports_kind_driver_on_task_payload(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """REL-6: GET /api/tasks (and the single-task endpoint) must surface
    active_job_kind == "driver" while a driver turn is executing, so the
    frontend's busy state / 1s polling can pick it up after a refresh or from
    any other entry point (UX-1)."""
    from marvis.agent import validation_app_service as vas

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "任务忙碌态", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    observed: dict[str, str | None] = {}
    real_dispatch = vas.dispatch_plan_driver_turn

    def observing_dispatch(runtime, repo, task, **kwargs):
        mid_turn = client.get(f"/api/tasks/{task_id}")
        assert mid_turn.status_code == 200, mid_turn.text
        observed["single"] = mid_turn.json()["active_job_kind"]
        listed = client.get("/api/tasks")
        listed_task = next(t for t in listed.json() if t["id"] == task_id)
        observed["listed"] = listed_task["active_job_kind"]
        return real_dispatch(runtime, repo, task, **kwargs)

    monkeypatch.setattr(vas, "dispatch_plan_driver_turn", observing_dispatch)

    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text

    assert observed["single"] == "driver"
    assert observed["listed"] == "driver"
    # The job is finished (synchronous turn) once the HTTP response returns.
    after = client.get(f"/api/tasks/{task_id}")
    assert after.json()["active_job_kind"] is None


def test_data_join_driver_turn_job_finishes_on_exception(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """REL-1: the job must be released (finish_job in the finally-equivalent
    except branch) even when the turn function raises an unexpected exception,
    so a single failed turn doesn't permanently lock the task behind the
    idx_jobs_active_task unique index."""
    from marvis.agent import validation_app_service as vas
    from marvis.app import create_app
    from marvis.db import TaskRepository

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "异常清理", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    def boom(runtime, repo, task, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(vas, "dispatch_plan_driver_turn", boom)

    # A fresh client sharing the same tmp_path-backed app/db, configured not to
    # re-raise the server exception, so the 500 can be asserted on the response
    # instead of propagating through TestClient.
    error_client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    resp = error_client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 500

    repo = TaskRepository(tmp_path / "marvis.sqlite")
    assert repo.get_active_job_kind(task_id) is None

    # The task is not stuck: a fresh confirm (now unpatched) can still claim
    # the job and complete the turn.
    monkeypatch.undo()
    resumed = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resumed.status_code == 202, resumed.text


def test_data_join_plan_payload_carries_started_at_for_running_step(
    client: TestClient, tmp_path: Path,
):
    """UX-1/REL-6: a step's plan payload gets a started_at field once it is
    RUNNING (sourced from plan_step_runs, already recorded per attempt), so the
    plan rail can show elapsed time via the same formatStepElapsed() range the
    validation stepper already uses, instead of a plain spinner."""
    from marvis.db import PlanRepository
    from marvis.orchestrator.contracts import StepStatus

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "耗时展示", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    plan = plans[-1]
    execute_step = next(step for step in plan["steps"] if step["tool_ref"]["tool"] == "execute_join")
    assert execute_step.get("started_at") in (None, "")  # not running yet

    repo = PlanRepository(tmp_path / "marvis.sqlite")
    loaded = repo.load_plan(plan["id"])
    running_step = next(step for step in loaded.steps if step.id == execute_step["id"])
    running_step.status = StepStatus.RUNNING
    repo.update_step(running_step)
    run_id = repo.start_step_run(
        plan_id=plan["id"],
        step_id=execute_step["id"],
        tool_ref="data_ops.execute_join",
        inputs={},
    )
    repo.update_step_run_progress(
        run_id,
        {
            "kind": "model_tuning",
            "algorithm": "xgb",
            "trial": 7,
            "trial_total": 40,
        },
    )

    refreshed = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
    refreshed_step = next(step for step in refreshed["steps"] if step["id"] == execute_step["id"])
    assert refreshed_step["status"] == "running"
    assert refreshed_step["started_at"]
    assert refreshed_step["progress"] == {
        "kind": "model_tuning",
        "algorithm": "xgb",
        "trial": 7,
        "trial_total": 40,
    }
    assert refreshed_step["progress_updated_at"]
