"""Deterministic diagnose / repair / verify for V2 workflows (B-line extension).

Spec: docs/superpowers/specs/2026-10-23-self-repair-extension-spec.md

Classifies a failed plan step from its tool, error text and error kind,
checks it with read-only probes, and offers repair options in the B-line
tiers (``auto`` / ``one_click`` / ``confirm``). An executor changes only the
failed step's declared inputs (or the task's own interpreter override), reruns
the step through ``PlanDriver``, verifies the result and rolls back when the
check fails. Nothing here calls an LLM or computes a deterministic metric; the
report text comes from what was executed and verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any
import uuid

from marvis.agent.validation_repair import (
    TIER_AUTO,
    TIER_CONFIRM,
    TIER_ONE_CLICK,
    UNDO_COMMAND,
    _option,
    _ran_automatically,
    _recovery_actions,
    _repair_reports,
    _undone_repair_ids,
    match_missing_fields,
    repair_budget,
)
from marvis.files import write_json_atomic


logger = logging.getLogger(__name__)

WORKFLOW_REPAIR_SCHEMA = "workflow_repair.v1"
STATE_SCHEMA = "workflow_repairs.v1"
DIAGNOSED_BY = "workflow_repair"
V2_WORKFLOWS = frozenset({"data_join", "feature_analysis", "modeling", "strategy"})
RETRY_COMMAND = "重试当前步骤"
_WORKFLOW_NAMES = {
    "data_join": "数据处理",
    "feature_analysis": "特征分析",
    "modeling": "模型开发",
    "strategy": "策略分析",
}
_PROBE_ROWS = 50_000
_TEXT_TO_FLOAT_RE = re.compile(r"could not convert string to float")
_TARGET_NOT_BINARY_RE = re.compile(r"target must be binary 0/1")
_UNPARSEABLE_STRING_RE = re.compile(r"Unable to parse string ['\"](?P<value>[^'\"]*)['\"]")
_NAN_LABEL_RE = re.compile(r"target '(?P<col>[^']+)' has (?P<nan>\d+)/(?P<total>\d+) NaN labels")
_PYTHON_REQUIRES_RE = re.compile(
    r"要求 Python (?P<req>[^，]+)，但当前工具运行环境是 Python (?P<ver>[\d.]+)（(?P<path>[^）]+)）"
)
_TIMEOUT_RE = re.compile(r"timed out after (?P<seconds>\d+)")
_EMPTY_SETS_RE = re.compile(r"切分后这些集合为空:\[(?P<sets>[^\]]*)\]")
_TARGET_THRESHOLDS = (1, 30, 90)
_IV_THRESHOLDS = (0.01, 0.005)
_SET_LABELS = {"train": "训练集", "test": "测试集", "oot": "OOT"}
UNDO_UNAVAILABLE_REPLY = (
    "现在没有可以撤销的处理：要么已经撤销过，要么流程已经往后走了（例如后面的确认已经做过）。"
    "平台没有做任何改动。"
)
STALE_COMMAND_REPLY = (
    "这个办法已经过期：流程在那张卡片之后有了新进展，平台没有执行它。请按最新的卡片或步骤操作。"
)


# ---------- task repair state (tasks/<id>/repairs/workflow_repairs.json) ----------


def repair_state_path(tasks_dir: Path, task_id: str) -> Path:
    return Path(tasks_dir) / task_id / "repairs" / "workflow_repairs.json"


def load_repair_state(tasks_dir: Path, task_id: str) -> dict:
    empty = {"schema_version": STATE_SCHEMA, "applied": [], "rejected_fingerprints": [], "audit": []}
    try:
        payload = json.loads(repair_state_path(tasks_dir, task_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return empty
    if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA:
        return empty
    return {
        "schema_version": STATE_SCHEMA,
        "applied": [item for item in payload.get("applied") or [] if isinstance(item, dict)],
        "rejected_fingerprints": [str(item) for item in payload.get("rejected_fingerprints") or [] if item],
        "audit": [item for item in payload.get("audit") or [] if isinstance(item, dict)],
    }


def save_repair_state(tasks_dir: Path, task_id: str, state: dict) -> None:
    path = repair_state_path(tasks_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, state, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------- failure context ----------


@dataclass(frozen=True)
class FailureContext:
    settings: Any
    task: Any
    plan_repo: Any
    plan: Any
    step: Any
    error: str
    error_kind: str
    run_seq: int


def failure_context(settings, task, plan_repo, metadata: dict) -> FailureContext | None:
    """The failed step a driver failure message points at, if it is still failed."""

    envelope = metadata.get("failure_envelope") if isinstance(metadata, dict) else None
    if not isinstance(envelope, dict):
        return None
    step_id = str(envelope.get("failed_step_id") or metadata.get("step_id") or "")
    plan_id = str(metadata.get("plan_id") or "")
    if not step_id or not plan_id:
        return None
    try:
        plan = plan_repo.load_plan(plan_id)
    except Exception:  # noqa: BLE001 - an unreadable plan keeps the original card
        return None
    step = next((item for item in plan.steps if item.id == step_id), None)
    if step is None or getattr(step.status, "value", step.status) != "failed":
        return None
    kind = ""
    reader = getattr(plan_repo, "latest_failed_step_run_error_kind", None)
    if callable(reader):
        kind = str(reader(step.id) or "")
    return FailureContext(
        settings=settings,
        task=task,
        plan_repo=plan_repo,
        plan=plan,
        step=step,
        error=str(step.error or ""),
        error_kind=kind,
        run_seq=int(envelope.get("run_seq") or metadata.get("run_seq") or 0),
    )


def _resolve_input(plan_repo, value):
    if not (isinstance(value, str) and value.startswith("$ref:")):
        return value
    from marvis.orchestrator.references import parse_step_output_ref

    step_id, field = parse_step_output_ref(value)
    loader = getattr(plan_repo, "load_bound_step_output", None) or plan_repo.load_step_output
    output = loader(step_id)
    for part in (field or "").split("."):
        if not part:
            continue
        output = output.get(part) if isinstance(output, dict) else None
    return output


def _registry(settings):
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository

    return DatasetRegistry(
        DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir
    )


def _tool_properties(plugin: str, tool: str) -> set[str]:
    manifest = Path(__file__).resolve().parents[1] / "packs" / plugin / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    for item in payload.get("tools") or []:
        if item.get("name") == tool:
            return set(((item.get("input_schema") or {}).get("properties") or {}).keys())
    return set()


def _column_families(dataset) -> dict[str, str]:
    from marvis.data.align import _dtype_family_from_str

    return {profile.name: _dtype_family_from_str(profile.dtype) for profile in dataset.columns}


# ---------- diagnostic builder ----------


def _diagnostic(ctx: FailureContext, *, code: str, title: str, summary: str, cause: str,
                location: str, evidence: list[dict], impact: str, actions: list[str],
                options: list[dict], agent_mode: bool, retryable: bool = True) -> dict:
    visible = list(options) if agent_mode else []
    recovery = _recovery_actions(visible)
    if retryable and agent_mode:
        recovery.append({"label": "按原样重试当前步骤", "command": RETRY_COMMAND, "tier": "", "repair_option_id": ""})
    elif retryable:
        recovery = [{"label": "由 Agent 重试当前步骤", "command": RETRY_COMMAND}]
    detail = ctx.error
    source_dir = str(getattr(ctx.task, "source_dir", "") or "")
    if source_dir:
        detail = detail.replace(source_dir, "<材料目录>")
    return {
        "schema_version": "workflow_error.v1",
        "diagnosed_by": DIAGNOSED_BY,
        "workflow": str(getattr(ctx.task, "task_type", "") or ""),
        "stage": ctx.step.tool_ref.label(),
        "plan_id": ctx.plan.id,
        "step_id": ctx.step.id,
        "code": code,
        "phase": "execution",
        "title": title,
        "summary": summary,
        "cause": cause,
        "location": location,
        "evidence": evidence,
        "impact": impact,
        "actions": actions,
        "retryable": retryable,
        "auto_recoverable": False,
        "error_kind": ctx.error_kind or "execution",
        "repair_options": visible,
        "recovery_actions": recovery,
        "technical_detail": detail[:2000],
    }


def diagnose_workflow_failure(ctx: FailureContext, *, agent_mode: bool) -> dict | None:
    """Classify one failed V2 step; ``None`` keeps the original failure card."""

    if str(getattr(ctx.task, "task_type", "")) not in V2_WORKFLOWS:
        return None
    tool = ctx.step.tool_ref.label()
    text = ctx.error
    try:
        if tool == "data_ops.confirm_join" and "needs_dtype_ack" in text:
            return _join_dtype_diagnostic(ctx, agent_mode)
        if ctx.error_kind == "environment" and _PYTHON_REQUIRES_RE.search(text):
            return _environment_diagnostic(ctx, agent_mode)
        if tool == "modeling.tune_hyperparameters" and (
            ctx.error_kind == "timeout" or "original error_kind=timeout" in text
        ):
            return _tuning_timeout_diagnostic(ctx, agent_mode)
        if ctx.step.tool_ref.plugin == "feature":
            if ctx.error_kind == "nan_label_not_confirmed" and _NAN_LABEL_RE.search(text):
                return _nan_label_diagnostic(ctx, agent_mode)
            if _TARGET_NOT_BINARY_RE.search(text) or _target_text_failure(ctx):
                return _target_definition_diagnostic(ctx, agent_mode)
            if _TEXT_TO_FLOAT_RE.search(text):
                return _text_feature_diagnostic(ctx, agent_mode)
        if "未找到可用候选特征列" in text:
            return _no_candidate_diagnostic(ctx, agent_mode)
        if tool == "modeling.make_split" and _EMPTY_SETS_RE.search(text):
            return _split_diagnostic(ctx, agent_mode)
        if tool == "modeling.select_features" and "selected empty" in text:
            return _iv_threshold_diagnostic(ctx, agent_mode)
    except Exception:  # noqa: BLE001 - a probe must never replace the failure itself
        logger.exception("workflow repair diagnosis failed task_id=%s", getattr(ctx.task, "id", ""))
        return None
    return None


# ---------- 1. join key text vs float ----------


def _join_dtype_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    join_plan_id = _resolve_input(ctx.plan_repo, (ctx.step.inputs or {}).get("join_plan_id"))
    ref = str((ctx.step.inputs or {}).get("join_plan_id") or "")
    from marvis.orchestrator.references import parse_step_output_ref

    propose_step_id = parse_step_output_ref(ref)[0] if ref.startswith("$ref:") else ""
    proposal = ctx.plan_repo.load_step_output(propose_step_id) if propose_step_id else {}
    anchor_id = str(proposal.get("anchor_dataset_id") or "")
    registry = _registry(ctx.settings)
    evidence: list[dict] = []
    safe_features: list[str] = []
    unsafe = False
    names: list[str] = []
    for join in proposal.get("joins") or []:
        divergences = [
            item for item in ((join.get("diagnostics") or {}).get("key_dtype_divergences") or [])
            if item.get("level") == "red"
        ]
        if not divergences:
            continue
        feature_id = str(join.get("feature_id") or "")
        feature_name = str(join.get("feature_name") or "特征表")
        match_rate = (join.get("diagnostics") or {}).get("match_rate")
        for item in divergences:
            anchor_col, feature_col = str(item["anchor_col"]), str(item["feature_col"])
            names.append(anchor_col if anchor_col == feature_col else f"{anchor_col} / {feature_col}")
            probe = _key_probe(
                registry, anchor_id, anchor_col, str(item.get("anchor_dtype")),
                feature_id, feature_col, str(item.get("feature_dtype")),
            )
            evidence.append({
                "label": f"{feature_name} 的拼接键",
                "value": f"样本 {anchor_col}（{_family_word(item.get('anchor_dtype'))}）↔ "
                         f"特征表 {feature_col}（{_family_word(item.get('feature_dtype'))}）",
            })
            evidence.append({"label": "数值侧", "value": probe["numeric_note"]})
            evidence.append({"label": "文本侧", "value": probe["text_note"]})
            if match_rate is not None:
                evidence.append({"label": "抽样命中率", "value": f"{float(match_rate):.2%}"})
            unsafe = unsafe or not probe["safe"]
        if feature_id:
            safe_features.append(feature_id)
    if not names or join_plan_id is None:
        return None
    key_text = "、".join(dict.fromkeys(names))
    options = []
    if safe_features and not unsafe:
        options.append(_option(
            "ack_key_dtype",
            tier=TIER_ONE_CLICK,
            label=f"确认 {key_text} 是同一编号，继续拼接",
            command=f"确认 {key_text} 两侧是同一编号，继续拼接",
            consequence=(
                f"按 {key_text} 的编号对齐两表；特征表里没有编号的行不会命中。"
                "执行前仍会停在「执行拼接」确认。"
            ),
            params={"feature_ids": sorted(safe_features), "join_plan_id": str(join_plan_id)},
            reversible=True,
        ))
    actions = (
        ["如果两侧不是同一个编号，请换拼接键或把特征表的编号列按文本重新导出后重新上传。"]
        if options
        else ["文本侧有前导零或数值侧有小数，确认会漏配；请把特征表的编号列按文本重新导出后重新上传。"]
    )
    return _diagnostic(
        ctx,
        code="join_key_dtype_mismatch",
        title="拼接键两侧类型不一致",
        summary=f"拼接键 {key_text} 一侧是文本、一侧是数值，平台需要你确认是同一个编号后才能拼接。",
        cause="数值列遇到空值会存成浮点，编号的前导零会丢失；不确认就拼接可能悄悄漏配。",
        location="确认拼接 · 拼接键",
        evidence=evidence,
        impact="拼接还没有执行，样本和特征表都没有改动。",
        actions=actions,
        options=options,
        agent_mode=agent_mode,
    )


def _family_word(dtype) -> str:
    from marvis.data.align import _dtype_family_from_str

    return {"text": "文本", "float": "浮点数", "int": "整数", "date": "日期"}.get(
        _dtype_family_from_str(str(dtype or "")), str(dtype or "未知")
    )


def _key_probe(registry, anchor_id, anchor_col, anchor_dtype, feature_id, feature_col, feature_dtype) -> dict:
    """Read both key columns and decide whether a text/number ack is safe.

    Tables longer than ``_PROBE_ROWS`` are checked on rows spread evenly over
    the whole table, and the notes say so.
    """

    import pandas as pd

    from marvis.data.align import _dtype_family_from_str

    sides = [(anchor_id, anchor_col, anchor_dtype), (feature_id, feature_col, feature_dtype)]
    text_side = next((item for item in sides if _dtype_family_from_str(item[2]) == "text"), None)
    numeric_side = next((item for item in sides if item is not text_side), None)
    if text_side is None or numeric_side is None:
        return {"safe": False, "numeric_note": "没有读到", "text_note": "没有读到"}
    numeric = registry.read_authenticated_parquet_snapshot(numeric_side[0], columns=[numeric_side[1]])[numeric_side[1]]
    text = registry.read_authenticated_parquet_snapshot(text_side[0], columns=[text_side[1]])[text_side[1]]
    sampled = max(len(numeric), len(text)) > _PROBE_ROWS
    numeric = pd.to_numeric(numeric.iloc[::max(1, -(-len(numeric) // _PROBE_ROWS))], errors="coerce")
    text = text.iloc[::max(1, -(-len(text) // _PROBE_ROWS))].dropna().astype(str).str.strip()
    scope = f"（全表均匀抽查 {_PROBE_ROWS // 10_000} 万行）" if sampled else ""
    present = numeric.dropna()
    integral = bool(len(present)) and bool(((present % 1) == 0).all())
    leading_zero = int(text.str.match(r"^0\d").sum())
    non_digit = int((~text.str.fullmatch(r"\d+")).sum())
    numeric_note = (
        f"{len(present)} 个非空值都是整数，{int(numeric.isna().sum())} 个空值"
        if integral
        else f"{int(((present % 1) != 0).sum())} 个值带小数"
    )
    text_note = (
        f"{leading_zero} 个编号有前导零" if leading_zero
        else f"{non_digit} 个编号不是纯数字" if non_digit
        else "全部是纯数字、没有前导零"
    )
    return {"safe": integral and not leading_zero and not non_digit,
            "numeric_note": numeric_note + scope, "text_note": text_note + scope}


# ---------- 2. text columns among the features ----------


def _feature_dataset(ctx: FailureContext):
    dataset_id = _resolve_input(ctx.plan_repo, (ctx.step.inputs or {}).get("dataset_id"))
    if not dataset_id:
        return None
    return _registry(ctx.settings).get(str(dataset_id))


def _text_feature_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    features = [str(item) for item in (ctx.step.inputs or {}).get("features") or [] if str(item).strip()]
    dataset = _feature_dataset(ctx)
    if not features or dataset is None:
        return None
    families = _column_families(dataset)
    text_columns = [name for name in features if families.get(name) == "text"]
    if not text_columns:
        return None
    remaining = [name for name in features if name not in text_columns]
    options = []
    if remaining:
        options.append(_option(
            "exclude_text_features",
            tier=TIER_AUTO,
            label=f"这次先不算文本列 {'、'.join(text_columns[:5])}",
            command=f"这次先不算文本列 {'、'.join(text_columns)}，其余特征照常计算",
            consequence=(
                f"{'、'.join(text_columns)} 不出现在本次指标表里，其余 {len(remaining)} 个特征的指标不受影响；可以撤销。"
            ),
            params={"exclude": text_columns, "features": remaining},
            reversible=True,
        ))
    return _diagnostic(
        ctx,
        code="feature_text_columns",
        title="特征里有文本列",
        summary=f"特征 {'、'.join(text_columns[:10])} 是文本（类别）列，数值指标算不了。",
        cause="IV、KS、分箱等指标按数值计算；文本列需要先编码，平台不会自行编码。",
        location=f"{ctx.step.title} · 特征列表",
        evidence=[
            {"label": "文本列", "value": "、".join(text_columns[:20])},
            {"label": "数值列", "value": f"{len(remaining)} 个"},
        ],
        impact="特征指标还没有算出来，后续分箱和报告都还没开始。",
        actions=["要分析文本列，先在数据处理里对它做 WOE 或独热编码，再把编码后的列加进来。"],
        options=options,
        agent_mode=agent_mode,
    )


# ---------- 3. no candidate features ----------


def _no_candidate_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    dataset = _feature_dataset(ctx)
    if dataset is None:
        return None
    families = _column_families(dataset)
    inputs = ctx.step.inputs or {}
    skipped = {str(inputs.get("target_col") or ""), str(inputs.get("split_col") or "")}
    text_columns = [name for name, family in families.items() if family == "text" and name not in skipped]
    date_columns = [name for name, family in families.items() if family == "date" and name not in skipped]
    return _diagnostic(
        ctx,
        code="feature_no_candidates",
        title="没有可用的数值特征",
        summary="平台在这份数据里没找到可以计算指标的数值特征。",
        cause="自动选特征只取数值列；这份数据除目标列和切分列外都是文本或日期列。",
        location=f"{ctx.step.title} · 候选特征",
        evidence=[
            {"label": "总列数", "value": str(len(families))},
            {"label": "文本列", "value": "、".join(text_columns[:20]) or "没有"},
            {"label": "日期列", "value": "、".join(date_columns[:10]) or "没有"},
        ],
        impact="特征指标还没有算出来。",
        actions=[
            "在消息里指定要分析的数值特征。",
            "要用文本列，先在数据处理里做编码，再重新发起特征分析。",
        ],
        options=[],
        agent_mode=agent_mode,
    )


# ---------- 4. modeling split leaves a set empty ----------


def _split_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    import pandas as pd

    from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED
    from marvis.packs.modeling.split_preflight import dry_run_split, split_config_columns, split_rejection

    inputs = ctx.step.inputs or {}
    config = dict(inputs.get("split_config") or {})
    time_col = str(config.get("oot_by_time") or "")
    if not time_col:
        return None
    dataset_id = _resolve_input(ctx.plan_repo, inputs.get("dataset_id"))
    registry = _registry(ctx.settings)
    dataset = registry.get(str(dataset_id))
    columns = split_config_columns(config)
    frame = registry.read_authenticated_parquet_snapshot(dataset.id, columns=columns)
    counts = frame[time_col].value_counts(dropna=True).sort_index()
    seed = int(inputs.get("seed") if inputs.get("seed") is not None else DEFAULT_RANDOM_SEED)
    match = _EMPTY_SETS_RE.search(ctx.error)
    empty = [item.strip(" '\"") for item in match.group("sets").split(",")] if match else []
    evidence = [
        {"label": "时间列", "value": time_col},
        {"label": "时间取值", "value": "、".join(f"{_plain(k)}：{int(v)} 行" for k, v in list(counts.items())[-6:])},
        {"label": "切分后为空", "value": "、".join(_SET_LABELS.get(item, item) for item in empty) or "—"},
    ]
    base = {key: value for key, value in config.items() if key not in {"oot_by_time", "oot_size", "random_oot"}}
    options = []
    if len(counts) >= 2:
        target = float(config.get("oot_size", 0.2) or 0.2)
        total = float(counts.sum())
        chosen = []
        share = 0.0
        for value, count in reversed(list(counts.items())):
            if len(chosen) >= len(counts) - 1:
                break
            chosen.append(value)
            share += float(count) / total
            if share >= target:
                break
        start = _plain(min(chosen))
        candidate = {**base, "rules": [*(base.get("rules") or []), {
            "when": [{"col": time_col, "op": "ge", "val": start}], "assign": "oot"}]}
        option_id, label = "split_latest_oot", f"用 {time_col} ≥ {start} 的样本作 OOT"
        consequence = f"OOT 改为 {time_col} ≥ {start} 的样本（约 {share:.0%}），train/test 从其余样本按原比例切；可以撤销。"
    else:
        candidate = dict(base)
        option_id, label = "split_without_oot", "不留 OOT，只切 train/test"
        consequence = "这份样本只有一个时间段，没有跨时间验证；模型报告会显示没有 OOT。可以撤销。"
    if split_rejection(frame, candidate, seed) is None:
        preview = dry_run_split(frame, candidate, seed).value_counts().to_dict()
        evidence.append({
            "label": "按新口径预演",
            "value": "、".join(f"{_SET_LABELS.get(k, k)} {int(v)} 行" for k, v in sorted(preview.items())),
        })
        options.append(_option(
            option_id,
            tier=TIER_ONE_CLICK,
            label=label,
            command=f"{label}，重新切分样本",
            consequence=consequence,
            params={"split_config": candidate},
            reversible=True,
        ))
    del pd
    return _diagnostic(
        ctx,
        code="modeling_split_empty",
        title="样本切分后有集合为空",
        summary=f"按 {time_col} 切 OOT 后，{'、'.join(_SET_LABELS.get(item, item) for item in empty)}没有样本。",
        cause="时间切分按时间先后的分位点划 OOT；时间分布不均（某个月占大多数或只有一个月）时，全部样本都会落进 OOT。",
        location="切分样本 · OOT 口径",
        evidence=evidence,
        impact="切分在写数据前就停下了，没有产生建模样本，后续步骤都还没开始。",
        actions=["也可以在消息里说明新的切分口径，例如指定 OOT 从哪个月开始。"],
        options=options,
        agent_mode=agent_mode,
    )


def _plain(value):
    return value.item() if hasattr(value, "item") else value


# ---------- 5. tuning timeout ----------


def _tuning_timeout_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    match = _TIMEOUT_RE.search(ctx.error)
    seconds = int(match.group("seconds")) if match else None
    config_step = next(
        (step for step in ctx.plan.steps if step.tool_ref.label() == "modeling.configure_tuning"), None
    )
    budgets: dict = {}
    if config_step is not None and config_step.output_ref:
        try:
            output = ctx.plan_repo.load_step_output(config_step.id)
            budgets = dict(output.get("n_trials_by_recipe") or {})
        except Exception:  # noqa: BLE001 - evidence only
            budgets = {}
    halved = {recipe: max(1, int(count) // 2) for recipe, count in budgets.items()}
    reconciliation = "explicit reconciliation required" in ctx.error.lower()
    evidence = [
        {"label": "超时上限", "value": f"{seconds // 3600} 小时" if seconds and seconds >= 3600 else f"{seconds} 秒" if seconds else "未知"},
        {"label": "调参预算", "value": "、".join(f"{k} {v} 轮" for k, v in budgets.items()) or "没有读到"},
    ]
    options = []
    if reconciliation:
        options.append(_option(
            "rebuild_with_smaller_budget",
            tier=TIER_CONFIRM,
            label="新建模型开发任务并减少调参轮数（需要你操作）",
            command="",
            consequence="原计划停在待核对状态，已有结果保留；新任务从头切分和筛选。",
        ))
    elif halved and halved != budgets:
        options.append(_option(
            "halve_tuning_budget",
            tier=TIER_ONE_CLICK,
            label="调参预算减半，回到配置调参确认",
            command="把调参预算减半，回到配置调参确认",
            consequence=f"预算改为 {'、'.join(f'{k} {v} 轮' for k, v in halved.items())}；确认后才会重新调参。",
            params={"n_trials_by_recipe": halved},
            reversible=False,
        ))
    actions = []
    if reconciliation:
        actions.append(
            "调参中途写出的记录平台无法证明没有生效，所以这一步不能直接重跑；"
            "请新建模型开发任务，在「配置调参」时把轮数减半"
            + (f"（{'、'.join(f'{k} {budgets[k]}→{v}' for k, v in halved.items())}）" if halved else "")
            + "，或减少特征、抽样后再调参。"
        )
    return _diagnostic(
        ctx,
        code="modeling_tuning_timeout",
        title="调参超时",
        summary="调参在时间上限内没有完成，已停止。",
        cause="数据量、特征数和调参轮数共同决定调参时长；超过上限后平台会停止调参进程。",
        location="调参 · 运行时长",
        evidence=evidence,
        impact="没有产出调参结果；切分、特征筛选等已完成步骤的结果都保留。",
        actions=actions,
        options=options,
        agent_mode=agent_mode,
        retryable=not reconciliation,
    )


# ---------- 6. worker environment ----------


def _environment_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    from marvis.task_environment import APPLIES_TO, probe_python_candidates

    match = _PYTHON_REQUIRES_RE.search(ctx.error)
    requirement, version, current = match.group("req"), match.group("ver"), match.group("path")
    plugin = ctx.step.tool_ref.plugin
    options = []
    candidates = []
    if plugin in APPLIES_TO and agent_mode:
        candidates = probe_python_candidates(python_requires=requirement, exclude=[current])[:2]
        for candidate in candidates:
            options.append(_option(
                "use_task_environment",
                tier=TIER_ONE_CLICK,
                label=f"这个任务改用 {candidate['label']} 运行",
                command=f"这个任务改用 {candidate['label']}（Python {candidate['python_version']}）运行建模和特征工具",
                consequence="只对这个任务的建模和特征步骤生效，全局执行环境不变；可以撤销。",
                params=candidate,
                reversible=True,
            ))
    options.append(_option(
        "change_global_environment",
        tier=TIER_CONFIRM,
        label="修改全局执行环境（需要你在设置里操作）",
        command="",
        consequence="影响所有任务，改完需要重启 MARVIS。",
    ))
    evidence = [
        {"label": "工具要求", "value": f"Python {requirement}"},
        {"label": "当前环境", "value": f"Python {version}"},
        {"label": "本机可用环境", "value": "、".join(
            f"{item['label']}（Python {item['python_version']}）" for item in candidates) or (
            "没有找到" if plugin in APPLIES_TO and agent_mode else "未检查")},
    ]
    return _diagnostic(
        ctx,
        code="workflow_environment_incompatible",
        title="工具运行环境的 Python 版本不兼容",
        summary=f"「{ctx.step.title}」要求 Python {requirement}，当前工具运行环境是 Python {version}。",
        cause="工具在独立的 Python 进程里运行；进程用的解释器版本不在工具支持的范围内，工具没有启动。",
        location=f"{ctx.step.title} · 运行环境",
        evidence=evidence,
        impact="这一步没有开始执行，数据和已完成步骤都没有改动。",
        actions=["也可以在「设置 → 执行环境」换成兼容环境并重启 MARVIS；平台不会替你改全局设置或安装 Python。"],
        options=options,
        agent_mode=agent_mode,
    )


# ---------- 7. NaN labels: drop rows ----------


def _nan_label_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    match = _NAN_LABEL_RE.search(ctx.error)
    target, n_nan, n_total = match.group("col"), int(match.group("nan")), int(match.group("total"))
    dataset = _feature_dataset(ctx)
    if dataset is None:
        return None
    column = _registry(ctx.settings).read_authenticated_parquet_snapshot(dataset.id, columns=[target])[target]
    observed = int(column.isna().sum())
    options = []
    if observed == n_nan and n_nan < n_total:
        options.append(_option(
            "drop_nan_label_rows",
            tier=TIER_ONE_CLICK,
            label=f"剔除目标为空的 {n_nan} 行后继续",
            command=f"剔除目标 {target} 为空的 {n_nan} 行后继续",
            consequence=f"本次分析只用目标有值的 {n_total - n_nan} 行，样本文件不变；可以撤销。",
            params={"target_col": target, "rows": n_nan, "total": n_total},
            reversible=True,
        ))
    return _diagnostic(
        ctx,
        code="feature_nan_labels",
        title="目标列有空值",
        summary=f"目标 {target} 有 {n_nan} / {n_total} 行是空值，平台不会把空值当成好或坏样本。",
        cause="空标签没有好坏信息；剔除这些行会改变样本口径，需要你确认。",
        location=f"{ctx.step.title} · 目标列",
        evidence=[
            {"label": "目标列", "value": target},
            {"label": "空值行数（复核）", "value": f"{observed} / {len(column)}"},
        ],
        impact="特征指标还没有算出来。",
        actions=["也可以补齐空标签后重新上传样本。"],
        options=options,
        agent_mode=agent_mode,
    )


# ---------- 8. target not 0/1: redefine the target ----------


def _target_text_failure(ctx: FailureContext) -> bool:
    match = _UNPARSEABLE_STRING_RE.search(ctx.error)
    if match is None:
        return False
    target = str((ctx.step.inputs or {}).get("target_col") or "")
    dataset = _feature_dataset(ctx)
    return bool(target and dataset is not None and _column_families(dataset).get(target) == "text")


def _target_definition_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    import pandas as pd

    target = str((ctx.step.inputs or {}).get("target_col") or "")
    dataset = _feature_dataset(ctx)
    if not target or dataset is None:
        return None
    values = _registry(ctx.settings).read_authenticated_parquet_snapshot(dataset.id, columns=[target])[target]
    present = values.dropna()
    numeric = pd.to_numeric(present, errors="coerce")
    options = []
    if len(present) and numeric.notna().all():
        distribution = "、".join(f"{_plain(k)}：{int(v)}" for k, v in numeric.value_counts().sort_index().head(8).items())
        for threshold in _TARGET_THRESHOLDS:
            bad = int((numeric >= threshold).sum())
            if 0 < bad < len(numeric):
                options.append(_option(
                    "redefine_target",
                    tier=TIER_ONE_CLICK,
                    label=f"把 {target} ≥ {threshold} 记为坏样本",
                    command=f"把 {target} ≥ {threshold} 记为坏样本（1），其余记为好样本（0）",
                    consequence=(
                        f"新建一份数据集，坏样本按 {target} ≥ {threshold} 计（坏率 {bad / len(numeric):.1%}），"
                        "原样本文件不变；可以撤销。"
                    ),
                    params={"target_col": target, "rule": {"kind": "ge", "threshold": threshold}},
                    reversible=True,
                ))
            if len(options) >= 2:
                break
    else:
        uniques = [str(item) for item in present.astype(str).unique().tolist()]
        distribution = "、".join(f"{k}：{int(v)}" for k, v in present.astype(str).value_counts().head(8).items())
        if len(uniques) == 2:
            for bad_value in sorted(uniques):
                share = float((present.astype(str) == bad_value).mean())
                options.append(_option(
                    "redefine_target",
                    tier=TIER_ONE_CLICK,
                    label=f"把 '{bad_value}' 记为坏样本",
                    command=f"把 {target} 为 '{bad_value}' 记为坏样本（1），另一值记为好样本（0）",
                    consequence=f"新建一份数据集，坏率 {share:.1%}，原样本文件不变；可以撤销。",
                    params={"target_col": target, "rule": {"kind": "equals", "bad_value": bad_value}},
                    reversible=True,
                ))
    return _diagnostic(
        ctx,
        code="feature_target_not_binary",
        title="目标列不是 0/1",
        summary=f"目标 {target} 不是 0/1 二分类，指标需要先定义好坏样本。",
        cause="IV、KS 等指标按好（0）坏（1）计算；坏样本怎么定义是业务口径，平台不会猜。",
        location=f"{ctx.step.title} · 目标列",
        evidence=[
            {"label": "目标列", "value": target},
            {"label": "取值分布", "value": distribution or "没有读到"},
        ],
        impact="特征指标还没有算出来。",
        actions=["也可以在样本里加一列 0/1 目标后重新上传。"],
        options=options,
        agent_mode=agent_mode,
    )


# ---------- 9. IV floor removed every feature ----------


def _iv_threshold_diagnostic(ctx: FailureContext, agent_mode: bool) -> dict | None:
    output = ctx.plan_repo.load_step_output(ctx.step.id)
    scores = output.get("scores") if isinstance(output, dict) else None
    if not isinstance(scores, dict) or not scores:
        return None
    current = float((ctx.step.inputs or {}).get("iv_min", 0.02))
    ivs = {name: float(item.get("iv")) for name, item in scores.items()
           if isinstance(item, dict) and item.get("iv") is not None}
    top = sorted(ivs.items(), key=lambda item: -item[1])[:5]
    options = []
    for threshold in _IV_THRESHOLDS:
        if threshold >= current:
            continue
        passing = sum(1 for value in ivs.values() if value >= threshold)
        if passing:
            options.append(_option(
                "lower_iv_threshold",
                tier=TIER_ONE_CLICK,
                label=f"把 IV 底线降到 {threshold}",
                command=f"把精选特征的 IV 底线从 {current} 降到 {threshold}",
                consequence=(
                    f"按平台已算的训练集 IV，有 {passing} 个特征能通过；这些特征区分度偏弱。"
                    "改动后停在「精选特征」确认，确认后才重新精选；可以撤销。"
                ),
                params={"iv_min": threshold, "passing": passing, "previous": current},
                reversible=True,
            ))
            break
    return _diagnostic(
        ctx,
        code="modeling_iv_floor_empty",
        title="精选后没有特征",
        summary=f"IV 底线 {current} 把全部 {len(ivs)} 个特征都筛掉了。",
        cause="精选按训练集 IV 设底线；特征区分度都低于底线时没有特征可以进入调参。",
        location="精选特征 · IV 底线",
        evidence=[
            {"label": "当前底线", "value": str(current)},
            {"label": "IV 最高的特征", "value": "、".join(f"{name} {value:.4f}" for name, value in top)},
        ],
        impact="调参和训练都还没开始。",
        actions=["特征区分度普遍偏低时，也请检查目标定义和拼接结果是否正确。"],
        options=options,
        agent_mode=agent_mode,
    )


# ---------- budget ----------


def plan_auto_repair(settings, task, diagnostic: dict | None, messages) -> dict | None:
    if not isinstance(diagnostic, dict) or getattr(task, "run_mode", "") != "agent":
        return None
    option = next((item for item in diagnostic.get("repair_options") or []
                   if isinstance(item, dict) and item.get("tier") == TIER_AUTO), None)
    if option is None:
        return None
    used, limit = repair_budget(settings, messages, task)
    if used >= limit:
        return None
    seen = {report.get("fingerprint") for report in _repair_reports(messages) if _ran_automatically(report)}
    rejected = set(load_repair_state(settings.tasks_dir, task.id)["rejected_fingerprints"])
    if option.get("fingerprint") in seen or option.get("fingerprint") in rejected:
        return None
    return dict(option)


# ---------- apply / verify / rollback ----------


@dataclass
class RepairOutcome:
    report: dict
    turn: Any = None


def _report(*, ctx: FailureContext, option: dict, finding: str, action: str, verification: str,
            verified: bool, log: list[str], undo: bool, trigger: str, source_message_id: str | None,
            audit: dict | None = None) -> dict:
    return {
        "schema_version": WORKFLOW_REPAIR_SCHEMA,
        "repair_id": uuid.uuid4().hex,
        "workflow": str(getattr(ctx.task, "task_type", "") or ""),
        "plan_id": ctx.plan.id,
        "step_id": ctx.step.id,
        "option_id": option.get("id"),
        "tier": option.get("tier"),
        "trigger": trigger,
        "fingerprint": option.get("fingerprint"),
        "finding": finding,
        "action": action,
        "verification": verification,
        "verified": bool(verified),
        "undo_command": UNDO_COMMAND if (undo and verified and option.get("reversible")) else "",
        "log": log,
        "source_message_id": source_message_id,
        **({"audit": audit} if audit else {}),
    }


def _actor(trigger: str, runtime) -> dict:
    if trigger == "auto":
        return {"kind": "agent", "label": "Agent（自动处理）"}
    principal = getattr(runtime, "local_principal", None)
    label = str(getattr(principal, "display_name", "") or "本机用户")
    return {"kind": "user", "label": label, "principal_id": str(getattr(principal, "id", "") or "")}


def _downstream_ids(plan, step_id: str) -> list[str]:
    found: set[str] = set()
    changed = True
    while changed:
        changed = False
        for step in plan.steps:
            if step.id == step_id or step.id in found:
                continue
            if set(step.depends_on or []) & (found | {step_id}):
                found.add(step.id)
                changed = True
    return [step.id for step in plan.steps if step.id in found]


def _not_done(step) -> bool:
    return getattr(step.status, "value", step.status) not in {"done", "skipped"}


def apply_workflow_repair(runtime, driver, ctx: FailureContext, option: dict, *,
                          finding: str, trigger: str, source_message_id: str | None = None) -> RepairOutcome:
    """Execute one option, verify it, and roll back when the check fails."""

    option_id = str(option.get("id") or "")
    params = dict(option.get("params") or {})
    settings, plan_repo, plan, step = ctx.settings, ctx.plan_repo, ctx.plan, ctx.step
    actor = _actor(trigger, runtime)
    log: list[str] = []
    original_inputs = {step.id: dict(step.inputs or {})}
    downstream_updates: dict[str, dict] = {}
    updates: dict = {}
    audit_entry: dict | None = None
    environment_previous = None
    derived_dataset_id = None

    if option_id == "halve_tuning_budget":
        turn = driver.rollback_failed_plan_to_tuning_config(
            plan.id, step.id, default_n_trials=None,
            n_trials_by_recipe=dict(params.get("n_trials_by_recipe") or {}), run_seq=ctx.run_seq + 1,
        )
        fresh = plan_repo.load_plan(plan.id)
        config = next((item for item in fresh.steps if item.tool_ref.label() == "modeling.configure_tuning"), None)
        verified = bool(
            config is not None
            and getattr(fresh.status, "value", fresh.status) == "awaiting_confirm"
            and dict((config.inputs or {}).get("n_trials_by_recipe") or {}) == dict(params["n_trials_by_recipe"])
        )
        budgets = "、".join(f"{k} {v} 轮" for k, v in params["n_trials_by_recipe"].items())
        report = _report(
            ctx=ctx, option=option, finding=finding,
            action=(f"把调参预算改为 {budgets}，计划回到「配置调参」确认。调参本身要等你确认后才会运行。"
                    if verified else
                    f"试过把调参预算改为 {budgets}，计划已经按新预算回退到「配置调参」，这项改动没有还原。"),
            verification=("配置调参已按新预算停在确认。" if verified else
                          "配置调参没有停在预期的确认状态；计划没有还原，请看计划里各步骤的状态再决定怎么继续。"),
            verified=verified, log=[f"新预算：{budgets}"], undo=False, trigger=trigger,
            source_message_id=source_message_id,
        )
        _record(settings, ctx, report, option, actor, original_inputs, None, None, verified)
        return RepairOutcome(report, turn if verified else None)

    if option_id == "use_task_environment":
        from marvis.task_environment import load_task_environment, probe_python_candidates, save_task_environment

        # Spec §4.4: probe again at write time; only a still-valid candidate is saved.
        match = _PYTHON_REQUIRES_RE.search(ctx.error)
        wanted = str(params["python_executable"])
        current = next((
            item for item in probe_python_candidates(
                python_requires=match.group("req") if match else "",
                exclude=[match.group("path")] if match else [],
            )
            if _same_path(item["python_executable"], wanted)
        ), None)
        if current is None:
            raise ValueError(f"{params.get('label') or wanted} 现在已经不是可用的执行环境")
        environment_previous = load_task_environment(settings.tasks_dir, ctx.task.id)
        repair_id = uuid.uuid4().hex
        save_task_environment(
            settings.tasks_dir, ctx.task.id,
            python_executable=wanted, label=str(params.get("label") or ""),
            python_version=str(current.get("python_version") or params.get("python_version") or ""),
            repair_id=repair_id,
        )
        log.append(f"任务级执行环境：{params.get('label')}（Python {params.get('python_version')}）")
        audit_entry = _audit_entry(
            kind="environment_override", actor=actor,
            changed=f"本任务的建模和特征工具改用 {params.get('label')}（Python {params.get('python_version')}）",
            before={"python_executable": (environment_previous or {}).get("python_executable") or "全局设置"},
            after={"python_executable": params["python_executable"]},
            undo="点「撤销」删除这个任务级覆盖，之后的步骤回到全局执行环境；已跑完的步骤不重跑。",
        )
        try:
            turn = driver.retry_failed_step(plan.id, step.id, run_seq=ctx.run_seq + 1,
                                            preserve_target_confirmation=True)
            verified = _step_done(plan_repo, plan.id, step.id)
        except Exception as exc:
            _fail_after_write(settings, plan_repo, ctx, option, actor, original_inputs, log, exc,
                              finding=finding, trigger=trigger, source_message_id=source_message_id,
                              audit_entry=audit_entry, environment=(environment_previous,), repair_id=repair_id)
        restored = True
        if not verified:
            restored = _restore_environment(settings, ctx.task.id, environment_previous, log)
            restored = _restore(plan_repo, ctx, original_inputs, log) and restored
        report = _report(
            ctx=ctx, option=option, finding=finding,
            action=(f"这个任务的建模和特征工具改用 {params.get('label')} 运行，全局执行环境没有改。"
                    if verified else f"试过让这个任务改用 {params.get('label')}，"
                    + ("已还原。" if restored else "但没能还原，请看日志。")),
            verification=(f"「{step.title}」在新环境里跑通了。" if verified
                          else f"核对没通过：「{step.title}」在新环境里仍未完成。"),
            verified=verified, log=log, undo=True, trigger=trigger,
            source_message_id=source_message_id, audit=audit_entry,
        )
        report["repair_id"] = repair_id
        _record(settings, ctx, report, option, actor, original_inputs, environment_previous, None, verified,
                audit_entry=audit_entry)
        return RepairOutcome(report, turn if verified else None)

    properties = _tool_properties(step.tool_ref.plugin, step.tool_ref.tool)
    update_keys = {
        "ack_key_dtype": ["ack_dtype_mismatch"],
        "exclude_text_features": ["features"],
        "split_latest_oot": ["split_config"],
        "split_without_oot": ["split_config"],
        "lower_iv_threshold": ["iv_min"],
        "drop_nan_label_rows": ["drop_nan_labels"],
        "redefine_target": ["dataset_id"],
    }.get(option_id)
    if update_keys is None:
        raise ValueError(f"repair option {option_id!r} has no executor")
    unknown = [key for key in update_keys if key not in properties]
    if unknown:
        raise ValueError(f"tool input schema does not declare {unknown}")
    if option_id == "ack_key_dtype":
        updates = {"ack_dtype_mismatch": list(params.get("feature_ids") or [])}
        log.append("确认文本与数值编号为同一标识")
    elif option_id == "exclude_text_features":
        updates = {"features": list(params.get("features") or [])}
        log.append(f"本步骤不再计算：{'、'.join(params.get('exclude') or [])}")
    elif option_id in {"split_latest_oot", "split_without_oot"}:
        updates = {"split_config": dict(params.get("split_config") or {})}
        log.append(f"新的切分口径：{json.dumps(updates['split_config'], ensure_ascii=False)}")
    elif option_id == "lower_iv_threshold":
        updates = {"iv_min": float(params["iv_min"])}
        log.append(f"IV 底线 {params.get('previous')} → {params['iv_min']}（训练集 IV 预演通过 {params.get('passing')} 个）")
        audit_entry = _audit_entry(
            kind="change_threshold", actor=actor,
            changed=f"「{step.title}」的 IV 底线从 {params.get('previous')} 改为 {params['iv_min']}",
            before={"iv_min": params.get("previous")}, after={"iv_min": params["iv_min"]},
            undo="点「撤销」恢复原底线，计划回到这一步失败时的状态。",
        )
    elif option_id == "drop_nan_label_rows":
        updates = {"drop_nan_labels": True}
        for later in plan.steps:
            if later.id in _downstream_ids(plan, step.id) and _not_done(later) and (
                "drop_nan_labels" in _tool_properties(later.tool_ref.plugin, later.tool_ref.tool)
            ):
                downstream_updates[later.id] = {**(later.inputs or {}), "drop_nan_labels": True}
        log.append(f"剔除目标 {params.get('target_col')} 为空的 {params.get('rows')} 行（由工具的确定性核心执行）")
        audit_entry = _audit_entry(
            kind="drop_rows", actor=actor,
            changed=f"本次分析剔除目标 {params.get('target_col')} 为空的 {params.get('rows')} 行（共 {params.get('total')} 行），样本文件不变",
            before={"rows": params.get("total"), "drop_nan_labels": False},
            after={"rows": int(params.get("total") or 0) - int(params.get("rows") or 0), "drop_nan_labels": True},
            undo="点「撤销」恢复原设置，计划回到这一步失败时的状态。",
        )
    else:  # redefine_target
        from marvis.agent.workflow_repair_actions import derive_binary_target

        raw_dataset = (step.inputs or {}).get("dataset_id")
        dataset_id = _resolve_input(plan_repo, raw_dataset)
        target_col, rule = str(params["target_col"]), dict(params["rule"])
        derived = derive_binary_target(
            settings=settings, task_id=ctx.task.id, dataset_id=str(dataset_id),
            target_col=target_col, rule=rule, actor=actor,
        )
        log.extend(derived["log"])
        if not derived["verified"]:
            report = _report(
                ctx=ctx, option=option, finding=finding,
                action="试过派生一份 0/1 目标的数据集，核对没通过，没有使用。",
                verification=derived["log"][-1], verified=False, log=log, undo=False,
                trigger=trigger, source_message_id=source_message_id,
            )
            _record(settings, ctx, report, option, actor, original_inputs, None, None, False)
            return RepairOutcome(report, None)
        derived_dataset_id = derived["dataset_id"]
        updates = {"dataset_id": derived_dataset_id}
        for later in plan.steps:
            if later.id in _downstream_ids(plan, step.id) and _not_done(later) and (
                (later.inputs or {}).get("dataset_id") == raw_dataset
            ):
                downstream_updates[later.id] = {**(later.inputs or {}), "dataset_id": derived_dataset_id}
        audit_entry = _audit_entry(
            kind="redefine_target", actor=actor,
            changed=derived["changed"],
            before={"dataset_id": dataset_id, "target": "原始取值"},
            after={"dataset_id": derived_dataset_id, "rule": rule, "bad_rows": derived["bad_rows"]},
            undo="点「撤销」让计划改回原数据集，回到这一步失败时的状态；派生数据集保留作审计证据。",
        )

    try:
        for later_id in downstream_updates:
            original_inputs[later_id] = dict(next(item for item in plan.steps if item.id == later_id).inputs or {})
        for later_id, inputs in downstream_updates.items():
            later = next(item for item in plan.steps if item.id == later_id)
            later.inputs = inputs
            plan_repo.update_step(later)
        turn = driver.retry_failed_step(plan.id, step.id, run_seq=ctx.run_seq + 1, inputs=updates,
                                        preserve_target_confirmation=True)
        verified, verification = _verify(plan_repo, ctx, option_id, params, updates)
    except Exception as exc:
        _fail_after_write(settings, plan_repo, ctx, option, actor, original_inputs, log, exc,
                          finding=finding, trigger=trigger, source_message_id=source_message_id,
                          audit_entry=audit_entry, derived_dataset_id=derived_dataset_id)
    restored = True
    if not verified:
        restored = _restore(plan_repo, ctx, original_inputs, log)
    report = _report(
        ctx=ctx, option=option, finding=finding,
        action=_action_text(option_id, params, step.title, verified, restored),
        verification=verification, verified=verified, log=log, undo=True, trigger=trigger,
        source_message_id=source_message_id, audit=audit_entry,
    )
    _record(settings, ctx, report, option, actor, original_inputs, None, derived_dataset_id, verified,
            audit_entry=audit_entry, repaired_inputs={**(step.inputs or {}), **updates})
    return RepairOutcome(report, turn if verified else None)


class RepairRolledBackError(RuntimeError):
    """A repair raised after its first write; ``remaining`` lists what could not be put back."""

    def __init__(self, message: str, *, remaining: list[str], derived_dataset_id: str | None = None):
        super().__init__(message)
        self.remaining = list(remaining)
        self.derived_dataset_id = derived_dataset_id

    @property
    def restored(self) -> bool:
        return not self.remaining


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def _restore_environment(settings, task_id: str, previous: dict | None, log: list[str]) -> bool:
    """Put the task-level interpreter override back to ``previous`` (``None`` = no override)."""

    from marvis.task_environment import clear_task_environment, save_task_environment

    try:
        if previous:
            save_task_environment(
                settings.tasks_dir, task_id,
                python_executable=previous["python_executable"],
                label=previous.get("label", ""),
                python_version=previous.get("python_version", ""),
                repair_id=previous.get("repair_id", ""),
            )
        else:
            clear_task_environment(settings.tasks_dir, task_id)
    except Exception as exc:  # noqa: BLE001 - the report must say the restore did not happen
        log.append(f"任务级执行环境没有还原：{str(exc)[:200]}")
        return False
    log.append("已还原：任务级执行环境回到处理前。")
    return True


def _fail_after_write(settings, plan_repo, ctx: FailureContext, option: dict, actor: dict,
                      original_inputs: dict, log: list[str], exc: Exception, *, finding: str, trigger: str,
                      source_message_id: str | None, audit_entry: dict | None,
                      environment: tuple | None = None, repair_id: str | None = None,
                      derived_dataset_id: str | None = None):
    """Roll back after a raise that followed the first write, audit it, and raise."""

    remaining: list[str] = []
    if environment is not None and not _restore_environment(settings, ctx.task.id, environment[0], log):
        remaining.append("任务级执行环境覆盖还在")
    if not _restore(plan_repo, ctx, original_inputs, log):
        remaining.extend(_unrestored_steps(plan_repo, ctx, original_inputs))
    if derived_dataset_id:
        log.append(f"派生数据集 {derived_dataset_id} 保留作审计证据，计划没有使用它。")
    detail = "；".join(remaining) if remaining else "已全部还原"
    report = _report(
        ctx=ctx, option=option, finding=finding,
        action="试过这项处理，执行中出错。" + ("已还原。" if not remaining else f"没能完全还原：{detail}。"),
        verification=f"执行出错：{str(exc)[:200]}", verified=False, log=log, undo=False,
        trigger=trigger, source_message_id=source_message_id,
    )
    if repair_id:
        report["repair_id"] = repair_id
    entry = audit_entry or _audit_entry(
        kind=str(option.get("id") or "workflow_repair"), actor=actor,
        changed=f"试过「{option.get('label') or option.get('id')}」", before={}, after={}, undo="",
    )
    _record(settings, ctx, report, option, actor, original_inputs, None, derived_dataset_id, False,
            audit_entry={**entry, "error": str(exc)[:500], "remaining": remaining})
    raise RepairRolledBackError(str(exc)[:500], remaining=remaining,
                                derived_dataset_id=derived_dataset_id) from exc


def _unrestored_steps(plan_repo, ctx: FailureContext, original_inputs: dict) -> list[str]:
    """After a refused restore, write back what can be and name what is still changed."""

    try:
        fresh = plan_repo.load_plan(ctx.plan.id)
    except Exception:  # noqa: BLE001
        return ["读不到计划，没法确认步骤是否还原"]
    remaining: list[str] = []
    for step_id, inputs in original_inputs.items():
        current = next((item for item in fresh.steps if item.id == step_id), None)
        if current is None or dict(current.inputs or {}) == dict(inputs):
            continue
        if step_id != ctx.step.id and _not_done(current):
            try:
                current.inputs = dict(inputs)
                plan_repo.update_step(current)
                continue
            except Exception:  # noqa: BLE001
                pass
        remaining.append(f"「{current.title}」的输入还是改过的")
    failed = next((item for item in fresh.steps if item.id == ctx.step.id), None)
    if failed is not None and getattr(failed.status, "value", failed.status) != "failed":
        remaining.append(f"「{failed.title}」的状态是 {getattr(failed.status, 'value', failed.status)}，不是失败")
    return remaining


def _audit_entry(*, kind: str, actor: dict, changed: str, before: dict, after: dict, undo: str) -> dict:
    return {"action_id": uuid.uuid4().hex, "kind": kind, "actor": actor, "at": _now(),
            "changed": changed, "before": before, "after": after, "undo": undo, "undone_at": None}


def _step_done(plan_repo, plan_id: str, step_id: str) -> bool:
    step = next((item for item in plan_repo.load_plan(plan_id).steps if item.id == step_id), None)
    return step is not None and getattr(step.status, "value", step.status) == "done"


def _verify(plan_repo, ctx: FailureContext, option_id: str, params: dict, updates: dict) -> tuple[bool, str]:
    fresh = plan_repo.load_plan(ctx.plan.id)
    step = next(item for item in fresh.steps if item.id == ctx.step.id)
    status = getattr(step.status, "value", step.status)
    if option_id == "lower_iv_threshold":
        ok = (
            getattr(fresh.status, "value", fresh.status) == "awaiting_confirm"
            and status in {"awaiting_confirm", "pending"}
            and float((step.inputs or {}).get("iv_min", -1)) == float(params["iv_min"])
        )
        return ok, (
            f"按平台已算的训练集 IV，新底线 {params['iv_min']} 下有 {params.get('passing')} 个特征通过；"
            "计划停在「精选特征」确认，确认后按新底线重新精选。"
            if ok else "「精选特征」没有按新底线停在确认。"
        )
    if status != "done":
        return False, f"核对没通过：「{step.title}」仍未完成（{str(step.error or '')[:200]}）。"
    if option_id == "drop_nan_label_rows":
        output = plan_repo.load_step_output(step.id)
        dropped = int((output or {}).get("nan_labels_dropped") or 0)
        if dropped != int(params.get("rows") or -1):
            return False, f"核对没通过：工具剔除了 {dropped} 行，和查证的 {params.get('rows')} 行不一致。"
        return True, f"「{step.title}」完成，工具确实只剔除了 {dropped} 行空标签。"
    return True, f"「{step.title}」重新运行完成。"


def _restore(plan_repo, ctx: FailureContext, original_inputs: dict, log: list[str]) -> bool:
    try:
        plan_repo.restore_failed_step(
            ctx.plan.id, ctx.step.id,
            inputs=original_inputs[ctx.step.id], error=ctx.error,
            downstream_inputs={key: value for key, value in original_inputs.items() if key != ctx.step.id},
            reason="workflow_repair_rollback",
        )
    except Exception as exc:  # noqa: BLE001 - the report must say the restore did not happen
        log.append(f"还原没有完成：{str(exc)[:200]}")
        return False
    log.append("已还原：步骤输入和失败状态回到处理前。")
    return True


def _action_text(option_id: str, params: dict, title: str, verified: bool, restored: bool) -> str:
    done = {
        "ack_key_dtype": "按你的确认，把文本和数值编号视为同一标识重新确认拼接；样本和特征表都没有改动。",
        "exclude_text_features": f"这次不计算文本列 {'、'.join(params.get('exclude') or [])}，其余特征照常计算。",
        "split_latest_oot": "按新的 OOT 口径重新切分样本。",
        "split_without_oot": "不留 OOT，只切 train/test 重新切分样本。",
        "lower_iv_threshold": f"把「{title}」的 IV 底线改为 {params.get('iv_min')}。",
        "drop_nan_label_rows": f"剔除目标为空的 {params.get('rows')} 行后重新计算，样本文件不变。",
        "redefine_target": "派生一份 0/1 目标的数据集，计划改用它重新计算；原样本文件不变。",
    }.get(option_id, "执行了这项处理。")
    if verified:
        return done
    return ("试过：" + done + ("已还原。" if restored else "但没能还原，请看日志。"))


def _record(settings, ctx: FailureContext, report: dict, option: dict, actor: dict, original_inputs: dict,
            environment_previous, derived_dataset_id, verified: bool, *, audit_entry: dict | None = None,
            repaired_inputs: dict | None = None) -> None:
    from marvis.repositories.plans import PlanRepository

    state = load_repair_state(settings.tasks_dir, ctx.task.id)
    if audit_entry is not None:
        state["audit"].append({**audit_entry, "repair_id": report["repair_id"],
                               "outcome": "applied" if verified else "rolled_back"})
    if verified:
        state["applied"].append({
            "repair_id": report["repair_id"],
            "option_id": option.get("id"),
            "fingerprint": option.get("fingerprint"),
            "plan_id": ctx.plan.id,
            "step_id": ctx.step.id,
            "original_inputs": original_inputs,
            "repaired_inputs_sha256": _sha(repaired_inputs) if repaired_inputs is not None else "",
            "original_error": ctx.error,
            "environment_previous": environment_previous,
            "derived_dataset_id": derived_dataset_id,
            "at": _now(),
            "actor": actor,
        })
    save_repair_state(settings.tasks_dir, ctx.task.id, state)
    PlanRepository(settings.db_path).write_audit(
        kind="workflow.repair.applied" if verified else "workflow.repair.rolled_back",
        target_ref=ctx.step.id,
        actor=str(actor.get("label") or actor.get("kind")),
        outcome="succeeded" if verified else "failed",
        detail={"plan_id": ctx.plan.id, "repair_id": report["repair_id"], "option_id": option.get("id"),
                "trigger": report.get("trigger"), "data_action": (audit_entry or {}).get("kind")},
    )


# ---------- undo ----------


def _applied_record(settings, task_id: str, repair_id: str) -> dict | None:
    return next((item for item in load_repair_state(settings.tasks_dir, task_id)["applied"]
                 if item.get("repair_id") == repair_id), None)


def undo_blocker(settings, plan_repo, task_id: str, report: dict) -> str | None:
    """Why this report can no longer be undone, or ``None``."""

    if not report.get("undo_command") or not report.get("verified"):
        return "这项处理不能撤销。"
    record = _applied_record(settings, task_id, str(report.get("repair_id") or ""))
    if record is None:
        return "没有找到这次处理的记录。"
    if record.get("option_id") == "use_task_environment":
        from marvis.task_environment import load_task_environment

        current = load_task_environment(settings.tasks_dir, task_id) or {}
        return None if current.get("repair_id") == report.get("repair_id") else "任务级执行环境已经变了。"
    try:
        plan = plan_repo.load_plan(str(record["plan_id"]))
    except Exception:  # noqa: BLE001
        return "计划已经不在了。"
    if getattr(plan.status, "value", plan.status) not in {"awaiting_confirm", "failed", "confirmed"}:
        return "计划已经往后走了。"
    step = next((item for item in plan.steps if item.id == record["step_id"]), None)
    if step is None or (record.get("repaired_inputs_sha256") and _sha(dict(step.inputs or {})) != record["repaired_inputs_sha256"]):
        return "这一步的输入已经变了。"
    if any(plan_repo.is_step_confirmed(later) for later in _downstream_ids(plan, step.id)):
        return "后面的确认已经做过。"
    return None


def undo_workflow_repair(settings, plan_repo, task, report: dict) -> dict:
    record = _applied_record(settings, task.id, str(report.get("repair_id") or ""))
    if record is None:
        raise ValueError("repair record not found")
    if record.get("option_id") == "use_task_environment":
        previous = record.get("environment_previous")
        undo_log: list[str] = []
        if not _restore_environment(settings, task.id, previous, undo_log):
            raise RuntimeError(undo_log[-1])
        finding = (
            f"已撤销任务级执行环境：之后的建模和特征步骤回到之前的 {previous.get('label') or previous['python_executable']}，已跑完的步骤不重跑。"
            if previous else
            "已撤销任务级执行环境：之后的建模和特征步骤回到全局执行环境，已跑完的步骤不重跑。"
        )
    else:
        plan_repo.restore_failed_step(
            record["plan_id"], record["step_id"],
            inputs=record["original_inputs"][record["step_id"]], error=record["original_error"],
            downstream_inputs={k: v for k, v in record["original_inputs"].items() if k != record["step_id"]},
            reason="workflow_repair_undo",
        )
        finding = "已撤销这次处理：步骤输入恢复原样，计划回到这一步失败时的状态。"
    state = load_repair_state(settings.tasks_dir, task.id)
    state["rejected_fingerprints"] = list(dict.fromkeys(
        [*state["rejected_fingerprints"], str(record.get("fingerprint") or "")]))
    for entry in state["audit"]:
        if entry.get("repair_id") == record["repair_id"] and not entry.get("undone_at"):
            entry["undone_at"] = _now()
    save_repair_state(settings.tasks_dir, task.id, state)
    plan_repo.write_audit(
        kind="workflow.repair.undone", target_ref=str(record["step_id"]), actor="user",
        outcome="succeeded", detail={"plan_id": record["plan_id"], "repair_id": record["repair_id"]},
    )
    return {
        "schema_version": WORKFLOW_REPAIR_SCHEMA,
        "repair_id": record["repair_id"],
        "finding": finding,
        "log": [finding, "这项处理不会再自动执行，仍可以在失败卡里手动点。"],
    }


# ---------- commands ----------


def _compact(text: str) -> str:
    return "".join(str(text or "").split())


def _is_workflow_report(report) -> bool:
    return isinstance(report, dict) and report.get("schema_version") == WORKFLOW_REPAIR_SCHEMA


def latest_workflow_report(messages) -> dict | None:
    for message in reversed(messages):
        report = (message.get("metadata") or {}).get("repair_report")
        if _is_workflow_report(report):
            return {**report, "message_id": message.get("id")}
    return None


def latest_repair_card(messages) -> dict | None:
    """The newest workflow failure card not superseded by later progress."""

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        metadata = message.get("metadata") or {}
        if metadata.get("repair_report") or metadata.get("repair_undo") or metadata.get("intent") in {
            "workflow_repair", "workflow_recovery_chat",
        }:
            continue
        diagnostic = metadata.get("error_diagnostic")
        if isinstance(diagnostic, dict) and diagnostic.get("diagnosed_by") == DIAGNOSED_BY:
            return message
        return None
    return None


def match_workflow_repair_command(settings, plan_repo, task_id: str, messages, content: str):
    """``("apply", option, message)`` / ``("undo", report)`` / ``("stale", kind)`` / ``None``."""

    compact = _compact(content)
    if not compact:
        return None
    if compact == _compact(UNDO_COMMAND):
        report = latest_workflow_report(messages)
        if report is None:
            return None
        if str(report.get("repair_id")) in _undone_repair_ids(messages):
            return ("stale", "undo")
        return ("undo", report) if undo_blocker(settings, plan_repo, task_id, report) is None else ("stale", "undo")
    card = latest_repair_card(messages)
    if card is not None:
        for option in card["metadata"]["error_diagnostic"].get("repair_options") or []:
            if option.get("command") and _compact(option["command"]) == compact:
                return ("apply", dict(option), card)
    for message in messages:
        diagnostic = (message.get("metadata") or {}).get("error_diagnostic")
        if not isinstance(diagnostic, dict) or diagnostic.get("diagnosed_by") != DIAGNOSED_BY:
            continue
        if any(option.get("command") and _compact(option["command"]) == compact
               for option in diagnostic.get("repair_options") or []):
            return ("stale", None)
    return None


def annotate_workflow_repair_reports(settings, plan_repo, task_id: str, messages: list[dict]) -> None:
    """Mark which workflow report can still be undone (read-time only)."""

    latest = latest_workflow_report(messages)
    undone = _undone_repair_ids(messages)
    for message in messages:
        metadata = message.get("metadata") or {}
        report = metadata.get("repair_report")
        if not _is_workflow_report(report):
            continue
        available = False
        if latest is not None and report.get("repair_id") == latest.get("repair_id") and (
            str(report.get("repair_id")) not in undone
        ):
            try:
                available = undo_blocker(settings, plan_repo, task_id, report) is None
            except Exception:  # noqa: BLE001 - read-time annotation only
                available = False
        message["metadata"] = {**metadata, "repair_report": {**report, "undo_available": available}}


def repair_card_without(diagnostic: dict, option_id: str, note: dict | None) -> dict:
    options = [item for item in diagnostic.get("repair_options") or [] if item.get("id") != option_id]
    retry = [item for item in diagnostic.get("recovery_actions") or [] if not item.get("repair_option_id")]
    return {
        **diagnostic,
        "repair_options": options,
        "recovery_actions": [*_recovery_actions(options), *retry],
        "evidence": [*(diagnostic.get("evidence") or []), *([note] if note else [])],
    }


def budget_note(settings, messages, task, diagnostic: dict) -> dict:
    used, limit = repair_budget(settings, messages, task)
    if used >= limit and any(item.get("tier") == TIER_AUTO for item in diagnostic.get("repair_options") or []):
        return {**diagnostic, "evidence": [*diagnostic.get("evidence", []),
                                           {"label": "自动处理额度", "value": f"本任务已用完（{used}/{limit}），这次需要你点一下"}]}
    return diagnostic


# ---------- strategy sample design ----------


_FIELD_ROLE_RE = re.compile(
    r"field_bindings\.(?P<role>\w+) 使用了数据集中不存在的列「(?P<column>[^」]+)」"
)
_ROLE_LABELS = {
    "time_field": "时间字段",
    "month_field": "月份字段",
    "entity_field": "实体字段",
    "group_field": "分组字段",
    "weight_field": "权重字段",
    "loan_amount_field": "放款金额字段",
    "overdue_amount_field": "逾期金额字段",
}
_DATE_NAME_RE = re.compile(r"(?:date|time|month|dt|日期|时间|月份)", re.IGNORECASE)


def strategy_sample_design_diagnostic(settings, task, *, message: str, messages) -> tuple[dict, list[dict]] | None:
    """Card + corrected-utterance replies for a sample-design clarification."""

    match = _FIELD_ROLE_RE.search(str(message or ""))
    target_invalid = "必须是数值 0/1 或真实空值" in str(message or "")
    if match is None and not target_invalid:
        return None
    try:
        from marvis.repositories.data_workspace import DataWorkspaceRepository

        snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
        registry = _registry(settings)
        dataset_id = snapshot.active_dataset_id
        if dataset_id is None:
            candidates = [item for item in registry.list_for_task(task.id)
                          if item.role in {"sample", "strategy_sample"}]
            dataset_id = candidates[0].id if len(candidates) == 1 else None
        if dataset_id is None:
            return None
        dataset = registry.get(dataset_id)
        columns = [profile.name for profile in dataset.columns]
    except Exception:  # noqa: BLE001 - clarification stays plain text
        logger.exception("strategy sample design diagnosis failed task_id=%s", getattr(task, "id", ""))
        return None
    request = next((str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"), "")
    replies: list[dict] = []
    if match is not None:
        role, wrong = match.group("role"), match.group("column")
        label = _ROLE_LABELS.get(role, role)
        near = match_missing_fields([wrong], columns)[wrong]
        alternatives = [near] if near else []
        if not alternatives and role in {"time_field", "month_field"}:
            alternatives = [name for name in columns if _DATE_NAME_RE.search(name)][:2]
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(wrong)}(?![A-Za-z0-9_])")
        for alternative in alternatives:
            if request and pattern.search(request):
                replies.append({"label": f"{label}改用 {alternative}", "command": pattern.sub(alternative, request)})
        evidence = [
            {"label": label, "value": f"原话里写的是 {wrong}，样本里没有这一列"},
            {"label": "样本里相近的列", "value": "、".join(alternatives) or "没有找到"},
        ]
        diagnostic = {
            "schema_version": "workflow_error.v1",
            "diagnosed_by": DIAGNOSED_BY,
            "workflow": "strategy",
            "code": "strategy_sample_design_field_missing",
            "title": "样本设计里的字段在样本中找不到",
            "summary": f"{label}写的是 {wrong}，当前样本没有这一列。",
            "cause": "样本设计的每个口径都按你的原话逐字绑定到样本列；平台不会替你改列名。",
            "location": f"策略样本设计 · {label}",
            "evidence": evidence,
            "impact": "样本设计还没有固化，也没有创建策略。",
            "actions": ["点下面的按钮会把原话里的列名换掉后重新发送；也可以自己改好原话再发。"],
            "retryable": True,
            "recovery_actions": [],
        }
    else:
        target = str(getattr(snapshot.semantic_mapping, "target_col", "") or "")
        values = registry.read_authenticated_parquet_snapshot(dataset.id, columns=[target])[target] if target in columns else None
        distribution = (
            "、".join(f"{k}：{int(v)}" for k, v in values.astype(str).value_counts().head(6).items())
            if values is not None else "没有读到"
        )
        diagnostic = {
            "schema_version": "workflow_error.v1",
            "diagnosed_by": DIAGNOSED_BY,
            "workflow": "strategy",
            "code": "strategy_sample_design_target_invalid",
            "title": "目标列不是数值 0/1",
            "summary": f"目标 {target} 不是数值 0/1，样本设计需要数值好坏标签。",
            "cause": "策略样本按数值 0/1 区分好坏；文本或其他编码平台不会自行转换。",
            "location": "策略样本设计 · 目标列",
            "evidence": [{"label": "目标列", "value": target}, {"label": "取值分布", "value": distribution}],
            "impact": "样本设计还没有固化，也没有创建策略。",
            "actions": ["在数据处理里把目标列转成数值 0/1（坏为 1）后，在数据工作区换绑新数据集，再发起样本设计。"],
            "retryable": True,
            "recovery_actions": [],
        }
    return diagnostic, replies[:2]


__all__ = [
    "DIAGNOSED_BY",
    "FailureContext",
    "RETRY_COMMAND",
    "RepairRolledBackError",
    "STALE_COMMAND_REPLY",
    "UNDO_UNAVAILABLE_REPLY",
    "V2_WORKFLOWS",
    "WORKFLOW_REPAIR_SCHEMA",
    "annotate_workflow_repair_reports",
    "apply_workflow_repair",
    "budget_note",
    "diagnose_workflow_failure",
    "failure_context",
    "latest_repair_card",
    "latest_workflow_report",
    "load_repair_state",
    "match_workflow_repair_command",
    "plan_auto_repair",
    "repair_card_without",
    "strategy_sample_design_diagnostic",
    "undo_blocker",
    "undo_workflow_repair",
]
