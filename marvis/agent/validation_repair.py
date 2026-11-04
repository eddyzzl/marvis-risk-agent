"""Deterministic diagnose / repair / verify loop for V2 model validation.

B-line of the 2026-10-14 workbench redesign (spec:
docs/superpowers/specs/2026-10-16-validation-agent-self-repair-spec.md).

The platform classifies a validation failure, checks it with read-only probes
and offers repair options in three tiers:

- ``auto``: reversible and local (a field alias in the task's own contract);
  done without asking in Agent mode, reported afterwards with an undo.
- ``one_click``: carries business meaning (time column, material choice);
  executed only when the user clicks the exact offered command.
- ``confirm``: changes the machine (installing Java); never executed here,
  only explained.

Nothing in this module calls an LLM or touches a deterministic metric. Every
report is built from what was executed and verified ("做了才说做了").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import uuid

from marvis.files import write_json_atomic


logger = logging.getLogger(__name__)


REPAIR_SCHEMA = "validation_repair.v1"
OVERRIDES_SCHEMA = "validation_repair_overrides.v1"
UNDO_COMMAND = "撤销这次自动处理"
RESELECT_MATERIALS_COMMAND = "重新选择验证材料"
RERUN_PMML_COMMAND = "重新跑 PMML 打分"
MAP_FIELDS_COMMAND = "按近似列名补上字段映射并重新扫描"
TIER_AUTO = "auto"
TIER_ONE_CLICK = "one_click"
TIER_CONFIRM = "confirm"

_MISSING_INPUTS_PREFIX = "sample missing required PMML inputs:"
_MISSING_INPUTS_RE = re.compile(
    r"sample missing required PMML inputs:\s*(?P<fields>[^；;\n]+)"
)
_UNPARSEABLE_TIME_RE = re.compile(
    r"time column (?P<quote>['\"])(?P<column>.+?)(?P=quote) contains unparseable "
    r"date values:\s*(?P<values>[^\n]*)"
)
# A material the platform could not find or pick, versus one it found but
# could not read; the card words them differently.
_MATERIAL_MISSING_RE = re.compile(
    r"selected (?:Notebook|sample|PMML|feature metadata) "
    r"(?:path is missing|file does not exist|escapes source directory|is not a regular file)"
    r"|缺少必需材料"
    r"|个候选，请显式指定"
)
_MATERIAL_INVALID_RE = re.compile(
    r"unsupported validation sample format"
    r"|(?:^|[；;：:]\s*)(?:sample|PMML|Notebook|feature metadata):\s"
    # The scan's status message is humanized before it reaches the Agent
    # (validation_report_copy.humanize_scan_check_message); without this the
    # missing-importance failure never got its 重新选择验证材料 option.
    r"|数据字典缺少"
)
_MATERIAL_RE = re.compile(f"{_MATERIAL_MISSING_RE.pattern}|{_MATERIAL_INVALID_RE.pattern}")
_MISSING_JAVA_RE = re.compile(
    r"No such file or directory:\s*'?java'?"
    r"|Unable to locate a Java Runtime"
    r"|JAVA_HOME"
    r"|java(?:\.exe)?['\"]?:? (?:command )?not found"
    r"|UnsupportedClassVersionError",
    re.IGNORECASE,
)
_MISSING_PYPMML_RE = re.compile(r"No module named ['\"]pypmml['\"]")
_TIME_LIKE_COLUMN_RE = re.compile(
    r"(?:date|time|month|day|dt|日期|时间|月份|月)", re.IGNORECASE
)
# Probes run inside a request, so they are bounded: the failing column is read
# up to 50k rows (each distinct value parsed once); a candidate replacement is
# read up to 5k rows and dropped at its first bad value. Confirming a
# replacement re-checks its whole column (validate_confirmation_against_materials).
_PROFILE_ROW_LIMIT = 50_000
_ALTERNATIVE_ROW_LIMIT = 5_000
_PROFILE_CHUNK_ROWS = 10_000
_MAX_ALTERNATIVE_TIME_COLUMNS = 2
_MAX_TIME_COLUMN_PROBES = 6
_MAX_EVIDENCE_NAMES = 5
_MATERIAL_ROLE_LABELS = (
    ("notebook", "Notebook"),
    ("sample", "样本数据"),
    ("model_pmml", "PMML"),
    ("data_dictionary", "数据字典"),
)


class RepairStale(ValueError):
    """The state an option was planned against has changed since."""


# ---------- Overrides (tasks/<id>/repairs/overrides.json) ----------


def repair_overrides_path(tasks_dir: Path, task_id: str) -> Path:
    return Path(tasks_dir) / task_id / "repairs" / "overrides.json"


def load_repair_overrides(tasks_dir: Path, task_id: str) -> dict:
    path = repair_overrides_path(tasks_dir, task_id)
    empty = {
        "schema_version": OVERRIDES_SCHEMA,
        "field_aliases": {},
        "applied": [],
        "rejected_fingerprints": [],
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return empty
    if not isinstance(payload, dict) or payload.get("schema_version") != OVERRIDES_SCHEMA:
        return empty
    aliases = payload.get("field_aliases")
    return {
        "schema_version": OVERRIDES_SCHEMA,
        "field_aliases": {
            str(key): str(value)
            for key, value in (aliases.items() if isinstance(aliases, dict) else ())
            if str(key).strip() and str(value).strip()
        },
        "applied": [item for item in payload.get("applied") or [] if isinstance(item, dict)],
        "rejected_fingerprints": [
            str(item) for item in payload.get("rejected_fingerprints") or [] if str(item)
        ],
    }


def save_repair_overrides(tasks_dir: Path, task_id: str, payload: dict) -> None:
    path = repair_overrides_path(tasks_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload, ensure_ascii=False, indent=2)


# ---------- Read-only probes ----------


def _normalized_field_key(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name or "")).casefold()


def match_missing_fields(
    missing: Sequence[str],
    available: Sequence[str],
    *,
    reserved: Sequence[str] = (),
) -> dict[str, str | None]:
    """Map each missing PMML input to its unique near-identical sample column.

    Near-identical means equal after case folding and dropping whitespace,
    ``_`` and ``-``. A column that is already a model input under its own name
    (``reserved``) is never borrowed, and two candidates mean no match.
    """

    reserved_set = set(reserved)
    by_key: dict[str, list[str]] = {}
    for column in available:
        if column in reserved_set:
            continue
        by_key.setdefault(_normalized_field_key(column), []).append(column)
    result: dict[str, str | None] = {}
    claimed: dict[str, int] = {}
    for name in missing:
        candidates = by_key.get(_normalized_field_key(name), [])
        result[name] = candidates[0] if len(candidates) == 1 else None
        if result[name] is not None:
            claimed[result[name]] = claimed.get(result[name], 0) + 1
    # Two model inputs must not share one sample column.
    for name, column in list(result.items()):
        if column is not None and claimed[column] > 1:
            result[name] = None
    return result


def _contract_record(db_path: Path, task_id: str):
    from marvis.repositories.validation_contracts import ValidationContractRepository

    return ValidationContractRepository(db_path).get(task_id)


def _resolved_materials(task):
    from marvis.validation_materials import resolve_selected_validation_materials

    return resolve_selected_validation_materials(task)


def sample_columns_and_inputs(db_path: Path, task) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sample column names and PMML raw inputs, from the contract when present."""

    record = _contract_record(db_path, task.id)
    contract = record.contract if record is not None else None
    columns: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    if contract is not None and contract.sample_schema is not None:
        columns = tuple(contract.sample_schema.columns)
    if contract is not None and contract.pmml_manifest is not None:
        inputs = tuple(contract.pmml_manifest.raw_required_fields)
    if columns and inputs:
        return columns, inputs
    materials = _resolved_materials(task)
    if not columns:
        from marvis.validation.sample_schema import inspect_sample_schema

        columns = tuple(inspect_sample_schema(materials.sample).columns)
    if not inputs:
        from marvis.validation.pmml_manifest import parse_pmml_input_manifest

        inputs = tuple(parse_pmml_input_manifest(materials.pmml).raw_required_fields)
    return columns, inputs


def _read_columns(task, columns: Sequence[str], *, schema=None, limit: int):
    """One bounded read of several sample columns (up to ``limit + 1`` rows).

    Every probe shares this read: Excel samples are parsed whole before the
    first row is yielded, so a read per column would multiply that cost.
    """

    import pandas as pd

    from marvis.validation.sample_schema import iter_sample_projection

    materials = _resolved_materials(task)
    wanted = tuple(dict.fromkeys(columns))
    frames = []
    read = 0
    for frame in iter_sample_projection(
        materials.sample,
        columns=wanted,
        chunk_size=min(_PROFILE_CHUNK_ROWS, limit + 1),
        schema=schema,
    ):
        remaining = limit + 1 - read
        if remaining <= 0:
            break
        frame = frame.iloc[:remaining]
        frames.append(frame)
        read += len(frame)
    if not frames:
        return pd.DataFrame(columns=list(wanted))
    return pd.concat(frames, ignore_index=True)


def _value_parses(value) -> bool:
    from marvis.validation.time_periods import (
        _normalize_time_value,
        _parse_normalized_time_value,
    )

    try:
        parsed = _parse_normalized_time_value(_normalize_time_value(value))
    except (TypeError, ValueError, OverflowError):
        return False
    return parsed is not None and str(parsed) != "NaT"


def _profile_values(values, *, column: str, limit: int, stop_at_first_bad: bool = False) -> dict:
    """Count values the time parser rejects, parsing each distinct value once."""

    import pandas as pd

    truncated = len(values) > limit
    values = values.iloc[:limit]
    bad_rows = 0
    examples: list[str] = []
    for value, count in values.value_counts(dropna=False).items():
        if not _value_parses(value):
            bad_rows += int(count)
            example = "空值" if pd.isna(value) else repr(value)[:40]
            if example not in examples and len(examples) < _MAX_EVIDENCE_NAMES:
                examples.append(example)
            if stop_at_first_bad:
                break
    return {
        "column": column,
        "rows": len(values),
        "bad_rows": bad_rows,
        "examples": examples,
        "truncated": truncated,
    }


def profile_time_column(task, column: str, *, schema=None, limit: int = _PROFILE_ROW_LIMIT) -> dict:
    frame = _read_columns(task, (column,), schema=schema, limit=limit)
    return _profile_values(frame[column], column=column, limit=limit)


def _could_hold_dates(column: str, schema) -> bool:
    """Time-like name, and not an ordinary numeric feature (``max_overdue_days``)."""

    if not _TIME_LIKE_COLUMN_RE.search(str(column)):
        return False
    dtype = str((getattr(schema, "dtypes", None) or {}).get(column, "")).lower()
    if any(token in dtype for token in ("float", "int")):
        # Numbers only count as dates when shaped like 202601 / 20260131.
        return bool(re.search(
            r"(?:^|_)(?:month|date|dt|ym|yyyymm|yyyymmdd)(?:_|$)|月份|日期",
            str(column),
            re.IGNORECASE,
        ))
    return True


def _candidate_time_columns(columns: Sequence[str], *, exclude: str, schema) -> list[str]:
    return [
        column for column in columns
        if column != exclude and _could_hold_dates(column, schema)
    ][:_MAX_TIME_COLUMN_PROBES]


def _alternatives_from_frame(frame, candidates: Sequence[str]) -> list[str]:
    found: list[str] = []
    head = frame.iloc[:_ALTERNATIVE_ROW_LIMIT]
    for column in candidates:
        if len(found) >= _MAX_ALTERNATIVE_TIME_COLUMNS:
            break
        profile = _profile_values(
            head[column], column=column, limit=_ALTERNATIVE_ROW_LIMIT, stop_at_first_bad=True
        )
        if profile["rows"] and not profile["bad_rows"]:
            found.append(column)
    return found


def alternative_time_columns(task, columns: Sequence[str], *, exclude: str, schema=None) -> list[str]:
    """Other date columns whose first 5k values all parse (checked again on confirm)."""

    candidates = _candidate_time_columns(columns, exclude=exclude, schema=schema)
    if not candidates:
        return []
    frame = _read_columns(task, candidates, schema=schema, limit=_ALTERNATIVE_ROW_LIMIT)
    return _alternatives_from_frame(frame, candidates)


def date_parse_probes(task, column: str, columns: Sequence[str], *, schema=None) -> tuple[dict, list[str]]:
    """Profile the failing time column and find replacements from one read."""

    candidates = _candidate_time_columns(columns, exclude=column, schema=schema)
    frame = _read_columns(task, (column, *candidates), schema=schema, limit=_PROFILE_ROW_LIMIT)
    profile = _profile_values(frame[column], column=column, limit=_PROFILE_ROW_LIMIT)
    return profile, _alternatives_from_frame(frame, candidates)


def material_inventory(source_dir: str) -> dict[str, list[str]]:
    """File names the scanner sees per material role (read-only listing)."""

    from marvis.files import scan_source_dir

    try:
        # Only names are needed here, so nothing is hashed.
        artifacts = scan_source_dir(Path(source_dir).expanduser().resolve(), hash_limit_bytes=0)
    except (OSError, ValueError):
        return {}
    grouped: dict[str, list[str]] = {}
    for artifact in artifacts:
        grouped.setdefault(str(artifact.role.value), []).append(artifact.path.name)
    return grouped


def probe_java_runtime() -> dict:
    executable = shutil.which("java")
    if not executable:
        return {"found": False, "version": ""}
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"found": True, "version": ""}
    lines = (completed.stderr or completed.stdout or "").strip().splitlines()
    return {"found": True, "version": lines[0][:120] if lines else ""}


# ---------- Options ----------


def _fingerprint(option_id: str, params: Mapping[str, Any]) -> str:
    payload = json.dumps({"id": option_id, "params": params}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _option(option_id: str, *, tier: str, label: str, command: str, consequence: str,
            params: Mapping[str, Any] | None = None, reversible: bool = False) -> dict:
    clean = dict(params or {})
    return {
        "id": option_id,
        "tier": tier,
        "label": label,
        "command": command,
        "consequence": consequence,
        "params": clean,
        "reversible": reversible,
        "fingerprint": _fingerprint(option_id, clean),
    }


def _recovery_actions(options: Sequence[dict]) -> list[dict]:
    return [
        {
            "label": option["label"],
            "command": option["command"],
            "consequence": option["consequence"],
            "tier": option["tier"],
            "repair_option_id": option["id"],
        }
        for option in options
        if option.get("command")
    ]


def _reselect_option() -> dict:
    return _option(
        "reselect_materials",
        tier=TIER_ONE_CLICK,
        label="重新选择验证材料",
        command=RESELECT_MATERIALS_COMMAND,
        consequence="会重新扫描新材料，之前的识别结果作废。",
    )


def _base_diagnostic(*, stage: str, code: str, title: str, summary: str, cause: str,
                     location: str, evidence: list[dict], impact: str, actions: list[str],
                     options: list[dict], error: str, task) -> dict:
    detail = str(error or "")
    source_dir = str(getattr(task, "source_dir", "") or "")
    if source_dir:
        detail = detail.replace(source_dir, "<材料目录>")
    return {
        "schema_version": "workflow_error.v1",
        "workflow": "validation",
        "stage": stage,
        "code": code,
        "phase": "validation_" + stage,
        "title": title,
        "summary": summary,
        "cause": cause,
        "location": location,
        "evidence": evidence,
        "impact": impact,
        "actions": actions,
        "retryable": True,
        "error_kind": code,
        "repair_options": options,
        "recovery_actions": _recovery_actions(options),
        "technical_detail": detail[:2000],
    }


# ---------- Diagnosis ----------


def diagnose_validation_failure(
    *,
    settings,
    task,
    stage: str,
    error: str,
    evidence: Mapping[str, Any] | None = None,
    requested_confirmation: Mapping[str, Any] | None = None,
    kinds: frozenset[str] | None = None,
) -> dict | None:
    """Classify one validation failure; ``None`` keeps the existing LLM text.

    ``kinds`` limits which classes may be produced (``field`` / ``date`` /
    ``dependency`` / ``material``); classification happens before any probe,
    so a caller that only wants cheap classes never pays for sample reads.
    """

    if getattr(task, "validation_workflow_version", 0) != 2:
        return None
    text = str(error or "")

    def allowed(kind: str) -> bool:
        return kinds is None or kind in kinds

    try:
        if allowed("field") and (match := _MISSING_INPUTS_RE.search(text)):
            fields = [item.strip() for item in match.group("fields").split(",") if item.strip()]
            return _field_mismatch_diagnostic(settings, task, missing=fields, error=text)
        if allowed("date") and (match := _UNPARSEABLE_TIME_RE.search(text)):
            return _date_parse_diagnostic(
                settings,
                task,
                column=match.group("column"),
                requested=requested_confirmation,
                error=text,
            )
        if allowed("dependency") and (_MISSING_JAVA_RE.search(text) or _MISSING_PYPMML_RE.search(text)):
            return _dependency_diagnostic(task, error=text)
        # A job exception (``kinds`` set) only counts as a material problem
        # when a material is missing or ambiguous; "feature metadata: ..." can
        # also be a contract message there.
        material_re = _MATERIAL_MISSING_RE if kinds is not None else _MATERIAL_RE
        if allowed("material") and (stage == "scan" or kinds is not None) and material_re.search(text):
            return _material_diagnostic(task, error=text, evidence=evidence)
    except Exception:  # noqa: BLE001 - a probe must never replace the failure itself
        # A probe that cannot read the materials must not hide the failure;
        # the caller falls back to the existing explanation.
        logger.exception("validation repair diagnosis failed task_id=%s", getattr(task, "id", ""))
        return None
    return None


def _contract_conflicts(db_path: Path, task_id: str) -> tuple[str, ...]:
    record = _contract_record(db_path, task_id)
    if record is None:
        return ()
    return tuple(record.contract.conflicts)


def _control_field_candidates(db_path: Path, task_id: str) -> list[str]:
    record = _contract_record(db_path, task_id)
    if record is None:
        return []
    names = []
    for key in ("target_col", "split_col", "time_col"):
        for candidate in record.contract.candidates.get(key, ()):
            if isinstance(candidate.value, str):
                names.append(candidate.value)
    return names


def _field_mismatch_diagnostic(settings, task, *, missing: list[str], error: str) -> dict:
    columns, inputs = sample_columns_and_inputs(settings.db_path, task)
    # Model inputs present under their own name, and the label / split /
    # time candidates, are never borrowed: mapping the label onto a model
    # input would leak it into the scores.
    reserved = [name for name in inputs if name in set(columns)]
    reserved.extend(_control_field_candidates(settings.db_path, task.id))
    matches = match_missing_fields(missing, columns, reserved=reserved)
    matched = {name: column for name, column in matches.items() if column}
    unmatched = [name for name, column in matches.items() if not column]
    conflicts = _contract_conflicts(settings.db_path, task.id)
    other_conflicts = [item for item in conflicts if not item.startswith(_MISSING_INPUTS_PREFIX)]
    evidence = [
        {"label": "模型需要但样本没有", "value": "、".join(missing[:20])},
        {"label": "样本列数", "value": str(len(columns))},
    ]
    if matched:
        evidence.append({
            "label": "样本里的近似列",
            "value": "、".join(f"{column} → {name}" for name, column in matched.items()),
        })
    if unmatched:
        evidence.append({"label": "找不到近似列", "value": "、".join(unmatched[:20])})
    if other_conflicts:
        evidence.append({"label": "其他冲突", "value": "；".join(other_conflicts)[:400]})
    options: list[dict] = []
    if matched and not unmatched and not other_conflicts:
        options.append(_option(
            "map_fields",
            tier=TIER_AUTO,
            label="按近似列名补上字段映射",
            command=MAP_FIELDS_COMMAND,
            consequence="只在验证合同里把样本列映射到模型输入名，样本文件不变；可以撤销。",
            params={"field_aliases": dict(sorted(matched.items()))},
            reversible=True,
        ))
    options.append(_reselect_option())
    actions = []
    if matched:
        actions.append("如果近似列其实不是同一个字段，请在样本里改成模型要求的列名后重新上传。")
    if unmatched:
        actions.append("在样本里补上缺少的列，或确认上传的 PMML 是否是这份样本对应的模型。")
    return _base_diagnostic(
        stage="scan",
        code="validation_field_mismatch",
        title="样本缺少模型需要的字段",
        summary=f"PMML 需要的输入 {'、'.join(missing[:20])} 在样本里找不到同名列。",
        cause=(
            "模型按列名取值，样本列名和 PMML 输入名不完全一致时，打分无法开始。"
            if matched
            else "样本里没有这些模型输入，可能样本和 PMML 不是同一个模型的材料。"
        ),
        location="样本数据 · 列名",
        evidence=evidence,
        impact="验证合同没有生成，PMML 打分和后续指标都还没开始。",
        actions=actions,
        options=options,
        error=error,
        task=task,
    )


def _date_parse_diagnostic(settings, task, *, column: str, requested, error: str) -> dict:
    record = _contract_record(settings.db_path, task.id)
    schema = record.contract.sample_schema if record is not None else None
    columns, _inputs = sample_columns_and_inputs(settings.db_path, task)
    profile, alternatives = date_parse_probes(task, column, columns, schema=schema)
    rows_label = f"{profile['bad_rows']} / {profile['rows']} 行"
    if profile["truncated"]:
        rows_label += f"（只查了前 {_PROFILE_ROW_LIMIT:,} 行）"
    evidence = [
        {"label": "时间列", "value": column},
        {"label": "无法解析", "value": rows_label},
    ]
    if profile["examples"]:
        evidence.append({"label": "样例", "value": "、".join(profile["examples"])})
    evidence.append({
        "label": "能完整解析的其他日期列",
        "value": "、".join(alternatives) if alternatives else "没有找到",
    })
    options: list[dict] = []
    if isinstance(requested, Mapping) and record is not None:
        for alternative in alternatives:
            options.append(_option(
                "switch_time_column",
                tier=TIER_ONE_CLICK,
                label=f"改用 {alternative} 作为时间列",
                command=f"改用 {alternative} 作为时间列并确认",
                consequence=f"月份划分、OOT 月份和月度稳定性都按 {alternative} 计算。",
                params={
                    "time_col": alternative,
                    "revision": record.revision,
                    "requested": dict(requested),
                },
            ))
    return _base_diagnostic(
        stage="input_confirmation",
        code="validation_time_parse_failed",
        title="时间列有无法解析的日期",
        summary=(
            f"时间列 {column} 里有 {profile['bad_rows']} 行无法解析为日期"
            + (f"（只查了前 {_PROFILE_ROW_LIMIT:,} 行）" if profile["truncated"] else "")
            + "，字段确认没有通过。"
        ),
        cause="平台按时间列划分月份和 OOT；有值解析不了时不会跳过这些行，以免悄悄改变样本。",
        location=f"样本数据 · {column}",
        evidence=evidence,
        impact="验证字段未确认，PMML 打分和后续指标都还没开始。",
        actions=[
            f"把 {column} 里无法解析的值改成日期（如 2026-01-31 或 202601）后重新上传样本。",
        ],
        options=options,
        error=error,
        task=task,
    )


def _dependency_diagnostic(task, *, error: str) -> dict:
    java = probe_java_runtime()
    pypmml_missing = bool(_MISSING_PYPMML_RE.search(error))
    evidence = [{
        "label": "Java",
        "value": java["version"] or ("已找到" if java["found"] else "没有找到 java 命令"),
    }]
    if pypmml_missing:
        evidence.append({"label": "pypmml", "value": "当前 Python 环境没有安装"})
        evidence.append({"label": "Python", "value": sys.executable})
    actions = []
    if pypmml_missing:
        actions.append(
            f"在终端执行 `{sys.executable} -m pip install pypmml` 安装打分依赖；"
            "会改动本机 Python 环境，平台不会替你安装。"
        )
    if not java["found"] or not pypmml_missing:
        actions.append(
            "安装 Java 11 或更高版本，并确认终端里 `java -version` 能看到版本号；"
            "会改动本机环境，平台不会替你安装。"
        )
    actions.append("装好后点「重新跑 PMML 打分」。")
    options = [
        _option(
            "install_dependency",
            tier=TIER_CONFIRM,
            label="安装打分依赖（需要你在本机操作）",
            command="",
            consequence="会改动本机环境，影响所有使用 PMML 打分的任务。",
        ),
        _option(
            "rerun_pmml_scoring",
            tier=TIER_ONE_CLICK,
            label="重新跑 PMML 打分",
            command=RERUN_PMML_COMMAND,
            consequence="用当前材料重新打分，字段确认结果保留。",
        ),
    ]
    return _base_diagnostic(
        stage="reproducibility",
        code="validation_dependency_missing",
        title="PMML 打分缺少运行依赖",
        summary="PMML 打分需要 pypmml 和 Java，当前环境缺少其中一项。",
        cause="PMML 由 pypmml 调用 Java 打分；依赖缺失时打分无法启动，与材料内容无关。",
        location="PMML 打分 · 运行环境",
        evidence=evidence,
        impact="PMML 打分未完成，后续指标都还没开始；字段确认结果已保留。",
        actions=actions,
        options=options,
        error=error,
        task=task,
    )


def _inventory_from_scan(scan: Mapping[str, Any] | None) -> dict[str, list[str]] | None:
    artifacts = scan.get("artifacts") if isinstance(scan, Mapping) else None
    if not isinstance(artifacts, list):
        return None
    grouped: dict[str, list[str]] = {}
    for artifact in artifacts:
        if isinstance(artifact, Mapping) and artifact.get("path"):
            grouped.setdefault(str(artifact.get("role") or ""), []).append(Path(str(artifact["path"])).name)
    return grouped


def _dictionary_metadata_problem(task) -> tuple[str, str] | None:
    """(file, Chinese reason) when no data-dictionary candidate is usable."""
    from marvis.api_scan_helpers import material_dictionary_diagnostics
    from marvis.validation_report_copy import METADATA_REASON_LABELS

    diagnostics = material_dictionary_diagnostics(task)
    rows = (diagnostics or {}).get("data_dictionary") or []
    if not rows or any(row.get("status") == "compatible" for row in rows):
        return None
    unusable = [row for row in rows if row.get("status") == "incompatible" and row.get("reasons")]
    if not unusable:
        return None
    row = next((item for item in unusable if item.get("selected")), unusable[0])
    reasons = row["reasons"]
    reason = "；".join(METADATA_REASON_LABELS[code] for code in reasons)
    if any(code.startswith("missing_") for code in reasons):
        reason += "，请补充后重新上传（平台不会自动填 importance）"
    return row.get("name") or "数据字典", reason


def _material_diagnostic(task, *, error: str, evidence: Mapping[str, Any] | None) -> dict:
    scan = evidence.get("scan") if isinstance(evidence, Mapping) else None
    # The scan that just failed already listed the directory; reuse it.
    inventory = _inventory_from_scan(scan)
    if inventory is None:
        inventory = material_inventory(getattr(task, "source_dir", ""))
    missing = bool(_MATERIAL_MISSING_RE.search(error))
    facts = []
    for role, label in _MATERIAL_ROLE_LABELS:
        names = inventory.get(role, [])
        value = "、".join(names[:_MAX_EVIDENCE_NAMES]) if names else "没有找到"
        if len(names) > _MAX_EVIDENCE_NAMES:
            value += f" 等 {len(names)} 个"
        facts.append({"label": f"目录里的{label}", "value": value})
    from marvis.validation_report_copy import humanize_scan_check_message

    check_facts = []
    if isinstance(scan, Mapping):
        for check in scan.get("checks") or []:
            if isinstance(check, Mapping) and check.get("status") == "error":
                check_facts.append({
                    "label": str(check.get("label") or "扫描检查"),
                    "value": humanize_scan_check_message(str(check.get("message") or ""))[:300],
                })
    metadata_problem = _dictionary_metadata_problem(task)
    if metadata_problem is not None:
        # The selection is incomplete because no dictionary is usable; say why
        # instead of guessing that files were moved or renamed. The scan's own
        # contract check says the same thing in platform English, so the
        # dictionary line replaces it.
        from marvis.validation_report_copy import dictionary_requirement_text

        name, reason = metadata_problem
        facts.append({"label": f"数据字典 {name}", "value": reason})
        return _base_diagnostic(
            stage="scan",
            code="validation_metadata_incomplete",
            title="数据字典不能用",
            summary=f"{name}：{reason}",
            cause=(
                "平台要求数据字典同时有 feature、category、importance 三类列，并覆盖所选 PMML 的入模特征；"
                "目录里没有满足要求的数据字典，所以材料选择没有完成。"
            ),
            location="材料选择 · 数据字典",
            evidence=facts,
            impact="材料扫描没有通过，后续验证步骤都还没开始。",
            actions=[
                "按上面的原因补齐数据字典，点「重新选择验证材料」，在弹窗里用「上传补充文件」传入并选中它。",
                dictionary_requirement_text(),
            ],
            options=[_reselect_option()],
            error=error,
            task=task,
        )
    facts.extend(check_facts)
    if missing:
        title = "验证材料没有找全"
        summary = "所选的 Notebook、样本、PMML 和数据字典里，有材料找不到或同类有多个候选。"
        cause = "文件可能被移动或改名，或者同一类材料有多个文件而没有指定用哪一个。"
        actions = ["每类材料各选一个文件；文件已移动的，重新上传或重新选择。"]
    else:
        title = "验证材料读不了"
        summary = "材料都找到了，但有文件的内容或格式平台读不了。"
        cause = "文件格式与扩展名不一致，或者内容结构不符合要求（见下方扫描检查）。"
        actions = ["按扫描检查里的提示修正对应文件，或换成正确的文件后重新选择。"]
    return _base_diagnostic(
        stage="scan",
        code="validation_material_unrecognized" if missing else "validation_material_unreadable",
        title=title,
        summary=summary,
        cause=cause,
        location="材料选择",
        evidence=facts,
        impact="材料扫描没有通过，后续验证步骤都还没开始。",
        actions=actions,
        options=[_reselect_option()],
        error=error,
        task=task,
    )


# ---------- Budget ----------


def _repair_reports(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    reports = []
    for message in messages:
        metadata = message.get("metadata") if isinstance(message, Mapping) else None
        report = metadata.get("repair_report") if isinstance(metadata, Mapping) else None
        if isinstance(report, dict):
            reports.append({**report, "message_id": message.get("id")})
    return reports


def _undone_repair_ids(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    undone = set()
    for message in messages:
        metadata = message.get("metadata") if isinstance(message, Mapping) else None
        undo = metadata.get("repair_undo") if isinstance(metadata, Mapping) else None
        if isinstance(undo, Mapping) and undo.get("repair_id"):
            undone.add(str(undo["repair_id"]))
    return undone


def _ran_automatically(report: Mapping[str, Any]) -> bool:
    # Reports written before ``trigger`` existed were all automatic.
    trigger = report.get("trigger")
    return trigger == "auto" if trigger else report.get("tier") == TIER_AUTO


def repair_budget(settings, messages: Sequence[Mapping[str, Any]], task=None) -> tuple[int, int]:
    """Automatic repairs used on this task and the capability tier's replan limit.

    A repair the user clicked does not use the automatic budget.
    """

    from marvis.orchestrator.capability import resolve_tier, tier_from_settings

    explicit = str(getattr(task, "capability_tier", "") or "")
    tier = resolve_tier(explicit) if explicit else tier_from_settings(settings)
    used = sum(1 for report in _repair_reports(messages) if _ran_automatically(report))
    return used, int(tier.max_replan_iterations)


def plan_auto_repair(settings, task, diagnostic: Mapping[str, Any] | None,
                     messages: Sequence[Mapping[str, Any]]) -> dict | None:
    """The auto option to run now, or ``None`` (not Agent mode, budget, seen)."""

    if not isinstance(diagnostic, Mapping) or getattr(task, "run_mode", "") != "agent":
        return None
    option = next(
        (item for item in diagnostic.get("repair_options") or []
         if isinstance(item, Mapping) and item.get("tier") == TIER_AUTO),
        None,
    )
    # Batch children repair in their own conversation like a single model;
    # the batch workbench shows the selected child's timeline and sends its
    # commands to that child (2026-10-23 extension spec §5.4).
    if option is None:
        return None
    used, limit = repair_budget(settings, messages, task)
    if used >= limit:
        return None
    seen = {report.get("fingerprint") for report in _repair_reports(messages)
            if _ran_automatically(report)}
    rejected = set(load_repair_overrides(settings.tasks_dir, task.id)["rejected_fingerprints"])
    if option.get("fingerprint") in seen or option.get("fingerprint") in rejected:
        return None
    return dict(option)


def budget_exhausted_note(settings, messages, task=None) -> dict | None:
    used, limit = repair_budget(settings, messages, task)
    if used < limit:
        return None
    return {"label": "自动处理额度", "value": f"本任务已用完（{used}/{limit}），这次需要你点一下"}


# ---------- Apply / verify / undo ----------


def _report(*, option: Mapping[str, Any], finding: str, action: str, verification: str,
            verified: bool, log: list[str], undo: bool, source_message_id: str | None,
            trigger: str = "user") -> dict:
    return {
        "schema_version": REPAIR_SCHEMA,
        "repair_id": uuid.uuid4().hex,
        "option_id": option.get("id"),
        "tier": option.get("tier"),
        "trigger": trigger,
        "fingerprint": option.get("fingerprint"),
        "finding": finding,
        "action": action,
        "verification": verification,
        "verified": bool(verified),
        "undo_command": UNDO_COMMAND if (undo and verified) else "",
        "log": log,
        "source_message_id": source_message_id,
    }


def _dry_run_contract(settings, task, field_aliases: Mapping[str, str]):
    from marvis.api_scan_helpers import build_validation_input_contract

    materials = _resolved_materials(task)
    return build_validation_input_contract(
        notebook_path=materials.notebook,
        sample_path=materials.sample,
        pmml_path=materials.pmml,
        dictionary_path=materials.dictionary,
        field_aliases=field_aliases,
    )


def apply_repair_option(settings, task, option: Mapping[str, Any], *,
                        finding: str = "", source_message_id: str | None = None,
                        trigger: str = "user") -> dict:
    """Execute one option and verify it; a failed check rolls the change back.

    ``trigger`` is ``"auto"`` when the Agent ran it within budget, ``"user"``
    when the user clicked it.
    """

    option_id = option.get("id")
    if option_id == "map_fields":
        report = _apply_field_aliases(settings, task, option, finding=finding,
                                      source_message_id=source_message_id)
    elif option_id == "switch_time_column":
        report = _apply_time_column(settings, task, option, finding=finding,
                                    source_message_id=source_message_id)
    else:
        raise ValueError(f"repair option {option_id!r} has no executor")
    return {**report, "trigger": trigger}


def _material_hashes(task) -> dict[str, str]:
    from marvis.files import sha256_file

    materials = _resolved_materials(task)
    return {"sample": sha256_file(materials.sample), "pmml": sha256_file(materials.pmml)}


def active_field_aliases(tasks_dir: Path, task_id: str, *, sample_sha256: str | None,
                         pmml_sha256: str | None) -> dict[str, str]:
    """Aliases from repairs made for exactly these sample and PMML files.

    After the user reselects or replaces a material, older repairs stop
    applying instead of silently carrying over to different files.
    """

    aliases: dict[str, str] = {}
    for record in load_repair_overrides(tasks_dir, task_id)["applied"]:
        hashes = record.get("material_hashes") or {}
        if hashes.get("sample") == sample_sha256 and hashes.get("pmml") == pmml_sha256:
            aliases.update({str(k): str(v) for k, v in (record.get("field_aliases") or {}).items()})
    return aliases


def _apply_field_aliases(settings, task, option, *, finding, source_message_id) -> dict:
    aliases = {
        str(key): str(value)
        for key, value in (option.get("params") or {}).get("field_aliases", {}).items()
    }
    if not aliases:
        raise ValueError("map_fields option has no field aliases")
    before = load_repair_overrides(settings.tasks_dir, task.id)
    hashes = _material_hashes(task)
    merged = {
        **active_field_aliases(settings.tasks_dir, task.id,
                               sample_sha256=hashes["sample"], pmml_sha256=hashes["pmml"]),
        **aliases,
    }
    mapping_text = "、".join(f"{column} → {name}" for name, column in aliases.items())
    log = [f"近似匹配：{mapping_text}"]
    contract = _dry_run_contract(settings, task, merged)
    remaining = [item for item in contract.conflicts if item.startswith(_MISSING_INPUTS_PREFIX)]
    verified = contract.status != "blocked" and not contract.conflicts
    if verified:
        log.append("核对：重新生成的验证合同没有冲突，模型需要的输入都能从样本得到。")
        repair_id_holder = _report(
            option=option,
            finding=finding or "样本列名与 PMML 输入名不一致，扫描没有通过。",
            action=f"在验证合同里补上字段映射：{mapping_text}。样本文件没有改动。",
            verification="重新生成的验证合同没有冲突，模型需要的输入都能从样本得到。",
            verified=True,
            log=log,
            undo=True,
            source_message_id=source_message_id,
        )
        save_repair_overrides(settings.tasks_dir, task.id, {
            **before,
            "field_aliases": {**before["field_aliases"], **aliases},
            "applied": [*before["applied"], {
                "repair_id": repair_id_holder["repair_id"],
                "fingerprint": option.get("fingerprint"),
                "field_aliases": aliases,
                "material_hashes": hashes,
            }],
        })
        return repair_id_holder
    reason = "；".join(remaining or contract.conflicts)[:300] or "验证合同仍未通过"
    log.append(f"核对没通过：{reason}")
    return _report(
        option=option,
        finding=finding or "样本列名与 PMML 输入名不一致，扫描没有通过。",
        action=f"试过补上字段映射：{mapping_text}，已还原，没有保存。",
        verification=f"核对没通过：{reason}",
        verified=False,
        log=log,
        undo=False,
        source_message_id=source_message_id,
    )


def confirmation_from_dict(payload: Mapping[str, Any]):
    from marvis.validation.input_contracts import (
        ValidationInputConfirmation,
        transformation_spec_from_dict,
    )

    values = dict(payload)
    transformations = tuple(
        transformation_spec_from_dict(item) for item in values.pop("transformations", []) or []
    )
    return ValidationInputConfirmation(**values, transformations=transformations)


def _apply_time_column(settings, task, option, *, finding, source_message_id) -> dict:
    from marvis.modeling_holdout.guards import record_exposure
    from marvis.repositories.validation_contracts import ValidationContractRepository
    from marvis.validation.input_confirmation import validate_confirmation_against_materials

    params = option.get("params") or {}
    time_col = str(params.get("time_col") or "")
    contracts = ValidationContractRepository(settings.db_path)
    record = contracts.get(task.id)
    if (
        record is None
        or record.status != "pending_confirmation"
        or record.revision != params.get("revision")
    ):
        raise RepairStale("字段确认已经变了，这个办法不再适用；请按最新的字段确认卡操作。")
    requested = replace(confirmation_from_dict(params.get("requested") or {}), time_col=time_col)
    materials = _resolved_materials(task)
    for path in (materials.sample, materials.dictionary):
        record_exposure(settings.db_path, settings.workspace, path)
    log = [f"改用时间列：{time_col}"]
    try:
        validated = validate_confirmation_against_materials(
            contract=record.contract,
            sample_path=materials.sample,
            dictionary_path=materials.dictionary,
            requested=requested,
        )
    except ValueError as exc:
        log.append(f"核对没通过：{exc}")
        return _report(
            option=option,
            finding=finding or "时间列有无法解析的日期，字段确认没有通过。",
            action=f"试过改用 {time_col} 作为时间列，字段没有保存。",
            verification=f"核对没通过：{str(exc)[:300]}",
            verified=False,
            log=log,
            undo=False,
            source_message_id=source_message_id,
        )
    contracts.confirm(
        task.id,
        validated.values,
        expected_revision=record.revision,
        resolved_sample_schema=validated.sample_schema,
        resolved_feature_metadata=validated.feature_metadata,
    )
    log.append(f"核对：{time_col} 整列都能解析为日期，字段已确认（第 {record.revision + 1} 版）。")
    return _report(
        option=option,
        finding=finding or "时间列有无法解析的日期，字段确认没有通过。",
        action=f"按你的选择改用 {time_col} 作为时间列，其余字段沿用你提交的确认。",
        verification=f"{time_col} 整列都能解析为日期，字段已确认。",
        verified=True,
        log=log,
        undo=False,
        source_message_id=source_message_id,
    )


# Stages after the field contract is confirmed. Undoing a field alias there
# would rescan and discard their results, so undo stops being offered.
_PAST_CONFIRMATION_STAGES = frozenset({"reproducibility", "metrics", "word_conclusion_draft"})


def latest_undoable_report(messages: Sequence[Mapping[str, Any]]) -> dict | None:
    """The newest repair report, if it can still be undone.

    Only the newest report is ever undoable (one undo button, one command),
    and only until a later stage has started.
    """

    undone = _undone_repair_ids(messages)
    latest_index = None
    for index in range(len(messages) - 1, -1, -1):
        metadata = messages[index].get("metadata") or {}
        if isinstance(metadata.get("repair_report"), dict):
            latest_index = index
            break
    if latest_index is None:
        return None
    message = messages[latest_index]
    report = {**message["metadata"]["repair_report"], "message_id": message.get("id")}
    if not report.get("undo_command") or str(report.get("repair_id")) in undone:
        return None
    if any(
        later.get("role") == "assistant"
        and str(later.get("stage") or "") in _PAST_CONFIRMATION_STAGES
        for later in messages[latest_index + 1:]
    ):
        return None
    return report


def undo_repair(settings, task, report: Mapping[str, Any]) -> dict:
    """Remove the aliases one repair added and never auto-apply it again."""

    if report.get("option_id") != "map_fields":
        raise ValueError("only field-alias repairs can be undone")
    before = load_repair_overrides(settings.tasks_dir, task.id)
    applied = [item for item in before["applied"]
               if item.get("repair_id") != report.get("repair_id")]
    aliases: dict[str, str] = {}
    for item in applied:
        aliases.update({str(k): str(v) for k, v in (item.get("field_aliases") or {}).items()})
    rejected = list(dict.fromkeys([*before["rejected_fingerprints"], str(report.get("fingerprint") or "")]))
    save_repair_overrides(settings.tasks_dir, task.id, {
        **before,
        "field_aliases": aliases,
        "applied": applied,
        "rejected_fingerprints": [item for item in rejected if item],
    })
    removed = [item for item in before["applied"] if item.get("repair_id") == report.get("repair_id")]
    mapping = {}
    for item in removed:
        mapping.update(item.get("field_aliases") or {})
    mapping_text = "、".join(f"{column} → {name}" for name, column in mapping.items())
    return {
        "schema_version": REPAIR_SCHEMA,
        "repair_id": report.get("repair_id"),
        "finding": f"已撤销字段映射：{mapping_text}" if mapping_text else "已撤销这次自动处理。",
        "log": [f"删除映射：{mapping_text}" if mapping_text else "没有需要删除的映射",
                "这项处理不会再自动执行，仍可以在失败卡里手动点。"],
    }


# ---------- Commands ----------


def _compact(text: str) -> str:
    return "".join(str(text or "").split())


def _is_progress_boundary(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    metadata = message.get("metadata") or {}
    if metadata.get("error") or metadata.get("repair_report") or metadata.get("repair_undo"):
        return False
    return str(message.get("stage") or "") in {"scan", "reproducibility", "metrics",
                                                 "input_confirmation", "word_conclusion_draft"}


def match_repair_command(messages: Sequence[Mapping[str, Any]], content: str):
    """Exact command match against the latest offered options or undo.

    Returns ``("undo", report)`` or ``("apply", option, failure_message)``.
    Only the newest validation failure card counts, and only until a later
    stage message supersedes it.
    """

    compact = _compact(content)
    if not compact:
        return None
    if compact == _compact(UNDO_COMMAND):
        report = latest_undoable_report(messages)
        return ("undo", report) if report is not None else ("stale", "undo")
    for message in reversed(messages):
        if _is_progress_boundary(message):
            break
        metadata = message.get("metadata") or {}
        diagnostic = metadata.get("error_diagnostic")
        if not isinstance(diagnostic, Mapping) or diagnostic.get("workflow") != "validation":
            continue
        for option in diagnostic.get("repair_options") or []:
            if isinstance(option, Mapping) and option.get("command") and (
                _compact(option["command"]) == compact
            ):
                if option.get("id") == "reselect_materials":
                    return None  # the existing material reselection branch owns it
                return ("apply", dict(option), message)
        break
    # An older card's command: answer that it expired instead of letting the
    # words fall through to other handlers (a plain rescan, or free chat that
    # might describe a change nobody made).
    if _is_historical_repair_command(messages, compact):
        return ("stale", None)
    return None


def _is_historical_repair_command(messages: Sequence[Mapping[str, Any]], compact: str) -> bool:
    for message in messages:
        diagnostic = (message.get("metadata") or {}).get("error_diagnostic")
        if not isinstance(diagnostic, Mapping) or diagnostic.get("workflow") != "validation":
            continue
        for option in diagnostic.get("repair_options") or []:
            if (
                isinstance(option, Mapping)
                # These two have their own handlers (material reselection,
                # the stage rerun), which stay correct at any point.
                and option.get("id") not in {"reselect_materials", "rerun_pmml_scoring"}
                and option.get("command")
                and _compact(option["command"]) == compact
            ):
                return True
    return False


UNDO_UNAVAILABLE_REPLY = (
    "现在没有可以撤销的处理：要么已经撤销过，要么流程已经往后跑了（撤销会清掉后面的结果）。"
    "平台没有做任何改动；如果要换回原来的字段，请重新选择验证材料。"
)
STALE_COMMAND_REPLY = (
    "这个办法已经过期：流程在那张卡片之后有了新进展，平台没有执行它。"
    "请按最新的卡片或步骤操作。"
)


# ---------- Message content ----------


def repair_report_content(report: Mapping[str, Any]) -> str:
    title = "遇到问题，已处理好" if report.get("verified") else "处理没成功，已还原"
    lines = [f"**{title}**", f"发现：{report.get('finding')}", f"处理：{report.get('action')}",
             f"核对：{report.get('verification')}"]
    if report.get("undo_command"):
        lines.append(f"如需撤销，点「{report['undo_command']}」。")
    return "\n".join(lines)


def repair_undo_content(undo: Mapping[str, Any]) -> str:
    return str(undo.get("finding") or "已撤销这次自动处理。")


def enrich_repair_reports(messages: list[dict]) -> None:
    """Mark undone reports on the way out without rewriting the audit record."""

    undone = _undone_repair_ids(messages)
    if not undone:
        return
    for message in messages:
        metadata = message.get("metadata") or {}
        report = metadata.get("repair_report")
        if isinstance(report, dict) and str(report.get("repair_id")) in undone:
            message["metadata"] = {**metadata, "repair_report": {**report, "undone": True}}


def strip_option(diagnostic: dict, option_id: str, note: dict | None = None) -> dict:
    """A copy of ``diagnostic`` without one option (tried and not verified)."""

    options = [item for item in diagnostic.get("repair_options") or [] if item.get("id") != option_id]
    result = {**diagnostic, "repair_options": options, "recovery_actions": _recovery_actions(options)}
    if note is not None:
        result["evidence"] = [*(diagnostic.get("evidence") or []), note]
    return result


__all__ = [
    "MAP_FIELDS_COMMAND",
    "REPAIR_SCHEMA",
    "RERUN_PMML_COMMAND",
    "RESELECT_MATERIALS_COMMAND",
    "RepairStale",
    "STALE_COMMAND_REPLY",
    "UNDO_UNAVAILABLE_REPLY",
    "active_field_aliases",
    "TIER_AUTO",
    "TIER_CONFIRM",
    "TIER_ONE_CLICK",
    "UNDO_COMMAND",
    "alternative_time_columns",
    "apply_repair_option",
    "budget_exhausted_note",
    "diagnose_validation_failure",
    "enrich_repair_reports",
    "latest_undoable_report",
    "load_repair_overrides",
    "match_missing_fields",
    "match_repair_command",
    "material_inventory",
    "plan_auto_repair",
    "probe_java_runtime",
    "profile_time_column",
    "repair_budget",
    "repair_report_content",
    "repair_undo_content",
    "save_repair_overrides",
    "strip_option",
    "undo_repair",
]
