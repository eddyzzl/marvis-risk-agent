from __future__ import annotations

from datetime import UTC, datetime
import re
import uuid

from jsonschema import Draft202012Validator

from marvis.agent.json_reply import load_json_object
from marvis.drafts.contracts import DraftTool, LearningNote
from marvis.drafts.errors import AuthoringError
from marvis.draft_language import (
    ALLOWED_IMPORT_ROOTS,
    DraftLanguageError,
    validate_draft_source,
)
from marvis.llm_prompts import AUTHOR_SYS as _AUTHOR_SYS_SPEC


TOOL_TEMPLATE = '''
def {entrypoint}(inputs: dict, ctx) -> dict:
    """{summary}"""
    {body}
    return {return_expr}
'''
# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of AUTHOR_SYS from here keep working unchanged.
AUTHOR_SYS = _AUTHOR_SYS_SPEC.text
REQUIRED_DRAFT_KEYS = (
    "name",
    "summary",
    "code",
    "input_schema",
    "output_schema",
    "determinism",
)
DETERMINISM_CHOICES = {"deterministic", "stochastic"}
DRAFT_MAX_ATTEMPTS = 2
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# Backward-compatible export for callers/tests that previously inspected the
# authoring gate's import set. It now comes from the shared language policy.
_ALLOWED_IMPORT_ROOTS = ALLOWED_IMPORT_ROOTS


def draft_script(
    task_id: str,
    goal: str,
    *,
    learning_note: LearningNote | None,
    llm_factory,
) -> DraftTool:
    """Generate one draft tool, with fence-tolerant JSON parsing + one retry.

    The Draft Language v1 gate is enforced on every attempt. On the first
    failure the model is fed its previous reply and the exact validation error
    for a targeted correction; the second failure raises.
    """
    base_prompt = _authoring_prompt(goal, learning_note)
    prompt = base_prompt
    last_error: str | None = None
    for attempt in range(DRAFT_MAX_ATTEMPTS):
        raw = llm_factory().complete(
            system_prompt=AUTHOR_SYS,
            user_prompt=prompt,
            response_format={"type": "json_object"},
            caller="author",
            stream=False,
        )
        try:
            return _build_draft_tool(str(task_id), learning_note, raw)
        except AuthoringError as exc:
            last_error = str(exc)
            if attempt + 1 >= DRAFT_MAX_ATTEMPTS:
                raise
            prompt = _retry_prompt(base_prompt, str(raw), last_error)
    # Unreachable: the loop either returns or raises above.
    raise AuthoringError(last_error or "draft authoring failed")


def _build_draft_tool(
    task_id: str,
    learning_note: LearningNote | None,
    raw,
) -> DraftTool:
    spec, error = load_json_object(raw)
    if spec is None:
        raise AuthoringError(f"LLM output is not valid JSON: {error}")
    _assert_required_keys(spec)
    _assert_name(str(spec["name"]))
    _assert_schema(spec["input_schema"], "input_schema")
    _assert_schema(spec["output_schema"], "output_schema")
    determinism = str(spec["determinism"])
    if determinism not in DETERMINISM_CHOICES:
        raise AuthoringError("determinism must be deterministic or stochastic")
    code = str(spec["code"])
    assert_draft_code_safe(code, entrypoint=str(spec["name"]))
    return DraftTool(
        id=_new_id(),
        task_id=task_id,
        name=str(spec["name"]),
        summary=str(spec["summary"]),
        code=code,
        input_schema=dict(spec["input_schema"]),
        output_schema=dict(spec["output_schema"]),
        determinism=determinism,
        source="web_learning" if learning_note else "llm_generated",
        learning_note_id=learning_note.id if learning_note else None,
        status="draft",
        created_at=_now(),
    )


def _retry_prompt(base_prompt: str, previous_reply: str, error: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"【上一次返回未通过校验】\n{previous_reply}\n\n"
        f"【校验错误】\n{error}\n\n"
        "请修正上述问题后，严格只返回一个 JSON 对象："
        "{name, summary, code, input_schema, output_schema, determinism}。"
    )


def assert_draft_code_safe(code: str, *, entrypoint: str | None = None) -> None:
    """Compatibility gate backed by the shared Draft Language v1 validator."""

    try:
        validate_draft_source(code, entrypoint=entrypoint)
    except DraftLanguageError as exc:
        raise AuthoringError(str(exc)) from exc


def _authoring_prompt(goal: str, learning_note: LearningNote | None) -> str:
    note_text = "无"
    if learning_note:
        note_text = (
            f"来源: {', '.join(learning_note.sources)}\n"
            f"学习笔记:\n{learning_note.distilled}"
        )
    return (
        f"目标: {goal}\n\n"
        f"可参考学习笔记:\n{note_text}\n\n"
        "请输出 JSON: {name, summary, code, input_schema, output_schema, determinism}。\n"
        f"工具模板:\n{TOOL_TEMPLATE}"
    )


def _assert_required_keys(payload: dict) -> None:
    missing = [key for key in REQUIRED_DRAFT_KEYS if key not in payload]
    if missing:
        raise AuthoringError(f"missing required draft keys: {', '.join(missing)}")


def _assert_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name):
        raise AuthoringError("name must be a Python function identifier")


def _assert_schema(schema, label: str) -> None:
    if not isinstance(schema, dict) or not schema:
        raise AuthoringError(f"{label} must be a non-empty JSON schema")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise AuthoringError(f"{label} is not a valid JSON schema") from exc


def _new_id() -> str:
    return f"draft-{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["TOOL_TEMPLATE", "assert_draft_code_safe", "draft_script"]
