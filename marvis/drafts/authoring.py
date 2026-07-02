from __future__ import annotations

import ast
from datetime import UTC, datetime
import re
import uuid

from jsonschema import Draft202012Validator

from marvis.agent.json_reply import load_json_object
from marvis.drafts.contracts import DraftTool, LearningNote
from marvis.drafts.errors import AuthoringError


TOOL_TEMPLATE = '''
def {entrypoint}(inputs: dict, ctx) -> dict:
    """{summary}"""
    {body}
    return {return_expr}
'''
AUTHOR_SYS = (
    "你在为 MARVIS 写一个数据/特征/分析工具。只用 pandas/numpy/标准库做纯计算；"
    "不读写任意文件、不联网、不执行系统命令。必须声明 input_schema/output_schema/determinism。"
)
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
_BANNED_SNIPPETS = (
    "os.system",
    "subprocess",
    "eval(",
    "exec(",
    "__import__",
    "socket",
    "shutil.rmtree",
    "requests.",
    "httpx.",
    "urllib.",
    "urlopen(",
    "open(",
    ".read_text(",
    ".read_bytes(",
    ".write_text(",
    ".write_bytes(",
    "os.remove",
    "os.unlink",
    "os.rmdir",
)
_BANNED_IMPORT_ROOTS = {
    "httpx",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
_BANNED_CALL_NAMES = {"eval", "exec", "open", "__import__"}
_BANNED_ATTR_CALLS = {
    "glob",
    "mkdir",
    "open",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "rglob",
    "rmdir",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}


def draft_script(
    task_id: str,
    goal: str,
    *,
    learning_note: LearningNote | None,
    llm_factory,
) -> DraftTool:
    """Generate one draft tool, with fence-tolerant JSON parsing + one retry.

    The safety floor (assert_draft_code_safe AST scan) is a hard gate on every
    attempt. On the first failure the model is fed its previous reply and the
    exact validation error for a targeted correction; the second failure raises.
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
    assert_draft_code_safe(code)
    if f"def {spec['name']}" not in code:
        raise AuthoringError("code must define the named tool function")
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


def assert_draft_code_safe(code: str) -> None:
    hits = [snippet for snippet in _BANNED_SNIPPETS if snippet in code]
    hits.extend(_ast_safety_hits(code))
    if hits:
        ordered_hits = list(dict.fromkeys(hits))
        raise AuthoringError(f"draft code contains banned calls: {', '.join(ordered_hits)}")


def _ast_safety_hits(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise AuthoringError(f"draft code is not valid Python: {exc.msg}") from exc
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name).split(".", 1)[0]
                if root in _BANNED_IMPORT_ROOTS:
                    hits.append(f"import {root}")
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if root in _BANNED_IMPORT_ROOTS:
                hits.append(f"from {root} import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALL_NAMES:
                hits.append(f"{node.func.id}(")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_ATTR_CALLS:
                hits.append(f".{node.func.attr}(")
    return hits


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
