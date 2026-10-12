from __future__ import annotations

from datetime import UTC, datetime
import uuid

from marvis.drafts.contracts import LearningNote
from marvis.agent.json_reply import load_json_object
from marvis.drafts.errors import DraftError
from marvis.llm_outbound import OutboundContextError, application_schema
from marvis.llm_outbound_learning import public_learning_prompt
from marvis.llm_prompts import LEARN_SYS as _LEARN_SYS_SPEC


MAX_CONTENT_CHARS = 5_000
MAX_JOINED_CHARS = 20_000
MAX_NOTE_CHARS = 4_000
# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of LEARN_SYS from here keep working unchanged.
LEARN_SYS = _LEARN_SYS_SPEC.text


def distill_learning(
    query: str,
    contents: list[str],
    sources: list[str],
    *,
    llm_factory,
    public_sources=(),
) -> LearningNote:
    try:
        prompt = public_learning_prompt(query, contents=contents, sources=sources, public_sources=public_sources)
    except OutboundContextError as exc:
        raise DraftError(str(exc)) from exc
    raw = llm_factory().complete(
        system_prompt=LEARN_SYS,
        user_prompt=prompt,
        response_format={"type": "json_object"},
        json_schema=application_schema("learning_note"),
        caller="learning_note",
        prompt_name=_LEARN_SYS_SPEC.name,
        prompt_version=_LEARN_SYS_SPEC.version,
        stream=False,
    )
    payload, _ = load_json_object(raw)
    note = payload.get("note") if isinstance(payload, dict) else None
    if not isinstance(note, str) or not note.strip() or len(note) > MAX_NOTE_CHARS:
        raise DraftError("学习摘要未通过输出契约校验，未保存学习成功状态；请重试。")
    return LearningNote(
        id=_new_id(),
        query=str(query),
        sources=tuple(str(source) for source in sources),
        distilled=_sanitize(note),
        created_at=_now(),
    )


def _sanitize(value: str) -> str:
    return "\n".join(line.strip() for line in value.replace("\x00", "").splitlines() if line.strip())


def _new_id() -> str:
    return f"note-{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["MAX_NOTE_CHARS", "distill_learning"]
