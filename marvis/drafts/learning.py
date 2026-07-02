from __future__ import annotations

from datetime import UTC, datetime
import uuid

from marvis.drafts.contracts import LearningNote
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
) -> LearningNote:
    joined = "\n---\n".join(str(content)[:MAX_CONTENT_CHARS] for content in contents)[:MAX_JOINED_CHARS]
    raw = llm_factory().complete(
        system_prompt=LEARN_SYS,
        user_prompt=_learn_prompt(query, joined),
        stream=False,
    )
    return LearningNote(
        id=_new_id(),
        query=str(query),
        sources=tuple(str(source) for source in sources),
        distilled=_sanitize(str(raw))[:MAX_NOTE_CHARS],
        created_at=_now(),
    )


def _learn_prompt(query: str, joined: str) -> str:
    return (
        f"学习目标：{query}\n\n"
        "请输出结构化实现笔记：\n"
        "1. 关键概念/公式\n"
        "2. 实现步骤\n"
        "3. 关键 API 或伪代码\n"
        "4. 风险与边界\n\n"
        f"资料正文（已截断）：\n{joined}"
    )


def _sanitize(value: str) -> str:
    return "\n".join(line.strip() for line in value.replace("\x00", "").splitlines() if line.strip())


def _new_id() -> str:
    return f"note-{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["MAX_NOTE_CHARS", "distill_learning"]
