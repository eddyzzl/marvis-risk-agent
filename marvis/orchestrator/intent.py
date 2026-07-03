from __future__ import annotations

from dataclasses import dataclass
import json
import re

from marvis.llm_prompts import CLASSIFY_SYS as _CLASSIFY_SYS_SPEC
from marvis.orchestrator.templates import WorkflowTemplate, get_template, list_templates


STRONG_MATCH_THRESHOLD = 0.75
# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of CLASSIFY_SYS from here keep working unchanged.
CLASSIFY_SYS = _CLASSIFY_SYS_SPEC.text


@dataclass(frozen=True)
class IntentResult:
    kind: str
    template_id: str | None
    slots: dict
    confidence: float
    rationale: str


class IntentRouter:
    def __init__(self, llm_factory, tool_registry):
        self._llm_factory = llm_factory
        self._tools = tool_registry

    def route(self, goal: str, task_context: dict) -> IntentResult:
        hit = self._match_templates(goal)
        if hit and hit[1] >= STRONG_MATCH_THRESHOLD:
            template = get_template(hit[0])
            return IntentResult(
                kind="template",
                template_id=template.id,
                slots=self._extract_slots(template, goal, task_context),
                confidence=hit[1],
                rationale=f"keyword match: {template.id}",
            )

        candidates = [template.id for template in list_templates()] + ["novel"]
        choice = self._llm_classify(goal, task_context, candidates)
        if choice != "novel":
            template = get_template(choice)
            return IntentResult(
                kind="template",
                template_id=choice,
                slots=self._extract_slots(template, goal, task_context),
                confidence=0.6,
                rationale="llm classified",
            )
        return IntentResult(
            kind="novel",
            template_id=None,
            slots={},
            confidence=0.5,
            rationale="no template matched",
        )

    def _match_templates(self, goal: str) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for template in list_templates():
            scores = [_pattern_score(pattern, goal) for pattern in template.goal_patterns]
            score = max(scores) if scores else 0.0
            if best is None or score > best[1]:
                best = (template.id, score)
        if best is None or best[1] <= 0:
            return None
        return best

    def _llm_classify(self, goal: str, task_context: dict, candidates: list[str]) -> str:
        if not candidates:
            return "novel"
        prompt = build_classify_prompt(goal, task_context, candidates)
        try:
            raw = self._llm_factory().complete(
                system_prompt=CLASSIFY_SYS,
                user_prompt=prompt,
                response_format={"type": "json_object"},
                caller="router_intent",
                stream=False,
            )
        except Exception:
            return "novel"
        return _extract_choice(str(raw), candidates)

    def _extract_slots(
        self,
        template: WorkflowTemplate,
        _goal: str,
        task_context: dict,
    ) -> dict:
        slots = {}
        for slot in template.slots:
            if slot.source in {"task_context", "infer", "user"}:
                value = task_context.get(slot.name)
            else:
                value = None
            if value is None and not slot.required:
                continue
            slots[slot.name] = value
        return slots


def build_classify_prompt(goal: str, task_context: dict, candidates: list[str]) -> str:
    return json.dumps(
        {
            "goal": goal,
            "task_context": task_context,
            "candidates": candidates,
            "instruction": "Return JSON like {\"choice\":\"candidate_id\"}.",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _pattern_score(pattern: str, goal: str) -> float:
    pattern_text = pattern.strip()
    if not pattern_text:
        return 0.0
    goal_text = goal.strip()
    if pattern_text.lower() == goal_text.lower():
        return 1.0
    if pattern_text.lower() in goal_text.lower():
        specificity = min(len(pattern_text) / max(len(goal_text), 1), 1.0)
        return 0.75 + 0.2 * specificity
    try:
        if re.search(pattern_text, goal_text, flags=re.IGNORECASE):
            return 0.85
    except re.error:
        return 0.0
    return 0.0


def _extract_choice(raw: str, candidates: list[str]) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        choice = data.get("choice") or data.get("template_id") or data.get("kind")
        if choice in candidates:
            return str(choice)
    cleaned = raw.strip().strip('"').strip("'")
    if cleaned in candidates:
        return cleaned
    for candidate in candidates:
        if candidate != "novel" and re.search(rf"\b{re.escape(candidate)}\b", raw):
            return candidate
    return "novel"
