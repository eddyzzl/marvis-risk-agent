from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import threading

from marvis.agent_memory.models import MEMORY_TYPES, normalize_memory_type


CONSOLIDATION_TRIGGERS = {"validation.completed", "report.after_generate", "memory.after_save"}


class ConsolidationScheduler:
    def __init__(
        self,
        distillation_engine,
        evolution_manager,
        store,
        *,
        throttle_seconds: int = 300,
        async_mode: bool = True,
        auto_enabled: Callable[[], bool] | None = None,
    ):
        self._distill = distillation_engine
        self._evolve = evolution_manager
        self._store = store
        self._throttle_seconds = int(throttle_seconds)
        self._async_mode = async_mode
        self._auto_enabled = auto_enabled or (lambda: True)

    def on_event(self, event: str, payload: dict) -> None:
        if event not in CONSOLIDATION_TRIGGERS:
            return
        if not self._auto_enabled():
            return
        categories = _categories_for_event(event, payload)
        for category in categories:
            if self._recently_consolidated(category):
                continue
            self._run(lambda category=category: self._consolidate(category))

    def consolidate_all(self, categories: list[str] | None = None) -> dict:
        result = {}
        selected = categories or [item for item in MEMORY_TYPES if item != "skill_experience_reserved"]
        for category in selected:
            normalized = normalize_memory_type(category)
            try:
                result[normalized] = self._consolidate(normalized)
            except Exception:
                result[normalized] = 0
        return result

    def _consolidate(self, category: str) -> int:
        count = 0
        candidates = self._distill.distill_category(category)
        for candidate in candidates:
            self._evolve.upsert_with_evolution(candidate)
            count += 1
        self._store.mark_consolidated(category, at=_now_iso())
        return count

    def _recently_consolidated(self, category: str) -> bool:
        last = self._store.last_consolidated_at(category)
        if not last:
            return False
        try:
            last_at = datetime.fromisoformat(last)
        except ValueError:
            return False
        return (_now() - last_at).total_seconds() < self._throttle_seconds

    def _run(self, func) -> None:
        if not self._async_mode:
            try:
                func()
            except Exception:
                return
            return
        thread = threading.Thread(target=_safe_call, args=(func,), daemon=True)
        thread.start()


def _categories_for_event(event: str, payload: dict) -> list[str]:
    if event == "validation.completed":
        return ["model_experience", "validation_pitfall", "field_convention", "task_experience"]
    if event == "report.after_generate":
        return ["task_experience"]
    category = payload.get("memory_type") or payload.get("category")
    if category:
        return [normalize_memory_type(str(category))]
    return [item for item in MEMORY_TYPES if item != "skill_experience_reserved"]


def _safe_call(func) -> None:
    try:
        func()
    except Exception:
        return


def _now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


__all__ = ["CONSOLIDATION_TRIGGERS", "ConsolidationScheduler"]
