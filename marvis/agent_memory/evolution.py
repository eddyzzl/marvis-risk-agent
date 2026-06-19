from __future__ import annotations

from marvis.agent_memory.distillation import MemoryDistillation


class EvolutionManager:
    def __init__(self, store):
        self._store = store

    def upsert_with_evolution(self, candidate: MemoryDistillation) -> MemoryDistillation:
        existing = self._store.get_active_distillation(candidate.scope_key)
        if existing is None:
            return self._store.create_distillation(candidate)
        if self._is_meaningful_update(existing, candidate):
            created = self._store.create_distillation(candidate)
            self._store.set_superseded(existing.id, by=created.id)
            return created
        return self._store.update_distillation_support(existing.id, candidate.support_count)

    def rollback(self, distillation_id: str) -> None:
        self._store.get_distillation(distillation_id)
        predecessor = self._store.find_superseded_by(distillation_id)
        if predecessor is not None:
            self._store.clear_superseded(predecessor.id)
        self._store.set_status_distillation(distillation_id, "rolled_back")

    def _is_meaningful_update(self, old: MemoryDistillation, new: MemoryDistillation) -> bool:
        if old.confidence != new.confidence:
            return True
        return _normalized_structured(old.structured) != _normalized_structured(new.structured)


def _normalized_structured(value: dict) -> dict:
    normalized = {}
    for key, item in value.items():
        if isinstance(item, list | tuple | set):
            normalized[key] = sorted(str(part) for part in item)
        elif isinstance(item, dict):
            normalized[key] = _normalized_structured(item)
        else:
            normalized[key] = item
    return normalized


__all__ = ["EvolutionManager"]
