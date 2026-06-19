from __future__ import annotations

from marvis.drafts.contracts import DraftTool
from marvis.drafts.errors import DraftNotFound


class DraftRegistry:
    def __init__(self, repo):
        self._repo = repo

    def add(self, draft: DraftTool) -> None:
        self._repo.save_draft(draft)

    def get(self, draft_id: str) -> DraftTool:
        draft = self._repo.get_draft(draft_id)
        if draft is None:
            raise DraftNotFound(draft_id)
        return draft

    def list_for_task(self, task_id: str, *, status: str | None = None) -> list[DraftTool]:
        return self._repo.list_drafts(task_id, status=status)

    def set_status(self, draft_id: str, status: str) -> None:
        self._repo.set_status(draft_id, status)


__all__ = ["DraftRegistry"]
