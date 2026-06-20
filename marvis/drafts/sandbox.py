from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from marvis.drafts.contracts import DRAFT_STATUS_DRAFT, DRAFT_STATUS_TESTED, DraftRun
from marvis.drafts.errors import DraftStateError


DRAFT_TIMEOUT_SECONDS = 30


class DraftSandbox:
    def __init__(self, tool_runner, draft_registry, repo):
        self._runner = tool_runner
        self._drafts = draft_registry
        self._repo = repo

    def run_draft(self, draft_id: str, inputs: dict, *, task_id: str) -> DraftRun:
        draft = self._drafts.get(draft_id)
        if str(draft.task_id) != str(task_id):
            raise DraftStateError(f"task mismatch for draft {draft_id}: {draft.task_id} != {task_id}")
        with tempfile.TemporaryDirectory(prefix="marvis-draft-") as temp_name:
            module_path = Path(temp_name) / f"{draft.name}.py"
            module_path.write_text(draft.code, encoding="utf-8")
            result = self._runner.invoke_adhoc(
                module=module_path,
                entrypoint=draft.name,
                inputs=inputs,
                input_schema=draft.input_schema,
                output_schema=draft.output_schema,
                timeout_seconds=DRAFT_TIMEOUT_SECONDS,
                task_id=task_id,
                mode="draft",
            )
        run = DraftRun(
            id=_new_id(),
            draft_id=draft_id,
            task_id=task_id,
            inputs_hash=_hash_inputs(inputs),
            ok=result.ok,
            output=result.output if result.ok else None,
            error=result.error,
            at=_now(),
        )
        self._repo.save_draft_run(run)
        if result.ok and draft.status in {DRAFT_STATUS_DRAFT, DRAFT_STATUS_TESTED}:
            self._drafts.set_status(draft_id, DRAFT_STATUS_TESTED)
        return run


def _new_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_inputs(inputs: dict) -> str:
    raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["DRAFT_TIMEOUT_SECONDS", "DraftSandbox"]
