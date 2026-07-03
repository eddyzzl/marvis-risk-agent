from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from marvis.drafts.authoring import assert_draft_code_safe
from marvis.drafts.contracts import DRAFT_STATUS_DRAFT, DRAFT_STATUS_TESTED, DraftRun
from marvis.drafts.errors import AuthoringError, DraftStateError


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
        try:
            assert_draft_code_safe(draft.code)
        except AuthoringError as exc:
            run = _failed_run(draft_id, inputs, task_id=task_id, error=str(exc))
            self._save_run_with_audit(run, status=None)
            return run
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
        next_status = (
            DRAFT_STATUS_TESTED
            if result.ok and draft.status in {DRAFT_STATUS_DRAFT, DRAFT_STATUS_TESTED}
            else None
        )
        self._save_run_with_audit(run, status=next_status)
        return run

    def _save_run_with_audit(self, run: DraftRun, *, status: str | None) -> None:
        self._repo.save_draft_run_with_status_audit(
            run,
            status=status,
            audit={
                "kind": "draft.run.record",
                "target_ref": run.draft_id,
                "outcome": "succeeded" if run.ok else "failed",
                "detail": {
                    "run_id": run.id,
                    "task_id": run.task_id,
                    "inputs_hash": run.inputs_hash,
                    "status": status,
                },
            },
        )


def _new_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_inputs(inputs: dict) -> str:
    raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _failed_run(draft_id: str, inputs: dict, *, task_id: str, error: str) -> DraftRun:
    return DraftRun(
        id=_new_id(),
        draft_id=draft_id,
        task_id=task_id,
        inputs_hash=_hash_inputs(inputs),
        ok=False,
        output=None,
        error=error,
        at=_now(),
    )


__all__ = ["DRAFT_TIMEOUT_SECONDS", "DraftSandbox"]
