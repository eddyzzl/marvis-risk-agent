"""CLI-facing helpers for production planning/guardrail evaluation (LLM-2).

Kept separate from ``marvis/__main__.py`` (which only does argument parsing +
thin dispatch) so this module can be imported and unit tested without
importing argparse/CLI plumbing.

Runs the *production* eval framework against a real, settings-configured LLM
model: builds an ``EvalOrchestrator`` backed by ``OpenAICompatibleLLMClient``,
runs ``calibrate_tier_for_model`` over the production case catalog, writes a unique JSON
report to ``workspace/eval/{model_id}-{timestamp}-{run_id}.json``, and -- when a baseline
report path is given -- runs ``regression_gate`` against it.

Not wired into CI (this makes real network calls to a configured local
inference server); it is the production-facing verification tool LLM-2 asked
for: "does my model swap regress" answered by an actual run, not a guess.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import uuid

from marvis.files import write_json_atomic
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import resolve_llm_model
from marvis.orchestrator.eval.cases import initial_eval_cases
from marvis.orchestrator.eval.runner import EvalOrchestrator
from marvis.orchestrator.eval.scoring import calibrate_tier_for_model, regression_gate


class EvalCliError(RuntimeError):
    pass


def run_eval_llm_cli(
    *,
    workspace: Path,
    model_id: str | None,
    baseline_path: Path | None,
) -> dict:
    """Run the planning/guardrail suite against a real configured model.

    Returns the report dict that was also written to disk. Raises
    ``EvalCliError`` (mapped to a non-zero exit by the caller) when a
    baseline is given and the regression gate fails.
    """
    baseline: dict | None = None
    baseline_ref = ""
    baseline_hash = ""
    if baseline_path is not None:
        try:
            baseline_bytes = baseline_path.read_bytes()
            loaded_baseline = json.loads(baseline_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvalCliError(f"cannot read eval baseline {baseline_path}: {exc}") from exc
        if not isinstance(loaded_baseline, dict):
            raise EvalCliError(f"eval baseline must be a JSON object: {baseline_path}")
        baseline = loaded_baseline
        baseline_ref = str(baseline_path.resolve())
        baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()

    profile = resolve_llm_model(workspace, model_id)
    resolved_model_id = str(profile.get("model_id") or model_id or "unknown")

    def llm_factory():
        return OpenAICompatibleLLMClient(profile)

    orchestrator = EvalOrchestrator(llm_factory)
    cases = list(initial_eval_cases())
    generated_at = datetime.now(UTC)
    run_id = uuid.uuid4().hex
    report = calibrate_tier_for_model(resolved_model_id, cases, orchestrator=orchestrator)
    report["generated_at"] = generated_at.isoformat()
    report["run_id"] = run_id
    report["case_ids"] = [case.id for case in cases]

    recommended = report.get("recommended_tier")
    if recommended is not None:
        report["overall_pass_rate"] = report["per_tier"][recommended]["pass_rate"]
        report["guardrail_pass_rate"] = report["per_tier"][recommended]["guardrail_pass_rate"]

    if baseline is not None:
        ok, problems = regression_gate(baseline, report)
        report["baseline_ref"] = baseline_ref
        report["baseline_sha256"] = baseline_hash
        report["regression_ok"] = ok
        report["regression_problems"] = problems

    out_path = _report_path(
        workspace,
        resolved_model_id,
        generated_at=generated_at,
        run_id=run_id,
    )
    report["report_path"] = str(out_path)
    write_json_atomic(out_path, report, ensure_ascii=False, indent=2)

    if baseline is not None:
        if report["regression_ok"] is not True:
            raise EvalCliError(
                f"eval regression against {baseline_path}: "
                f"{'; '.join(report['regression_problems'])}"
            )
    return report


def _report_path(
    workspace: Path,
    model_id: str,
    *,
    generated_at: datetime,
    run_id: str,
) -> Path:
    safe_model_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in model_id)
    timestamp = generated_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_run_id = "".join(char for char in run_id.lower() if char in "0123456789abcdef")
    if len(safe_run_id) < 8:
        raise ValueError("eval run_id must contain at least 8 hexadecimal characters")
    return workspace / "eval" / f"{safe_model_id}-{timestamp}-{safe_run_id[:12]}.json"


__all__ = ["EvalCliError", "run_eval_llm_cli"]
