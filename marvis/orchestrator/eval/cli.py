"""CLI-facing helpers for ``marvis eval-llm`` (LLM-2).

Kept separate from ``marvis/__main__.py`` (which only does argument parsing +
thin dispatch) so this module can be imported and unit tested without
importing argparse/CLI plumbing.

Runs the *production* eval framework against a real, settings-configured LLM
model: builds an ``EvalOrchestrator`` backed by ``OpenAICompatibleLLMClient``,
runs ``calibrate_tier_for_model`` over ``INITIAL_EVAL_CASES``, writes a JSON
report to ``workspace/eval/{model_id}-{date}.json``, and -- when a baseline
report path is given -- runs ``regression_gate`` against it.

Not wired into CI (this makes real network calls to a configured local
inference server); it is the production-facing verification tool LLM-2 asked
for: "does my model swap regress" answered by an actual run, not a guess.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

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
    """Run the full eval suite against a real, settings-configured model.

    Returns the report dict that was also written to disk. Raises
    ``EvalCliError`` (mapped to a non-zero exit by the caller) when a
    baseline is given and the regression gate fails.
    """
    profile = resolve_llm_model(workspace, model_id)
    resolved_model_id = str(profile.get("model_id") or model_id or "unknown")

    def llm_factory():
        return OpenAICompatibleLLMClient(profile)

    orchestrator = EvalOrchestrator(llm_factory)
    cases = list(initial_eval_cases())
    report = calibrate_tier_for_model(resolved_model_id, cases, orchestrator=orchestrator)
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["case_ids"] = [case.id for case in cases]

    recommended = report.get("recommended_tier")
    if recommended is not None:
        report["overall_pass_rate"] = report["per_tier"][recommended]["pass_rate"]
        report["guardrail_pass_rate"] = report["per_tier"][recommended]["guardrail_pass_rate"]

    out_path = _report_path(workspace, resolved_model_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(out_path)

    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        ok, problems = regression_gate(baseline, report)
        report["regression_ok"] = ok
        report["regression_problems"] = problems
        if not ok:
            raise EvalCliError(
                f"eval regression against {baseline_path}: {'; '.join(problems)}"
            )
    return report


def _report_path(workspace: Path, model_id: str) -> Path:
    safe_model_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in model_id)
    date = datetime.now(UTC).strftime("%Y%m%d")
    return workspace / "eval" / f"{safe_model_id}-{date}.json"


__all__ = ["EvalCliError", "run_eval_llm_cli"]
