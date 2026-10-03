#!/usr/bin/env python3
"""Recompute retained V2 evidence against independently held bindings.

Exit 0 means the requested checks agree, never that audit acceptance is closed.
Without a final-run manifest this checks domain evidence only; execution budgets,
the original model connection and independent business acceptance remain unproved.
With final records, only the selected case's domain is recomputed; other cases'
recorded outcomes are retained but their domain results are not revalidated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from marvis.orchestrator.eval.runtime_archive_validation import (  # noqa: E402
    FrozenValidationBinding,
    revalidate_validation_archive,
)
from marvis.orchestrator.eval.runtime_contracts import digest  # noqa: E402


class ReviewInputError(ValueError):
    """Only fixed reason codes, never submitted values or parser diagnostics."""


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return value


def _load_binding(path: Path, expected_digest: str) -> FrozenValidationBinding:
    with path.open("rb") as handle:
        raw = handle.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ReviewInputError("binding_size_limit")
    if digest(raw) != expected_digest:
        raise ReviewInputError("binding_digest_mismatch")
    try:
        return FrozenValidationBinding.model_validate_json(raw)
    except ValueError as exc:
        raise ReviewInputError("binding_schema_invalid") from exc


def _write_output(path: Path, payload: str, *, protected_paths: tuple[Path, ...]) -> None:
    target = path.absolute()
    resolved = target.resolve()
    if any(resolved == item.resolve() or resolved.is_relative_to(item.resolve()) for item in protected_paths):
        raise ReviewInputError("output_overlaps_source_evidence")
    # No replacement or chmod of an existing file, including symbolic links.
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def _final_run_binding(run_dir: Path, expected_digest: str, binding: FrozenValidationBinding,
                       custody_digest: str) -> dict:
    from marvis.orchestrator.eval.runtime_run_manifest import RunManifestUnavailable, verify_run_manifest

    try:
        verified = verify_run_manifest(run_dir, expected_manifest_sha256=expected_digest)
    except RunManifestUnavailable:
        return {"status": "unavailable", "reason": "original_final_run_manifest_missing"}
    manifest = verified["manifest"]
    frozen = manifest["frozen"]
    if manifest["run_id"] != binding.run_id or any(
        frozen.get(key) != getattr(binding, key)
        for key in ("cases_sha256", "expected_sha256", "source", "model_connection_sha256")
    ):
        raise ReviewInputError("final_run_identity_mismatch")
    case = verified["executions"].get(binding.case.id)
    if not isinstance(case, dict) or case.get("case_sha256") != digest(binding.case.model_dump()):
        raise ReviewInputError("final_case_identity_mismatch")
    if case.get("budget") != binding.case.budget.model_dump():
        raise ReviewInputError("final_case_budget_mismatch")
    custody = case.get("evidence_custody", {})
    if custody.get("status") != "retained" or custody.get("manifest_sha256") != custody_digest:
        raise ReviewInputError("final_custody_digest_mismatch")
    execution = case.get("execution", {})
    pipeline = execution.get("validation_pipeline", {})
    if execution.get("task_id") != binding.task_id or any(
        pipeline.get(key) != getattr(binding, key)
        for key in ("input_contract_sha256", "confirmed_draft_sha256", "report_revision")
    ):
        raise ReviewInputError("final_task_contract_or_confirmation_mismatch")
    report_hashes = {"word": None, "excel": None}
    kinds = set()
    for item in pipeline.get("report_files", []):
        kind = item.get("kind")
        if kind not in report_hashes or kind in kinds:
            raise ReviewInputError("final_report_binding_ambiguous")
        kinds.add(kind)
        report_hashes[kind] = item.get("sha256")
    if report_hashes != binding.report_sha256:
        raise ReviewInputError("final_report_download_mismatch")
    score = verified["scores"][binding.case.id]
    successful_record = case.get("runtime_status") == "completed" and score.get("passed") is True
    return {
        "status": "bound", "manifest_sha256": expected_digest,
        "scope": "complete_run_file_consistency_and_selected_case_domain_binding; recorded_outcomes_not_recomputed",
        "recorded_runtime_status": case.get("runtime_status"),
        "recorded_case_passed": score.get("passed") is True,
        "recorded_run_passed": verified["report"].get("all_passed") is True,
        "run_denominator": manifest["denominator"],
        "recorded_case_failure": not successful_record,
        "independent_runtime_authentication": "not_established",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--binding", required=True, type=Path,
                        help="Independently retained FrozenValidationBinding JSON")
    parser.add_argument("--binding-sha256", required=True, type=_sha256,
                        help="Binding file digest held outside the archive")
    parser.add_argument("--run-dir", type=Path, help="Optional original final run records")
    parser.add_argument("--run-manifest-sha256", type=_sha256,
                        help="Independently held digest of the original final-run manifest")
    parser.add_argument("--output", type=Path, help="Optional new private JSON file; never overwritten")
    args = parser.parse_args(argv)
    if bool(args.run_dir) != bool(args.run_manifest_sha256):
        parser.error("--run-dir and --run-manifest-sha256 must be supplied together")
    try:
        if args.binding.resolve().is_relative_to(args.archive.resolve()):
            raise ReviewInputError("binding_must_be_outside_archive")
        binding = _load_binding(args.binding, args.binding_sha256)
        final_run = (_final_run_binding(args.run_dir, args.run_manifest_sha256, binding, args.manifest_sha256)
                     if args.run_dir else {"status": "unavailable", "reason": "no_final_run_manifest_supplied"})
        result = revalidate_validation_archive(
            args.archive, expected_manifest_sha256=args.manifest_sha256, frozen_binding=binding,
        )
        result.update(
            requested_scope="selected_case_domain_and_final_run_records" if args.run_dir else "domain_only",
            binding_file_sha256=args.binding_sha256,
            final_runtime_evidence=final_run,
        )
        result["requested_checks_verified"] = (
            result["supported_checks_verified"]
            and (not args.run_dir or final_run.get("status") == "bound")
            and not final_run.get("recorded_case_failure", False)
        )
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            protected = (args.archive, args.binding, args.run_dir) if args.run_dir else (args.archive, args.binding)
            _write_output(args.output, payload, protected_paths=protected)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # Validation errors may embed submitted JSON. Emit no raw exception or
        # material content, including malformed externally supplied bindings.
        reason = (str(exc) if isinstance(exc, ReviewInputError) else
                  "output_already_exists" if isinstance(exc, FileExistsError) else "input_or_output_rejected")
        print(json.dumps({"error": "archive_review_input_or_output_rejected", "error_type": type(exc).__name__, "reason": reason,
                          "source_authentication": "not_established", "acceptance_claim": "not_established"}))
        return 2
    print(payload)
    return int(not result["requested_checks_verified"])


if __name__ == "__main__":
    raise SystemExit(main())
