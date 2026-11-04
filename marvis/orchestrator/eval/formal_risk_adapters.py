"""Fixed risk-family dispatch for externally bound portfolio and Vintage originals."""
from pydantic import TypeAdapter

from .acceptance_trust import TrustInputError, _require
from .runtime_archive_portfolio import FrozenPortfolioBinding
from .runtime_archive_vintage import FrozenVintageBinding, revalidate_vintage_archive


class FrozenRiskBinding:
    """Parse only installed, strict schemas; the archive cannot choose a reader."""

    @staticmethod
    def model_validate_json(raw):
        return TypeAdapter(FrozenPortfolioBinding | FrozenVintageBinding).validate_json(raw)


def vintage_domain(case, expected, item, archive, binding, observed):
    from .formal_acceptance import _domain_result
    from .formal_domain_adapters import _final_native_binding, _predicate_value

    _require(isinstance(binding, FrozenVintageBinding), "formal_vintage_binding_required")
    _require(binding.case.task.task_type == "vintage", "formal_vintage_task_mismatch")
    tool = "strategy.vintage_curve"
    _final_native_binding(binding, observed, native_tools={step: tool for step in binding.step_ids.values()})
    recomputed = revalidate_vintage_archive(
        archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding,
    )
    answers = []
    if recomputed["supported_checks_verified"]:
        for assertion in expected.assertions:
            if assertion.kind == "tool_succeeded" and assertion.tool == tool:
                answers.append(recomputed["checks"]["native_steps"]["status"] == "verified")
            elif assertion.kind in {"output_equals", "output_close", "output_length"} and assertion.tool == tool:
                answers.append(_predicate_value(assertion, recomputed["values"]))
            elif assertion.kind == "http_status":
                codes = [event.get("status_code") for event in observed.get("http_events", [])
                         if event.get("stage") == assertion.stage]
                answers.append(bool(codes) and all(type(code) is type(assertion.value) and code == assertion.value
                                                  for code in codes))
            else:
                raise TrustInputError("formal_vintage_predicate_unavailable")
    return _domain_result(recomputed, expected.result == "done" and bool(answers) and all(answers)), None
