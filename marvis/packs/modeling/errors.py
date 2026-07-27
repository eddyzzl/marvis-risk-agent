from marvis.error_kinds import ErrorKind


class ModelingError(ValueError):
    pass


class ReportScoreMissingError(ModelingError):
    """Raised when generate_model_report has no way to obtain a real model score.

    Neither an explicit ``score`` column nor a trained ``artifact_id`` is available, so
    there is no model score to report on. Silently substituting the first feature column
    would compute plausible-looking KS/PSI/bin numbers with the wrong semantics (DOM-10) —
    the report is a formal deliverable, so this must fail loudly instead.
    """

    def __init__(self, *, experiment_id: str, dataset_id: str) -> None:
        self.experiment_id = str(experiment_id)
        self.dataset_id = str(dataset_id)
        super().__init__(
            f"experiment {self.experiment_id!r} has no artifact and dataset "
            f"{self.dataset_id!r} has no `score` column; cannot generate score-based "
            "report sections. Train a model for this experiment first, or register a "
            "dataset that already carries a `score` column."
        )

    def to_detail(self) -> dict:
        """Structured diagnostics (never parsed from free text)."""
        return {
            "kind": ErrorKind.REPORT_SCORE_MISSING,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
        }


class SpecialValueDecisionRequiredError(ModelingError):
    """Raised when detected special values lack a complete, explicit policy.

    The error is deliberately typed so Agent/manual UIs can render the exact
    columns, detected values and allowed actions as a human-in-the-loop gate.
    AUTO mode cannot turn this into an implicit confirmation: the workflow must
    be resumed with a concrete decision for every affected selected feature.
    """

    def __init__(
        self,
        *,
        columns: list[str],
        sentinel_columns: dict,
        problems: dict[str, str] | None = None,
    ) -> None:
        self.columns = [str(column) for column in columns]
        self.sentinel_columns = dict(sentinel_columns)
        self.problems = dict(problems or {})
        preview = "、".join(self.columns[:12])
        suffix = f" 等 {len(self.columns)} 列" if len(self.columns) > 12 else ""
        super().__init__(
            "检测到需要治理的哨兵/特殊值，但决策尚未完整确认："
            f"{preview}{suffix}。请逐列选择 mask（转为空值）、retain（保留并说明原因）"
            "或 drop（剔除特征）；retain 必须由用户明确确认。"
        )

    def to_detail(self) -> dict:
        return {
            "kind": ErrorKind.SPECIAL_VALUE_DECISION_REQUIRED,
            "columns": list(self.columns),
            "sentinel_columns": dict(self.sentinel_columns),
            "problems": dict(self.problems),
            "allowed_actions": ["mask", "retain", "drop"],
            "human_confirmation_required_for": ["retain"],
        }


__all__ = [
    "ModelingError",
    "ReportScoreMissingError",
    "SpecialValueDecisionRequiredError",
]
