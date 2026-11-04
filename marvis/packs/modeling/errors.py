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


_PMML_ENVIRONMENT_CAUSES = frozenset({
    "unsupported_library_version",
    "java_unavailable",
    "java_unsupported",
})


class PmmlConverterError(ModelingError):
    """Raised when the sklearn2pmml/JPMML converter (a Java subprocess) fails.

    sklearn2pmml itself only raises a generic RuntimeError and prints the
    converter's diagnosis to stdout, so the real cause never reached the user.
    This error carries the converter's own exception and suggested solution.
    ``environment_issue`` marks causes that no model or data change can fix --
    the converter rejecting an installed library version, or no usable Java --
    so the user is told to repair the runtime instead of retrying blindly.
    """

    def __init__(
        self,
        *,
        cause: str,
        converter_version: str = "",
        library: str = "",
        installed_version: str = "",
        java_exception: str = "",
        java_message: str = "",
        solution: str = "",
        output_tail: str = "",
    ) -> None:
        self.cause = str(cause)
        self.converter_version = str(converter_version)
        self.library = str(library)
        self.installed_version = str(installed_version)
        self.java_exception = str(java_exception)
        self.java_message = str(java_message)
        self.solution = str(solution)
        self.output_tail = str(output_tail)
        super().__init__(self._message())

    @property
    def environment_issue(self) -> bool:
        return self.cause in _PMML_ENVIRONMENT_CAUSES

    def _message(self) -> str:
        converter = f"sklearn2pmml {self.converter_version}".strip()
        if self.cause == "unsupported_library_version":
            advice = f"（转换器建议：{self.solution}）" if self.solution else ""
            return (
                f"PMML 导出环境不兼容：{converter} 内置的 JPMML 转换器不支持已安装的 "
                f"{self.library} {self.installed_version}{advice}。请把运行环境恢复到 "
                "pyproject.toml / uv.lock 锁定的依赖版本后重试；这不是模型或数据问题。"
            )
        if self.cause == "java_unavailable":
            return (
                "PMML 导出需要 Java，但 MARVIS 进程的 PATH 上找不到 java 可执行文件。"
                "请安装 Java 11 及以上（部署清单推荐 17）并把 java 加入 PATH 后重试；"
                "只设置 JAVA_HOME 不够。"
            )
        if self.cause == "java_unsupported":
            return (
                f"PMML 导出所用的 Java 版本过旧，无法运行 {converter} 的 JPMML 转换器："
                f"{self.java_message}。请改用 Java 11 及以上（部署清单推荐 17）后重试。"
            )
        reason = (
            f"{self.java_exception}: {self.java_message}"
            if self.java_exception
            else self.java_message or "转换器没有给出可解析的异常"
        )
        text = f"PMML 转换失败（{converter} / JPMML）：{reason}"
        return f"{text}；转换器建议：{self.solution}" if self.solution else text

    def to_detail(self) -> dict:
        return {
            "kind": (
                ErrorKind.PMML_EXPORT_ENVIRONMENT
                if self.environment_issue
                else ErrorKind.PMML_CONVERSION_FAILED
            ),
            "cause": self.cause,
            "environment_issue": self.environment_issue,
            "converter": "sklearn2pmml",
            "converter_version": self.converter_version,
            "library": self.library,
            "installed_version": self.installed_version,
            "java_exception": self.java_exception,
            "java_message": self.java_message,
            "solution": self.solution,
            "output_tail": self.output_tail,
        }


__all__ = [
    "ModelingError",
    "PmmlConverterError",
    "ReportScoreMissingError",
    "SpecialValueDecisionRequiredError",
]
