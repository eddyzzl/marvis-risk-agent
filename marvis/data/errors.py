from __future__ import annotations


class DataLayerError(RuntimeError):
    """Base error for deterministic data-layer operations."""


class DataBackendError(DataLayerError):
    """Raised when the tabular backend cannot complete a data operation."""


class DataIngestError(DataLayerError):
    """Raised when a source file cannot be normalized into a dataset."""


class DedupRequiredError(DataLayerError):
    """Raised when a non-unique feature key needs a user-selected dedup strategy."""


class FanOutError(DataLayerError):
    """Raised when a join would expand beyond the anchor row count."""


class KeyDtypeMismatchError(DataLayerError):
    """T1-B8: raised when a proposed join key has a RED dtype divergence (one side text, the
    other float) and the caller has not acknowledged it. A text<->float key can silently lose
    precision / leading zeros and mis-match rows, so execute must not proceed silently. Mirrors
    DedupRequiredError's pattern; pass ``ack_dtype_mismatch=True`` to confirm past it."""

    def __init__(
        self,
        *,
        feature_dataset_id: str,
        divergences: list | None = None,
    ) -> None:
        self.feature_dataset_id = str(feature_dataset_id)
        self.divergences = list(divergences or [])
        columns = ", ".join(
            f"{getattr(d, 'anchor_col', '?')}={getattr(d, 'feature_col', '?')}"
            for d in self.divergences
        )
        super().__init__(
            f"feature {self.feature_dataset_id} join key has a dtype mismatch "
            f"(text vs float, possible precision/leading-zero loss): {columns or '?'}; "
            f"pass ack_dtype_mismatch=true to proceed anyway, or re-import the column as string"
        )

    def to_detail(self) -> dict:
        return {
            "kind": "key_dtype_mismatch",
            "feature_dataset_id": self.feature_dataset_id,
            "divergences": [
                {
                    "anchor_col": getattr(d, "anchor_col", None),
                    "feature_col": getattr(d, "feature_col", None),
                    "anchor_dtype": getattr(d, "anchor_dtype", None),
                    "feature_dtype": getattr(d, "feature_dtype", None),
                    "level": getattr(d, "level", None),
                }
                for d in self.divergences
            ],
        }


class JoinNotConfirmedError(DataLayerError):
    """Raised when executing a join plan before every join is confirmed."""


class NanLabelNotConfirmedError(DataLayerError):
    """Raised when a target column has NaN labels and the caller has not confirmed dropping them.

    A NaN target carries no supervision signal; it must NEVER be silently coerced
    to a class (INV-1 / INV-2). By default label-dependent operations stop and hand
    these structured diagnostics to the user, who chooses to drop the rows
    (``drop_nan_labels=True``) or fix the sample and retry.
    """

    def __init__(
        self,
        *,
        target_col: str,
        n_total: int,
        n_nan: int,
        scope: str = "dataset",
        by_split: dict | None = None,
    ) -> None:
        self.target_col = str(target_col)
        self.n_total = int(n_total)
        self.n_nan = int(n_nan)
        self.scope = str(scope)
        self.by_split = dict(by_split or {})
        super().__init__(
            f"target {self.target_col!r} has {self.n_nan}/{self.n_total} NaN labels "
            f"in {self.scope}; pass drop_nan_labels=true to drop these rows and continue, "
            f"or fix the sample and retry"
        )

    def to_detail(self) -> dict:
        """Structured diagnostics for the confirmation gate (never parsed from free text)."""
        return {
            "kind": "nan_label_not_confirmed",
            "target_col": self.target_col,
            "n_total": self.n_total,
            "n_nan": self.n_nan,
            "scope": self.scope,
            "by_split": self.by_split,
        }


class LabelSemanticsNotDeclaredError(DataLayerError):
    """T1-A1: raised by the strategy vintage path when the caller has not declared
    whether the bad column carries INCREMENTAL (每期当期新发生的坏, 会累加) or SNAPSHOT
    (每列/行=截至该MOB是否曾坏, 已单调) semantics.

    The vintage kernel ALWAYS accumulates the target across MOBs to form cum_bad_rate;
    on a snapshot/ever-bad flag that double-counts (a 3% rate can virtually balloon to
    36% over 12 MOBs) and only clips >1.0 in extreme cases -- a silent corruption. So
    the strategy tool refuses to guess: it stops and hands structured diagnostics to the
    user (mirrors NanLabelNotConfirmedError's typed-error + to_detail() pattern), who
    picks incremental or snapshot. ``monotone_heuristic`` reports whether the data looks
    snapshot-shaped (bad_count non-decreasing across every cohort), an advisory hint only.
    """

    def __init__(
        self,
        *,
        target_col: str,
        n_cohorts: int,
        monotone_heuristic: bool,
    ) -> None:
        self.target_col = str(target_col)
        self.n_cohorts = int(n_cohorts)
        self.monotone_heuristic = bool(monotone_heuristic)
        super().__init__(
            f"vintage bad column {self.target_col!r} has undeclared label_semantics; "
            f"pass label_semantics='incremental' (每期=当期新发生的坏, 会累加) or "
            f"'snapshot' (每列/行=截至该MOB是否曾坏, 已单调, 不再累加) and retry"
            + ("; data looks like a SNAPSHOT flag" if self.monotone_heuristic else "")
        )

    def to_detail(self) -> dict:
        """Structured diagnostics for the confirmation gate (never parsed from free text)."""
        return {
            "kind": "label_semantics_not_declared",
            "target_col": self.target_col,
            "n_cohorts": self.n_cohorts,
            "monotone_heuristic": self.monotone_heuristic,
            "examples": {
                "incremental": "每行=该MOB当期新发生的坏 (会逐期累加成 cum_bad_rate)",
                "snapshot": "每行/列=截至该MOB累计是否曾坏 (已单调, 不再累加, 直接读该期率)",
            },
        }


class CohortMaturityNotConfirmedError(DataLayerError):
    """C1: raised when constructing labels over放款 cohorts whose 表现期 has not
    closed to the 定坏判定 MOB, and the caller has not confirmed including them.

    An immature cohort (its loans haven't been observed through the required MOB) has
    an UNRELIABLE bad label: a loan that will go bad later is still counted as good,
    so including immature cohorts UNDERSTATES the bad rate. Rather than silently
    include them, the label tool stops and hands structured diagnostics to the user
    (mirrors NanLabelNotConfirmedError / LabelSemanticsNotDeclaredError's typed-error +
    ``to_detail()`` pattern), who either confirms including them anyway
    (``confirm_immature_cohorts=True``) or waits/backfills the performance window.
    """

    def __init__(
        self,
        *,
        required_mob: int,
        immature_cohorts: list[str],
        cohort_diagnostics: list[dict] | None = None,
    ) -> None:
        self.required_mob = int(required_mob)
        self.immature_cohorts = [str(cohort) for cohort in immature_cohorts]
        self.cohort_diagnostics = list(cohort_diagnostics or [])
        preview = ", ".join(self.immature_cohorts[:5])
        more = "" if len(self.immature_cohorts) <= 5 else f" (+{len(self.immature_cohorts) - 5} more)"
        super().__init__(
            f"{len(self.immature_cohorts)} cohort(s) have not matured to mob "
            f"{self.required_mob} for the bad definition ({preview}{more}); including "
            f"them will UNDERSTATE the bad rate. Pass confirm_immature_cohorts=true to "
            f"include them anyway, or wait for the performance window to close"
        )

    def to_detail(self) -> dict:
        """Structured diagnostics for the confirmation gate (never parsed from free text)."""
        return {
            "kind": "cohort_maturity_not_confirmed",
            "required_mob": self.required_mob,
            "immature_cohorts": self.immature_cohorts,
            "cohort_diagnostics": self.cohort_diagnostics,
        }


class ScoreDirectionConflictError(DataLayerError):
    """Raised when a declared/default score_direction contradicts the empirical
    corr(score, target) sign beyond the configured threshold (S1a determinism gate).

    Mirrors NanLabelNotConfirmedError's pattern: typed error + to_detail() payload,
    default is to stop and hand structured diagnostics to the user, who confirms
    (confirm_direction_conflict=True) to proceed with the declared direction anyway,
    or fixes score_direction and retries. Also covers the rule-based build_strategy
    case, where there is no single "declared" direction to confirm past -- see
    ``reason``/``conflicting_rules`` for that variant.
    """

    def __init__(
        self,
        *,
        tool: str,
        score_col: str,
        target_col: str | None = None,
        declared_direction: str | None = None,
        implied_direction: str | None = None,
        corr: float | None = None,
        n_labeled: int = 0,
        conflicting_rules: list[str] | None = None,
        reason: str | None = None,
    ) -> None:
        self.tool = str(tool)
        self.score_col = str(score_col)
        self.target_col = str(target_col) if target_col else None
        self.declared_direction = declared_direction
        self.implied_direction = implied_direction
        self.corr = float(corr) if corr is not None else None
        self.n_labeled = int(n_labeled)
        self.conflicting_rules = list(conflicting_rules or [])
        self.reason = reason
        super().__init__(
            f"{tool}: score_direction conflict on {score_col!r}"
            + (
                f" (declared={declared_direction}, data implies={implied_direction}, "
                f"corr={corr:.3f}, n={n_labeled})"
                if declared_direction
                else f" ({reason or 'rules disagree on direction'})"
            )
            + "; pass confirm_direction_conflict=true to proceed anyway, or fix score_direction/rules and retry"
        )

    def to_detail(self) -> dict:
        return {
            "kind": "score_direction_conflict",
            "tool": self.tool,
            "score_col": self.score_col,
            "target_col": self.target_col,
            "declared_direction": self.declared_direction,
            "implied_direction": self.implied_direction,
            "corr": self.corr,
            "n_labeled": self.n_labeled,
            "conflicting_rules": self.conflicting_rules,
            "reason": self.reason,
        }


class DataSecurityError(DataLayerError):
    """Raised when untrusted input would enter a backend query."""


class DatasetTooLargeError(DataIngestError):
    """TST-2: raised when an uploaded/registered file exceeds the configured
    size or row-count guardrail. Carries structured diagnostics (mirrors
    NanLabelNotConfirmedError's pattern) so the router can surface a 413 with
    both the configured limit and the actual size/row count, instead of a
    generic message the caller has to parse.
    """

    def __init__(
        self,
        *,
        reason: str,
        limit: int,
        actual: int | None = None,
        unit: str = "bytes",
    ) -> None:
        self.reason = str(reason)
        self.limit = int(limit)
        self.actual = None if actual is None else int(actual)
        self.unit = str(unit)
        actual_text = "unknown" if self.actual is None else str(self.actual)
        super().__init__(
            f"{self.reason}: limit={self.limit} {self.unit}, actual={actual_text} {self.unit}"
        )

    def to_detail(self) -> dict:
        return {
            "kind": "dataset_too_large",
            "reason": self.reason,
            "limit": self.limit,
            "actual": self.actual,
            "unit": self.unit,
        }


class InvalidDatasetPathError(DataSecurityError):
    """TST-2: raised when a local-path dataset registration request names a
    path that fails validation (does not exist, is not a regular file,
    unsupported extension, or escapes the expected location)."""


class PerformanceFrameError(DataLayerError):
    """S3: raised when a performance (表现期快照) frame violates its minimum
    column contract (missing column, unparseable snapshot month, bucket value
    outside the declared ``states``, or unparseable balance).

    Mirrors NanLabelNotConfirmedError's typed-error + ``to_detail()`` pattern so
    the structured diagnostics (中文 reason + 截断样例值) cross the subprocess
    boundary as data instead of free text. ``problem`` is the machine-readable
    sub-kind; ``missing_columns`` / ``column`` / ``samples`` are populated per
    problem so a caller (or a gate) never has to parse the message.
    """

    def __init__(
        self,
        *,
        reason: str,
        problem: str,
        missing_columns: list[str] | None = None,
        column: str | None = None,
        samples: list | None = None,
    ) -> None:
        self.reason = str(reason)
        self.problem = str(problem)
        self.missing_columns = list(missing_columns or [])
        self.column = str(column) if column else None
        self.samples = list(samples or [])
        super().__init__(self.reason)

    def to_detail(self) -> dict:
        return {
            "kind": "performance_frame_invalid",
            "problem": self.problem,
            "reason": self.reason,
            "missing_columns": self.missing_columns,
            "column": self.column,
            "samples": [str(sample) for sample in self.samples],
        }


__all__ = [
    "CohortMaturityNotConfirmedError",
    "DataBackendError",
    "DataIngestError",
    "DataLayerError",
    "DataSecurityError",
    "DatasetTooLargeError",
    "DedupRequiredError",
    "FanOutError",
    "InvalidDatasetPathError",
    "JoinNotConfirmedError",
    "KeyDtypeMismatchError",
    "LabelSemanticsNotDeclaredError",
    "NanLabelNotConfirmedError",
    "PerformanceFrameError",
    "ScoreDirectionConflictError",
]
