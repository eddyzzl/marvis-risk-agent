import math
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from marvis.domain import TASK_TYPE_VALIDATION


StrictJsonScalar = StrictStr | StrictInt | StrictFloat | StrictBool | None
StrictRatio = Annotated[StrictInt | StrictFloat, Field(ge=0.0, le=1.0)]
StrictNonNegativeNumber = Annotated[StrictInt | StrictFloat, Field(ge=0.0)]
StrictNonEmptyStr = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1),
]
StrictSha256 = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


def _canonical_non_empty_string(value: str) -> str:
    if value == "" or value != value.strip() or "\x00" in value:
        raise ValueError("must be canonical non-empty text")
    return value


StrictCanonicalNonEmptyStr = Annotated[
    StrictStr,
    AfterValidator(_canonical_non_empty_string),
]
DataWorkspacePage = Literal[
    "overview",
    "fields",
    "semantics",
    "history",
    "statistics",
]


class DataSemanticMappingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    target_col: StrictCanonicalNonEmptyStr | None
    field_roles: dict[StrictCanonicalNonEmptyStr, StrictCanonicalNonEmptyStr]
    business_names: dict[StrictCanonicalNonEmptyStr, StrictCanonicalNonEmptyStr]


class DataWorkspaceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    active_dataset_id: StrictCanonicalNonEmptyStr | None
    active_dataset_content_hash: StrictSha256 | None
    page: DataWorkspacePage
    selected_field: StrictCanonicalNonEmptyStr | None
    semantic_mapping: DataSemanticMappingRequest


class DataWorkspaceSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["data-workspace.v1"]
    task_id: StrictNonEmptyStr
    revision: StrictInt = Field(ge=0)
    active_dataset_id: StrictNonEmptyStr | None
    active_dataset_content_hash: StrictSha256 | None
    analysis_generation: StrictInt = Field(ge=0)
    page: DataWorkspacePage
    selected_field: StrictNonEmptyStr | None
    semantic_mapping: DataSemanticMappingRequest
    updated_at: StrictNonEmptyStr


class StrategyProfitInputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ead_col: StrictNonEmptyStr
    pd_col: StrictNonEmptyStr
    annual_rate: StrictRatio
    funding_rate: StrictRatio
    lgd: StrictRatio
    operating_cost_per_loan: StrictNonNegativeNumber
    term_months: StrictInt = Field(ge=1)


class StrategyTaskInputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_mode: Literal["strategy_development", "strategy_analysis"] = "strategy_development"
    strategy_type: Literal[
        "approval", "reject", "limit", "pricing", "segmentation"
    ] = "approval"
    objective: Literal["", "max_profit", "max_approval"] = ""
    max_bad_rate: StrictRatio | None = None
    min_approval_rate: StrictRatio | None = None
    baseline_strategy_id: StrictNonEmptyStr | None = None
    profit: StrategyProfitInputRequest | None = None


class CreateTaskRequest(BaseModel):
    task_type: str = TASK_TYPE_VALIDATION
    model_name: str
    model_version: str = ""
    validator: str
    source_dir: str
    algorithm: str = ""
    target_col: str = "y"
    score_col: str = "pred"
    split_col: str = "split"
    time_col: str = "apply_month"
    run_mode: str = "manual"
    feature_columns: list[str] = Field(default_factory=list)
    target_type: str = ""
    recipes: list[str] = Field(default_factory=list)
    sample_weight_col: str = ""
    # AGT-4 (optional, modeling tasks only): None/absent → no success criterion is
    # injected into the plan. Never defaulted to a platform-chosen number.
    oot_ks_min: float | None = None
    strategy_input: StrategyTaskInputRequest | None = None
    metrics: list[str] = Field(default_factory=list)
    # Per-task capability tier (conservative/balanced/aggressive); "" → global default.
    capability_tier: str = ""
    notebook_path: str | None = None
    sample_path: str | None = None
    pmml_path: str | None = None
    dictionary_path: str | None = None
    report_values: dict[str, str] = Field(default_factory=dict)


class ValidateRequest(BaseModel):
    feature_columns: list[str] | None = None


class MaterialSelectionRequest(BaseModel):
    notebook_path: str
    sample_path: str
    pmml_path: str
    dictionary_path: str


class ValidationInputConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: StrictInt = Field(gt=0)
    target_col: str
    positive_label: StrictJsonScalar
    negative_label: StrictJsonScalar = None
    split_col: str
    split_value_mapping: dict[str, StrictJsonScalar]
    time_col: str
    time_granularity: str
    pmml_output_field: str
    model_params: dict[str, Any]
    metadata_sheet: str | None = None
    feature_col: str
    category_col: str
    importance_col: str
    transformations: list[dict[str, Any]] = Field(default_factory=list)


class ExecutionEnvironmentRequest(BaseModel):
    execution_mode: str = "jupyter_kernel"
    kernel_name: str = "python3"
    conda_env_name: str = ""
    python_executable: str = ""
    # Soft RSS cap (MB) for the notebook kernel; None / 0 / omitted = no cap.
    notebook_memory_limit_mb: int | None = None


class LLMSettingsRequest(BaseModel):
    default_model_id: str = ""
    capability_tier: str = ""
    # LLM-4: caller-role -> model_id routing (e.g. {"planner": "model-a",
    # "gate": "model-b"}); unmapped roles fall back to default_model_id.
    role_overrides: dict[str, str] = Field(default_factory=dict)
    models: list[dict] = Field(default_factory=list)


class LLMConnectionTestRequest(BaseModel):
    # GAP-8: test a candidate profile before it is saved (e.g. from the "add
    # model" dialog) by supplying the connection fields directly; alternatively
    # pass model_id to test an already-saved model (has_api_key must be true
    # for that model, since its api_key is never round-tripped to the client).
    model_id: str = ""
    api_base_url: str = ""
    model_name: str = ""
    api_key: str = ""
    timeout_seconds: int = 10


class MemoryPolicyRequest(BaseModel):
    reference_cross_task: bool = True
    auto_distill: bool = True


class ReportFieldsUpdateRequest(BaseModel):
    text_values: dict[str, str] = Field(default_factory=dict)


ManualStrategyWorkflow = Literal[
    "univariate_candidate_analysis",
    "cross_matrix_analysis",
    "automatic_tree_candidate_build",
    "univariate_candidate_refinement",
    "scorecard_band_build",
    "scorecard_cutoff_selection",
    "candidate_monthly_stability",
    "voting_candidate_search",
    "voting_candidate_build_from_search",
    "interactive_tree_revision",
    "interactive_tree_frontier_group_materialization",
    "interactive_tree_frontier_materialization",
    "strategy_pool_apply",
]

ManualUnivariateRefinementMethod = Literal[
    "equal_frequency",
    "equal_width",
    "chimerge",
    "tree",
    "manual",
    "categorical",
]
ManualUnivariateAnalysisMethod = Literal[
    "equal_frequency",
    "equal_width",
    "chimerge",
    "tree",
    "manual",
]
ManualCandidateId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^candidate-[0-9a-f]{32}$"),
]
ManualScorecardAssetId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^scorecard-band-asset-[0-9a-f]{32}$"),
]
ManualScorecardCutoffId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^scorecard-cutoff-[0-9a-f]{32}$"),
]
ManualCandidateAssetId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^candidate-asset-[0-9a-f]{32}$"),
]
ManualPoolEntryId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^pool-entry-[0-9a-f]{32}$"),
]
ManualInteractiveTreeSourceId = Annotated[
    StrictStr,
    StringConstraints(
        pattern=(
            r"^(?:candidate-asset-[0-9a-f]{32}|"
            r"interactive-tree-revision-[0-9a-f]{32})$"
        )
    ),
]
ManualInteractiveTreeNodeId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^node-[0-9a-f]{20}$"),
]
ManualInteractiveTreeRevisionId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^interactive-tree-revision-[0-9a-f]{32}$"),
]
ManualInteractiveTreeFrontierNodeId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^(?:node|leaf)-[0-9a-f]{20}$"),
]
ManualVotingRuleId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^candidate-rule-[0-9a-f]{32}$"),
]
ManualVotingSearchId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^voting-search-[0-9a-f]{32}$"),
]
ManualVotingComboId = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^voting-combo-[0-9a-f]{32}$"),
]
ManualVotingMetric = Literal[
    "hit_count",
    "hit_share",
    "good_count",
    "bad_count",
    "bad_rate",
    "lift",
    "bad_capture_rate",
    "weighted_hit_total",
    "weighted_hit_share",
    "weighted_good_total",
    "weighted_bad_total",
    "weighted_bad_rate",
    "weighted_bad_capture_rate",
    "hit_amount",
    "hit_amount_share",
    "good_amount",
    "bad_amount",
    "bad_amount_rate",
    "bad_amount_capture_rate",
]
_MANUAL_VOTING_RATE_METRICS = frozenset(
    {
        "hit_share",
        "bad_rate",
        "bad_capture_rate",
        "weighted_hit_share",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount_share",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }
)
_MANUAL_VOTING_REQUIRED_MINIMUM_SHARE = {
    "bad_rate": "hit_share",
    "weighted_bad_rate": "weighted_hit_share",
    "bad_amount_rate": "hit_amount_share",
}
ManualStrategyType = Literal[
    "approval",
    "reject",
    "limit",
    "pricing",
    "segmentation",
]
ManualSelectionReason = Annotated[
    StrictStr,
    AfterValidator(_canonical_non_empty_string),
    StringConstraints(max_length=500),
]


def _finite_strictly_increasing_numbers(
    values: list[int | float],
) -> list[int | float]:
    previous: float | None = None
    for value in values:
        if isinstance(value, int) and abs(value) > 2**53 - 1:
            raise ValueError("integer exceeds the exact JSON number range")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("numbers must be finite")
        if previous is not None and previous >= number:
            raise ValueError("numbers must be strictly increasing and unique")
        previous = number
    return values


def _unique_strings(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError("values must be unique")
    return values


def _raw_pd_band_edges(
    values: list[int | float],
) -> list[int | float]:
    normalized = _finite_strictly_increasing_numbers(values)
    if float(normalized[0]) != 0.0 or float(normalized[-1]) != 1.0:
        raise ValueError("raw_pd_band_edges must start at 0 and end at 1")
    return normalized


ManualBreakpointList = Annotated[
    list[StrictInt | StrictFloat],
    Field(min_length=1, max_length=19),
    AfterValidator(_finite_strictly_increasing_numbers),
]
ManualRawPdBandEdges = Annotated[
    list[StrictInt | StrictFloat],
    Field(min_length=3, max_length=21),
    AfterValidator(_raw_pd_band_edges),
]
ManualBinId = Annotated[
    StrictStr,
    AfterValidator(_canonical_non_empty_string),
    StringConstraints(max_length=128),
]
ManualMergeGroup = Annotated[
    list[ManualBinId],
    Field(min_length=2, max_length=20),
    AfterValidator(_unique_strings),
]


class ManualRiskThresholdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    operator: Literal[">=", ">", "<=", "<"]
    value: StrictRatio


class ManualScorecardBandBuildInputs(BaseModel):
    """Only user-owned scorecard band controls cross the manual API."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    bin_count: StrictInt | None = Field(default=None, ge=2, le=20)
    raw_pd_band_edges: ManualRawPdBandEdges | None = None

    @model_validator(mode="after")
    def validate_optional_and_mutually_exclusive_controls(self) -> Self:
        explicit_nulls = sorted(
            field
            for field in self.model_fields_set
            if getattr(self, field) is None
        )
        if explicit_nulls:
            raise ValueError(
                "optional fields must be omitted instead of null: "
                + ", ".join(explicit_nulls)
            )
        if self.bin_count is not None and self.raw_pd_band_edges is not None:
            raise ValueError(
                "bin_count and raw_pd_band_edges are mutually exclusive"
            )
        return self


class ManualScorecardCutoffSelectionInputs(BaseModel):
    """One visible asset/cutoff pointer plus an optional operator reason."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    asset_id: ManualScorecardAssetId
    cutoff_id: ManualScorecardCutoffId
    reason: ManualSelectionReason | None = None

    @model_validator(mode="after")
    def reject_explicit_null_reason(self) -> Self:
        if "reason" in self.model_fields_set and self.reason is None:
            raise ValueError("optional fields must be omitted instead of null: reason")
        return self


class ManualCandidateStabilityAssetInputs(BaseModel):
    """One visible standalone univariate candidate pointer."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    asset_id: ManualCandidateAssetId


class ManualCandidateStabilityPoolEntryInputs(BaseModel):
    """One visible entry in the current task-owned Strategy Pool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy_type: Literal[
        "approval",
        "reject",
        "limit",
        "pricing",
        "segmentation",
    ]
    entry_id: ManualPoolEntryId


ManualCandidateStabilityInputs = (
    ManualCandidateStabilityAssetInputs
    | ManualCandidateStabilityPoolEntryInputs
)


class ManualInteractiveTreeRevisionInputs(BaseModel):
    """One visible tree/node pointer for an immutable prune revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_tree_id: ManualInteractiveTreeSourceId
    node_id: ManualInteractiveTreeNodeId
    operation: Literal["prune_subtree"]
    reason: ManualSelectionReason | None = None

    @model_validator(mode="after")
    def reject_explicit_null_reason(self) -> Self:
        if "reason" in self.model_fields_set and self.reason is None:
            raise ValueError("optional fields must be omitted instead of null: reason")
        return self


class ManualInteractiveTreeFrontierMaterializationInputs(BaseModel):
    """One visible revision/frontier pointer for immutable materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    revision_id: ManualInteractiveTreeRevisionId
    source_node_id: ManualInteractiveTreeFrontierNodeId
    selection_reason: ManualSelectionReason | None = None

    @model_validator(mode="after")
    def reject_explicit_null_reason(self) -> Self:
        if (
            "selection_reason" in self.model_fields_set
            and self.selection_reason is None
        ):
            raise ValueError(
                "optional fields must be omitted instead of null: selection_reason"
            )
        return self


class ManualInteractiveTreeFrontierGroupMaterializationInputs(BaseModel):
    """Visible revision/frontier pointers for one immutable OR group."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    revision_id: ManualInteractiveTreeRevisionId
    source_node_ids: Annotated[
        list[ManualInteractiveTreeFrontierNodeId],
        Field(min_length=2, max_length=50),
    ]
    selection_reason: ManualSelectionReason | None = None

    @model_validator(mode="after")
    def reject_invalid_optional_reason_or_duplicate_members(self) -> Self:
        if (
            "selection_reason" in self.model_fields_set
            and self.selection_reason is None
        ):
            raise ValueError(
                "optional fields must be omitted instead of null: selection_reason"
            )
        if len(self.source_node_ids) != len(set(self.source_node_ids)):
            raise ValueError("source_node_ids cannot contain duplicate node ids")
        return self


class ManualVotingObjective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    metric: ManualVotingMetric
    direction: Literal["maximize", "minimize"]


class ManualVotingConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    metric: ManualVotingMetric
    operator: Literal["gte", "lte"]
    value: Annotated[StrictInt | StrictFloat, Field(ge=0.0)]


class ManualVotingCandidateSearchInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy_type: ManualStrategyType
    member_count: StrictInt = Field(ge=2, le=50)
    n: StrictInt = Field(ge=1, le=50)
    objective: ManualVotingObjective
    constraints: Annotated[
        list[ManualVotingConstraint],
        Field(max_length=32),
    ] = Field(default_factory=list)
    include_rule_ids: Annotated[
        list[ManualVotingRuleId],
        Field(max_length=50),
        AfterValidator(_unique_strings),
    ] = Field(default_factory=list)
    exclude_rule_ids: Annotated[
        list[ManualVotingRuleId],
        Field(max_length=50),
        AfterValidator(_unique_strings),
    ] = Field(default_factory=list)
    max_combinations: StrictInt = Field(default=10_000, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_relational_controls(self) -> Self:
        if self.n > self.member_count:
            raise ValueError("n cannot exceed member_count")
        if len(self.include_rule_ids) > self.member_count:
            raise ValueError("include_rule_ids cannot exceed member_count")
        if set(self.include_rule_ids) & set(self.exclude_rule_ids):
            raise ValueError("include_rule_ids and exclude_rule_ids cannot overlap")
        identities = [
            (constraint.metric, constraint.operator)
            for constraint in self.constraints
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("constraints cannot repeat metric/operator")
        for constraint in self.constraints:
            try:
                number = float(constraint.value)
            except OverflowError as exc:
                raise ValueError("constraint values must be finite") from exc
            if not math.isfinite(number):
                raise ValueError("constraint values must be finite")
            if (
                constraint.metric in _MANUAL_VOTING_RATE_METRICS
                and number > 1.0
            ):
                raise ValueError("rate constraint values cannot exceed 1")
        required_share = _MANUAL_VOTING_REQUIRED_MINIMUM_SHARE.get(
            self.objective.metric
        )
        if self.objective.direction == "minimize" and required_share is not None:
            has_positive_share = any(
                constraint.metric == required_share
                and constraint.operator == "gte"
                and float(constraint.value) > 0.0
                for constraint in self.constraints
            )
            if not has_positive_share:
                raise ValueError(
                    "minimizing a rate requires its positive share constraint"
                )
        return self


class ManualVotingCandidateBuildFromSearchInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    search_id: ManualVotingSearchId
    combo_id: ManualVotingComboId
    strategy_type: ManualStrategyType | None = None

    @model_validator(mode="after")
    def reject_explicit_null_strategy_type(self) -> Self:
        if "strategy_type" in self.model_fields_set and self.strategy_type is None:
            raise ValueError(
                "optional fields must be omitted instead of null: strategy_type"
            )
        return self


class ManualStrategyPoolApplyInputs(BaseModel):
    """Only the Pool type and optional safe column prefix are user-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy_type: ManualStrategyType
    output_prefix: Annotated[
        StrictStr,
        StringConstraints(
            pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,47}$",
            max_length=48,
        ),
    ] | None = None

    @model_validator(mode="after")
    def reject_explicit_null_output_prefix(self) -> Self:
        if "output_prefix" in self.model_fields_set and self.output_prefix is None:
            raise ValueError(
                "optional fields must be omitted instead of null: output_prefix"
            )
        return self


class ManualRiskThresholdSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    risk_threshold: ManualRiskThresholdRequest


class ManualSourceBinSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_bin_ids: Annotated[
        list[ManualBinId],
        Field(min_length=1, max_length=50),
        AfterValidator(_unique_strings),
    ]


class ManualFreshUnivariateRefinementInputs(BaseModel):
    """User controls for re-analysing one feature before refinement."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    feature: StrictCanonicalNonEmptyStr
    method: ManualUnivariateRefinementMethod
    selection: ManualRiskThresholdSelectionRequest
    features: Annotated[
        list[StrictCanonicalNonEmptyStr],
        Field(min_length=1, max_length=50),
    ] | None = None
    methods: Annotated[
        list[ManualUnivariateAnalysisMethod],
        Field(min_length=1, max_length=5),
    ] | None = None
    bin_count: StrictInt | None = Field(default=None, ge=3, le=20)
    min_bin_pct: Annotated[
        StrictInt | StrictFloat,
        Field(ge=0.0, le=0.5),
    ] | None = None
    loan_amount_col: StrictCanonicalNonEmptyStr | None = None
    overdue_amount_col: StrictCanonicalNonEmptyStr | None = None
    sentinel_values: Annotated[
        list[StrictStr | StrictInt | StrictFloat],
        Field(max_length=20),
    ] | None = None
    manual_breakpoints: (
        dict[StrictCanonicalNonEmptyStr, ManualBreakpointList] | None
    ) = None
    selection_reason: ManualSelectionReason | None = None

    @model_validator(mode="after")
    def reject_explicit_nulls(self) -> Self:
        explicit_nulls = sorted(
            field
            for field in self.model_fields_set
            if getattr(self, field) is None
        )
        if explicit_nulls:
            raise ValueError(
                "optional fields must be omitted instead of null: "
                + ", ".join(explicit_nulls)
            )
        return self


class ManualExistingUnivariateRefinementInputs(BaseModel):
    """User controls bound to an immutable candidate the user inspected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    feature: StrictCanonicalNonEmptyStr
    method: ManualUnivariateRefinementMethod
    source_candidate_id: ManualCandidateId
    selection: ManualSourceBinSelectionRequest | ManualRiskThresholdSelectionRequest
    merge_groups: Annotated[
        list[ManualMergeGroup],
        Field(max_length=20),
    ] | None = None
    selection_reason: ManualSelectionReason | None = None

    @model_validator(mode="after")
    def validate_optional_fields_and_merge_groups(self) -> Self:
        explicit_nulls = sorted(
            field
            for field in self.model_fields_set
            if getattr(self, field) is None
        )
        if explicit_nulls:
            raise ValueError(
                "optional fields must be omitted instead of null: "
                + ", ".join(explicit_nulls)
            )
        if self.merge_groups is not None:
            flattened = [
                bin_id for group in self.merge_groups for bin_id in group
            ]
            if len(flattened) != len(set(flattened)):
                raise ValueError(
                    "merge_groups cannot reuse a source bin id across groups"
                )
        return self


ManualUnivariateRefinementInputs = (
    ManualFreshUnivariateRefinementInputs
    | ManualExistingUnivariateRefinementInputs
)
_MANUAL_UNIVARIATE_REFINEMENT_INPUTS = TypeAdapter(
    ManualUnivariateRefinementInputs
)
_MANUAL_SCORECARD_BAND_BUILD_INPUTS = TypeAdapter(
    ManualScorecardBandBuildInputs
)
_MANUAL_SCORECARD_CUTOFF_SELECTION_INPUTS = TypeAdapter(
    ManualScorecardCutoffSelectionInputs
)
_MANUAL_CANDIDATE_STABILITY_INPUTS = TypeAdapter(
    ManualCandidateStabilityInputs
)
_MANUAL_INTERACTIVE_TREE_REVISION_INPUTS = TypeAdapter(
    ManualInteractiveTreeRevisionInputs
)
_MANUAL_INTERACTIVE_TREE_FRONTIER_MATERIALIZATION_INPUTS = TypeAdapter(
    ManualInteractiveTreeFrontierMaterializationInputs
)
_MANUAL_INTERACTIVE_TREE_FRONTIER_GROUP_MATERIALIZATION_INPUTS = TypeAdapter(
    ManualInteractiveTreeFrontierGroupMaterializationInputs
)
_MANUAL_VOTING_CANDIDATE_SEARCH_INPUTS = TypeAdapter(
    ManualVotingCandidateSearchInputs
)
_MANUAL_VOTING_CANDIDATE_BUILD_FROM_SEARCH_INPUTS = TypeAdapter(
    ManualVotingCandidateBuildFromSearchInputs
)
_MANUAL_STRATEGY_POOL_APPLY_INPUTS = TypeAdapter(
    ManualStrategyPoolApplyInputs
)

_MANUAL_STRATEGY_PLATFORM_FIELDS = frozenset(
    {
        "analysis_generation",
        "artifact_id",
        "artifact_ids",
        "candidate_id",
        "content_hash",
        "dataset_id",
        "dataset_ref",
        "evidence_hash",
        "expected_artifact_content_hash",
        "expected_content_hash",
        "hashes",
        "metric",
        "metrics",
        "output",
        "outputs",
        "refs",
        "result",
        "results",
        "revision",
        "revision_id",
        "revisions",
        "sample_design_ref",
        "semantic_mapping_hash",
        "source_artifact_id",
        "task_id",
        "target_col",
        "workspace_revision",
    }
)


class ManualStrategyRequest(BaseModel):
    """User-owned controls for a deterministic Candidate Lab workflow.

    The envelope deliberately carries the compiler's canonical request shape,
    but leaves workflow-specific validation to ``validate_strategy_request``.
    Platform bindings and calculated evidence never cross this API boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_kind: Literal["standard_workflow"]
    workflow: ManualStrategyWorkflow
    workflow_inputs: dict[StrictCanonicalNonEmptyStr, Any]

    @model_validator(mode="after")
    def reject_platform_owned_inputs(self) -> Self:
        if self.workflow == "univariate_candidate_refinement":
            _MANUAL_UNIVARIATE_REFINEMENT_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "scorecard_band_build":
            _MANUAL_SCORECARD_BAND_BUILD_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "scorecard_cutoff_selection":
            _MANUAL_SCORECARD_CUTOFF_SELECTION_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "candidate_monthly_stability":
            _MANUAL_CANDIDATE_STABILITY_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "interactive_tree_revision":
            _MANUAL_INTERACTIVE_TREE_REVISION_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "interactive_tree_frontier_materialization":
            _MANUAL_INTERACTIVE_TREE_FRONTIER_MATERIALIZATION_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "interactive_tree_frontier_group_materialization":
            (
                _MANUAL_INTERACTIVE_TREE_FRONTIER_GROUP_MATERIALIZATION_INPUTS
                .validate_python(
                    self.workflow_inputs,
                    strict=True,
                )
            )
        elif self.workflow == "voting_candidate_search":
            _MANUAL_VOTING_CANDIDATE_SEARCH_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "voting_candidate_build_from_search":
            _MANUAL_VOTING_CANDIDATE_BUILD_FROM_SEARCH_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        elif self.workflow == "strategy_pool_apply":
            _MANUAL_STRATEGY_POOL_APPLY_INPUTS.validate_python(
                self.workflow_inputs,
                strict=True,
            )
        user_owned_identity_fields = {
            "univariate_candidate_refinement": {"source_candidate_id"},
            "scorecard_cutoff_selection": {"asset_id", "cutoff_id"},
            "candidate_monthly_stability": {"asset_id", "entry_id"},
            "interactive_tree_revision": {"source_tree_id", "node_id"},
            "interactive_tree_frontier_materialization": {
                "revision_id",
                "source_node_id",
            },
            "interactive_tree_frontier_group_materialization": {
                "revision_id",
                "source_node_ids",
            },
            "voting_candidate_search": {
                "include_rule_ids",
                "exclude_rule_ids",
            },
            "voting_candidate_build_from_search": {
                "search_id",
                "combo_id",
            },
        }.get(self.workflow, set())
        forbidden = sorted(
            key
            for key in self.workflow_inputs
            if key not in user_owned_identity_fields
            and (
                key in _MANUAL_STRATEGY_PLATFORM_FIELDS
                or key.startswith(("artifact_", "dataset_", "expected_", "workspace_"))
                or key.endswith(
                    (
                        "_hash",
                        "_id",
                        "_ids",
                        "_metrics",
                        "_ref",
                        "_results",
                        "_revision",
                    )
                )
            )
        )
        if forbidden:
            raise ValueError(
                "strategy_request.workflow_inputs cannot include platform-owned "
                "fields: " + ", ".join(forbidden)
            )
        return self


class AgentMessageRequest(BaseModel):
    content: str
    model_id: str | None = None
    effort: str | None = None
    acceptance_mode: str | None = None
    # Structured continuation for a strategy setup clarification. This is kept
    # separate from free text so the backend never has to infer business targets
    # or constraints from the conversation.
    strategy_input: StrategyTaskInputRequest | None = None
    # Candidate Lab manual controls use the same canonical request and trusted
    # execution kernel as natural-language strategy requests. The free-text
    # content remains a user-visible action label, not executable business input.
    strategy_request: ManualStrategyRequest | None = None
    # Optional edited feature set from the §4 interactive screening table; when a
    # screening gate is confirmed this overrides the screen's proposed `selected`.
    selection: list[str] | None = None
    # Optional per-feature dedup strategy map (feature_id -> first|last) from the §4
    # join dedup picker; re-confirms confirm_join to resolve non-unique-key conflicts.
    dedup_strategies: dict[str, str] | None = None
    # Optional structured parameter overrides from manual controls (for example the
    # feature-screening leakage/missing thresholds). These bypass LLM text routing.
    adjust_params: dict[str, object] | None = None
    # Optional optimistic-lock token for structured gate controls. The frontend sends
    # the gate step id it rendered; the backend rejects stale tabs/buttons.
    expected_step_id: str | None = None


class AgentModelRequest(BaseModel):
    model_id: str | None = None
    effort: str | None = None
    acceptance_mode: str | None = None


class AgentReportDraftConfirmRequest(BaseModel):
    revision: int
    text_values: dict[str, str] = Field(default_factory=dict)


def model_payload(payload: BaseModel) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()
