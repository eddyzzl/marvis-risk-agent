"""Strict deidentified reference protocol; provider claims are not external truth."""

from datetime import UTC, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from marvis.decision_twin._canonical import canonical_json as canonical_json
from marvis.decision_twin._canonical import content_hash as content_hash

Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SCHEMA = "risk-source.response.v1"
PROVIDER_VERSION = "reference.v1"


class SourceError(ValueError):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def instant(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO string") from exc
    if parsed.utcoffset() is None:
        raise ValueError("timestamp timezone required")
    return parsed.astimezone(UTC).timestamp()


def iso(value):
    return datetime.fromtimestamp(value, UTC).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourceProfile(Contract):
    profile_id: Id
    mode: Literal["reference_http", "historical_file"]
    endpoint: str | None = None
    provider_version: Literal["reference.v1"] = PROVIDER_VERSION
    schema_version: Literal["risk-source.response.v1"] = SCHEMA
    requests_per_minute: int = Field(default=30, ge=1, le=120)
    timeout_seconds: float = Field(default=2.0, ge=0.05, le=10)
    breaker_threshold: int = Field(default=3, ge=1, le=10)
    breaker_seconds: int = Field(default=30, ge=1, le=300)

    @model_validator(mode="after")
    def address(self):
        if self.mode == "historical_file":
            if self.endpoint is not None:
                raise ValueError("historical profile has no network endpoint")
        else:
            url = urlsplit(self.endpoint or "")
            if (
                url.scheme != "http"
                or url.hostname != "127.0.0.1"
                or not url.port
                or url.username
                or url.password
                or url.path not in ("", "/")
                or url.query
                or url.fragment
            ):
                raise ValueError("reference endpoint must be http://127.0.0.1:port")
        return self


class RegisterProfile(Contract):
    profile: SourceProfile
    token: SecretStr | None = None


class SourceGrant(Contract):
    grant_id: Id
    profile_id: Id
    task_id: Id
    grantee_id: Id
    subject_namespace: Id
    # Tokens must be generated upstream, never raw ID numbers or contact strings.
    subject_token: Hash
    purpose: Literal["kyc", "credit_review", "fraud_review"]
    basis_artifact_id: Id
    starts_at: str
    expires_at: str

    @model_validator(mode="after")
    def times(self):
        if instant(self.starts_at) >= instant(self.expires_at):
            raise ValueError("grant interval must be nonempty")
        return self


class HistoricalSource(Contract):
    dataset_id: Id
    expected_content_hash: Hash
    row_id_column: Id
    row_id: Id
    response_column: Id


class SourceQuery(Contract):
    request_id: Id
    grant_id: Id
    expires_in_seconds: int = Field(default=300, ge=1, le=600)
    historical: HistoricalSource | None = None


class AuthorizationBasis(Contract):
    basis_id: Id
    reference: Id
    declaration: Literal[
        "synthetic_reference_test",
        "deidentified_history_review",
        "documented_authorization",
    ]


class ProviderResult(Contract):
    request_id: Hash
    request_hash: Hash
    status: Literal["found", "no_record"]
    envelope: "SourceEnvelope | None"

    @model_validator(mode="after")
    def record(self):
        if (self.status == "found") != (self.envelope is not None):
            raise ValueError("provider status and record disagree")
        return self


class KycClaims(Contract):
    identity_match: Literal["match", "mismatch", "unknown", "no_record"] = "unknown"
    document_status: Literal["valid", "invalid", "expired", "unknown", "no_record"] = (
        "unknown"
    )
    liveness: Literal["pass", "fail", "unknown", "no_record"] = "unknown"
    watchlist: Literal["hit", "clear", "unknown", "no_record"] = "unknown"


class CreditAccount(Contract):
    account_token: Hash
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    # Integer minor units avoid implicit rounding, NaN and currency conversion.
    balance_minor: int | None = Field(default=None, ge=0, le=10**15)
    past_due_minor: int | None = Field(default=None, ge=0, le=10**15)
    days_past_due: int | None = Field(default=None, ge=0, le=36500)


class CreditInquiry(Contract):
    inquiry_id: Id
    queried_at: str
    category: Literal["application", "review", "other", "unknown"]

    @field_validator("queried_at")
    @classmethod
    def time(cls, value):
        instant(value)
        return value


class BureauClaims(Contract):
    coverage: Literal["complete", "partial", "unknown", "no_record"] = "unknown"
    accounts: list[CreditAccount] = Field(default_factory=list, max_length=1000)
    inquiries: list[CreditInquiry] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def unique(self):
        if len({a.account_token for a in self.accounts}) != len(self.accounts) or len(
            {q.inquiry_id for q in self.inquiries}
        ) != len(self.inquiries):
            raise ValueError("duplicate credit record identity")
        if self.coverage == "no_record" and (self.accounts or self.inquiries):
            raise ValueError("no_record cannot contain credit records")
        return self


class SourceEnvelope(Contract):
    schema_version: Literal["risk-source.response.v1"] = SCHEMA
    provider_version: Literal["reference.v1"] = PROVIDER_VERSION
    subject_namespace: Id
    subject_token: Hash
    record_version: int = Field(ge=1)
    observed_at: str
    available_at: str
    expires_at: str
    kyc: KycClaims = Field(default_factory=KycClaims)
    bureau: BureauClaims = Field(default_factory=BureauClaims)

    @model_validator(mode="after")
    def times(self):
        if (
            not instant(self.observed_at)
            <= instant(self.available_at)
            < instant(self.expires_at)
        ):
            raise ValueError("source timestamps out of order")
        if any(
            instant(q.queried_at) > instant(self.observed_at)
            for q in self.bureau.inquiries
        ):
            raise ValueError("inquiry cannot follow source observation")
        return self


class ProviderQuery(Contract):
    request_id: Hash
    subject_namespace: Id
    subject_token: Hash
    purpose: Literal["kyc", "credit_review", "fraud_review"]
    provider_version: Literal["reference.v1"] = PROVIDER_VERSION
    schema_version: Literal["risk-source.response.v1"] = SCHEMA
    expires_at: str

    @field_validator("expires_at")
    @classmethod
    def time(cls, value):
        instant(value)
        return value


def assess(envelope: SourceEnvelope, now: float):
    """No composite identity pass, inferred debt zero, or LLM-derived metric."""
    reasons = []
    if instant(envelope.available_at) > now:
        reasons.append("not_yet_available")
    if instant(envelope.expires_at) <= now:
        reasons.append("expired")
    kyc = envelope.kyc.model_dump()
    for dimension, value in kyc.items():
        if value in {"unknown", "no_record"}:
            reasons.append(f"kyc.{dimension}:{value}")
    bureau = envelope.bureau
    if bureau.coverage != "complete":
        reasons.append(f"bureau:{bureau.coverage}")
    totals = {}
    for account in bureau.accounts:
        amount = totals.setdefault(
            account.currency, {"balance_minor": 0, "past_due_minor": 0}
        )
        for field in amount:
            value = getattr(account, field)
            if value is None:
                reasons.append(f"bureau.{field}:missing")
                amount[field] = None
            elif amount[field] is not None:
                amount[field] += value
    complete = bureau.coverage == "complete"
    return {
        "status": "expired"
        if "expired" in reasons
        else ("unknown" if "not_yet_available" in reasons else "available"),
        "kyc_claims": kyc,
        "bureau_coverage": bureau.coverage,
        "account_count": len(bureau.accounts) if complete else None,
        "inquiry_count": len(bureau.inquiries) if complete else None,
        "totals_by_currency": totals if complete else None,
        "missing_reasons": sorted(set(reasons)),
        "finding_codes": sorted(
            f"kyc.{key}:{value}"
            for key, value in kyc.items()
            if value in {"mismatch", "invalid", "expired", "fail", "hit"}
        ),
        "automated_clearance": False,
    }


ProviderResult.model_rebuild()
