"""Fixed source, registry and deployment protocol contracts."""
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marvis.reference_decision.contracts import FeatureField

Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PROTOCOL = "marvis.institution-adapter.v1"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class EndpointConfig(Contract):
    """Only trusted local startup configuration may construct this object."""
    alias: Identity
    instance_id: Identity
    environment: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    base_url: str
    credential_file: Path
    mode: Literal["local_reference", "institution_https"]
    ca_file: Path | None = None
    timeout_seconds: float = Field(default=3.0, ge=0.1, le=10)

    @model_validator(mode="after")
    def trusted_origin(self):
        url = urlsplit(self.base_url)
        if url.username or url.password or url.path not in ("", "/") or url.query or url.fragment or not url.hostname:
            raise ValueError("configured origin must contain no credentials, route or query")
        if self.mode == "local_reference":
            if url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port:
                raise ValueError("local reference origin must be explicit loopback HTTP")
        elif url.scheme != "https":
            raise ValueError("institution origin requires certificate-verified HTTPS")
        if self.environment == "local-reference":
            raise ValueError("the built-in local-reference environment is reserved")
        return self


class RegistryUpload(Contract):
    operation_id: Identity
    model_artifact_id: Identity
    raw_schema: list[FeatureField] = Field(min_length=1, max_length=5000)


class RegistryTransition(Contract):
    operation_id: Identity
    upload_operation_id: Identity
    promotion_id: Identity


class DeploymentInstall(Contract):
    operation_id: Identity
    promotion_id: Identity
    raw_schema: list[FeatureField] = Field(min_length=1, max_length=500)
    probe_features: dict = Field(max_length=500)


class DeploymentActivate(Contract):
    operation_id: Identity
    install_operation_id: Identity


class DeploymentRollback(Contract):
    operation_id: Identity
    deployment_operation_id: Identity
    expected_current_ref: Identity


class SourceRequest(Contract):
    operation_id: Identity
    grant_id: Identity
    expires_in_seconds: int = Field(default=300, ge=1, le=600)


class SourceGrantRevocation(Contract):
    grant_id: Identity


class RemoteMutation(Contract):
    protocol: Literal["marvis.institution-adapter.v1"] = PROTOCOL
    operation_id: Identity
    kind: Literal["registry_upload", "registry_promote", "registry_revoke", "deployment_install", "deployment_activate", "deployment_rollback", "source_query"]
    payload: dict


class RemoteReceipt(Contract):
    protocol: Literal["marvis.institution-adapter.v1"] = PROTOCOL
    instance_id: Identity
    operation_id: Identity
    request_hash: Hash
    kind: str
    state: Literal["completed", "rejected"]
    result: dict


class RemoteReadback(Contract):
    protocol: Literal["marvis.institution-adapter.v1"] = PROTOCOL
    instance_id: Identity
    resource: Literal["registry", "deployment"]
    identity: Identity
    payload: dict


class RemoteCapabilities(Contract):
    protocol: Literal["marvis.institution-adapter.v1"] = PROTOCOL
    instance_id: Identity
    registry_execution: Literal["opaque_native_storage_only"]
    deployment_execution: Literal["rule_only"]


class AdapterError(ValueError):
    def __init__(self, code, status=409, *, next_action="检查绑定和远端状态；不确定操作请显式读回"):
        self.code, self.status, self.next_action = code, status, next_action
        super().__init__(code)
