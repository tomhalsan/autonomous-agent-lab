"""Provider-neutral contracts for tool authorization and security audit decisions."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .tools import ToolOperationType


class ToolAuthorizationOutcome(str, Enum):
    """Whether a proposed tool invocation is permitted by its trusted scope."""

    ALLOWED = "allowed"
    DENIED = "denied"


class ToolAuthorizationReason(str, Enum):
    """Stable, non-sensitive reasons an invocation could not be authorized."""

    IDENTITY_MISSING = "identity_missing"
    PERMISSION_SCOPE_MISSING = "permission_scope_missing"
    AUTHORIZATION_INPUT_INVALID = "authorization_input_invalid"
    OPERATION_TYPE_UNCLASSIFIED = "operation_type_unclassified"
    TOOL_GRANT_REQUIRED = "tool_grant_required"
    MUTATION_GRANT_REQUIRED = "mutation_grant_required"
    AUTHORIZATION_UNAVAILABLE = "authorization_unavailable"


class ToolAuditDeliveryOutcome(str, Enum):
    """Delivery outcome for the required authorization audit event."""

    NOT_ATTEMPTED = "not_attempted"
    DELIVERED = "delivered"
    FAILED = "failed"


class ToolExecutionState(str, Enum):
    """Whether the coordinator crossed the execution boundary."""

    NOT_ATTEMPTED = "not_attempted"
    ATTEMPTED = "attempted"


def _parse_enum(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, Enum) or not isinstance(value, str):
        raise TypeError(f"{field_name} must be a {enum_type.__name__} or string")
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unknown {field_name} {value!r}") from exc


@dataclass(frozen=True)
class ToolPermissionGrant:
    """Permission for one exact tool identity and one operation type."""

    tool_name: str
    operation_type: ToolOperationType

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        object.__setattr__(
            self,
            "operation_type",
            _parse_enum(self.operation_type, ToolOperationType, "operation_type"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_name": self.tool_name,
            "operation_type": self.operation_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPermissionGrant":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(
            tool_name=data["tool_name"],
            operation_type=data["operation_type"],
        )


@dataclass(frozen=True, init=False)
class ToolPermissionScope:
    """Immutable set of exact-tool grants supplied by trusted runtime context."""

    grants: tuple[ToolPermissionGrant, ...]

    def __init__(self, grants: Iterable[ToolPermissionGrant]) -> None:
        if isinstance(grants, (str, bytes)) or not isinstance(grants, Iterable):
            raise TypeError("grants must be an iterable of ToolPermissionGrant values")
        grant_tuple = tuple(grants)
        if any(not isinstance(grant, ToolPermissionGrant) for grant in grant_tuple):
            raise TypeError("grants must contain only ToolPermissionGrant values")
        if len(set(grant_tuple)) != len(grant_tuple):
            raise ValueError("grants must not contain duplicates")
        object.__setattr__(self, "grants", grant_tuple)

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"grants": [grant.to_dict() for grant in self.grants]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPermissionScope":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        raw_grants = data["grants"]
        if not isinstance(raw_grants, list):
            raise TypeError("grants must be a list")
        return cls(ToolPermissionGrant.from_dict(grant) for grant in raw_grants)


@dataclass(frozen=True)
class ToolAuthorizationContext:
    """Trusted runtime identity and permission scope, separate from tool metadata."""

    agent_id: str | None
    permission_scope: ToolPermissionScope | None

    def __post_init__(self) -> None:
        if self.agent_id is not None and not isinstance(self.agent_id, str):
            raise TypeError("agent_id must be a string or None")
        if self.permission_scope is not None and not isinstance(
            self.permission_scope, ToolPermissionScope
        ):
            raise TypeError("permission_scope must be a ToolPermissionScope or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "permission_scope": (
                self.permission_scope.to_dict() if self.permission_scope is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolAuthorizationContext":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        raw_scope = data.get("permission_scope")
        return cls(
            agent_id=data.get("agent_id"),
            permission_scope=(
                ToolPermissionScope.from_dict(raw_scope) if raw_scope is not None else None
            ),
        )


@dataclass(frozen=True)
class ToolAuthorizationDecision:
    """Authorization result with a stable reason and safe identifying context."""

    outcome: ToolAuthorizationOutcome
    agent_id: str | None
    tool_name: str
    operation_type: ToolOperationType | None
    reason_code: ToolAuthorizationReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            _parse_enum(self.outcome, ToolAuthorizationOutcome, "outcome"),
        )
        if self.agent_id is not None and not isinstance(self.agent_id, str):
            raise TypeError("agent_id must be a string or None")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if self.operation_type is not None:
            object.__setattr__(
                self,
                "operation_type",
                _parse_enum(self.operation_type, ToolOperationType, "operation_type"),
            )
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                _parse_enum(self.reason_code, ToolAuthorizationReason, "reason_code"),
            )
        if self.outcome is ToolAuthorizationOutcome.ALLOWED and self.reason_code is not None:
            raise ValueError("allowed decisions must not have a denial reason")
        if self.outcome is ToolAuthorizationOutcome.DENIED and self.reason_code is None:
            raise ValueError("denied decisions require a denial reason")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "outcome": self.outcome.value,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "operation_type": (
                self.operation_type.value if self.operation_type is not None else None
            ),
            "reason_code": self.reason_code.value if self.reason_code is not None else None,
        }


@dataclass(frozen=True)
class ToolSecurityAuditEvent:
    """Sanitized authorization decision; deliberately excludes prompts and arguments."""

    call_id: str
    decision: ToolAuthorizationDecision
    execution_state: ToolExecutionState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.decision, ToolAuthorizationDecision):
            raise TypeError("decision must be a ToolAuthorizationDecision")
        if self.decision.outcome is ToolAuthorizationOutcome.DENIED:
            if self.execution_state is not ToolExecutionState.NOT_ATTEMPTED:
                raise ValueError("denial audit events must record execution as not attempted")
        elif self.execution_state is not None:
            raise ValueError("allowed audit events must not claim execution state")

    def to_dict(self) -> dict[str, str | None]:
        data: dict[str, str | None] = {"call_id": self.call_id, **self.decision.to_dict()}
        data["execution_state"] = (
            self.execution_state.value if self.execution_state is not None else None
        )
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
