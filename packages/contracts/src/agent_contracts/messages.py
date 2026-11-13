import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .tools import ToolErrorCode


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TaskRequested:
    task_id: str
    prompt: str
    created_at: str
    version: int = 1

    @classmethod
    def create(cls, prompt: str) -> "TaskRequested":
        return cls(task_id=str(uuid4()), prompt=prompt.strip(), created_at=now_iso())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TaskRequested":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "TaskRequested":
        return cls(**json.loads(value))


@dataclass(frozen=True)
class FallbackTransition:
    """Operational record of switching from a failed provider to a target provider."""

    failed_provider: str
    target_provider: str
    reason: str
    attempt_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.failed_provider, str) or not self.failed_provider.strip():
            raise ValueError("failed_provider must be a non-empty string")
        object.__setattr__(self, "failed_provider", self.failed_provider.strip().lower())

        if not isinstance(self.target_provider, str) or not self.target_provider.strip():
            raise ValueError("target_provider must be a non-empty string")
        object.__setattr__(self, "target_provider", self.target_provider.strip().lower())

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())

        if (
            type(self.attempt_index) is bool
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 0
        ):
            raise ValueError(
                f"attempt_index must be a non-negative integer, got {self.attempt_index!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FallbackTransition":
        return cls(**data)


VALID_INFERENCE_EXECUTION_STATUSES = frozenset({
    "succeeded",
    "fallback_eligible_failure",
    "non_fallback_failure",
    "unexpected_failure",
})


@dataclass(frozen=True)
class InferenceExecutionTelemetry:
    """Sanitized operational execution telemetry for an individual inference attempt."""

    provider_name: str
    attempt_index: int
    latency_ms: float
    status: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    model_name: str | None = None
    failure_reason: str | None = None
    status_code: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider_name, str) or not self.provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        object.__setattr__(self, "provider_name", self.provider_name.strip().lower())

        if (
            type(self.attempt_index) is bool
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 0
        ):
            raise ValueError(
                f"attempt_index must be a non-negative integer, got {self.attempt_index!r}"
            )

        if (
            type(self.latency_ms) is bool
            or not isinstance(self.latency_ms, (int, float))
            or self.latency_ms < 0
        ):
            raise ValueError(
                f"latency_ms must be a non-negative number, got {self.latency_ms!r}"
            )
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string")
        norm_status = self.status.strip().lower()
        if norm_status not in VALID_INFERENCE_EXECUTION_STATUSES:
            raise ValueError(
                f"Invalid status: {self.status!r}. Must be one of: {sorted(VALID_INFERENCE_EXECUTION_STATUSES)}"
            )
        object.__setattr__(self, "status", norm_status)

        for token_field, val in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if val is not None:
                if type(val) is bool or not isinstance(val, int):
                    raise TypeError(f"{token_field} must be an int or None, got {type(val).__name__}")
                if val < 0:
                    raise ValueError(f"{token_field} must be a non-negative integer, got {val}")

        if self.model_name is not None:
            if not isinstance(self.model_name, str) or not self.model_name.strip():
                raise ValueError("model_name must be a non-empty string if specified")
            object.__setattr__(self, "model_name", self.model_name.strip())

        if self.finish_reason is not None:
            if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
                raise ValueError("finish_reason must be a non-empty string if specified")
            object.__setattr__(self, "finish_reason", self.finish_reason.strip().lower())

        if self.status_code is not None:
            if type(self.status_code) is bool or not isinstance(self.status_code, int):
                raise TypeError(
                    f"status_code must be an int or None, got {type(self.status_code).__name__}"
                )
            if not (100 <= self.status_code <= 599):
                raise ValueError(
                    f"status_code must be a valid HTTP status (100-599), got {self.status_code}"
                )

        if norm_status == "succeeded":
            if self.failure_reason is not None:
                raise ValueError("succeeded status requires failure_reason to be None")
            if self.status_code is not None:
                raise ValueError("succeeded status requires status_code to be None")
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError(f"{norm_status} status requires a non-empty failure_reason")
            if not re.fullmatch(r"[a-z0-9_]+", self.failure_reason):
                raise ValueError(
                    "failure_reason must contain only lowercase letters, digits, and underscores"
                )
            object.__setattr__(self, "failure_reason", self.failure_reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceExecutionTelemetry":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "InferenceExecutionTelemetry":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ToolExecutionTelemetry:
    """Sanitized operational measurements for one terminal tool execution."""

    tool_name: str
    agent_id: str
    call_id: str
    duration_ms: float
    outcome: str
    argument_size_bytes: int | None
    result_size_bytes: int | None
    task_id: str | None = None
    started_at: str | None = None
    error_code: ToolErrorCode | None = None

    def __post_init__(self) -> None:
        for field_name in ("tool_name", "agent_id", "call_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, (int, float)):
            raise TypeError("duration_ms must be a finite non-negative number")
        try:
            is_finite_duration = math.isfinite(self.duration_ms)
        except OverflowError:
            is_finite_duration = False
        if not is_finite_duration or self.duration_ms < 0:
            raise ValueError("duration_ms must be a finite non-negative number")
        object.__setattr__(self, "duration_ms", float(self.duration_ms))

        if not isinstance(self.outcome, str) or self.outcome not in {"succeeded", "failed"}:
            raise ValueError("outcome must be 'succeeded' or 'failed'")

        for field_name in ("argument_size_bytes", "result_size_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None")

        for field_name in ("task_id", "started_at"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")

        error_code = self.error_code
        if isinstance(error_code, str) and not isinstance(error_code, ToolErrorCode):
            try:
                error_code = ToolErrorCode(error_code)
            except ValueError as exc:
                raise ValueError(f"Unknown tool error code {error_code!r}") from exc
            object.__setattr__(self, "error_code", error_code)
        elif error_code is not None and not isinstance(error_code, ToolErrorCode):
            raise TypeError("error_code must be a ToolErrorCode, string, or None")

        if self.outcome == "succeeded" and self.error_code is not None:
            raise ValueError("successful telemetry must not include an error_code")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "tool_name": self.tool_name,
            "agent_id": self.agent_id,
            "call_id": self.call_id,
            "duration_ms": self.duration_ms,
            "argument_size_bytes": self.argument_size_bytes,
            "result_size_bytes": self.result_size_bytes,
            "outcome": self.outcome,
        }
        if self.task_id is not None:
            data["task_id"] = self.task_id
        if self.started_at is not None:
            data["started_at"] = self.started_at
        if self.error_code is not None:
            data["error_code"] = self.error_code.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolExecutionTelemetry":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "ToolExecutionTelemetry":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class AgentEvent:
    agent_id: str
    state: str | None
    detail: str
    occurred_at: str
    task_id: str | None = None
    routing_decision: dict[str, Any] | None = None
    fallback_transition: FallbackTransition | None = None
    inference_telemetry: InferenceExecutionTelemetry | None = None
    version: int = 1
    tool_execution_telemetry: ToolExecutionTelemetry | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fallback_transition, dict):
            object.__setattr__(
                self,
                "fallback_transition",
                FallbackTransition.from_dict(self.fallback_transition),
            )
        elif self.fallback_transition is not None and not isinstance(
            self.fallback_transition, FallbackTransition
        ):
            raise TypeError(
                f"fallback_transition must be FallbackTransition, dict, or None, got {type(self.fallback_transition).__name__}"
            )

        if isinstance(self.inference_telemetry, dict):
            object.__setattr__(
                self,
                "inference_telemetry",
                InferenceExecutionTelemetry.from_dict(self.inference_telemetry),
            )
        elif self.inference_telemetry is not None and not isinstance(
            self.inference_telemetry, InferenceExecutionTelemetry
        ):
            raise TypeError(
                f"inference_telemetry must be InferenceExecutionTelemetry, dict, or None, got {type(self.inference_telemetry).__name__}"
            )

        if isinstance(self.tool_execution_telemetry, dict):
            object.__setattr__(
                self,
                "tool_execution_telemetry",
                ToolExecutionTelemetry.from_dict(self.tool_execution_telemetry),
            )
        elif self.tool_execution_telemetry is not None and not isinstance(
            self.tool_execution_telemetry, ToolExecutionTelemetry
        ):
            raise TypeError(
                "tool_execution_telemetry must be ToolExecutionTelemetry, dict, or None"
            )

        if (
            self.tool_execution_telemetry is not None
            and self.tool_execution_telemetry.agent_id != self.agent_id
        ):
            raise ValueError("tool execution telemetry agent_id must match the event agent_id")
        if self.tool_execution_telemetry is not None and self.state is not None:
            raise ValueError("tool execution telemetry events must not carry agent state")
        if self.state is None and self.tool_execution_telemetry is None:
            raise ValueError("events without state require tool execution telemetry")

    @classmethod
    def create(
        cls,
        agent_id: str,
        state: str | None,
        detail: str,
        task_id: str | None = None,
        routing_decision: dict[str, Any] | None = None,
        fallback_transition: FallbackTransition | None = None,
        inference_telemetry: InferenceExecutionTelemetry | None = None,
        tool_execution_telemetry: ToolExecutionTelemetry | None = None,
    ) -> "AgentEvent":
        return cls(
            agent_id=agent_id,
            state=state,
            detail=detail,
            occurred_at=now_iso(),
            task_id=task_id,
            routing_decision=routing_decision,
            fallback_transition=fallback_transition,
            inference_telemetry=inference_telemetry,
            tool_execution_telemetry=tool_execution_telemetry,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.state is None:
            data.pop("state", None)
        if self.routing_decision is None:
            data.pop("routing_decision", None)
        if self.fallback_transition is None:
            data.pop("fallback_transition", None)
        if self.inference_telemetry is None:
            data.pop("inference_telemetry", None)
        if self.tool_execution_telemetry is None:
            data.pop("tool_execution_telemetry", None)
        else:
            data["tool_execution_telemetry"] = self.tool_execution_telemetry.to_dict()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvent":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed_fields = {
            "agent_id",
            "state",
            "detail",
            "occurred_at",
            "task_id",
            "routing_decision",
            "fallback_transition",
            "inference_telemetry",
            "version",
            "tool_execution_telemetry",
        }
        unexpected_fields = data.keys() - allowed_fields
        if unexpected_fields:
            raise TypeError(f"unexpected AgentEvent fields: {sorted(unexpected_fields)}")
        return cls(
            agent_id=data["agent_id"],
            state=data.get("state"),
            detail=data["detail"],
            occurred_at=data["occurred_at"],
            task_id=data.get("task_id"),
            routing_decision=data.get("routing_decision"),
            fallback_transition=data.get("fallback_transition"),
            inference_telemetry=data.get("inference_telemetry"),
            version=data.get("version", 1),
            tool_execution_telemetry=data.get("tool_execution_telemetry"),
        )

    @classmethod
    def from_json(cls, value: str) -> "AgentEvent":
        return cls.from_dict(json.loads(value))
