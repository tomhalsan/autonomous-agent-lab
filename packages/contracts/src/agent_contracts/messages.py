import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


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
class AgentEvent:
    agent_id: str
    state: str
    detail: str
    occurred_at: str
    task_id: str | None = None
    routing_decision: dict[str, Any] | None = None
    fallback_transition: FallbackTransition | None = None
    inference_telemetry: InferenceExecutionTelemetry | None = None
    version: int = 1

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

    @classmethod
    def create(
        cls,
        agent_id: str,
        state: str,
        detail: str,
        task_id: str | None = None,
        routing_decision: dict[str, Any] | None = None,
        fallback_transition: FallbackTransition | None = None,
        inference_telemetry: InferenceExecutionTelemetry | None = None,
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
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.routing_decision is None:
            data.pop("routing_decision", None)
        if self.fallback_transition is None:
            data.pop("fallback_transition", None)
        if self.inference_telemetry is None:
            data.pop("inference_telemetry", None)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AgentEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "AgentEvent":
        return cls(**json.loads(value))
