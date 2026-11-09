import json
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


@dataclass(frozen=True)
class AgentEvent:
    agent_id: str
    state: str
    detail: str
    occurred_at: str
    task_id: str | None = None
    routing_decision: dict[str, Any] | None = None
    fallback_transition: FallbackTransition | None = None
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

    @classmethod
    def create(
        cls,
        agent_id: str,
        state: str,
        detail: str,
        task_id: str | None = None,
        routing_decision: dict[str, Any] | None = None,
        fallback_transition: FallbackTransition | None = None,
    ) -> "AgentEvent":
        return cls(
            agent_id=agent_id,
            state=state,
            detail=detail,
            occurred_at=now_iso(),
            task_id=task_id,
            routing_decision=routing_decision,
            fallback_transition=fallback_transition,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.routing_decision is None:
            data.pop("routing_decision", None)
        if self.fallback_transition is None:
            data.pop("fallback_transition", None)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AgentEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "AgentEvent":
        return cls(**json.loads(value))
