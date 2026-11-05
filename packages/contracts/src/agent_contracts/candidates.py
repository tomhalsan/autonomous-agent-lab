import json
from dataclasses import dataclass, field
from typing import Any

from .availability import AgentAvailability, AgentExecutionState
from .capabilities import DISALLOWED_METADATA_KEYS


@dataclass(frozen=True)
class AgentCandidate:
    """Structured candidate metadata for an eligible agent providing a required capability."""

    agent_id: str
    capabilities: list[str]
    availability: AgentAvailability
    agent_type: str = "agent"
    execution_state: AgentExecutionState = AgentExecutionState.IDLE
    providers: dict[str, str] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")

        if not isinstance(self.capabilities, list):
            raise TypeError("capabilities must be a list of strings")

        for cap in self.capabilities:
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError("Capability entries must be non-empty strings")

        if isinstance(self.execution_state, str) and not isinstance(
            self.execution_state, AgentExecutionState
        ):
            object.__setattr__(
                self, "execution_state", AgentExecutionState.from_str(self.execution_state)
            )
        elif not isinstance(self.execution_state, AgentExecutionState):
            raise TypeError(
                f"execution_state must be an AgentExecutionState, got {type(self.execution_state).__name__}"
            )

        if isinstance(self.availability, dict):
            object.__setattr__(
                self, "availability", AgentAvailability.from_dict(self.availability)
            )
        elif not isinstance(self.availability, AgentAvailability):
            raise TypeError(
                f"availability must be an AgentAvailability, got {type(self.availability).__name__}"
            )

        # Validate providers against sensitive keys
        for key in self.providers:
            normalized_key = key.lower()
            for disallowed in DISALLOWED_METADATA_KEYS:
                if disallowed in normalized_key:
                    raise ValueError(
                        f"Sensitive key {key!r} must not be exposed in provider metadata"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": list(self.capabilities),
            "execution_state": self.execution_state.value,
            "availability": self.availability.to_dict(),
            "providers": dict(self.providers),
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCandidate":
        raw_avail = data["availability"]
        availability = (
            raw_avail
            if isinstance(raw_avail, AgentAvailability)
            else AgentAvailability.from_dict(raw_avail)
        )
        return cls(
            agent_id=str(data["agent_id"]),
            agent_type=str(data.get("agent_type", "agent")),
            capabilities=list(data.get("capabilities", [])),
            execution_state=AgentExecutionState.from_str(str(data.get("execution_state", "idle"))),
            availability=availability,
            providers=dict(data.get("providers", {})),
            version=int(data.get("version", 1)),
        )

    @classmethod
    def from_json(cls, value: str) -> "AgentCandidate":
        return cls.from_dict(json.loads(value))

