from enum import StrEnum

from agent_contracts import AgentCandidate, AgentCapabilityMetadata

__all__ = [
    "AgentCandidate",
    "AgentCapability",
    "AgentCapabilityMetadata",
    "CapabilityUnavailableError",
]


class AgentCapability(StrEnum):
    GENERAL = "general"
    RESEARCH = "research"


class CapabilityUnavailableError(Exception):
    """Raised when no currently eligible agent is available to satisfy a requested capability."""

    def __init__(
        self,
        capability: AgentCapability | str,
        message: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self.capability = (
            capability
            if isinstance(capability, AgentCapability)
            else AgentCapability(str(capability).lower())
            if str(capability).lower() in [c.value for c in AgentCapability]
            else str(capability)
        )
        self.task_id = task_id
        detail = (
            message
            or f"No eligible agent is currently available for capability '{self.capability}'"
        )
        super().__init__(detail)
        self.message = detail
