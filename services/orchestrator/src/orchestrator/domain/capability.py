from enum import StrEnum

from agent_contracts import AgentCandidate, AgentCapabilityMetadata

__all__ = ["AgentCandidate", "AgentCapability", "AgentCapabilityMetadata"]


class AgentCapability(StrEnum):
    GENERAL = "general"
    RESEARCH = "research"
