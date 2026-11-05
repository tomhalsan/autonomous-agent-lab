from enum import StrEnum

from agent_contracts import AgentCapabilityMetadata

__all__ = ["AgentCapability", "AgentCapabilityMetadata"]


class AgentCapability(StrEnum):
    GENERAL = "general"
    RESEARCH = "research"
