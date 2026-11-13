"""Domain models for deterministic routing assessment and ambiguity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from orchestrator.domain.capability import AgentCapability

__all__ = [
    "RoutingAssessment",
    "RoutingClarity",
    "RoutingSignal",
    "RoutingSignalType",
    "SignalStrength",
]


class RoutingClarity(StrEnum):
    """Clarity assessment of a deterministic routing decision."""

    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"


class SignalStrength(StrEnum):
    """Strength and authoritative weight of a routing signal."""

    STRONG = "strong"
    WEAK = "weak"


class RoutingSignalType(StrEnum):
    """Taxonomy of inspectable deterministic routing signals."""

    # Strong Research signals
    EXPLICIT_EXTERNAL_RETRIEVAL = "explicit_external_retrieval"
    EXPLICIT_SOURCES_OR_CITATIONS = "explicit_sources_or_citations"
    FRESHNESS_OR_CURRENT_INFO = "freshness_or_current_info"
    CHANGING_EXTERNAL_INFO = "changing_external_info"
    OUTSIDE_AUTHORITY_OR_PROVIDER = "outside_authority_or_provider"
    RESEARCH_SYNTHESIS = "research_synthesis"

    # Strong General signals
    SUPPLIED_LOCAL_ARTIFACT = "supplied_local_artifact"
    TRANSFORMATION_OR_REWRITE = "transformation_or_rewrite"
    LOCAL_CODE_OR_DEBUGGING = "local_code_or_debugging"
    ESTABLISHED_CONCEPTUAL_EXPLANATION = "established_conceptual_explanation"
    ALL_DATA_SUPPLIED_INLINE = "all_data_supplied_inline"

    # Weak contextual signals
    WEAK_RESEARCH_VERB = "weak_research_verb"

    # Mixed-task compound signal
    COMPOUND_MIXED_TASK = "compound_mixed_task"


@dataclass(frozen=True)
class RoutingSignal:
    """An individual inspectable deterministic routing signal."""

    signal_type: RoutingSignalType
    target_capability: AgentCapability
    strength: SignalStrength
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.signal_type, RoutingSignalType):
            raise TypeError(
                f"signal_type must be RoutingSignalType, got {type(self.signal_type)}"
            )
        if not isinstance(self.target_capability, AgentCapability):
            raise TypeError(
                f"target_capability must be AgentCapability, got {type(self.target_capability)}"
            )
        if not isinstance(self.strength, SignalStrength):
            raise TypeError(
                f"strength must be SignalStrength, got {type(self.strength)}"
            )


@dataclass(frozen=True)
class RoutingAssessment:
    """Immutable assessment of a prompt's routing capability and clarity."""

    provisional_capability: AgentCapability
    clarity: RoutingClarity
    signals: tuple[RoutingSignal, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.provisional_capability, AgentCapability):
            raise TypeError(
                f"provisional_capability must be AgentCapability, got {type(self.provisional_capability)}"
            )
        if not isinstance(self.clarity, RoutingClarity):
            raise TypeError(
                f"clarity must be RoutingClarity, got {type(self.clarity)}"
            )
        if not isinstance(self.signals, tuple):
            raise TypeError(
                f"signals must be a tuple, got {type(self.signals)}"
            )
        for s in self.signals:
            if not isinstance(s, RoutingSignal):
                raise TypeError(
                    f"all elements in signals must be RoutingSignal, got {type(s)}"
                )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
