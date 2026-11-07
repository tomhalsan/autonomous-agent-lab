"""Domain models and exceptions for routing advisory boundary."""

from dataclasses import dataclass

from .capability import AgentCapability

MAX_ADVISOR_REASON_LENGTH = 4096


class AdvisorError(Exception):
    """Base exception for routing advisor failures."""


class AdvisorUnavailableError(AdvisorError):
    """Raised when routing advisor provider is unavailable or unreachable."""


class AdvisorTimeoutError(AdvisorError):
    """Raised when routing advisor fails to respond within the configured deadline."""


class AdvisorExecutionError(AdvisorError):
    """Raised when routing advisor encounters an execution failure."""


@dataclass(frozen=True)
class AdvisorRecommendation:
    """Immutable recommendation produced by a RoutingAdvisor."""

    recommended_capability: AgentCapability
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.recommended_capability, AgentCapability):
            raise TypeError(
                f"recommended_capability must be an AgentCapability instance, got {type(self.recommended_capability).__name__}"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if len(self.reason) > MAX_ADVISOR_REASON_LENGTH:
            raise ValueError(
                f"reason exceeds maximum allowed length of {MAX_ADVISOR_REASON_LENGTH} characters"
            )
