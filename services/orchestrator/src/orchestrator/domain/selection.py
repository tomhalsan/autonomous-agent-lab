"""Candidate selection policies for dynamic capability-based orchestration."""

from typing import Protocol

from agent_contracts import AgentCandidate


class CandidateSelectionPolicy(Protocol):
    """Protocol for selecting an executor candidate from eligible candidates."""

    def select(self, candidates: list[AgentCandidate]) -> AgentCandidate:
        """Select an agent candidate from a non-empty list of eligible candidates."""
        ...


class DeterministicSelectionPolicy:
    """Deterministic selection policy sorting by agent_id for repeatable selection.

    Does not depend on hardcoded service identities or topology assumptions.
    """

    def select(self, candidates: list[AgentCandidate]) -> AgentCandidate:
        if not candidates:
            raise ValueError("Cannot select candidate from an empty candidate list")

        return min(candidates, key=lambda c: c.agent_id)

