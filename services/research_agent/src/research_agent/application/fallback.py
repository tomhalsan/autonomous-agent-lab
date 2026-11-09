"""Deterministic inference provider fallback traversal for Research Agent (Issue #16 Increment 2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_contracts import (
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
    NonFallbackInferenceError,
    ProviderFailureSummary,
)
from agent_policy import RankedProviderCandidate

from research_agent.application.ports import InferenceProvider
from research_agent.domain.models import SynthesisResult


class FallbackResearchSynthesizer:
    """Application adapter implementing InferenceProvider with deterministic fallback traversal.

    Traverses pre-ranked candidates exactly in the order supplied by provider selection policy.
    Resolves candidates lazily only when reached.
    Stops immediately on first success, NonFallbackInferenceError, resolver failure, or unexpected exception.
    Advances to next candidate only when an already-resolved provider raises FallbackEligibleInferenceError.
    """

    def __init__(
        self,
        candidates: Sequence[RankedProviderCandidate],
        resolver: Callable[[str], InferenceProvider],
    ) -> None:
        if not callable(resolver):
            raise TypeError(
                f"resolver must be a Callable[[str], InferenceProvider], got {type(resolver).__name__}"
            )
        for idx, cand in enumerate(candidates):
            if not isinstance(cand, RankedProviderCandidate):
                raise TypeError(
                    f"candidates item at index {idx} must be a RankedProviderCandidate, got {type(cand).__name__}"
                )

        self._candidates: tuple[RankedProviderCandidate, ...] = tuple(candidates)
        self._resolver = resolver

    @property
    def candidates(self) -> tuple[RankedProviderCandidate, ...]:
        """Immutable sequence of ranked provider candidates."""
        return self._candidates

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        """Execute research synthesis with deterministic fallback traversal."""
        if not self._candidates:
            raise AllInferenceProvidersFailedError(failures=())

        failures: list[ProviderFailureSummary] = []

        for candidate in self._candidates:
            # 1. Resolve candidate lazily (resolver failures propagate immediately without fallback)
            provider = self._resolver(candidate.provider_name)
            if provider is None:
                raise TypeError(
                    f"Resolver returned None for candidate provider {candidate.provider_name!r}"
                )

            # 2. Invoke resolved provider
            try:
                return await provider.synthesize(question, evidence)
            except FallbackEligibleInferenceError as exc:
                # Candidate provider identity is strictly authoritative; do not overwrite with exc.provider_name
                reason = (
                    exc.reason.strip()
                    if exc.reason and exc.reason.strip()
                    else "fallback_eligible_error"
                )
                failures.append(
                    ProviderFailureSummary(
                        provider_name=candidate.provider_name,
                        reason=reason,
                        status_code=exc.status_code,
                    )
                )
                continue
            except NonFallbackInferenceError:
                raise
            except (ValueError, TypeError):
                raise
            # Any unexpected exception escapes immediately without catching or reclassifying

        raise AllInferenceProvidersFailedError(failures=failures)
