"""Deterministic inference provider fallback traversal for Research Agent (Issue #16 Increment 2 & 3)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence

from agent_contracts import (
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
    FallbackTransition,
    NonFallbackInferenceError,
    ProviderFailureSummary,
)
from agent_policy import RankedProviderCandidate

from research_agent.application.ports import InferenceProvider
from research_agent.domain.models import SynthesisResult

logger = logging.getLogger(__name__)


class FallbackResearchSynthesizer:
    """Application adapter implementing InferenceProvider with deterministic fallback traversal.

    Traverses pre-ranked candidates exactly in the order supplied by provider selection policy.
    Resolves candidates lazily only when reached (resolving target provider exactly once).
    Stops immediately on first success, NonFallbackInferenceError, resolver failure, or unexpected exception.
    Advances to next candidate only when an already-resolved provider raises FallbackEligibleInferenceError.
    Emits FallbackTransition best-effort before invoking the already-resolved target provider.
    """

    def __init__(
        self,
        candidates: Sequence[RankedProviderCandidate],
        resolver: Callable[[str], InferenceProvider],
        on_transition: Callable[[FallbackTransition], Awaitable[None]] | None = None,
    ) -> None:
        if not callable(resolver):
            raise TypeError(
                f"resolver must be a Callable[[str], InferenceProvider], got {type(resolver).__name__}"
            )
        if on_transition is not None and not callable(on_transition):
            raise TypeError(
                f"on_transition must be a Callable or None, got {type(on_transition).__name__}"
            )
        for idx, cand in enumerate(candidates):
            if not isinstance(cand, RankedProviderCandidate):
                raise TypeError(
                    f"candidates item at index {idx} must be a RankedProviderCandidate, got {type(cand).__name__}"
                )

        self._candidates: tuple[RankedProviderCandidate, ...] = tuple(candidates)
        self._resolver = resolver
        self._on_transition = on_transition

    @property
    def candidates(self) -> tuple[RankedProviderCandidate, ...]:
        """Immutable sequence of ranked provider candidates."""
        return self._candidates

    async def _notify_transition(self, transition: FallbackTransition) -> None:
        if self._on_transition is None:
            return
        try:
            res = self._on_transition(transition)
            if inspect.isawaitable(res):
                await res
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Fallback transition notification failed from %s to %s (error_type=%s)",
                transition.failed_provider,
                transition.target_provider,
                type(exc).__name__,
            )

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        """Execute research synthesis with deterministic fallback traversal."""
        if not self._candidates:
            raise AllInferenceProvidersFailedError(failures=())

        current_candidate = self._candidates[0]
        current_provider = self._resolver(current_candidate.provider_name)
        if current_provider is None:
            raise TypeError(
                f"Resolver returned None for candidate provider {current_candidate.provider_name!r}"
            )

        failures: list[ProviderFailureSummary] = []

        for attempt_idx, candidate in enumerate(self._candidates):
            try:
                return await current_provider.synthesize(question, evidence)
            except FallbackEligibleInferenceError as exc:
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

                next_idx = attempt_idx + 1
                if next_idx >= len(self._candidates):
                    break

                next_candidate = self._candidates[next_idx]
                next_provider = self._resolver(next_candidate.provider_name)
                if next_provider is None:
                    raise TypeError(
                        f"Resolver returned None for candidate provider {next_candidate.provider_name!r}"
                    )

                transition = FallbackTransition(
                    failed_provider=candidate.provider_name,
                    target_provider=next_candidate.provider_name,
                    reason=reason,
                    attempt_index=attempt_idx,
                )
                await self._notify_transition(transition)

                current_candidate = next_candidate
                current_provider = next_provider
            except NonFallbackInferenceError:
                raise
            except (ValueError, TypeError):
                raise
            # Any unexpected exception escapes immediately without catching or reclassifying

        raise AllInferenceProvidersFailedError(failures=failures)
