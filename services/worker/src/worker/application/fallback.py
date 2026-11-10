"""Deterministic inference provider fallback traversal for Worker (Issue #16 Increment 2 & 3, Issue #17 Increment 2)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from agent_contracts import (
    DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
    AgentEvent,
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
    FallbackTransition,
    InferenceExecutionTelemetry,
    InferenceProviderHealth,
    InferenceResponse,
    NonFallbackInferenceError,
    ProviderFailureSummary,
    TaskRequested,
    parse_health_probe_timeout,
)
from agent_policy import (
    DeterministicProviderSelector,
    RankedProviderCandidate,
    SelectionStrategy,
    parse_provider_priority,
)

from worker.application.ports import (
    InferenceHealthProbe,
    InferenceProvider,
    MessageBus,
    RuntimeInferenceProvider,
)

logger = logging.getLogger(__name__)


def _sanitize_failure_reason(reason: str | None, default: str) -> str:
    if isinstance(reason, str):
        cleaned = reason.strip().lower()
        if re.fullmatch(r"[a-z0-9_]+", cleaned):
            return cleaned
    return default


def _sanitize_status_code(code: Any) -> int | None:
    if isinstance(code, int) and type(code) is not bool and 100 <= code <= 599:
        return code
    return None


class FallbackInferenceProvider:
    """Application adapter implementing InferenceProvider with deterministic fallback traversal.

    Traverses pre-ranked candidates exactly in the order supplied by provider selection policy.
    Resolves candidates lazily only when reached (resolving target provider exactly once).
    Stops immediately on first success, NonFallbackInferenceError, resolver failure, or unexpected exception.
    Advances to next candidate only when an already-resolved provider raises FallbackEligibleInferenceError.
    Emits FallbackTransition best-effort before invoking the already-resolved target provider.
    Measures and emits InferenceExecutionTelemetry best-effort for every actual provider attempt.
    """

    def __init__(
        self,
        candidates: Sequence[RankedProviderCandidate],
        resolver: Callable[[str], InferenceProvider],
        on_transition: Callable[[FallbackTransition], Awaitable[None]] | None = None,
        on_telemetry: Callable[[InferenceExecutionTelemetry], Awaitable[None]] | None = None,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not callable(resolver):
            raise TypeError(
                f"resolver must be a Callable[[str], InferenceProvider], got {type(resolver).__name__}"
            )
        if on_transition is not None and not callable(on_transition):
            raise TypeError(
                f"on_transition must be a Callable or None, got {type(on_transition).__name__}"
            )
        if on_telemetry is not None and not callable(on_telemetry):
            raise TypeError(
                f"on_telemetry must be a Callable or None, got {type(on_telemetry).__name__}"
            )
        if not callable(clock):
            raise TypeError(
                f"clock must be a Callable[[], int], got {type(clock).__name__}"
            )
        for idx, cand in enumerate(candidates):
            if not isinstance(cand, RankedProviderCandidate):
                raise TypeError(
                    f"candidates item at index {idx} must be a RankedProviderCandidate, got {type(cand).__name__}"
                )

        self._candidates: tuple[RankedProviderCandidate, ...] = tuple(candidates)
        self._resolver = resolver
        self._on_transition = on_transition
        self._on_telemetry = on_telemetry
        self._clock = clock

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

    async def _notify_telemetry(self, telemetry: InferenceExecutionTelemetry) -> None:
        if self._on_telemetry is None:
            return
        try:
            res = self._on_telemetry(telemetry)
            if inspect.isawaitable(res):
                await res
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference telemetry notification failed for provider %s (error_type=%s)",
                telemetry.provider_name,
                type(exc).__name__,
            )

    async def generate(self, prompt: str) -> str:
        """Execute prompt generation with deterministic fallback traversal and attempt telemetry."""
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
            t_start = self._clock()
            try:
                response = await current_provider.generate(prompt)
                t_end = self._clock()
                latency_ms = max(0.0, (t_end - t_start) / 1_000_000.0)

                input_tokens = None
                output_tokens = None
                total_tokens = None
                finish_reason = None
                model_name = None
                response_text = ""

                if isinstance(response, InferenceResponse):
                    input_tokens = response.input_tokens
                    output_tokens = response.output_tokens
                    total_tokens = response.total_tokens
                    finish_reason = response.finish_reason
                    model_name = response.model_name
                    response_text = response.text
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)

                telemetry = InferenceExecutionTelemetry(
                    provider_name=candidate.provider_name,
                    attempt_index=attempt_idx,
                    latency_ms=latency_ms,
                    status="succeeded",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    finish_reason=finish_reason,
                    model_name=model_name,
                    failure_reason=None,
                    status_code=None,
                )
                await self._notify_telemetry(telemetry)
                return response_text
            except FallbackEligibleInferenceError as exc:
                t_end = self._clock()
                latency_ms = max(0.0, (t_end - t_start) / 1_000_000.0)
                reason = _sanitize_failure_reason(exc.reason, "fallback_eligible_error")
                status_code = _sanitize_status_code(exc.status_code)

                telemetry = InferenceExecutionTelemetry(
                    provider_name=candidate.provider_name,
                    attempt_index=attempt_idx,
                    latency_ms=latency_ms,
                    status="fallback_eligible_failure",
                    failure_reason=reason,
                    status_code=status_code,
                )
                await self._notify_telemetry(telemetry)

                failures.append(
                    ProviderFailureSummary(
                        provider_name=candidate.provider_name,
                        reason=reason,
                        status_code=status_code,
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
            except NonFallbackInferenceError as exc:
                t_end = self._clock()
                latency_ms = max(0.0, (t_end - t_start) / 1_000_000.0)
                reason = _sanitize_failure_reason(exc.reason, "non_fallback_error")
                status_code = _sanitize_status_code(exc.status_code)

                telemetry = InferenceExecutionTelemetry(
                    provider_name=candidate.provider_name,
                    attempt_index=attempt_idx,
                    latency_ms=latency_ms,
                    status="non_fallback_failure",
                    failure_reason=reason,
                    status_code=status_code,
                )
                await self._notify_telemetry(telemetry)
                raise
            except Exception:
                t_end = self._clock()
                latency_ms = max(0.0, (t_end - t_start) / 1_000_000.0)

                telemetry = InferenceExecutionTelemetry(
                    provider_name=candidate.provider_name,
                    attempt_index=attempt_idx,
                    latency_ms=latency_ms,
                    status="unexpected_failure",
                    failure_reason="unexpected_error",
                    status_code=None,
                )
                await self._notify_telemetry(telemetry)
                raise

        raise AllInferenceProvidersFailedError(failures=failures)


class WorkerFallbackCoordinator:
    """Application coordinator managing per-task concurrent health probes, fallback, and telemetry."""

    def __init__(
        self,
        agent_id: str,
        bus: MessageBus,
        preferences: Sequence[str],
        providers: Mapping[str, RuntimeInferenceProvider],
        selector: DeterministicProviderSelector | None = None,
        probe_timeout_seconds: float = DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        self.probe_timeout_seconds = parse_health_probe_timeout(
            probe_timeout_seconds, default=DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS
        )
        if not callable(clock):
            raise TypeError(
                f"clock must be a Callable[[], int], got {type(clock).__name__}"
            )

        self.agent_id = agent_id.strip()
        self.bus = bus
        self.preferences = parse_provider_priority(None, default=preferences)
        self.selector = selector or DeterministicProviderSelector()
        self._clock = clock

        normalized_providers: dict[str, RuntimeInferenceProvider] = {}
        for name, provider in providers.items():
            normalized_providers[name.strip().lower()] = provider

        for pref in self.preferences:
            if pref not in normalized_providers:
                raise ValueError(
                    f"Configured provider {pref!r} is missing from provided runtime providers"
                )

        self._providers = normalized_providers

    @property
    def providers(self) -> Mapping[str, RuntimeInferenceProvider]:
        """Immutable mapping of retained runtime providers."""
        return self._providers

    async def _probe_single(
        self,
        provider_name: str,
        probe: InferenceHealthProbe,
        timeout_seconds: float,
    ) -> InferenceProviderHealth | None:
        try:
            return await asyncio.wait_for(probe.check_health(), timeout=timeout_seconds)
        except TimeoutError:
            logger.warning(
                "Health probe timed out for provider %s (timeout=%.2fs)",
                provider_name,
                timeout_seconds,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Health probe failed for provider %s (error_type=%s)",
                provider_name,
                type(exc).__name__,
            )
            return None

    async def generate(self, task: TaskRequested) -> str:
        """Execute task generation with per-task concurrent health probing, fallback, and attempt telemetry."""
        probe_tasks = [
            self._probe_single(name, self._providers[name], self.probe_timeout_seconds)
            for name in self.preferences
        ]
        probe_results = await asyncio.gather(*probe_tasks)
        snapshots = [s for s in probe_results if s is not None]

        decision = self.selector.select(
            self.preferences,
            snapshots,
            strategy=SelectionStrategy.PRIORITY,
        )

        if not decision.ranked_candidates:
            raise AllInferenceProvidersFailedError(failures=())

        async def on_transition(transition: FallbackTransition) -> None:
            event = AgentEvent.create(
                agent_id=self.agent_id,
                state="fallback",
                detail=(
                    f"Inference fallback: {transition.failed_provider} -> "
                    f"{transition.target_provider} ({transition.reason})"
                ),
                task_id=task.task_id,
                fallback_transition=transition,
            )
            await self.bus.publish(event)

        async def on_telemetry(telemetry: InferenceExecutionTelemetry) -> None:
            event = AgentEvent.create(
                agent_id=self.agent_id,
                state="working",
                detail=(
                    f"Inference attempt {telemetry.attempt_index} "
                    f"{telemetry.status} ({telemetry.provider_name})"
                ),
                task_id=task.task_id,
                inference_telemetry=telemetry,
            )
            await self.bus.publish(event)

        provider = FallbackInferenceProvider(
            candidates=decision.ranked_candidates,
            resolver=lambda name: self._providers[name],
            on_transition=on_transition,
            on_telemetry=on_telemetry,
            clock=self._clock,
        )
        return await provider.generate(task.prompt)
