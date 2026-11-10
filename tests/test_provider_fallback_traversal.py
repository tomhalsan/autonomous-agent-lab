"""Deterministic inference provider fallback traversal tests (Issue #16 Increment 2)."""

from __future__ import annotations

import asyncio
import json

import pytest
from agent_contracts import (
    AgentEvent,
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
    FallbackTransition,
    InferenceError,
    InferenceExecutionTelemetry,
    InferenceHealthStatus,
    InferenceProviderHealth,
    InferenceResponse,
    NonFallbackInferenceError,
    ProviderFailureSummary,
    TaskRequested,
    TaskResult,
)
from agent_policy import (
    RankedProviderCandidate,
)
from research_agent.application.fallback import (
    FallbackResearchSynthesizer,
    ResearchFallbackCoordinator,
)
from research_agent.application.processor import (
    ResearchProcessor,
)
from research_agent.domain.models import ResearchResult, SynthesisResult
from worker.application.fallback import (
    FallbackInferenceProvider,
    WorkerFallbackCoordinator,
)
from worker.application.processor import (
    TaskExecutionError,
    TaskProcessor,
)


def _make_candidate(provider_name: str, priority_index: int = 0) -> RankedProviderCandidate:
    return RankedProviderCandidate(
        provider_name=provider_name,
        health_status=InferenceHealthStatus.HEALTHY,
        priority_index=priority_index,
    )


class RecordingMockWorkerProvider:
    def __init__(self, name: str, behavior: str | Exception) -> None:
        self.name = name
        self.behavior = behavior
        self.invoked = False
        self.received_prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.invoked = True
        self.received_prompts.append(prompt)
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


class RecordingMockResearchProvider:
    def __init__(self, name: str, behavior: SynthesisResult | Exception) -> None:
        self.name = name
        self.behavior = behavior
        self.invoked = False
        self.received_calls: list[tuple[str, str]] = []

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        self.invoked = True
        self.received_calls.append((question, evidence))
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


# ==============================================================================
# Contract Validation Tests: ProviderFailureSummary & AllInferenceProvidersFailedError
# ==============================================================================


def test_provider_failure_summary_normalization_and_validation() -> None:
    summary = ProviderFailureSummary(
        provider_name="  CLOUDFLARE  ",
        reason="  rate_limited  ",
        status_code=429,
    )
    assert summary.provider_name == "cloudflare"
    assert summary.reason == "rate_limited"
    assert summary.status_code == 429

    # Validation: provider_name must be non-empty string
    with pytest.raises(ValueError, match="provider_name"):
        ProviderFailureSummary(provider_name="   ", reason="timeout")

    with pytest.raises(ValueError, match="provider_name"):
        ProviderFailureSummary(provider_name="", reason="timeout")

    # Validation: reason must be non-empty string
    with pytest.raises(ValueError, match="reason"):
        ProviderFailureSummary(provider_name="gemini", reason="   ")

    # Validation: status_code must be valid int HTTP status (100-599) or None
    with pytest.raises(TypeError, match="status_code"):
        ProviderFailureSummary(provider_name="gemini", reason="timeout", status_code=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="status_code"):
        ProviderFailureSummary(provider_name="gemini", reason="timeout", status_code="429")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="status_code"):
        ProviderFailureSummary(provider_name="gemini", reason="timeout", status_code=99)

    with pytest.raises(ValueError, match="status_code"):
        ProviderFailureSummary(provider_name="gemini", reason="timeout", status_code=600)


def test_all_inference_providers_failed_error_contracts() -> None:
    summary1 = ProviderFailureSummary(provider_name="cloudflare", reason="timeout")
    summary2 = ProviderFailureSummary(provider_name="gemini", reason="rate_limited", status_code=429)

    err = AllInferenceProvidersFailedError(failures=[summary1, summary2])
    assert isinstance(err, InferenceError)
    assert isinstance(err, RuntimeError)
    assert err.failures == (summary1, summary2)
    assert str(err) == AllInferenceProvidersFailedError.MESSAGE
    assert str(err) == "All inference providers failed"

    # Default constructor with no arguments produces empty failures tuple and generic message
    err_empty = AllInferenceProvidersFailedError()
    assert err_empty.failures == ()
    assert str(err_empty) == "All inference providers failed"

    # Callers cannot inject arbitrary terminal error text through constructor
    with pytest.raises(TypeError):
        AllInferenceProvidersFailedError("Arbitrary terminal error text")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        AllInferenceProvidersFailedError(message="Custom text")  # type: ignore[call-arg]

    # Rejects invalid failure elements
    with pytest.raises(TypeError, match="ProviderFailureSummary"):
        AllInferenceProvidersFailedError(failures=["invalid"])  # type: ignore[list-item]


# ==============================================================================
# Worker Fallback Adapter Tests
# ==============================================================================


@pytest.mark.anyio
async def test_worker_fallback_primary_success() -> None:
    p1 = RecordingMockWorkerProvider("p1", "success from p1")
    p2 = RecordingMockWorkerProvider("p2", "success from p2")
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    result = await adapter.generate("Test prompt")
    assert result == "success from p1"
    assert p1.invoked is True
    assert p2.invoked is False
    assert resolved == ["p1"]


@pytest.mark.anyio
async def test_worker_fallback_primary_failure_secondary_success() -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("Transport timed out", provider_name="p1", reason="timeout"),
    )
    p2 = RecordingMockWorkerProvider("p2", "success from p2")
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    result = await adapter.generate("Prompt")
    assert result == "success from p2"
    assert p1.invoked is True
    assert p2.invoked is True
    assert resolved == ["p1", "p2"]


@pytest.mark.anyio
async def test_worker_fallback_multiple_failures_then_success() -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("500 internal error", provider_name="p1", reason="server_error_500", status_code=500),
    )
    p2 = RecordingMockWorkerProvider(
        "p2",
        FallbackEligibleInferenceError("429 rate limit", provider_name="p2", reason="rate_limited", status_code=429),
    )
    p3 = RecordingMockWorkerProvider("p3", "success from p3")
    p4 = RecordingMockWorkerProvider("p4", "success from p4")
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2, "p3": p3, "p4": p4}[name]

    candidates = [
        _make_candidate("p1", 0),
        _make_candidate("p2", 1),
        _make_candidate("p3", 2),
        _make_candidate("p4", 3),
    ]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    result = await adapter.generate("Prompt")
    assert result == "success from p3"
    assert resolved == ["p1", "p2", "p3"]
    assert p1.invoked and p2.invoked and p3.invoked
    assert p4.invoked is False


@pytest.mark.anyio
async def test_worker_fallback_complete_exhaustion() -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("503 unavailable", provider_name="p1", reason="server_error_503", status_code=503),
    )
    p2 = RecordingMockWorkerProvider(
        "p2",
        FallbackEligibleInferenceError("malformed json", provider_name="p2", reason="malformed_response"),
    )
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.generate("Prompt")

    err = exc_info.value
    assert isinstance(err, RuntimeError)
    assert str(err) == "All inference providers failed"
    assert len(err.failures) == 2
    assert err.failures[0].provider_name == "p1"
    assert err.failures[0].reason == "server_error_503"
    assert err.failures[0].status_code == 503
    assert err.failures[1].provider_name == "p2"
    assert err.failures[1].reason == "malformed_response"
    assert err.failures[1].status_code is None
    # No raw exception object attached
    for f in err.failures:
        assert not hasattr(f, "original_error")
        assert not hasattr(f, "exception")


@pytest.mark.anyio
async def test_worker_fallback_empty_candidate_sequence() -> None:
    def resolver(name: str) -> RecordingMockWorkerProvider:
        raise AssertionError("Resolver must not be called when candidates are empty")

    adapter = FallbackInferenceProvider(candidates=[], resolver=resolver)

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.generate("Prompt")

    assert str(exc_info.value) == "All inference providers failed"
    assert exc_info.value.failures == ()


@pytest.mark.anyio
async def test_worker_fallback_non_fallback_invocation_stops_immediately() -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        NonFallbackInferenceError("HTTP 401 Unauthorized", provider_name="p1", reason="client_error_401", status_code=401),
    )
    p2 = RecordingMockWorkerProvider("p2", "success from p2")
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await adapter.generate("Prompt")

    assert exc_info.value.status_code == 401
    assert p1.invoked is True
    assert p2.invoked is False
    assert resolved == ["p1"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "unexpected_exc",
    [
        ValueError("Invalid prompt parameter"),
        TypeError("Expected string prompt"),
        RuntimeError("Unexpected threadpool error"),
        ZeroDivisionError("Math error"),
    ],
)
async def test_worker_fallback_unexpected_invocation_exception_stops_immediately(
    unexpected_exc: Exception,
) -> None:
    p1 = RecordingMockWorkerProvider("p1", unexpected_exc)
    p2 = RecordingMockWorkerProvider("p2", "success from p2")
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    with pytest.raises(type(unexpected_exc)):
        await adapter.generate("Prompt")

    assert p1.invoked is True
    assert p2.invoked is False
    assert resolved == ["p1"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "resolver_exc",
    [
        FallbackEligibleInferenceError("Provider unreachable during init", reason="network_error"),
        NonFallbackInferenceError("Missing credentials", reason="auth_error"),
        ValueError("Bad provider configuration"),
        TypeError("Invalid config types"),
        KeyError("unknown_provider"),
        RuntimeError("Container startup failure"),
    ],
)
async def test_worker_fallback_resolver_failures_stop_immediately(
    resolver_exc: Exception,
) -> None:
    p2 = RecordingMockWorkerProvider("p2", "success from p2")

    def resolver(name: str) -> RecordingMockWorkerProvider:
        if name == "p1":
            raise resolver_exc
        return p2

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    with pytest.raises(type(resolver_exc)):
        await adapter.generate("Prompt")

    assert p2.invoked is False


@pytest.mark.anyio
async def test_worker_fallback_resolver_returning_none_raises_type_error() -> None:
    def resolver(name: str) -> None:  # type: ignore[return-value]
        return None

    candidates = [_make_candidate("p1", 0)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="returned None"):
        await adapter.generate("Prompt")


def test_worker_fallback_adapter_constructor_type_validation() -> None:
    # 1. Non-callable resolver
    with pytest.raises(TypeError, match="Callable"):
        FallbackInferenceProvider(candidates=[], resolver={"stub": None})  # type: ignore[arg-type]

    # 2. Bare strings in candidates list rejected
    with pytest.raises(TypeError, match="RankedProviderCandidate"):
        FallbackInferenceProvider(
            candidates=["stub"],  # type: ignore[list-item]
            resolver=lambda name: RecordingMockWorkerProvider(name, "ok"),
        )


@pytest.mark.anyio
async def test_worker_fallback_candidate_provider_name_is_authoritative() -> None:
    # When exc.provider_name is missing or differs from candidate.provider_name,
    # candidate.provider_name remains strictly authoritative in the summary.
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("Error from internal proxy", provider_name="internal-proxy", reason="timeout"),
    )

    def resolver(name: str) -> RecordingMockWorkerProvider:
        return p1

    candidates = [_make_candidate("configured-candidate-p1", 0)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver)

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.generate("Prompt")

    summary = exc_info.value.failures[0]
    assert summary.provider_name == "configured-candidate-p1"
    assert summary.reason == "timeout"


# ==============================================================================
# Research Agent Fallback Adapter Tests (Result Parity & SynthesisResult Preservation)
# ==============================================================================


@pytest.mark.anyio
async def test_research_fallback_primary_success() -> None:
    synthesis = SynthesisResult(
        question="What is consensus?",
        content="Consensus findings.",
        evidence="Evidence text",
        provider="p1",
        model="model-1",
    )
    p1 = RecordingMockResearchProvider("p1", synthesis)
    p2 = RecordingMockResearchProvider("p2", synthesis)
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockResearchProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackResearchSynthesizer(candidates=candidates, resolver=resolver)

    result = await adapter.synthesize("What is consensus?", "Evidence text")
    # Result parity: SynthesisResult is preserved unmodified
    assert isinstance(result, SynthesisResult)
    assert result == synthesis
    assert p1.invoked is True
    assert p2.invoked is False
    assert resolved == ["p1"]


@pytest.mark.anyio
async def test_research_fallback_primary_failure_secondary_success() -> None:
    synthesis = SynthesisResult(
        question="What is consensus?",
        content="Secondary consensus findings.",
        evidence="Evidence text",
        provider="p2",
        model="model-2",
    )
    p1 = RecordingMockResearchProvider(
        "p1",
        FallbackEligibleInferenceError("Rate limit reached", provider_name="p1", reason="rate_limited", status_code=429),
    )
    p2 = RecordingMockResearchProvider("p2", synthesis)
    resolved: list[str] = []

    def resolver(name: str) -> RecordingMockResearchProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackResearchSynthesizer(candidates=candidates, resolver=resolver)

    result = await adapter.synthesize("What is consensus?", "Evidence text")
    assert result == synthesis
    assert p1.invoked is True
    assert p2.invoked is True
    assert resolved == ["p1", "p2"]


@pytest.mark.anyio
async def test_research_fallback_complete_exhaustion() -> None:
    p1 = RecordingMockResearchProvider(
        "p1",
        FallbackEligibleInferenceError("500 internal", provider_name="p1", reason="server_error_500", status_code=500),
    )
    p2 = RecordingMockResearchProvider(
        "p2",
        FallbackEligibleInferenceError("timeout", provider_name="p2", reason="timeout"),
    )

    def resolver(name: str) -> RecordingMockResearchProvider:
        return {"p1": p1, "p2": p2}[name]

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackResearchSynthesizer(candidates=candidates, resolver=resolver)

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.synthesize("Question", "Evidence")

    assert str(exc_info.value) == "All inference providers failed"
    assert len(exc_info.value.failures) == 2
    assert exc_info.value.failures[0].provider_name == "p1"
    assert exc_info.value.failures[0].status_code == 500
    assert exc_info.value.failures[1].provider_name == "p2"


@pytest.mark.anyio
async def test_research_fallback_empty_candidate_sequence() -> None:
    def resolver(name: str) -> RecordingMockResearchProvider:
        raise AssertionError("Resolver must not be called when candidates are empty")

    adapter = FallbackResearchSynthesizer(candidates=[], resolver=resolver)

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.synthesize("Question", "Evidence")

    assert str(exc_info.value) == "All inference providers failed"
    assert exc_info.value.failures == ()


def test_research_fallback_adapter_constructor_type_validation() -> None:
    with pytest.raises(TypeError, match="Callable"):
        FallbackResearchSynthesizer(candidates=[], resolver={"stub": None})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RankedProviderCandidate"):
        FallbackResearchSynthesizer(
            candidates=["stub"],  # type: ignore[list-item]
            resolver=lambda name: RecordingMockResearchProvider(name, None),  # type: ignore[arg-type]
        )


# ==============================================================================
# Operational Fallback-Transition Events Tests (Issue #16 Increment 3)
# ==============================================================================


@pytest.mark.parametrize("adapter_cls", [FallbackInferenceProvider, FallbackResearchSynthesizer])
def test_fallback_adapter_on_transition_validation(adapter_cls: type) -> None:
    # 1. None is valid
    adapter_cls(candidates=[], resolver=lambda name: None, on_transition=None)  # type: ignore[arg-type]

    # 2. Callable is valid
    async def dummy_cb(transition: FallbackTransition) -> None:
        pass

    adapter_cls(candidates=[], resolver=lambda name: None, on_transition=dummy_cb)  # type: ignore[arg-type]

    # 3. Non-None, non-callable raises TypeError
    with pytest.raises(TypeError, match="on_transition must be a Callable or None"):
        adapter_cls(candidates=[], resolver=lambda name: None, on_transition="not-callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="on_transition must be a Callable or None"):
        adapter_cls(candidates=[], resolver=lambda name: None, on_transition=123)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_worker_fallback_transition_emission_on_primary_failure() -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("500 internal", provider_name="p1", reason="server_error_500", status_code=500),
    )
    p2 = RecordingMockWorkerProvider("p2", "success from p2")
    resolved: list[str] = []
    transitions: list[FallbackTransition] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2}[name]

    async def on_transition(t: FallbackTransition) -> None:
        transitions.append(t)

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver, on_transition=on_transition)

    result = await adapter.generate("Test prompt")
    assert result == "success from p2"
    # Target provider p2 was resolved exactly once
    assert resolved == ["p1", "p2"]
    # Exactly one transition emitted
    assert len(transitions) == 1
    t = transitions[0]
    assert t.failed_provider == "p1"
    assert t.target_provider == "p2"
    assert t.reason == "server_error_500"
    assert t.attempt_index == 0


@pytest.mark.anyio
async def test_worker_fallback_transition_multiple_switches() -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("Transport timed out", provider_name="p1", reason="timeout"),
    )
    p2 = RecordingMockWorkerProvider(
        "p2",
        FallbackEligibleInferenceError("429 limit", provider_name="p2", reason="rate_limited", status_code=429),
    )
    p3 = RecordingMockWorkerProvider("p3", "success from p3")
    resolved: list[str] = []
    transitions: list[FallbackTransition] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        resolved.append(name)
        return {"p1": p1, "p2": p2, "p3": p3}[name]

    async def on_transition(t: FallbackTransition) -> None:
        transitions.append(t)

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1), _make_candidate("p3", 2)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver, on_transition=on_transition)

    result = await adapter.generate("Test prompt")
    assert result == "success from p3"
    assert resolved == ["p1", "p2", "p3"]
    assert len(transitions) == 2
    assert transitions[0] == FallbackTransition(
        failed_provider="p1",
        target_provider="p2",
        reason="timeout",
        attempt_index=0,
    )
    assert transitions[1] == FallbackTransition(
        failed_provider="p2",
        target_provider="p3",
        reason="rate_limited",
        attempt_index=1,
    )


@pytest.mark.anyio
async def test_worker_fallback_transition_reason_normalization() -> None:
    # If exc.reason is None or whitespace, fallback_eligible_error is normalized
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("Connection reset without reason code", provider_name="p1", reason="   "),
    )
    p2 = RecordingMockWorkerProvider("p2", "success from p2")
    transitions: list[FallbackTransition] = []

    def resolver(name: str) -> RecordingMockWorkerProvider:
        return {"p1": p1, "p2": p2}[name]

    async def on_transition(t: FallbackTransition) -> None:
        transitions.append(t)

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(candidates=candidates, resolver=resolver, on_transition=on_transition)

    await adapter.generate("Test prompt")
    assert len(transitions) == 1
    assert transitions[0].reason == "fallback_eligible_error"


@pytest.mark.anyio
async def test_worker_fallback_transition_suppression() -> None:
    transitions: list[FallbackTransition] = []

    async def on_transition(t: FallbackTransition) -> None:
        transitions.append(t)

    # Case 1: Primary succeeds -> 0 transitions
    p1_ok = RecordingMockWorkerProvider("p1", "ok")
    adapter1 = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0), _make_candidate("p2", 1)],
        resolver=lambda name: p1_ok,
        on_transition=on_transition,
    )
    await adapter1.generate("Prompt")
    assert len(transitions) == 0

    # Case 2: Complete exhaustion (final candidate fails) -> only transitions for valid next providers, none for final
    p1_fail = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("fail 1", reason="timeout"),
    )
    p2_fail = RecordingMockWorkerProvider(
        "p2",
        FallbackEligibleInferenceError("fail 2", reason="timeout"),
    )

    def resolver_exhaust(name: str) -> RecordingMockWorkerProvider:
        return {"p1": p1_fail, "p2": p2_fail}[name]

    adapter2 = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0), _make_candidate("p2", 1)],
        resolver=resolver_exhaust,
        on_transition=on_transition,
    )
    with pytest.raises(AllInferenceProvidersFailedError):
        await adapter2.generate("Prompt")
    # Emitted transition from p1 -> p2, but NO transition after p2 fails because there is no next provider
    assert len(transitions) == 1
    assert transitions[0].failed_provider == "p1"
    assert transitions[0].target_provider == "p2"
    transitions.clear()

    # Case 3: NonFallbackInferenceError -> 0 transitions
    p1_non_fallback = RecordingMockWorkerProvider(
        "p1",
        NonFallbackInferenceError("401 Unauthorized", reason="client_error_401", status_code=401),
    )
    adapter3 = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0), _make_candidate("p2", 1)],
        resolver=lambda name: {"p1": p1_non_fallback, "p2": p1_ok}[name],
        on_transition=on_transition,
    )
    with pytest.raises(NonFallbackInferenceError):
        await adapter3.generate("Prompt")
    assert len(transitions) == 0

    # Case 4: Target-provider resolution fails -> 0 transitions
    def resolver_fail_on_p2(name: str) -> RecordingMockWorkerProvider:
        if name == "p1":
            return p1_fail
        raise RuntimeError("p2 resolution failed in factory")

    adapter4 = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0), _make_candidate("p2", 1)],
        resolver=resolver_fail_on_p2,
        on_transition=on_transition,
    )
    with pytest.raises(RuntimeError, match="p2 resolution failed"):
        await adapter4.generate("Prompt")
    assert len(transitions) == 0

    # Case 5: Empty candidate sequence -> 0 transitions
    adapter5 = FallbackInferenceProvider(
        candidates=[],
        resolver=lambda name: p1_ok,
        on_transition=on_transition,
    )
    with pytest.raises(AllInferenceProvidersFailedError):
        await adapter5.generate("Prompt")
    assert len(transitions) == 0


@pytest.mark.anyio
async def test_worker_fallback_transition_callback_failure_containment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("Internal 500", reason="server_error_500", status_code=500),
    )
    p2 = RecordingMockWorkerProvider("p2", "recovered value")

    def resolver(name: str) -> RecordingMockWorkerProvider:
        return {"p1": p1, "p2": p2}[name]

    secret_raw_text = "SECRET_PAYLOAD_UNAUTHORIZED_LEAK"

    async def failing_on_transition(t: FallbackTransition) -> None:
        raise RuntimeError(f"Event bus failure with {secret_raw_text}")

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackInferenceProvider(
        candidates=candidates,
        resolver=resolver,
        on_transition=failing_on_transition,
    )

    # Traversal continues successfully despite callback exception
    with caplog.at_level("WARNING"):
        result = await adapter.generate("Test prompt")

    assert result == "recovered value"
    assert p2.invoked is True
    # Logger recorded failure with sanitized provider identities and exception type only
    assert "Fallback transition notification failed from p1 to p2 (error_type=RuntimeError)" in caplog.text
    # Raw exception text and sensitive payload are never leaked into logs
    assert secret_raw_text not in caplog.text


@pytest.mark.anyio
async def test_research_fallback_transition_parity() -> None:
    synthesis = SynthesisResult(
        question="What is consensus?",
        content="Consensus findings.",
        evidence="Evidence text",
        provider="p2",
        model="model-2",
    )
    p1 = RecordingMockResearchProvider(
        "p1",
        FallbackEligibleInferenceError("503 unavailable", reason="server_error_503", status_code=503),
    )
    p2 = RecordingMockResearchProvider("p2", synthesis)
    transitions: list[FallbackTransition] = []

    def resolver(name: str) -> RecordingMockResearchProvider:
        return {"p1": p1, "p2": p2}[name]

    async def on_transition(t: FallbackTransition) -> None:
        transitions.append(t)

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]
    adapter = FallbackResearchSynthesizer(
        candidates=candidates,
        resolver=resolver,
        on_transition=on_transition,
    )

    result = await adapter.synthesize("What is consensus?", "Evidence text")
    assert result == synthesis
    assert len(transitions) == 1
    assert transitions[0] == FallbackTransition(
        failed_provider="p1",
        target_provider="p2",
        reason="server_error_503",
        attempt_index=0,
    )


# ==============================================================================
# Runtime Health & Fallback Coordinator Tests (Issue #16 Increment 4)
# ==============================================================================


class MockRuntimeWorkerProvider:
    def __init__(
        self,
        name: str,
        health_status: InferenceHealthStatus = InferenceHealthStatus.HEALTHY,
        health_delay: float = 0.0,
        health_error: Exception | None = None,
        generate_behavior: str | Exception = "mock worker output",
    ) -> None:
        self.name = name
        self.health_status = health_status
        self.health_delay = health_delay
        self.health_error = health_error
        self.generate_behavior = generate_behavior
        self.health_checked = False
        self.generate_called = False
        self.received_prompts: list[str] = []

    async def check_health(self) -> InferenceProviderHealth:
        self.health_checked = True
        if self.health_delay > 0:
            await asyncio.sleep(self.health_delay)
        if self.health_error is not None:
            raise self.health_error
        return InferenceProviderHealth(
            provider_name=self.name,
            status=self.health_status,
            reason=f"Status is {self.health_status.value}",
        )

    async def generate(self, prompt: str) -> str:
        self.generate_called = True
        self.received_prompts.append(prompt)
        if isinstance(self.generate_behavior, Exception):
            raise self.generate_behavior
        return self.generate_behavior


class MockRuntimeResearchProvider:
    def __init__(
        self,
        name: str,
        health_status: InferenceHealthStatus = InferenceHealthStatus.HEALTHY,
        health_delay: float = 0.0,
        health_error: Exception | None = None,
        synthesize_behavior: SynthesisResult | Exception | None = None,
    ) -> None:
        self.name = name
        self.health_status = health_status
        self.health_delay = health_delay
        self.health_error = health_error
        self.synthesize_behavior = synthesize_behavior or SynthesisResult(
            question="default question",
            content="synthesized content",
            evidence="grounded evidence",
            provider=name,
            model="default-model",
        )
        self.health_checked = False
        self.synthesize_called = False
        self.received_calls: list[tuple[str, str]] = []

    async def check_health(self) -> InferenceProviderHealth:
        self.health_checked = True
        if self.health_delay > 0:
            await asyncio.sleep(self.health_delay)
        if self.health_error is not None:
            raise self.health_error
        return InferenceProviderHealth(
            provider_name=self.name,
            status=self.health_status,
            reason=f"Status is {self.health_status.value}",
        )

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        self.synthesize_called = True
        self.received_calls.append((question, evidence))
        if isinstance(self.synthesize_behavior, Exception):
            raise self.synthesize_behavior
        return self.synthesize_behavior


class MockMessageBus:
    def __init__(self, fail_publish: bool = False) -> None:
        self.published: list[AgentEvent] = []
        self.saved_results: dict[str, TaskResult | ResearchResult] = {}
        self.fail_publish = fail_publish

    async def publish(self, event: AgentEvent) -> None:
        if self.fail_publish:
            raise RuntimeError("Simulated bus transport failure")
        self.published.append(event)

    async def save(self, result: TaskResult | ResearchResult) -> None:
        self.saved_results[result.task_id] = result

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError


class MockResearchExecutor:
    def __init__(self, output: str = "Research ground evidence") -> None:
        self.output = output

    async def execute(self, task: TaskRequested) -> ResearchResult:
        return ResearchResult(
            task_id=task.task_id,
            output=self.output,
            acknowledged=True,
            sources=["https://example.com/ref1"],
        )


def test_worker_coordinator_validation() -> None:
    bus = MockMessageBus()
    provider = MockRuntimeWorkerProvider("p1")

    with pytest.raises(ValueError, match="agent_id must be a non-empty string"):
        WorkerFallbackCoordinator(
            agent_id="  ",
            bus=bus,
            preferences=["p1"],
            providers={"p1": provider},
        )

    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        WorkerFallbackCoordinator(
            agent_id="worker-1",
            bus=bus,
            preferences=["p1"],
            providers={"p1": provider},
            probe_timeout_seconds=-0.5,
        )

    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        WorkerFallbackCoordinator(
            agent_id="worker-1",
            bus=bus,
            preferences=["p1"],
            providers={"p1": provider},
            probe_timeout_seconds=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Configured provider 'p2' is missing"):
        WorkerFallbackCoordinator(
            agent_id="worker-1",
            bus=bus,
            preferences=["p1", "p2"],
            providers={"p1": provider},
        )


@pytest.mark.anyio
async def test_worker_coordinator_healthy_priority_order() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider("p1", health_status=InferenceHealthStatus.HEALTHY, generate_behavior="out-p1")
    p2 = MockRuntimeWorkerProvider("p2", health_status=InferenceHealthStatus.HEALTHY, generate_behavior="out-p2")

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("Test prompt")
    result = await coordinator.generate(task)

    assert result == "out-p1"
    assert p1.health_checked is True
    assert p2.health_checked is True
    assert p1.generate_called is True
    assert p2.generate_called is False
    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.state == "working"
    assert event.task_id == task.task_id
    assert event.inference_telemetry is not None
    assert event.inference_telemetry.status == "succeeded"
    assert event.inference_telemetry.provider_name == "p1"
    assert event.inference_telemetry.attempt_index == 0


@pytest.mark.anyio
async def test_worker_coordinator_concurrent_probing_and_timeout(caplog: pytest.LogCaptureFixture) -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        health_delay=0.01,
        generate_behavior="fast output",
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        health_delay=0.5,
        generate_behavior="slow output",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
        probe_timeout_seconds=0.05,
    )
    task = TaskRequested.create("Fast prompt")

    with caplog.at_level("WARNING"):
        result = await coordinator.generate(task)

    assert result == "fast output"
    assert p1.generate_called is True
    assert p2.generate_called is False
    assert "Health probe timed out for provider p2 (timeout=0.05s)" in caplog.text


@pytest.mark.anyio
async def test_worker_coordinator_probe_failure_isolation(caplog: pytest.LogCaptureFixture) -> None:
    bus = MockMessageBus()
    secret_text = "SECRET_TOKEN_DO_NOT_LEAK"
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_error=RuntimeError(f"Connection dropped with {secret_text}"),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior="recovered output",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("Isolation prompt")

    with caplog.at_level("WARNING"):
        result = await coordinator.generate(task)

    assert result == "recovered output"
    assert p2.generate_called is True
    assert "Health probe failed for provider p1 (error_type=RuntimeError)" in caplog.text
    assert secret_text not in caplog.text


@pytest.mark.anyio
async def test_worker_coordinator_degraded_outranked_by_healthy() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.DEGRADED,
        generate_behavior="p1 output",
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior="p2 output",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("Check degradation ranking")
    result = await coordinator.generate(task)

    assert result == "p2 output"
    assert p2.generate_called is True
    assert p1.generate_called is False


@pytest.mark.anyio
async def test_worker_coordinator_fallback_publishes_event_with_task_id() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=FallbackEligibleInferenceError("Rate limited", reason="rate_limited", status_code=429),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior="secondary output",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("Fallback prompt")
    result = await coordinator.generate(task)

    assert result == "secondary output"
    assert p1.generate_called is True
    assert p2.generate_called is True

    assert len(bus.published) == 3
    ev_telemetry_0, ev_transition, ev_telemetry_1 = bus.published

    assert ev_telemetry_0.state == "working"
    assert ev_telemetry_0.task_id == task.task_id
    assert ev_telemetry_0.inference_telemetry is not None
    assert ev_telemetry_0.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telemetry_0.inference_telemetry.provider_name == "p1"
    assert ev_telemetry_0.inference_telemetry.attempt_index == 0
    assert ev_telemetry_0.inference_telemetry.failure_reason == "rate_limited"
    assert ev_telemetry_0.inference_telemetry.status_code == 429

    assert ev_transition.state == "fallback"
    assert ev_transition.agent_id == "worker-1"
    assert ev_transition.task_id == task.task_id
    assert "rate_limited" in ev_transition.detail
    assert ev_transition.fallback_transition == FallbackTransition(
        failed_provider="p1",
        target_provider="p2",
        reason="rate_limited",
        attempt_index=0,
    )

    assert ev_telemetry_1.state == "working"
    assert ev_telemetry_1.task_id == task.task_id
    assert ev_telemetry_1.inference_telemetry is not None
    assert ev_telemetry_1.inference_telemetry.status == "succeeded"
    assert ev_telemetry_1.inference_telemetry.provider_name == "p2"
    assert ev_telemetry_1.inference_telemetry.attempt_index == 1


@pytest.mark.anyio
async def test_worker_coordinator_non_fallback_error_stops() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=NonFallbackInferenceError("400 Bad Request", reason="bad_request", status_code=400),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior="should not run",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("Bad request prompt")

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await coordinator.generate(task)

    assert exc_info.value.reason == "bad_request"
    assert p1.generate_called is True
    assert p2.generate_called is False
    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.state == "working"
    assert event.task_id == task.task_id
    assert event.inference_telemetry is not None
    assert event.inference_telemetry.status == "non_fallback_failure"
    assert event.inference_telemetry.provider_name == "p1"
    assert event.inference_telemetry.attempt_index == 0
    assert event.inference_telemetry.failure_reason == "bad_request"
    assert event.inference_telemetry.status_code == 400


@pytest.mark.anyio
async def test_worker_coordinator_all_probes_unavailable_raises_generic_exhaustion() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider("p1", health_status=InferenceHealthStatus.UNAVAILABLE)
    p2 = MockRuntimeWorkerProvider("p2", health_status=InferenceHealthStatus.UNAVAILABLE)

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("No candidates prompt")

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await coordinator.generate(task)

    assert str(exc_info.value) == AllInferenceProvidersFailedError.MESSAGE
    assert exc_info.value.failures == ()
    assert p1.generate_called is False
    assert p2.generate_called is False


@pytest.mark.anyio
async def test_worker_coordinator_event_publish_failure_does_not_abort_recovery() -> None:
    bus = MockMessageBus(fail_publish=True)
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=FallbackEligibleInferenceError("Timeout", reason="timeout"),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior="recovered value",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    task = TaskRequested.create("Publish failure prompt")
    result = await coordinator.generate(task)

    assert result == "recovered value"
    assert p2.generate_called is True


@pytest.mark.anyio
async def test_worker_task_processor_end_to_end_fallback_lifecycle() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=FallbackEligibleInferenceError("503", reason="server_error_503", status_code=503),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior="final answer",
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    processor = TaskProcessor(
        agent_id="worker-1",
        bus=bus,
        result_store=bus,
        fallback_coordinator=coordinator,
    )
    task = TaskRequested.create("End to end prompt")
    await processor.run_once(task)

    # Event sequence: task working -> telemetry p1 -> fallback -> telemetry p2 -> idle
    assert len(bus.published) == 5
    ev_task_working, ev_telem1, ev_fallback, ev_telem2, ev_idle = bus.published

    assert ev_task_working.state == "working"
    assert ev_task_working.task_id == task.task_id
    assert ev_task_working.inference_telemetry is None

    assert ev_telem1.state == "working"
    assert ev_telem1.task_id == task.task_id
    assert ev_telem1.inference_telemetry is not None
    assert ev_telem1.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telem1.inference_telemetry.provider_name == "p1"
    assert ev_telem1.inference_telemetry.attempt_index == 0

    assert ev_fallback.state == "fallback"
    assert ev_fallback.task_id == task.task_id
    assert ev_fallback.fallback_transition == FallbackTransition("p1", "p2", "server_error_503", 0)

    assert ev_telem2.state == "working"
    assert ev_telem2.task_id == task.task_id
    assert ev_telem2.inference_telemetry is not None
    assert ev_telem2.inference_telemetry.status == "succeeded"
    assert ev_telem2.inference_telemetry.provider_name == "p2"
    assert ev_telem2.inference_telemetry.attempt_index == 1

    assert ev_idle.state == "idle"
    assert ev_idle.task_id == task.task_id

    assert task.task_id in bus.saved_results
    assert bus.saved_results[task.task_id].output == "final answer"


@pytest.mark.anyio
async def test_worker_task_processor_end_to_end_exhaustion_lifecycle() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=FallbackEligibleInferenceError("503", reason="server_error_503"),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=FallbackEligibleInferenceError("429", reason="rate_limited"),
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    processor = TaskProcessor(
        agent_id="worker-1",
        bus=bus,
        result_store=bus,
        fallback_coordinator=coordinator,
    )
    task = TaskRequested.create("Exhaustion prompt")

    with pytest.raises(TaskExecutionError) as exc_info:
        await processor.run_once(task)

    assert isinstance(exc_info.value.original_error, AllInferenceProvidersFailedError)
    assert exc_info.value.task_id == task.task_id

    # Event sequence: task working -> telemetry p1 -> fallback -> telemetry p2 -> error -> idle
    assert len(bus.published) == 6
    ev_task_working, ev_telem1, ev_fallback, ev_telem2, ev_error, ev_idle = bus.published

    assert ev_task_working.state == "working"
    assert ev_task_working.inference_telemetry is None

    assert ev_telem1.state == "working"
    assert ev_telem1.inference_telemetry is not None
    assert ev_telem1.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telem1.inference_telemetry.provider_name == "p1"

    assert ev_fallback.state == "fallback"
    assert ev_fallback.fallback_transition == FallbackTransition("p1", "p2", "server_error_503", 0)

    assert ev_telem2.state == "working"
    assert ev_telem2.inference_telemetry is not None
    assert ev_telem2.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telem2.inference_telemetry.provider_name == "p2"

    assert ev_error.state == "error"
    assert ev_error.detail == AllInferenceProvidersFailedError.MESSAGE
    assert ev_idle.state == "idle"


@pytest.mark.anyio
async def test_research_fallback_coordinator_parity() -> None:
    bus = MockMessageBus()
    synthesis = SynthesisResult(
        question="Consensus models",
        content="Paxos and Raft comparison",
        evidence="Evidence text",
        provider="p2",
        model="m2",
        truncated=False,
    )
    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("500", reason="server_error_500", status_code=500),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=synthesis,
    )

    coordinator = ResearchFallbackCoordinator(
        agent_id="research-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    result = await coordinator.synthesize("Consensus models", "Evidence text", task_id="task-r-1")

    assert result == synthesis
    assert p1.synthesize_called is True
    assert p2.synthesize_called is True
    assert len(bus.published) == 3
    ev_telemetry_0, ev_transition, ev_telemetry_1 = bus.published

    assert ev_telemetry_0.state == "working"
    assert ev_telemetry_0.agent_id == "research-1"
    assert ev_telemetry_0.task_id == "task-r-1"
    assert ev_telemetry_0.inference_telemetry is not None
    assert ev_telemetry_0.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telemetry_0.inference_telemetry.provider_name == "p1"
    assert ev_telemetry_0.inference_telemetry.attempt_index == 0
    assert ev_telemetry_0.inference_telemetry.failure_reason == "server_error_500"
    assert ev_telemetry_0.inference_telemetry.status_code == 500

    assert ev_transition.state == "fallback"
    assert ev_transition.agent_id == "research-1"
    assert ev_transition.task_id == "task-r-1"
    assert ev_transition.fallback_transition == FallbackTransition("p1", "p2", "server_error_500", 0)

    assert ev_telemetry_1.state == "working"
    assert ev_telemetry_1.agent_id == "research-1"
    assert ev_telemetry_1.task_id == "task-r-1"
    assert ev_telemetry_1.inference_telemetry is not None
    assert ev_telemetry_1.inference_telemetry.status == "succeeded"
    assert ev_telemetry_1.inference_telemetry.provider_name == "p2"
    assert ev_telemetry_1.inference_telemetry.attempt_index == 1
    assert ev_telemetry_1.inference_telemetry.model_name == "m2"


@pytest.mark.anyio
async def test_research_processor_end_to_end_fallback_lifecycle() -> None:
    bus = MockMessageBus()
    synthesis = SynthesisResult(
        question="Grounded query",
        content="Synthesized research findings",
        evidence="Ground evidence",
        provider="p2",
        model="m2",
    )
    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("Timeout", reason="timeout"),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=synthesis,
    )

    coordinator = ResearchFallbackCoordinator(
        agent_id="research-1",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    executor = MockResearchExecutor(output="Ground evidence")
    processor = ResearchProcessor(
        agent_id="research-1",
        bus=bus,
        executor=executor,
        result_store=bus,
        fallback_coordinator=coordinator,
    )
    task = TaskRequested.create("Grounded query")
    await processor.run_once(task)

    # Lifecycle: task working -> telemetry p1 -> fallback -> telemetry p2 -> idle
    assert len(bus.published) == 5
    ev_task_working, ev_telem1, ev_fallback, ev_telem2, ev_idle = bus.published

    assert ev_task_working.state == "working"
    assert ev_task_working.task_id == task.task_id
    assert ev_task_working.inference_telemetry is None

    assert ev_telem1.state == "working"
    assert ev_telem1.task_id == task.task_id
    assert ev_telem1.inference_telemetry is not None
    assert ev_telem1.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telem1.inference_telemetry.provider_name == "p1"
    assert ev_telem1.inference_telemetry.attempt_index == 0

    assert ev_fallback.state == "fallback"
    assert ev_fallback.fallback_transition == FallbackTransition("p1", "p2", "timeout", 0)

    assert ev_telem2.state == "working"
    assert ev_telem2.task_id == task.task_id
    assert ev_telem2.inference_telemetry is not None
    assert ev_telem2.inference_telemetry.status == "succeeded"
    assert ev_telem2.inference_telemetry.provider_name == "p2"
    assert ev_telem2.inference_telemetry.attempt_index == 1

    assert ev_idle.state == "idle"
    assert ev_idle.task_id == task.task_id

    assert task.task_id in bus.saved_results
    saved = bus.saved_results[task.task_id]
    assert saved.output == "Synthesized research findings"
    assert saved.sources == ("https://example.com/ref1",)


# ==============================================================================
# Issue #17 Increment 2: Inference Execution Telemetry Traversal Tests
# ==============================================================================


class DeterministicClock:
    def __init__(self, timestamps_ns: list[int]) -> None:
        self._timestamps = list(timestamps_ns)
        self._idx = 0

    def __call__(self) -> int:
        if self._idx < len(self._timestamps):
            val = self._timestamps[self._idx]
            self._idx += 1
            return val
        return self._timestamps[-1] if self._timestamps else 0


@pytest.mark.anyio
async def test_fallback_telemetry_single_successful_attempt() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    class MockProvider:
        async def generate(self, prompt: str) -> InferenceResponse:
            return InferenceResponse(
                text=f"Response for {prompt}",
                provider_name="p1",
                model_name="test-model-v1",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                finish_reason="stop",
            )

    clock = DeterministicClock([10_000_000, 35_000_000])  # 25.0 ms delta
    candidate = _make_candidate("p1", 0)

    async def on_telemetry(t: InferenceExecutionTelemetry) -> None:
        telemetry_records.append(t)

    adapter = FallbackInferenceProvider(
        candidates=[candidate],
        resolver=lambda name: MockProvider() if name == "p1" else None,  # type: ignore[return-value]
        on_telemetry=on_telemetry,
        clock=clock,
    )
    result = await adapter.generate("Test prompt")

    assert type(result) is str
    assert result == "Response for Test prompt"
    assert len(telemetry_records) == 1

    t = telemetry_records[0]
    assert t.attempt_index == 0
    assert t.status == "succeeded"
    assert t.provider_name == "p1"
    assert t.model_name == "test-model-v1"
    assert t.input_tokens == 100
    assert t.output_tokens == 50
    assert t.total_tokens == 150
    assert t.finish_reason == "stop"
    assert t.failure_reason is None
    assert t.status_code is None
    assert t.latency_ms == 25.0


@pytest.mark.anyio
async def test_fallback_telemetry_missing_metadata_defaults_none() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    class MockProvider:
        async def generate(self, prompt: str) -> InferenceResponse:
            return InferenceResponse(
                text="Bare response",
                provider_name="p1",
            )

    candidate = _make_candidate("p1", 0)

    adapter = FallbackInferenceProvider(
        candidates=[candidate],
        resolver=lambda name: MockProvider() if name == "p1" else None,  # type: ignore[return-value]
        on_telemetry=lambda t: telemetry_records.append(t),  # type: ignore[arg-type,return-value]
    )
    result = await adapter.generate("Test prompt")
    assert result == "Bare response"
    assert len(telemetry_records) == 1

    t = telemetry_records[0]
    assert t.input_tokens is None
    assert t.output_tokens is None
    assert t.total_tokens is None
    assert t.finish_reason is None
    assert t.model_name is None


@pytest.mark.anyio
async def test_fallback_telemetry_sequence_primary_fail_secondary_success() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []
    transitions: list[FallbackTransition] = []

    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("Rate limited", reason="rate_limited", status_code=429),
    )
    p2 = RecordingMockWorkerProvider("p2", "secondary success")

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]

    adapter = FallbackInferenceProvider(
        candidates=candidates,
        resolver=lambda name: p1 if name == "p1" else p2,
        on_transition=lambda tr: transitions.append(tr),  # type: ignore[arg-type,return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )
    result = await adapter.generate("Sequence prompt")

    assert result == "secondary success"
    assert len(telemetry_records) == 2

    # Attempt 0: p1 fallback-eligible failure
    t0 = telemetry_records[0]
    assert t0.attempt_index == 0
    assert t0.provider_name == "p1"
    assert t0.status == "fallback_eligible_failure"
    assert t0.failure_reason == "rate_limited"
    assert t0.status_code == 429

    # Attempt 1: p2 success
    t1 = telemetry_records[1]
    assert t1.attempt_index == 1
    assert t1.provider_name == "p2"
    assert t1.status == "succeeded"
    assert t1.failure_reason is None
    assert t1.status_code is None

    # Transition occurred truthfully between attempt 0 and 1
    assert len(transitions) == 1
    assert transitions[0] == FallbackTransition(
        failed_provider="p1",
        target_provider="p2",
        reason="rate_limited",
        attempt_index=0,
    )


@pytest.mark.anyio
async def test_fallback_telemetry_exhaustion_all_candidates_fail() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    p1 = RecordingMockWorkerProvider(
        "p1",
        FallbackEligibleInferenceError("500", reason="server_error_500", status_code=500),
    )
    p2 = RecordingMockWorkerProvider(
        "p2",
        FallbackEligibleInferenceError("503", reason="server_error_503", status_code=503),
    )
    p3 = RecordingMockWorkerProvider(
        "p3",
        FallbackEligibleInferenceError("429", reason="rate_limited", status_code=429),
    )

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1), _make_candidate("p3", 2)]
    providers = {"p1": p1, "p2": p2, "p3": p3}

    adapter = FallbackInferenceProvider(
        candidates=candidates,
        resolver=lambda name: providers[name],
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.generate("Exhaustion test")

    assert len(telemetry_records) == 3
    assert [t.attempt_index for t in telemetry_records] == [0, 1, 2]
    assert [t.provider_name for t in telemetry_records] == ["p1", "p2", "p3"]
    assert all(t.status == "fallback_eligible_failure" for t in telemetry_records)
    # Ensure no duplicate terminal telemetry was emitted
    assert len(exc_info.value.failures) == 3
    assert exc_info.value.failures[0].provider_name == "p1"
    assert exc_info.value.failures[1].provider_name == "p2"
    assert exc_info.value.failures[2].provider_name == "p3"


@pytest.mark.anyio
async def test_fallback_telemetry_non_fallback_failure_stops() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []
    transitions: list[FallbackTransition] = []

    p1 = RecordingMockWorkerProvider(
        "p1",
        NonFallbackInferenceError("Safety blocked", reason="blocked_safety", status_code=400),
    )
    p2 = RecordingMockWorkerProvider("p2", "should not run")

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]

    adapter = FallbackInferenceProvider(
        candidates=candidates,
        resolver=lambda name: p1 if name == "p1" else p2,
        on_transition=lambda tr: transitions.append(tr),  # type: ignore[arg-type,return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await adapter.generate("Safety prompt")

    assert exc_info.value.reason == "blocked_safety"
    assert p1.invoked is True
    assert p2.invoked is False
    assert len(transitions) == 0

    assert len(telemetry_records) == 1
    t0 = telemetry_records[0]
    assert t0.attempt_index == 0
    assert t0.provider_name == "p1"
    assert t0.status == "non_fallback_failure"
    assert t0.failure_reason == "blocked_safety"
    assert t0.status_code == 400


@pytest.mark.anyio
async def test_fallback_telemetry_unexpected_exception_re_raised() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []
    transitions: list[FallbackTransition] = []

    p1 = RecordingMockWorkerProvider(
        "p1",
        RuntimeError("Database crashed: https://internal.secret.db/tokens?key=abc"),
    )
    p2 = RecordingMockWorkerProvider("p2", "should not run")

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]

    adapter = FallbackInferenceProvider(
        candidates=candidates,
        resolver=lambda name: p1 if name == "p1" else p2,
        on_transition=lambda tr: transitions.append(tr),  # type: ignore[arg-type,return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(RuntimeError) as exc_info:
        await adapter.generate("Unexpected failure prompt")

    assert "https://internal.secret.db" in str(exc_info.value)
    assert p1.invoked is True
    assert p2.invoked is False
    assert len(transitions) == 0

    assert len(telemetry_records) == 1
    t0 = telemetry_records[0]
    assert t0.attempt_index == 0
    assert t0.provider_name == "p1"
    assert t0.status == "unexpected_failure"
    assert t0.failure_reason == "unexpected_error"
    assert t0.status_code is None
    # Verify raw exception text is NOT present in telemetry
    assert "internal.secret.db" not in t0.to_json()
    assert "tokens?key=abc" not in t0.to_json()


@pytest.mark.anyio
async def test_fallback_telemetry_no_telemetry_on_resolver_failure() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    candidates = [_make_candidate("missing_provider", 0)]

    adapter = FallbackInferenceProvider(
        candidates=candidates,
        resolver=lambda name: None,  # type: ignore[return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(TypeError, match="Resolver returned None"):
        await adapter.generate("Prompt")

    assert len(telemetry_records) == 0


@pytest.mark.anyio
async def test_fallback_telemetry_callback_failure_containment(caplog: pytest.LogCaptureFixture) -> None:
    # 8a. Successful attempt with callback failure does not prevent return
    class SuccessProvider:
        async def generate(self, prompt: str) -> str:
            return "Clean output"

    async def broken_telemetry(t: InferenceExecutionTelemetry) -> None:
        raise RuntimeError("Telemetry bus failure")

    adapter = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0)],
        resolver=lambda name: SuccessProvider(),  # type: ignore[return-value]
        on_telemetry=broken_telemetry,
    )
    result = await adapter.generate("Prompt")
    assert result == "Clean output"
    assert "Inference telemetry notification failed for provider p1 (error_type=RuntimeError)" in caplog.text
    assert "Telemetry bus failure" not in caplog.text

    # 8b. Fallback progression is not stopped by callback failure
    p1 = RecordingMockWorkerProvider("p1", FallbackEligibleInferenceError("429", reason="rate_limited"))
    p2 = RecordingMockWorkerProvider("p2", "recovered output")
    adapter_fallback = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0), _make_candidate("p2", 1)],
        resolver=lambda name: p1 if name == "p1" else p2,
        on_telemetry=broken_telemetry,
    )
    res_fallback = await adapter_fallback.generate("Prompt")
    assert res_fallback == "recovered output"

    # 8c. Non-fallback error is preserved and not replaced by callback failure
    p_non_fallback = RecordingMockWorkerProvider(
        "p1",
        NonFallbackInferenceError("Bad request", reason="bad_request", status_code=400),
    )
    adapter_nf = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0)],
        resolver=lambda name: p_non_fallback,
        on_telemetry=broken_telemetry,
    )
    with pytest.raises(NonFallbackInferenceError) as exc_nf:
        await adapter_nf.generate("Prompt")
    assert exc_nf.value.reason == "bad_request"

    # 8d. Unexpected exception is preserved and not replaced by callback failure
    p_unexpected = RecordingMockWorkerProvider("p1", ValueError("Invalid argument in provider"))
    adapter_un = FallbackInferenceProvider(
        candidates=[_make_candidate("p1", 0)],
        resolver=lambda name: p_unexpected,
        on_telemetry=broken_telemetry,
    )
    with pytest.raises(ValueError, match="Invalid argument in provider"):
        await adapter_un.generate("Prompt")


@pytest.mark.anyio
async def test_worker_coordinator_publishes_working_event_with_telemetry() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=InferenceResponse(
            text="Coordinator success",
            provider_name="p1",
            model_name="p1-model",
            input_tokens=12,
            output_tokens=6,
            total_tokens=18,
            finish_reason="stop",
        ),
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-agent-99",
        bus=bus,
        preferences=["p1"],
        providers={"p1": p1},
    )
    task = TaskRequested.create("Verify coordinator publication prompt")
    result = await coordinator.generate(task)

    assert result == "Coordinator success"
    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.state == "working"
    assert event.agent_id == "worker-agent-99"
    assert event.task_id == task.task_id
    assert event.detail == "Inference attempt 0 succeeded (p1)"
    # Verify sanitized detail: no prompt, output, URLs, etc.
    assert "Verify coordinator publication prompt" not in event.detail
    assert "Coordinator success" not in event.detail
    assert event.inference_telemetry is not None
    assert event.inference_telemetry.status == "succeeded"
    assert event.inference_telemetry.provider_name == "p1"
    assert event.inference_telemetry.model_name == "p1-model"
    assert event.inference_telemetry.input_tokens == 12
    assert event.inference_telemetry.output_tokens == 6
    assert event.inference_telemetry.total_tokens == 18


@pytest.mark.anyio
async def test_worker_coordinator_and_task_processor_runtime_compatibility() -> None:
    bus = MockMessageBus()
    p1 = MockRuntimeWorkerProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=FallbackEligibleInferenceError("503", reason="server_error_503"),
    )
    p2 = MockRuntimeWorkerProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        generate_behavior=InferenceResponse(
            text="End-to-end preserved plain string",
            provider_name="p2",
        ),
    )

    coordinator = WorkerFallbackCoordinator(
        agent_id="worker-compat",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    processor = TaskProcessor(
        agent_id="worker-compat",
        bus=bus,
        result_store=bus,
        fallback_coordinator=coordinator,
    )
    task = TaskRequested.create("Runtime compat task")
    await processor.run_once(task)

    assert task.task_id in bus.saved_results
    saved = bus.saved_results[task.task_id]
    assert isinstance(saved, TaskResult)
    assert type(saved.output) is str
    assert saved.output == "End-to-end preserved plain string"

    # Event sequence: task working -> telemetry p1 -> fallback -> telemetry p2 -> idle
    assert len(bus.published) == 5
    assert bus.published[0].state == "working"
    assert bus.published[0].inference_telemetry is None

    assert bus.published[1].state == "working"
    assert bus.published[1].inference_telemetry is not None
    assert bus.published[1].inference_telemetry.status == "fallback_eligible_failure"

    assert bus.published[2].state == "fallback"
    assert bus.published[2].fallback_transition is not None

    assert bus.published[3].state == "working"
    assert bus.published[3].inference_telemetry is not None
    assert bus.published[3].inference_telemetry.status == "succeeded"

    assert bus.published[4].state == "idle"


# ==============================================================================
# Issue #17 Increment 3: Research Agent Inference Execution Telemetry Tests
# ==============================================================================


@pytest.mark.anyio
async def test_research_fallback_telemetry_single_successful_attempt() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    synthesis = SynthesisResult(
        question="What is consensus?",
        content="Paxos and Raft comparison",
        evidence="Evidence text",
        provider="r1",
        model="r1-model-v2",
        finish_reason="stop",
        usage={"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
    )

    class MockResearchProvider:
        async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
            return synthesis

    clock = DeterministicClock([5_000_000, 30_000_000])  # 25.0 ms delta
    candidate = _make_candidate("r1", 0)

    adapter = FallbackResearchSynthesizer(
        candidates=[candidate],
        resolver=lambda name: MockResearchProvider() if name == "r1" else None,  # type: ignore[return-value]
        on_telemetry=lambda t: telemetry_records.append(t),  # type: ignore[arg-type,return-value]
        clock=clock,
    )
    result = await adapter.synthesize("What is consensus?", "Evidence text")

    assert isinstance(result, SynthesisResult)
    assert result == synthesis
    assert len(telemetry_records) == 1

    t = telemetry_records[0]
    assert t.attempt_index == 0
    assert t.status == "succeeded"
    assert t.provider_name == "r1"
    assert t.model_name == "r1-model-v2"
    assert t.finish_reason == "stop"
    assert t.input_tokens == 80
    assert t.output_tokens == 40
    assert t.total_tokens == 120
    assert t.failure_reason is None
    assert t.status_code is None
    assert t.latency_ms == 25.0


@pytest.mark.anyio
async def test_research_fallback_telemetry_missing_metadata_defaults_none() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    synthesis = SynthesisResult(
        question="Question",
        content="Content without usage or model",
        evidence="Evidence",
        provider="r1",
        model="",
        finish_reason=None,
        usage=None,
    )

    class MockResearchProvider:
        async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
            return synthesis

    candidate = _make_candidate("r1", 0)

    adapter = FallbackResearchSynthesizer(
        candidates=[candidate],
        resolver=lambda name: MockResearchProvider() if name == "r1" else None,  # type: ignore[return-value]
        on_telemetry=lambda t: telemetry_records.append(t),  # type: ignore[arg-type,return-value]
    )
    result = await adapter.synthesize("Question", "Evidence")
    assert result == synthesis
    assert len(telemetry_records) == 1

    t = telemetry_records[0]
    assert t.input_tokens is None
    assert t.output_tokens is None
    assert t.total_tokens is None
    assert t.finish_reason is None
    assert t.model_name is None


@pytest.mark.anyio
async def test_research_fallback_telemetry_sequence_primary_fail_secondary_success() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []
    transitions: list[FallbackTransition] = []

    synthesis = SynthesisResult(
        question="What is Paxos?",
        content="Paxos details",
        evidence="Evidence",
        provider="p2",
        model="p2-model",
    )
    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("429 Rate limited", reason="rate_limited", status_code=429),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=synthesis,
    )

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]

    adapter = FallbackResearchSynthesizer(
        candidates=candidates,
        resolver=lambda name: p1 if name == "p1" else p2,
        on_transition=lambda tr: transitions.append(tr),  # type: ignore[arg-type,return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )
    result = await adapter.synthesize("What is Paxos?", "Evidence")

    assert result == synthesis
    assert len(telemetry_records) == 2

    # Attempt 0: p1 fallback-eligible failure
    t0 = telemetry_records[0]
    assert t0.attempt_index == 0
    assert t0.provider_name == "p1"
    assert t0.status == "fallback_eligible_failure"
    assert t0.failure_reason == "rate_limited"
    assert t0.status_code == 429

    # Transition occurred truthfully between attempt 0 and 1
    assert len(transitions) == 1
    assert transitions[0] == FallbackTransition(
        failed_provider="p1",
        target_provider="p2",
        reason="rate_limited",
        attempt_index=0,
    )

    # Attempt 1: p2 success
    t1 = telemetry_records[1]
    assert t1.attempt_index == 1
    assert t1.provider_name == "p2"
    assert t1.status == "succeeded"
    assert t1.failure_reason is None
    assert t1.status_code is None


@pytest.mark.anyio
async def test_research_fallback_telemetry_exhaustion_all_candidates_fail() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("500", reason="server_error_500", status_code=500),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("503", reason="server_error_503", status_code=503),
    )
    p3 = MockRuntimeResearchProvider(
        "p3",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("429", reason="rate_limited", status_code=429),
    )

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1), _make_candidate("p3", 2)]
    providers = {"p1": p1, "p2": p2, "p3": p3}

    adapter = FallbackResearchSynthesizer(
        candidates=candidates,
        resolver=lambda name: providers[name],
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(AllInferenceProvidersFailedError) as exc_info:
        await adapter.synthesize("Research question", "Evidence text")

    assert len(telemetry_records) == 3
    assert [t.attempt_index for t in telemetry_records] == [0, 1, 2]
    assert [t.provider_name for t in telemetry_records] == ["p1", "p2", "p3"]
    assert all(t.status == "fallback_eligible_failure" for t in telemetry_records)
    # Ensure no duplicate terminal telemetry was emitted
    assert len(exc_info.value.failures) == 3


@pytest.mark.anyio
async def test_research_fallback_telemetry_non_fallback_failure_stops() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []
    transitions: list[FallbackTransition] = []

    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=NonFallbackInferenceError("Policy block", reason="blocked_safety", status_code=400),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior="should not be called",
    )

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]

    adapter = FallbackResearchSynthesizer(
        candidates=candidates,
        resolver=lambda name: p1 if name == "p1" else p2,
        on_transition=lambda tr: transitions.append(tr),  # type: ignore[arg-type,return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await adapter.synthesize("Blocked question", "Evidence")

    assert exc_info.value.reason == "blocked_safety"
    assert p1.synthesize_called is True
    assert p2.synthesize_called is False
    assert len(transitions) == 0

    assert len(telemetry_records) == 1
    t0 = telemetry_records[0]
    assert t0.attempt_index == 0
    assert t0.provider_name == "p1"
    assert t0.status == "non_fallback_failure"
    assert t0.failure_reason == "blocked_safety"
    assert t0.status_code == 400


@pytest.mark.anyio
async def test_research_fallback_telemetry_unexpected_failure_re_raised() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []
    transitions: list[FallbackTransition] = []

    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=RuntimeError("Internal GPU failure: https://internal.cluster/api?secret=tok"),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior="should not run",
    )

    candidates = [_make_candidate("p1", 0), _make_candidate("p2", 1)]

    adapter = FallbackResearchSynthesizer(
        candidates=candidates,
        resolver=lambda name: p1 if name == "p1" else p2,
        on_transition=lambda tr: transitions.append(tr),  # type: ignore[arg-type,return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(RuntimeError) as exc_info:
        await adapter.synthesize("Question", "Evidence")

    assert "https://internal.cluster" in str(exc_info.value)
    assert p1.synthesize_called is True
    assert p2.synthesize_called is False
    assert len(transitions) == 0

    assert len(telemetry_records) == 1
    t0 = telemetry_records[0]
    assert t0.attempt_index == 0
    assert t0.provider_name == "p1"
    assert t0.status == "unexpected_failure"
    assert t0.failure_reason == "unexpected_error"
    assert t0.status_code is None
    # Verify raw exception text is absent from telemetry
    assert "internal.cluster" not in t0.to_json()
    assert "secret=tok" not in t0.to_json()


@pytest.mark.anyio
async def test_research_fallback_telemetry_no_telemetry_on_resolver_failure() -> None:
    telemetry_records: list[InferenceExecutionTelemetry] = []

    candidates = [_make_candidate("missing_research_provider", 0)]

    adapter = FallbackResearchSynthesizer(
        candidates=candidates,
        resolver=lambda name: None,  # type: ignore[return-value]
        on_telemetry=lambda te: telemetry_records.append(te),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(TypeError, match="Resolver returned None"):
        await adapter.synthesize("Question", "Evidence")

    assert len(telemetry_records) == 0


@pytest.mark.anyio
async def test_research_fallback_telemetry_callback_failure_containment(caplog: pytest.LogCaptureFixture) -> None:
    synthesis = SynthesisResult(
        question="Question",
        content="Synthesis content",
        evidence="Evidence",
        provider="p1",
    )

    # 8a. Successful attempt with callback failure does not prevent return
    class SuccessProvider:
        async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
            return synthesis

    async def broken_telemetry(t: InferenceExecutionTelemetry) -> None:
        raise RuntimeError("Telemetry collector crashed")

    adapter = FallbackResearchSynthesizer(
        candidates=[_make_candidate("p1", 0)],
        resolver=lambda name: SuccessProvider(),  # type: ignore[return-value]
        on_telemetry=broken_telemetry,
    )
    result = await adapter.synthesize("Question", "Evidence")
    assert result == synthesis
    assert "Inference telemetry notification failed for provider p1 (error_type=RuntimeError)" in caplog.text
    assert "Telemetry collector crashed" not in caplog.text

    # 8b. Fallback progression continues despite callback failure
    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("429", reason="rate_limited"),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=synthesis,
    )
    adapter_fallback = FallbackResearchSynthesizer(
        candidates=[_make_candidate("p1", 0), _make_candidate("p2", 1)],
        resolver=lambda name: p1 if name == "p1" else p2,
        on_telemetry=broken_telemetry,
    )
    res_fallback = await adapter_fallback.synthesize("Question", "Evidence")
    assert res_fallback == synthesis

    # 8c. Non-fallback error is preserved despite callback failure
    p_non_fallback = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=NonFallbackInferenceError("400", reason="bad_request", status_code=400),
    )
    adapter_nf = FallbackResearchSynthesizer(
        candidates=[_make_candidate("p1", 0)],
        resolver=lambda name: p_non_fallback,
        on_telemetry=broken_telemetry,
    )
    with pytest.raises(NonFallbackInferenceError) as exc_nf:
        await adapter_nf.synthesize("Question", "Evidence")
    assert exc_nf.value.reason == "bad_request"

    # 8d. Unexpected exception is preserved despite callback failure
    p_unexpected = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=ValueError("Value error in research synthesis"),
    )
    adapter_un = FallbackResearchSynthesizer(
        candidates=[_make_candidate("p1", 0)],
        resolver=lambda name: p_unexpected,
        on_telemetry=broken_telemetry,
    )
    with pytest.raises(ValueError, match="Value error in research synthesis"):
        await adapter_un.synthesize("Question", "Evidence")


@pytest.mark.anyio
async def test_research_coordinator_publishes_working_event_with_telemetry() -> None:
    bus = MockMessageBus()
    synthesis = SynthesisResult(
        question="Analysis question",
        content="Research synthesis findings",
        evidence="Grounded evidence",
        provider="p1",
        model="gemini-2.5-flash",
        finish_reason="stop",
        usage={"promptTokenCount": 55, "candidatesTokenCount": 25, "totalTokenCount": 80},
    )
    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=synthesis,
    )

    coordinator = ResearchFallbackCoordinator(
        agent_id="research-coordinator-test",
        bus=bus,
        preferences=["p1"],
        providers={"p1": p1},
    )
    result = await coordinator.synthesize(
        question="Analysis question",
        evidence="Grounded evidence",
        task_id="task-rc-101",
    )

    assert result == synthesis
    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.state == "working"
    assert event.agent_id == "research-coordinator-test"
    assert event.task_id == "task-rc-101"
    assert event.detail == "Inference attempt 0 succeeded (p1)"
    # Verify sanitized detail: no question, evidence, synthesis content, URLs, headers, or credentials
    assert "Analysis question" not in event.detail
    assert "Grounded evidence" not in event.detail
    assert "Research synthesis findings" not in event.detail

    assert event.inference_telemetry is not None
    assert event.inference_telemetry.status == "succeeded"
    assert event.inference_telemetry.provider_name == "p1"
    assert event.inference_telemetry.model_name == "gemini-2.5-flash"
    assert event.inference_telemetry.finish_reason == "stop"
    assert event.inference_telemetry.input_tokens == 55
    assert event.inference_telemetry.output_tokens == 25
    assert event.inference_telemetry.total_tokens == 80


@pytest.mark.anyio
async def test_inference_telemetry_event_stream_serialization() -> None:
    """Verifies that inference_telemetry survives AgentEvent -> Redis mapping serialization."""
    telemetry = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=1,
        latency_ms=45.5,
        status="succeeded",
        input_tokens=150,
        output_tokens=75,
        total_tokens=225,
        finish_reason="stop",
        model_name="@cf/meta/llama-3.1-8b-instruct",
    )
    event = AgentEvent.create(
        agent_id="research-agent-1",
        state="working",
        detail="Inference attempt 1 succeeded (cloudflare)",
        task_id="task-serialization-1",
        inference_telemetry=telemetry,
    )

    # Use existing Redis mapping serialization logic from RedisBus
    mapping = {
        key: ""
        if value is None
        else (
            json.dumps(value)
            if isinstance(value, (dict, list))
            else str(value)
        )
        for key, value in event.to_dict().items()
    }
    mapping["updated_at"] = event.occurred_at

    assert "inference_telemetry" in mapping
    raw_telem_json = mapping["inference_telemetry"]
    assert isinstance(raw_telem_json, str)
    deserialized_data = json.loads(raw_telem_json)
    restored_telem = InferenceExecutionTelemetry.from_dict(deserialized_data)
    assert restored_telem == telemetry
    assert restored_telem.provider_name == "cloudflare"
    assert restored_telem.attempt_index == 1
    assert restored_telem.latency_ms == 45.5
    assert restored_telem.input_tokens == 150
    assert restored_telem.output_tokens == 75
    assert restored_telem.total_tokens == 225


@pytest.mark.anyio
async def test_issue_19_fallback_telemetry_coexistence_and_ordering() -> None:
    """Verifies #19 evidence: FallbackTransition and per-attempt InferenceExecutionTelemetry coexist in order."""
    bus = MockMessageBus()
    synthesis = SynthesisResult(
        question="Consensus models",
        content="Raft comparison findings",
        evidence="Evidence text",
        provider="p2",
        model="m2",
    )
    p1 = MockRuntimeResearchProvider(
        "p1",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=FallbackEligibleInferenceError("429 Rate limit", reason="rate_limited", status_code=429),
    )
    p2 = MockRuntimeResearchProvider(
        "p2",
        health_status=InferenceHealthStatus.HEALTHY,
        synthesize_behavior=synthesis,
    )

    coordinator = ResearchFallbackCoordinator(
        agent_id="research-issue-19",
        bus=bus,
        preferences=["p1", "p2"],
        providers={"p1": p1, "p2": p2},
    )
    result = await coordinator.synthesize("Consensus models", "Evidence text", task_id="task-19-order")
    assert result == synthesis

    # Event sequence:
    # 1. Primary attempt telemetry (fallback_eligible_failure)
    # 2. Fallback transition event
    # 3. Secondary attempt telemetry (succeeded)
    assert len(bus.published) == 3

    ev_telem0 = bus.published[0]
    assert ev_telem0.state == "working"
    assert ev_telem0.task_id == "task-19-order"
    assert ev_telem0.inference_telemetry is not None
    assert ev_telem0.inference_telemetry.status == "fallback_eligible_failure"
    assert ev_telem0.inference_telemetry.provider_name == "p1"
    assert ev_telem0.inference_telemetry.attempt_index == 0
    assert ev_telem0.inference_telemetry.failure_reason == "rate_limited"
    assert ev_telem0.inference_telemetry.status_code == 429

    ev_transition = bus.published[1]
    assert ev_transition.state == "fallback"
    assert ev_transition.task_id == "task-19-order"
    assert ev_transition.fallback_transition == FallbackTransition("p1", "p2", "rate_limited", 0)

    ev_telem1 = bus.published[2]
    assert ev_telem1.state == "working"
    assert ev_telem1.task_id == "task-19-order"
    assert ev_telem1.inference_telemetry is not None
    assert ev_telem1.inference_telemetry.status == "succeeded"
    assert ev_telem1.inference_telemetry.provider_name == "p2"
    assert ev_telem1.inference_telemetry.attempt_index == 1
