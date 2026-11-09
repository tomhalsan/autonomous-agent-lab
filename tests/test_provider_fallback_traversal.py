"""Deterministic inference provider fallback traversal tests (Issue #16 Increment 2)."""

from __future__ import annotations

import asyncio

import pytest
from agent_contracts import (
    AgentEvent,
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
    FallbackTransition,
    InferenceError,
    InferenceHealthStatus,
    InferenceProviderHealth,
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
    assert len(bus.published) == 0


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

    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.state == "fallback"
    assert event.agent_id == "worker-1"
    assert event.task_id == task.task_id
    assert "rate_limited" in event.detail
    assert event.fallback_transition == FallbackTransition(
        failed_provider="p1",
        target_provider="p2",
        reason="rate_limited",
        attempt_index=0,
    )


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
    assert len(bus.published) == 0


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

    # Event sequence: working -> fallback -> idle
    assert len(bus.published) == 3
    ev_working, ev_fallback, ev_idle = bus.published

    assert ev_working.state == "working"
    assert ev_working.task_id == task.task_id

    assert ev_fallback.state == "fallback"
    assert ev_fallback.task_id == task.task_id
    assert ev_fallback.fallback_transition == FallbackTransition("p1", "p2", "server_error_503", 0)

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

    # Event sequence: working -> fallback -> error -> idle
    assert len(bus.published) == 4
    ev_working, ev_fallback, ev_error, ev_idle = bus.published

    assert ev_working.state == "working"
    assert ev_fallback.state == "fallback"
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
    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.state == "fallback"
    assert event.agent_id == "research-1"
    assert event.task_id == "task-r-1"
    assert event.fallback_transition == FallbackTransition("p1", "p2", "server_error_500", 0)


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

    # Lifecycle: working -> fallback -> idle
    assert len(bus.published) == 3
    ev_working, ev_fallback, ev_idle = bus.published

    assert ev_working.state == "working"
    assert ev_fallback.state == "fallback"
    assert ev_fallback.fallback_transition == FallbackTransition("p1", "p2", "timeout", 0)
    assert ev_idle.state == "idle"

    assert task.task_id in bus.saved_results
    saved = bus.saved_results[task.task_id]
    assert saved.output == "Synthesized research findings"
    assert saved.sources == ("https://example.com/ref1",)
