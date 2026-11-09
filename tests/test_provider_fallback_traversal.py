"""Deterministic inference provider fallback traversal tests (Issue #16 Increment 2)."""

from __future__ import annotations

import pytest
from agent_contracts import (
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
    FallbackTransition,
    InferenceError,
    InferenceHealthStatus,
    NonFallbackInferenceError,
    ProviderFailureSummary,
)
from agent_policy import RankedProviderCandidate
from research_agent.application.fallback import FallbackResearchSynthesizer
from research_agent.domain.models import SynthesisResult
from worker.application.fallback import FallbackInferenceProvider


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
