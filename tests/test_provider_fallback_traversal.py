"""Deterministic inference provider fallback traversal tests (Issue #16 Increment 2)."""

from __future__ import annotations

import pytest
from agent_contracts import (
    AllInferenceProvidersFailedError,
    FallbackEligibleInferenceError,
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
