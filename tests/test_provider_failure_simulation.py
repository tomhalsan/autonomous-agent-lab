"""Deterministic offline provider failure simulation and fault injection tests (Issue #68).

Validates that:
1. Protocol-level provider faults (HTTP 429, 500, 503, timeouts, malformed responses)
   can be deterministically simulated without live credentials or network access.
2. Provider failures do not crash the Worker or Research Agent processes.
3. Provider failures never corrupt task or result state (zero persistence of failed tasks).
4. Simulated failures emit appropriate observability telemetry with credentials redacted.
5. The Worker and Research Agent recover cleanly to process subsequent valid tasks.
6. The architectural boundary for fallback (raw fault -> health probe -> InferenceHealthStatus
   -> DeterministicProviderSelector) operates correctly across simulated faults.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from agent_contracts import (
    AgentEvent,
    FallbackEligibleInferenceError,
    InferenceError,
    NonFallbackInferenceError,
    TaskRequested,
    TaskResult,
)
from agent_infrastructure import CloudflareHealthProbe, GeminiHealthProbe
from agent_policy import DeterministicProviderSelector
from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as ResearchCloudflareProvider,
)
from research_agent.infrastructure.gemini_inference import (
    GeminiInferenceProvider as ResearchGeminiProvider,
)
from research_agent.infrastructure.stub_executor import StubResearchExecutor
from research_agent.infrastructure.stub_inference import (
    StubInferenceProvider as ResearchStubProvider,
)
from worker.application.processor import TaskExecutionError, TaskProcessor
from worker.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as WorkerCloudflareProvider,
)
from worker.infrastructure.gemini_inference import (
    GeminiInferenceProvider as WorkerGeminiProvider,
)
from worker.infrastructure.stub_inference import (
    StubInferenceProvider as WorkerStubProvider,
)

# ==============================================================================
# Test Fakes for MessageBus and ResultStore
# ==============================================================================


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError


class FakeWorkerResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, TaskResult] = {}

    async def save(self, result: TaskResult) -> None:
        self.saved[result.task_id] = result


class FakeResearchResultStore:
    def __init__(self) -> None:
        self.saved: dict[str, ResearchResult] = {}

    async def save(self, result: ResearchResult) -> None:
        self.saved[result.task_id] = result


# ==============================================================================
# Deterministic Fault-Injecting HTTP Client
# ==============================================================================


class FaultInjectingClient:
    """Mock HTTP client that injects a specific fault mode on first call, then recovers on subsequent calls."""

    def __init__(
        self,
        provider: str,
        fault_mode: str,
        secret_token: str,
        recovery_response: str = "Recovered response content.",
    ) -> None:
        self.provider = provider
        self.fault_mode = fault_mode
        self.secret_token = secret_token
        self.recovery_response = recovery_response
        self.call_count = 0

    async def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        self.call_count += 1
        req = httpx.Request("POST", url, headers=headers)
        if self.call_count == 1:
            return self._inject_fault(req)
        return self._make_success_response(req)

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: Any = None,
    ) -> httpx.Response:
        self.call_count += 1
        req = httpx.Request("GET", url, headers=headers)
        if self.call_count == 1:
            return self._inject_fault(req)
        return self._make_success_probe_response(req)

    def _inject_fault(self, req: httpx.Request) -> httpx.Response:
        if self.fault_mode == "timeout":
            raise httpx.TimeoutException(f"Transport timeout occurred with {self.secret_token}")

        if self.fault_mode == "malformed_json":
            return httpx.Response(
                status_code=200,
                content=b"<html>502 Bad Gateway</html>",
                headers={"content-type": "text/html"},
                request=req,
            )

        if self.fault_mode == "empty_candidates":
            return httpx.Response(
                status_code=200,
                json={"candidates": []},
                request=req,
            )

        if self.fault_mode == "safety_block" and self.provider == "gemini":
            return httpx.Response(
                status_code=200,
                json={
                    "promptFeedback": {
                        "blockReason": f"SAFETY: {self.secret_token}",
                    },
                },
                request=req,
            )

        if self.fault_mode == "safety_finish_reason" and self.provider == "gemini":
            return httpx.Response(
                status_code=200,
                json={
                    "candidates": [
                        {
                            "finishReason": "SAFETY",
                            "content": {"parts": []},
                        }
                    ]
                },
                request=req,
            )

        if self.fault_mode == "unclassified_error_envelope":
            if self.provider == "cloudflare":
                return httpx.Response(
                    status_code=200,
                    json={
                        "success": False,
                        "errors": [{"code": 99999, "message": f"Unrecognized API error with {self.secret_token}"}],
                    },
                    request=req,
                )
            if self.provider == "gemini":
                return httpx.Response(
                    status_code=200,
                    json={
                        "error": {
                            "code": 99999,
                            "message": f"Unrecognized API error with {self.secret_token}",
                        }
                    },
                    request=req,
                )

        if self.provider == "cloudflare":
            if self.fault_mode in ("http_400", "http_401", "http_403", "http_404"):
                code = int(self.fault_mode.split("_")[1])
                return httpx.Response(
                    status_code=code,
                    json={
                        "success": False,
                        "errors": [{"code": 10000 + code, "message": f"Client error {code} with {self.secret_token}"}],
                    },
                    request=req,
                )
            if self.fault_mode == "http_429":
                return httpx.Response(
                    status_code=429,
                    json={
                        "success": False,
                        "errors": [{"code": 10014, "message": f"Rate limit exceeded with {self.secret_token}"}],
                    },
                    headers={"Retry-After": "30"},
                    request=req,
                )
            if self.fault_mode == "http_500":
                return httpx.Response(
                    status_code=500,
                    json={
                        "success": False,
                        "errors": [{"code": 10000, "message": f"Internal server error with {self.secret_token}"}],
                    },
                    request=req,
                )
            if self.fault_mode == "http_503":
                return httpx.Response(
                    status_code=503,
                    json={
                        "success": False,
                        "errors": [{"code": 10001, "message": f"Service unavailable with {self.secret_token}"}],
                    },
                    headers={"Retry-After": "60"},
                    request=req,
                )

        elif self.provider == "gemini":
            if self.fault_mode == "http_402":
                return httpx.Response(
                    status_code=402,
                    json={
                        "error": {
                            "code": 402,
                            "message": f"Prepayment credits depleted with {self.secret_token}",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    },
                    request=req,
                )
            if self.fault_mode in ("http_400", "http_401", "http_403", "http_404"):
                code = int(self.fault_mode.split("_")[1])
                return httpx.Response(
                    status_code=code,
                    json={
                        "error": {
                            "code": code,
                            "message": f"Client error {code} with {self.secret_token}",
                            "status": "INVALID_ARGUMENT" if code == 400 else "PERMISSION_DENIED",
                        }
                    },
                    request=req,
                )
            if self.fault_mode == "http_429":
                return httpx.Response(
                    status_code=429,
                    json={
                        "error": {
                            "code": 429,
                            "message": f"Quota exceeded with {self.secret_token}",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    },
                    headers={"Retry-After": "30"},
                    request=req,
                )
            if self.fault_mode == "http_500":
                return httpx.Response(
                    status_code=500,
                    json={
                        "error": {
                            "code": 500,
                            "message": f"Internal server error with {self.secret_token}",
                            "status": "INTERNAL",
                        }
                    },
                    request=req,
                )
            if self.fault_mode == "http_503":
                return httpx.Response(
                    status_code=503,
                    json={
                        "error": {
                            "code": 503,
                            "message": f"Service unavailable with {self.secret_token}",
                            "status": "UNAVAILABLE",
                        }
                    },
                    headers={"Retry-After": "60"},
                    request=req,
                )

        raise ValueError(f"Unsupported provider/fault mode: {self.provider}/{self.fault_mode}")

    def _make_success_response(self, req: httpx.Request) -> httpx.Response:
        if self.provider == "cloudflare":
            return httpx.Response(
                status_code=200,
                json={
                    "result": {"response": self.recovery_response},
                    "success": True,
                    "errors": [],
                    "messages": [],
                },
                request=req,
            )
        if self.provider == "gemini":
            return httpx.Response(
                status_code=200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": self.recovery_response}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
                },
                request=req,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _make_success_probe_response(self, req: httpx.Request) -> httpx.Response:
        if self.provider == "cloudflare":
            return httpx.Response(
                status_code=200,
                json={
                    "result": [{"name": "@cf/meta/llama-3.1-8b-instruct"}],
                    "success": True,
                },
                request=req,
            )
        if self.provider == "gemini":
            return httpx.Response(
                status_code=200,
                json={
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
                request=req,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")


# ==============================================================================
# Helper Instantiators
# ==============================================================================


def _create_worker_faulty_provider(provider_type: str, client: FaultInjectingClient) -> Any:
    if provider_type == "cloudflare":
        return WorkerCloudflareProvider(
            account_id="test-acc",
            api_token=client.secret_token,
            client=client,
        )
    if provider_type == "gemini":
        return WorkerGeminiProvider(
            api_key=client.secret_token,
            client=client,
        )
    raise ValueError(f"Unknown provider type: {provider_type}")


def _create_research_faulty_provider(provider_type: str, client: FaultInjectingClient) -> Any:
    if provider_type == "cloudflare":
        return ResearchCloudflareProvider(
            account_id="test-acc",
            api_token=client.secret_token,
            client=client,
        )
    if provider_type == "gemini":
        return ResearchGeminiProvider(
            api_key=client.secret_token,
            client=client,
        )
    raise ValueError(f"Unknown provider type: {provider_type}")


# ==============================================================================
# Group 1: Infrastructure Adapter Fault Simulation & Credential Redaction
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["cloudflare", "gemini"])
@pytest.mark.parametrize(
    "fault_mode",
    ["http_429", "http_500", "http_503", "timeout", "malformed_json"],
)
async def test_adapter_protocol_fault_simulation_and_sanitization(
    provider_type: str,
    fault_mode: str,
) -> None:
    """Verifies infrastructure adapters transform protocol faults into FallbackEligibleInferenceError without claiming provider recovery."""
    secret_token = f"super-secret-{provider_type}-token-xyz"
    client = FaultInjectingClient(
        provider=provider_type,
        fault_mode=fault_mode,
        secret_token=secret_token,
    )

    worker_provider = _create_worker_faulty_provider(provider_type, client)

    with pytest.raises(FallbackEligibleInferenceError) as exc_info:
        await worker_provider.generate("Test prompt")

    err_text = str(exc_info.value)
    # 1. Error message must not leak credentials
    assert secret_token not in err_text
    # 2. Inheritance and typed attributes
    assert isinstance(exc_info.value, FallbackEligibleInferenceError)
    assert isinstance(exc_info.value, InferenceError)
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, NonFallbackInferenceError)
    assert exc_info.value.provider_name == provider_type
    # 3. Error message captures fault characteristics
    if fault_mode == "http_429":
        assert "429" in err_text
        assert exc_info.value.status_code == 429
        assert exc_info.value.reason == "rate_limited"
    elif fault_mode == "http_500":
        assert "500" in err_text
        assert exc_info.value.status_code == 500
        assert exc_info.value.reason == "server_error_500"
    elif fault_mode == "http_503":
        assert "503" in err_text
        assert exc_info.value.status_code == 503
        assert exc_info.value.reason == "server_error_503"
    elif fault_mode == "timeout":
        assert "timed out" in err_text.lower() or "timeout" in err_text.lower()
        assert exc_info.value.reason == "timeout"
    elif fault_mode == "malformed_json":
        assert "invalid json" in err_text.lower()
        assert exc_info.value.reason == "malformed_response"


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["cloudflare", "gemini"])
@pytest.mark.parametrize(
    "fault_mode",
    ["http_400", "http_401", "http_403", "http_404"],
)
async def test_adapter_non_fallback_fault_simulation(
    provider_type: str,
    fault_mode: str,
) -> None:
    """Verifies infrastructure adapters cleanly transform permanent 4xx protocol faults into NonFallbackInferenceError."""
    secret_token = f"super-secret-perm-{provider_type}-token-xyz"
    client = FaultInjectingClient(
        provider=provider_type,
        fault_mode=fault_mode,
        secret_token=secret_token,
    )

    worker_provider = _create_worker_faulty_provider(provider_type, client)

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await worker_provider.generate("Test prompt")

    err_text = str(exc_info.value)
    # 1. Error message must not leak credentials
    assert secret_token not in err_text
    # 2. Inheritance contracts
    assert isinstance(exc_info.value, InferenceError)
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, FallbackEligibleInferenceError)
    # 3. Attributes
    assert exc_info.value.provider_name == provider_type
    expected_status = int(fault_mode.split("_")[1])
    assert exc_info.value.status_code == expected_status
    assert exc_info.value.reason == f"client_error_{expected_status}"
    assert str(expected_status) in err_text


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["cloudflare", "gemini"])
@pytest.mark.parametrize(
    "fault_mode",
    ["http_429", "http_500", "http_503", "timeout", "malformed_json"],
)
async def test_research_adapter_protocol_fault_simulation(
    provider_type: str,
    fault_mode: str,
) -> None:
    """Verifies research adapters transform protocol faults into FallbackEligibleInferenceError."""
    secret_token = f"super-secret-research-{provider_type}-xyz"
    client = FaultInjectingClient(
        provider=provider_type,
        fault_mode=fault_mode,
        secret_token=secret_token,
    )

    research_provider = _create_research_faulty_provider(provider_type, client)

    with pytest.raises(FallbackEligibleInferenceError) as exc_info:
        await research_provider.synthesize("Test question", "Test evidence")

    assert secret_token not in str(exc_info.value)
    assert isinstance(exc_info.value, FallbackEligibleInferenceError)
    assert isinstance(exc_info.value, InferenceError)
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, NonFallbackInferenceError)
    assert exc_info.value.provider_name == provider_type


@pytest.mark.anyio
async def test_gemini_http_402_payment_required_fallback_classification() -> None:
    """Verifies Gemini HTTP 402 generation failure raises FallbackEligibleInferenceError with reason payment_required."""
    secret_token = "secret-gemini-payment-token-402"
    client = FaultInjectingClient(
        provider="gemini",
        fault_mode="http_402",
        secret_token=secret_token,
    )

    # 1. Worker adapter
    worker_provider = _create_worker_faulty_provider("gemini", client)
    with pytest.raises(FallbackEligibleInferenceError) as exc_info:
        await worker_provider.generate("Test prompt for worker")

    err_text = str(exc_info.value)
    assert secret_token not in err_text
    assert "Prepayment credits depleted" in err_text
    assert "402" in err_text
    assert isinstance(exc_info.value, FallbackEligibleInferenceError)
    assert isinstance(exc_info.value, InferenceError)
    assert not isinstance(exc_info.value, NonFallbackInferenceError)
    assert exc_info.value.provider_name == "gemini"
    assert exc_info.value.reason == "payment_required"
    assert exc_info.value.status_code == 402
    assert client.call_count == 1

    # 2. Research adapter
    research_client = FaultInjectingClient(
        provider="gemini",
        fault_mode="http_402",
        secret_token=secret_token,
    )
    research_provider = _create_research_faulty_provider("gemini", research_client)
    with pytest.raises(FallbackEligibleInferenceError) as res_exc_info:
        await research_provider.synthesize("Research question", "Evidence text")

    res_err_text = str(res_exc_info.value)
    assert secret_token not in res_err_text
    assert "Prepayment credits depleted" in res_err_text
    assert "402" in res_err_text
    assert isinstance(res_exc_info.value, FallbackEligibleInferenceError)
    assert res_exc_info.value.provider_name == "gemini"
    assert res_exc_info.value.reason == "payment_required"
    assert res_exc_info.value.status_code == 402
    assert research_client.call_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("block_mode", ["safety_block", "safety_finish_reason"])
async def test_gemini_blocked_generation_is_non_fallback(block_mode: str) -> None:
    """Verifies explicit policy/safety blocks are classified as NonFallbackInferenceError, not fallback eligible."""
    secret_token = "secret-safety-prompt-leak-token"
    client = FaultInjectingClient(
        provider="gemini",
        fault_mode=block_mode,
        secret_token=secret_token,
    )
    provider = _create_worker_faulty_provider("gemini", client)

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await provider.generate("Test prompt triggering safety policy")

    err_text = str(exc_info.value)
    # 1. Credentials/tokens are sanitized
    assert secret_token not in err_text
    # 2. Inheritance
    assert isinstance(exc_info.value, InferenceError)
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, FallbackEligibleInferenceError)
    # 3. Explicit block characteristics
    assert "blocked" in err_text.lower()
    assert exc_info.value.reason is not None and "blocked" in exc_info.value.reason.lower()
    assert exc_info.value.provider_name == "gemini"


@pytest.mark.anyio
async def test_gemini_malformed_response_distinguishable_from_block() -> None:
    """Verifies missing candidates without safety blockReason is classified as FallbackEligibleInferenceError."""
    secret_token = "secret-malformed-response-token"
    client = FaultInjectingClient(
        provider="gemini",
        fault_mode="empty_candidates",
        secret_token=secret_token,
    )
    provider = _create_worker_faulty_provider("gemini", client)

    with pytest.raises(FallbackEligibleInferenceError) as exc_info:
        await provider.generate("Test prompt")

    err_text = str(exc_info.value)
    assert secret_token not in err_text
    assert isinstance(exc_info.value, InferenceError)
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, NonFallbackInferenceError)
    assert exc_info.value.reason == "malformed_response"
    assert "candidates" in err_text.lower()


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["cloudflare", "gemini"])
async def test_adapter_unclassified_error_envelope_is_non_fallback(
    provider_type: str,
) -> None:
    """Verifies error envelopes with unrecognized non-fallback status codes default to NonFallbackInferenceError."""
    secret_token = f"super-secret-unclassified-{provider_type}-xyz"
    client = FaultInjectingClient(
        provider=provider_type,
        fault_mode="unclassified_error_envelope",
        secret_token=secret_token,
    )
    provider = _create_worker_faulty_provider(provider_type, client)

    with pytest.raises(NonFallbackInferenceError) as exc_info:
        await provider.generate("Test prompt")

    err_text = str(exc_info.value)
    # 1. Credentials are not leaked
    assert secret_token not in err_text
    # 2. Inheritance
    assert isinstance(exc_info.value, InferenceError)
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, FallbackEligibleInferenceError)
    # 3. Explicit unclassified reason
    assert exc_info.value.provider_name == provider_type
    assert exc_info.value.reason == "unclassified_api_error"


# ==============================================================================
# Group 2: Worker TaskProcessor Fault Containment, Telemetry & Recovery
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["cloudflare", "gemini"])
@pytest.mark.parametrize(
    "fault_mode",
    ["http_429", "http_500", "http_503", "timeout", "malformed_json"],
)
async def test_worker_task_processor_fault_containment_and_recovery(
    provider_type: str,
    fault_mode: str,
) -> None:
    """Verifies that worker provider failure does not crash the agent, corrupt state, and allows recovery."""
    secret_token = f"secret-worker-key-{provider_type}-{fault_mode}"
    client = FaultInjectingClient(
        provider=provider_type,
        fault_mode=fault_mode,
        secret_token=secret_token,
        recovery_response="Recovered worker task response.",
    )
    provider = _create_worker_faulty_provider(provider_type, client)
    bus = FakeMessageBus()
    store = FakeWorkerResultStore()

    processor = TaskProcessor(
        agent_id="worker-failure-test",
        bus=bus,
        result_store=store,
        inference_provider=provider,
    )

    # 1. Dispatch failing task
    task_failing = TaskRequested.create("Execute database index optimization")
    with pytest.raises(TaskExecutionError) as exc_info:
        await processor.run_once(task_failing)

    # Assertion A: TaskExecutionError correlates to the task
    assert exc_info.value.task_id == task_failing.task_id

    # Assertion B: No result is persisted (state is not corrupted)
    assert task_failing.task_id not in store.saved
    assert len(store.saved) == 0

    # Assertion C: Lifecycle telemetry records working -> error -> idle
    assert len(bus.published) == 3
    assert bus.published[0].state == "working"
    assert bus.published[0].task_id == task_failing.task_id

    assert bus.published[1].state == "error"
    assert bus.published[1].task_id == task_failing.task_id
    error_detail = bus.published[1].detail

    assert bus.published[2].state == "idle"
    assert bus.published[2].detail == "Waiting for work"

    # Assertion D: Error detail is sanitized (secrets are redacted)
    assert secret_token not in error_detail
    assert secret_token not in str(exc_info.value)

    # Assertion E: Worker recovers and successfully processes a subsequent valid task
    task_recovery = TaskRequested.create("Analyze connection pool metrics")
    await processor.run_once(task_recovery)

    assert task_recovery.task_id in store.saved
    saved_result = store.saved[task_recovery.task_id]
    assert isinstance(saved_result, TaskResult)
    assert saved_result.task_id == task_recovery.task_id
    assert saved_result.status == "completed"
    assert saved_result.output == "Recovered worker task response."

    # Verify event stream for recovered task (working -> idle)
    assert len(bus.published) == 5
    assert bus.published[3].state == "working"
    assert bus.published[3].task_id == task_recovery.task_id
    assert bus.published[4].state == "idle"
    assert bus.published[4].task_id == task_recovery.task_id


# ==============================================================================
# Group 3: Research Agent ResearchProcessor Fault Containment, Telemetry & Recovery
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("provider_type", ["cloudflare", "gemini"])
@pytest.mark.parametrize(
    "fault_mode",
    ["http_429", "http_500", "http_503", "timeout", "malformed_json"],
)
async def test_research_processor_fault_containment_and_recovery(
    provider_type: str,
    fault_mode: str,
) -> None:
    """Verifies that research provider failure does not crash the agent, corrupt state, and allows recovery."""
    secret_token = f"secret-research-key-{provider_type}-{fault_mode}"
    client = FaultInjectingClient(
        provider=provider_type,
        fault_mode=fault_mode,
        secret_token=secret_token,
        recovery_response="Recovered research synthesis findings.\n\nSources:\n- https://example.com",
    )
    provider = _create_research_faulty_provider(provider_type, client)
    bus = FakeMessageBus()
    store = FakeResearchResultStore()
    executor = StubResearchExecutor()

    processor = ResearchProcessor(
        agent_id="research-failure-test",
        bus=bus,
        executor=executor,
        result_store=store,
        inference_provider=provider,
    )

    # 1. Dispatch failing task
    task_failing = TaskRequested.create("Investigate distributed deadlock resolution")
    with pytest.raises(ResearchExecutionError) as exc_info:
        await processor.run_once(task_failing)

    # Assertion A: ResearchExecutionError correlates to the task
    assert exc_info.value.task_id == task_failing.task_id

    # Assertion B: No result is persisted (state is not corrupted)
    assert task_failing.task_id not in store.saved
    assert len(store.saved) == 0

    # Assertion C: Lifecycle telemetry records the architecture's existing failure state behavior (working -> error)
    assert len(bus.published) == 2
    assert bus.published[0].state == "working"
    assert bus.published[0].task_id == task_failing.task_id

    assert bus.published[1].state == "error"
    assert bus.published[1].task_id == task_failing.task_id
    error_detail = bus.published[1].detail

    # Assertion D: Error detail is sanitized (secrets are redacted)
    assert secret_token not in error_detail
    assert secret_token not in str(exc_info.value)

    # Assertion E: Research Agent recovers and successfully processes a subsequent valid task
    task_recovery = TaskRequested.create("Investigate state machine replication")
    await processor.run_once(task_recovery)

    assert task_recovery.task_id in store.saved
    saved_result = store.saved[task_recovery.task_id]
    assert isinstance(saved_result, ResearchResult)
    assert saved_result.task_id == task_recovery.task_id
    assert saved_result.acknowledged is True
    assert "Recovered research synthesis" in saved_result.output

    # Verify event stream for recovered task (working -> idle)
    assert len(bus.published) == 4
    assert bus.published[2].state == "working"
    assert bus.published[2].task_id == task_recovery.task_id
    assert bus.published[3].state == "idle"
    assert bus.published[3].task_id == task_recovery.task_id


# ==============================================================================
# Group 4: Stub Provider Failure Containment & Recovery
# ==============================================================================


@pytest.mark.anyio
async def test_stub_provider_failure_containment_and_recovery() -> None:
    """Verifies that deterministic stub failure triggers preserve state and allow recovery across both agents."""
    trigger = "SIMULATED_CRASH_TRIGGER"

    # 1. Worker with Stub failure trigger
    worker_stub = WorkerStubProvider(provider_name="stub", failure_trigger=trigger)
    worker_bus = FakeMessageBus()
    worker_store = FakeWorkerResultStore()
    worker_proc = TaskProcessor("worker-stub", worker_bus, result_store=worker_store, inference_provider=worker_stub)

    failing_worker_task = TaskRequested.create(f"Task with {trigger}")
    with pytest.raises(TaskExecutionError):
        await worker_proc.run_once(failing_worker_task)

    assert len(worker_store.saved) == 0
    assert [e.state for e in worker_bus.published] == ["working", "error", "idle"]

    # Recovery
    valid_worker_task = TaskRequested.create("Normal worker task")
    await worker_proc.run_once(valid_worker_task)
    assert valid_worker_task.task_id in worker_store.saved

    # 2. Research Agent with Stub failure trigger
    research_stub = ResearchStubProvider(provider_name="stub", failure_trigger=trigger)
    research_bus = FakeMessageBus()
    research_store = FakeResearchResultStore()
    research_proc = ResearchProcessor(
        "research-stub",
        research_bus,
        StubResearchExecutor(),
        result_store=research_store,
        inference_provider=research_stub,
    )

    failing_research_task = TaskRequested.create(f"Question with {trigger}")
    with pytest.raises(ResearchExecutionError):
        await research_proc.run_once(failing_research_task)

    assert len(research_store.saved) == 0
    assert [e.state for e in research_bus.published] == ["working", "error"]

    # Recovery
    valid_research_task = TaskRequested.create("Normal research question")
    await research_proc.run_once(valid_research_task)
    assert valid_research_task.task_id in research_store.saved


# ==============================================================================
# Group 5: Architectural Boundary: Raw Fault -> Health Probe -> Selector Fallback
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fault_mode", "expected_unhealthy_status"),
    [
        ("http_429", "rate_limited"),
        ("http_500", "unavailable"),
        ("http_503", "unavailable"),
        ("timeout", "unavailable"),
    ],
)
async def test_provider_fault_to_selection_fallback_boundary(
    fault_mode: str,
    expected_unhealthy_status: str,
) -> None:
    """Demonstrates architectural boundary: raw HTTP fault -> health probe -> InferenceHealthStatus -> selector fallback.

    Respects Clean Architecture:
    - Raw HTTP faults are classified by infrastructure probes into normalized InferenceHealthStatus.
    - DeterministicProviderSelector consumes normalized health snapshots (not HTTP responses)
      and deterministically falls back to the healthy secondary candidate.
    """
    # 1. Primary provider probe encounters simulated fault
    primary_client = FaultInjectingClient(
        provider="cloudflare",
        fault_mode=fault_mode,
        secret_token="primary-secret",
    )
    primary_probe = CloudflareHealthProbe(
        account_id="acc-1",
        api_token="primary-secret",
        client=primary_client,
    )
    primary_health = await primary_probe.check_health()
    assert primary_health.status.value == expected_unhealthy_status
    assert not primary_health.status.is_available

    # 2. Secondary provider probe is healthy (call_count set to 1 so probe succeeds)
    secondary_client = FaultInjectingClient(
        provider="gemini",
        fault_mode="http_500",
        secret_token="sec-secret",
    )
    secondary_client.call_count = 1
    secondary_probe = GeminiHealthProbe(
        api_key="sec-secret",
        client=secondary_client,
    )
    secondary_health = await secondary_probe.check_health()
    assert secondary_health.status.is_available
    assert secondary_health.status.value == "healthy"

    # 3. DeterministicProviderSelector receives normalized snapshots and falls back to secondary
    selector = DeterministicProviderSelector()
    decision = selector.select(
        preferences=["cloudflare", "gemini"],
        health_snapshots=[primary_health, secondary_health],
    )

    # Fallback verification: secondary provider selected because primary is constrained/unavailable
    assert decision.selected_provider == "gemini"
    assert decision.status.value == "healthy"
    assert any(c.provider_name == "cloudflare" for c in decision.excluded_candidates)
    assert decision.ranked_candidates[0].provider_name == "gemini"
