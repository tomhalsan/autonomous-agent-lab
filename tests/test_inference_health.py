from dataclasses import FrozenInstanceError
from typing import Any

import httpx
import pytest
from agent_contracts import (
    InferenceHealthStatus,
    InferenceProviderHealth,
)
from agent_infrastructure import CloudflareHealthProbe
from research_agent.application.ports import (
    InferenceHealthProbe as ResearchInferenceHealthProbe,
)
from research_agent.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as ResearchCloudflareProvider,
)
from research_agent.infrastructure.stub_inference import (
    StubInferenceProvider as ResearchStubProvider,
)
from worker.application.ports import InferenceHealthProbe as WorkerInferenceHealthProbe
from worker.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as WorkerCloudflareProvider,
)
from worker.infrastructure.stub_inference import (
    StubInferenceProvider as WorkerStubProvider,
)


class DummyAsyncClient:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.calls.append({"method": "GET", "url": url, "headers": headers, "params": params})
        return self.handler("GET", url, headers, params=params)

    async def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json})
        return self.handler("POST", url, headers, json_data=json)


def test_inference_health_status_properties() -> None:
    # Availability: HEALTHY and DEGRADED can accept inference
    assert InferenceHealthStatus.HEALTHY.is_available is True
    assert InferenceHealthStatus.DEGRADED.is_available is True
    assert InferenceHealthStatus.RATE_LIMITED.is_available is False
    assert InferenceHealthStatus.QUOTA_EXHAUSTED.is_available is False
    assert InferenceHealthStatus.UNAVAILABLE.is_available is False

    # Connectivity: All except UNAVAILABLE are connected/communicating
    assert InferenceHealthStatus.HEALTHY.is_connected is True
    assert InferenceHealthStatus.DEGRADED.is_connected is True
    assert InferenceHealthStatus.RATE_LIMITED.is_connected is True
    assert InferenceHealthStatus.QUOTA_EXHAUSTED.is_connected is True
    assert InferenceHealthStatus.UNAVAILABLE.is_connected is False

    # Constrained: RATE_LIMITED and QUOTA_EXHAUSTED
    assert InferenceHealthStatus.HEALTHY.is_constrained is False
    assert InferenceHealthStatus.DEGRADED.is_constrained is False
    assert InferenceHealthStatus.RATE_LIMITED.is_constrained is True
    assert InferenceHealthStatus.QUOTA_EXHAUSTED.is_constrained is True
    assert InferenceHealthStatus.UNAVAILABLE.is_constrained is False


def test_inference_provider_health_defaults_and_immutability() -> None:
    health = InferenceProviderHealth(
        provider_name="test-provider",
        status=InferenceHealthStatus.HEALTHY,
    )
    assert health.provider_name == "test-provider"
    assert health.status == InferenceHealthStatus.HEALTHY
    assert health.model is None
    assert health.latency_ms is None
    assert health.retry_after_seconds is None
    assert health.reason == ""
    assert isinstance(health.checked_at, str) and len(health.checked_at) > 0

    with pytest.raises(FrozenInstanceError):
        health.status = InferenceHealthStatus.UNAVAILABLE  # type: ignore[misc]


def test_inference_provider_health_status_string_coercion() -> None:
    health = InferenceProviderHealth(
        provider_name="test",
        status="healthy",  # type: ignore[arg-type]
    )
    assert health.status == InferenceHealthStatus.HEALTHY


def test_inference_provider_health_validation_provider_name() -> None:
    with pytest.raises(ValueError, match="provider_name must be a non-empty string"):
        InferenceProviderHealth(provider_name="", status=InferenceHealthStatus.HEALTHY)

    with pytest.raises(ValueError, match="provider_name must be a non-empty string"):
        InferenceProviderHealth(provider_name="   ", status=InferenceHealthStatus.HEALTHY)

    with pytest.raises(ValueError, match="provider_name must be a non-empty string"):
        InferenceProviderHealth(provider_name=None, status=InferenceHealthStatus.HEALTHY)  # type: ignore[arg-type]


def test_inference_provider_health_validation_status() -> None:
    with pytest.raises(ValueError, match="Invalid health status"):
        InferenceProviderHealth(provider_name="test", status="invalid_status")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="status must be an InferenceHealthStatus enum"):
        InferenceProviderHealth(provider_name="test", status=123)  # type: ignore[arg-type]


def test_inference_provider_health_validation_model() -> None:
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        InferenceProviderHealth(
            provider_name="test", status=InferenceHealthStatus.HEALTHY, model="   "
        )


def test_inference_provider_health_validation_numbers() -> None:
    with pytest.raises(TypeError, match="latency_ms must be a non-negative number"):
        InferenceProviderHealth(
            provider_name="test", status=InferenceHealthStatus.HEALTHY, latency_ms="fast"  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="latency_ms must be a non-negative number"):
        InferenceProviderHealth(
            provider_name="test", status=InferenceHealthStatus.HEALTHY, latency_ms=True  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="latency_ms must be non-negative"):
        InferenceProviderHealth(
            provider_name="test", status=InferenceHealthStatus.HEALTHY, latency_ms=-1.0
        )

    with pytest.raises(TypeError, match="retry_after_seconds must be a non-negative number"):
        InferenceProviderHealth(
            provider_name="test",
            status=InferenceHealthStatus.RATE_LIMITED,
            reason="rate limit",
            retry_after_seconds=False,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="retry_after_seconds must be non-negative"):
        InferenceProviderHealth(
            provider_name="test",
            status=InferenceHealthStatus.RATE_LIMITED,
            reason="rate limit",
            retry_after_seconds=-5.0,
        )


def test_inference_provider_health_reason_required_for_constrained_and_unavailable() -> None:
    with pytest.raises(
        ValueError, match="reason must be provided when health status is rate_limited"
    ):
        InferenceProviderHealth(
            provider_name="test",
            status=InferenceHealthStatus.RATE_LIMITED,
            reason="",
        )

    with pytest.raises(
        ValueError, match="reason must be provided when health status is quota_exhausted"
    ):
        InferenceProviderHealth(
            provider_name="test",
            status=InferenceHealthStatus.QUOTA_EXHAUSTED,
            reason="  ",
        )

    with pytest.raises(ValueError, match="reason must be provided when health status is unavailable"):
        InferenceProviderHealth(
            provider_name="test",
            status=InferenceHealthStatus.UNAVAILABLE,
            reason="",
        )

    # Healthy and Degraded allow empty reason
    h1 = InferenceProviderHealth(
        provider_name="test", status=InferenceHealthStatus.HEALTHY, reason=""
    )
    assert h1.reason == ""

    h2 = InferenceProviderHealth(
        provider_name="test", status=InferenceHealthStatus.DEGRADED, reason=""
    )
    assert h2.reason == ""


def test_inference_provider_health_serialization_round_trip() -> None:
    original = InferenceProviderHealth(
        provider_name="cloudflare",
        status=InferenceHealthStatus.RATE_LIMITED,
        model="@cf/meta/llama-3.1-8b-instruct",
        checked_at="2026-11-08T00:00:00+00:00",
        latency_ms=45.2,
        retry_after_seconds=30.0,
        reason="Rate limit reached: 3021",
    )
    data = original.to_dict()
    assert data == {
        "provider_name": "cloudflare",
        "status": "rate_limited",
        "model": "@cf/meta/llama-3.1-8b-instruct",
        "checked_at": "2026-11-08T00:00:00+00:00",
        "latency_ms": 45.2,
        "retry_after_seconds": 30.0,
        "reason": "Rate limit reached: 3021",
    }

    json_str = original.to_json()
    restored = InferenceProviderHealth.from_json(json_str)
    assert restored == original
    assert restored.status == InferenceHealthStatus.RATE_LIMITED


def test_inference_provider_health_from_dict_validation() -> None:
    with pytest.raises(TypeError, match="data must be a dict"):
        InferenceProviderHealth.from_dict("not a dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Missing required field 'provider_name'"):
        InferenceProviderHealth.from_dict({"status": "healthy"})

    with pytest.raises(ValueError, match="Missing required field 'status'"):
        InferenceProviderHealth.from_dict({"provider_name": "stub"})

    with pytest.raises(TypeError, match="provider_name must be a string"):
        InferenceProviderHealth.from_dict({"provider_name": 123, "status": "healthy"})

    with pytest.raises(TypeError, match="status must be a str or InferenceHealthStatus"):
        InferenceProviderHealth.from_dict({"provider_name": "stub", "status": 999})

    with pytest.raises(ValueError, match="Invalid health status"):
        InferenceProviderHealth.from_dict({"provider_name": "stub", "status": "nonexistent"})

    with pytest.raises(TypeError, match="model must be a string or None"):
        InferenceProviderHealth.from_dict(
            {"provider_name": "stub", "status": "healthy", "model": 123}
        )

    with pytest.raises(TypeError, match="reason must be a string"):
        InferenceProviderHealth.from_dict(
            {"provider_name": "stub", "status": "healthy", "reason": 123}
        )


def test_inference_health_probe_protocol_compliance() -> None:
    worker_stub = WorkerStubProvider()
    assert isinstance(worker_stub, WorkerInferenceHealthProbe)

    research_stub = ResearchStubProvider()
    assert isinstance(research_stub, ResearchInferenceHealthProbe)

    worker_cf = WorkerCloudflareProvider(account_id="acc", api_token="tok")
    assert isinstance(worker_cf, WorkerInferenceHealthProbe)

    research_cf = ResearchCloudflareProvider(account_id="acc", api_token="tok")
    assert isinstance(research_cf, ResearchInferenceHealthProbe)

    class NotAProbe:
        pass

    assert not isinstance(NotAProbe(), WorkerInferenceHealthProbe)
    assert not isinstance(NotAProbe(), ResearchInferenceHealthProbe)


@pytest.mark.anyio
async def test_stub_inference_provider_health_check() -> None:
    # Worker stub
    worker_stub = WorkerStubProvider(provider_name="worker-stub", model_name="test-model")
    health = await worker_stub.check_health()
    assert health.provider_name == "worker-stub"
    assert health.model == "test-model"
    assert health.status == InferenceHealthStatus.HEALTHY
    assert health.latency_ms == 1.0

    # Simulate rate limited status
    worker_stub.set_health(InferenceHealthStatus.RATE_LIMITED, retry_after_seconds=60.0)
    health = await worker_stub.check_health()
    assert health.status == InferenceHealthStatus.RATE_LIMITED
    assert health.retry_after_seconds == 60.0
    assert "rate_limited" in health.reason

    # Research agent stub
    research_stub = ResearchStubProvider(provider_name="research-stub", model_name="synth-model")
    r_health = await research_stub.check_health()
    assert r_health.provider_name == "research-stub"
    assert r_health.model == "synth-model"
    assert r_health.status == InferenceHealthStatus.HEALTHY

    research_stub.set_health(InferenceHealthStatus.QUOTA_EXHAUSTED, reason="Monthly quota reached")
    r_health2 = await research_stub.check_health()
    assert r_health2.status == InferenceHealthStatus.QUOTA_EXHAUSTED
    assert r_health2.reason == "Monthly quota reached"


def test_adapters_delegate_to_shared_cloudflare_probe() -> None:
    worker_cf = WorkerCloudflareProvider(account_id="acc1", api_token="tok1", model="m1")
    research_cf = ResearchCloudflareProvider(account_id="acc2", api_token="tok2", model="m2")

    assert isinstance(worker_cf._health_probe, CloudflareHealthProbe)
    assert isinstance(research_cf._health_probe, CloudflareHealthProbe)
    assert worker_cf._health_probe.account_id == "acc1"
    assert research_cf._health_probe.account_id == "acc2"


@pytest.mark.anyio
async def test_cloudflare_health_check_successful_catalog_access() -> None:
    calls = []

    def mock_handler(method, url, headers, params=None, json_data=None):
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "json": json_data,
            }
        )
        return httpx.Response(
            200,
            json={"success": True, "result": [{"id": "@cf/meta/llama-3.1-8b-instruct"}]},
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.HEALTHY
    assert health.provider_name == "cloudflare"
    assert health.model == "@cf/meta/llama-3.1-8b-instruct"
    assert "verified model '@cf/meta/llama-3.1-8b-instruct' is accessible" in health.reason
    assert health.latency_ms is not None and health.latency_ms >= 0.0

    # Ensure zero-token verification: GET /ai/models/search was called, NOT /ai/run
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/models/search"
    assert call["params"] == {"search": "@cf/meta/llama-3.1-8b-instruct"}
    assert call["json"] is None  # No prompt or payload passed


@pytest.mark.anyio
async def test_cloudflare_health_check_configured_model_absent() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={"success": True, "result": [{"id": "@cf/meta/other-model"}]},
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "absent from Cloudflare Workers AI catalog response" in health.reason
    assert health.status.is_available is False


@pytest.mark.anyio
async def test_cloudflare_health_check_uninterpretable_schema() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={"success": True, "result": "unexpected_string_format"},
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "catalog response schema unrecognized" in health.reason


@pytest.mark.anyio
async def test_cloudflare_health_check_auth_failure_401() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            401, json={"success": False, "errors": [{"message": "Invalid token"}]}
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "authentication failed (HTTP 401)" in health.reason
    assert health.status.is_available is False
    assert health.status.is_connected is False


@pytest.mark.anyio
async def test_cloudflare_health_check_authorization_failure_403() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            403, json={"success": False, "errors": [{"message": "Forbidden"}]}
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "authorization failed (HTTP 403)" in health.reason
    assert health.status.is_available is False
    assert health.status.is_connected is False


@pytest.mark.anyio
async def test_cloudflare_health_check_rate_limited() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            429,
            headers={"Retry-After": "45"},
            json={
                "success": False,
                "errors": [{"code": 3021, "message": "Rate limit exceeded"}],
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.RATE_LIMITED
    assert health.retry_after_seconds == 45.0
    assert "Rate limit exceeded" in health.reason
    assert health.status.is_available is False
    assert health.status.is_constrained is True
    assert health.status.is_connected is True


@pytest.mark.anyio
async def test_cloudflare_health_check_quota_exhausted() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            429,
            json={
                "success": False,
                "errors": [
                    {
                        "code": 4006,
                        "message": (
                            "AiError: you have used up your daily free allocation of 10,000 neurons"
                        ),
                    }
                ],
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.QUOTA_EXHAUSTED
    assert "daily free allocation of 10,000 neurons" in health.reason
    assert health.status.is_available is False
    assert health.status.is_constrained is True


@pytest.mark.anyio
async def test_cloudflare_health_check_server_error_unavailable() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(502, text="Bad Gateway")

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "server error (HTTP 502)" in health.reason


@pytest.mark.anyio
async def test_cloudflare_health_check_unsuccessful_payload_is_unavailable_not_degraded() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "success": False,
                "errors": [{"message": "catalog partially unavailable"}],
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "catalog search unsuccessful" in health.reason
    assert health.status != InferenceHealthStatus.DEGRADED


@pytest.mark.anyio
async def test_cloudflare_health_check_transport_errors_and_credential_sanitization() -> None:
    # 1. TimeoutException
    class TimeoutClient:
        async def get(self, url, headers=None, params=None):
            raise httpx.ReadTimeout("Timed out waiting for response")

    timeout_probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=TimeoutClient(),
    )
    t_health = await timeout_probe.check_health()
    assert t_health.status == InferenceHealthStatus.UNAVAILABLE
    assert "timed out" in t_health.reason

    # 2. RequestError with token in message
    class ConnectionFailClient:
        async def get(self, url, headers=None, params=None):
            raise httpx.ConnectError(
                "Failed to connect to api.cloudflare.com with token secret-token-xyz"
            )

    fail_probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=ConnectionFailClient(),
    )
    c_health = await fail_probe.check_health()
    assert c_health.status == InferenceHealthStatus.UNAVAILABLE
    assert "secret-token-xyz" not in c_health.reason
    assert "[REDACTED]" in c_health.reason

    # 3. Programming defect raises rather than turning into UNAVAILABLE
    class BuggyClient:
        async def get(self, url, headers=None, params=None):
            raise TypeError("programming defect in client code")

    buggy_probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=BuggyClient(),
    )
    with pytest.raises(TypeError, match="programming defect"):
        await buggy_probe.check_health()


@pytest.mark.anyio
async def test_service_adapters_cloudflare_health_delegation() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={"success": True, "result": [{"id": "@cf/meta/llama-3.1-8b-instruct"}]},
        )

    client = DummyAsyncClient(mock_handler)
    worker_provider = WorkerCloudflareProvider(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )
    research_provider = ResearchCloudflareProvider(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=client,
    )

    w_health = await worker_provider.check_health()
    assert w_health.status == InferenceHealthStatus.HEALTHY
    assert w_health.provider_name == "cloudflare"

    r_health = await research_provider.check_health()
    assert r_health.status == InferenceHealthStatus.HEALTHY
    assert r_health.provider_name == "cloudflare"
