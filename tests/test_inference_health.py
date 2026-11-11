from dataclasses import FrozenInstanceError
from typing import Any, Self

import httpx
import pytest
from agent_contracts import (
    DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
    FallbackEligibleInferenceError,
    InferenceHealthStatus,
    InferenceProviderHealth,
    NonFallbackInferenceError,
    parse_health_probe_timeout,
)
from agent_infrastructure import (
    CloudflareHealthProbe,
    GeminiClient,
    GeminiHealthProbe,
    canonical_gemini_model,
    normalize_gemini_model,
)
from research_agent.application.ports import (
    InferenceHealthProbe as ResearchInferenceHealthProbe,
)
from research_agent.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as ResearchCloudflareProvider,
)
from research_agent.infrastructure.gemini_inference import (
    GeminiInferenceProvider as ResearchGeminiProvider,
)
from research_agent.infrastructure.stub_inference import (
    StubInferenceProvider as ResearchStubProvider,
)
from worker.application.ports import InferenceHealthProbe as WorkerInferenceHealthProbe
from worker.infrastructure.cloudflare_inference import (
    CloudflareInferenceProvider as WorkerCloudflareProvider,
)
from worker.infrastructure.gemini_inference import (
    GeminiInferenceProvider as WorkerGeminiProvider,
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


def test_parse_health_probe_timeout_defaults_and_explicit() -> None:
    assert DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS == 3.0
    assert parse_health_probe_timeout(None) == 3.0
    assert parse_health_probe_timeout("") == 3.0
    assert parse_health_probe_timeout("   ") == 3.0
    assert parse_health_probe_timeout(None, default=5.0) == 5.0
    assert parse_health_probe_timeout("", default=5.0) == 5.0

    assert parse_health_probe_timeout(3.0) == 3.0
    assert parse_health_probe_timeout(5) == 5.0
    assert parse_health_probe_timeout("2.5") == 2.5
    assert parse_health_probe_timeout(" 10 ") == 10.0
    assert parse_health_probe_timeout(0.1) == 0.1


def test_parse_health_probe_timeout_invalid_values() -> None:
    # Zero
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(0)
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("0")
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(0.0)

    # Negative
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(-1)
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("-1.5")

    # Booleans
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(False)  # type: ignore[arg-type]

    # Malformed non-numeric
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("abc")
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("10s")
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout([])  # type: ignore[arg-type]

    # NaN
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(float("nan"))
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("nan")
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("NaN")

    # Positive Infinity
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(float("inf"))
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("inf")
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("+infinity")

    # Negative Infinity
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout(float("-inf"))
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("-inf")
    with pytest.raises(ValueError, match="probe_timeout_seconds must be a positive number"):
        parse_health_probe_timeout("-infinity")


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

    worker_gemini = WorkerGeminiProvider(api_key="tok")
    assert isinstance(worker_gemini, WorkerInferenceHealthProbe)

    research_gemini = ResearchGeminiProvider(api_key="tok")
    assert isinstance(research_gemini, ResearchInferenceHealthProbe)

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
async def test_cloudflare_health_check_exact_catalog_match_with_uuid_id() -> None:
    calls = []

    def mock_handler(method, url, headers, params=None, json_data=None):
        calls.append({"method": method, "url": url, "headers": headers, "params": params})
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": [
                    {
                        "id": "9b9c87c6-d4b7-494c-b177-87feab5904db",
                        "name": "@cf/meta/llama-3.1-8b-instruct",
                    }
                ],
            },
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
    assert "catalog probe verified model '@cf/meta/llama-3.1-8b-instruct' is accessible" in health.reason
    assert len(calls) == 1
    assert calls[0]["url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/models/search"


@pytest.mark.anyio
async def test_cloudflare_health_check_exact_schema_metadata_confirmation_healthy() -> None:
    calls = []

    def mock_handler(method, url, headers, params=None, json_data=None):
        calls.append({"method": method, "url": url, "headers": headers, "params": params})
        if "/ai/models/search" in url:
            return httpx.Response(
                200,
                json={"success": True, "result": [{"id": "@cf/meta/other-model"}]},
            )
        if "/ai/models/schema" in url:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "errors": [],
                    "messages": [],
                    "result": {
                        "input": {
                            "type": "object",
                            "properties": {"prompt": {"type": "string"}},
                            "required": ["prompt"],
                        },
                        "output": {
                            "type": "object",
                            "properties": {"response": {"type": "string"}},
                        },
                    },
                },
            )
        raise ValueError(f"Unexpected url: {url}")

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.HEALTHY
    assert health.model == "@cf/meta/llama-3.1-8b-instruct"
    assert "schema probe verified model '@cf/meta/llama-3.1-8b-instruct' is accessible" in health.reason
    assert "catalog probe verified" not in health.reason
    assert len(calls) == 2


@pytest.mark.anyio
async def test_cloudflare_health_check_observed_home_core_variant_false_negative_avoided() -> None:
    calls = []

    def mock_handler(method, url, headers, params=None, json_data=None):
        calls.append({"method": method, "url": url, "headers": headers, "params": params})
        if "/ai/models/search" in url:
            # Observed home-core response: search returned only the -fp8 variant with UUID id
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": [
                        {
                            "id": "9b9c87c6-d4b7-494c-b177-87feab5904db",
                            "name": "@cf/meta/llama-3.1-8b-instruct-fp8",
                        }
                    ],
                },
            )
        if "/ai/models/schema" in url:
            # Observed home-core response: schema lookup for base alias returns 404 code 6002
            return httpx.Response(
                404,
                json={
                    "success": False,
                    "errors": [{"code": 6002, "message": "Model schema not found"}],
                    "result": None,
                },
            )
        raise ValueError(f"Unexpected url: {url}")

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=client,
    )

    health = await probe.check_health()
    # Verified semantics: DEGRADED, neither false-positive HEALTHY nor false-negative UNAVAILABLE
    assert health.status == InferenceHealthStatus.DEGRADED
    assert health.status != InferenceHealthStatus.HEALTHY
    assert health.status != InferenceHealthStatus.UNAVAILABLE
    assert health.status.is_available is True
    assert health.model == "@cf/meta/llama-3.1-8b-instruct"

    # Reason asserts narrow, truthful semantics
    assert "API reachability and authentication established" in health.reason
    assert "could not be verified" in health.reason
    assert "runtime inference may still succeed" in health.reason
    assert "Model schema not found" in health.reason
    # -fp8 variant was NOT accepted as an exact catalog or schema match
    assert "catalog probe verified" not in health.reason
    assert "schema probe verified" not in health.reason

    assert len(calls) == 2
    assert client.calls == calls
    assert calls[0]["url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/models/search"
    assert calls[0]["params"] == {"search": "@cf/meta/llama-3.1-8b-instruct"}
    assert calls[1]["url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/models/schema"
    assert calls[1]["params"] == {"model": "@cf/meta/llama-3.1-8b-instruct"}


@pytest.mark.anyio
async def test_cloudflare_health_check_client_lifecycle_and_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Injected client handles both requests in exact order and is not closed
    class TrackingInjectedClient:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.closed = False

        async def get(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            params: dict[str, str] | None = None,
        ) -> httpx.Response:
            self.calls.append(url)
            if "/ai/models/search" in url:
                return httpx.Response(200, json={"success": True, "result": []})
            if "/ai/models/schema" in url:
                return httpx.Response(
                    404,
                    json={
                        "success": False,
                        "errors": [{"code": 6002, "message": "Model schema not found"}],
                    },
                )
            raise ValueError(f"Unexpected url {url}")

        async def aclose(self) -> None:
            self.closed = True

    injected = TrackingInjectedClient()
    probe_injected = CloudflareHealthProbe(
        account_id="acc123",
        api_token="token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=injected,
    )
    res_injected = await probe_injected.check_health()
    assert res_injected.status == InferenceHealthStatus.DEGRADED
    assert len(injected.calls) == 2
    assert "/ai/models/search" in injected.calls[0]
    assert "/ai/models/schema" in injected.calls[1]
    assert injected.closed is False

    # 2. When no client is injected, probe creates exactly ONE AsyncClient and closes it
    created_clients: list[Any] = []

    class MockAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: list[str] = []
            self.entered = False
            self.exited = False
            created_clients.append(self)

        async def __aenter__(self) -> Self:
            self.entered = True
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_val: object,
            exc_tb: object,
        ) -> None:
            self.exited = True

        async def get(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            params: dict[str, str] | None = None,
        ) -> httpx.Response:
            self.calls.append(url)
            if "/ai/models/search" in url:
                return httpx.Response(200, json={"success": True, "result": []})
            if "/ai/models/schema" in url:
                return httpx.Response(
                    404,
                    json={
                        "success": False,
                        "errors": [{"code": 6002, "message": "Model schema not found"}],
                    },
                )
            raise ValueError(f"Unexpected url {url}")

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    probe_owned = CloudflareHealthProbe(
        account_id="acc123",
        api_token="token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=None,
    )
    res_owned = await probe_owned.check_health()
    assert res_owned.status == InferenceHealthStatus.DEGRADED
    assert len(created_clients) == 1
    client_instance = created_clients[0]
    assert client_instance.entered is True
    assert client_instance.exited is True
    assert len(client_instance.calls) == 2
    assert "/ai/models/search" in client_instance.calls[0]
    assert "/ai/models/schema" in client_instance.calls[1]


@pytest.mark.anyio
async def test_cloudflare_health_check_unverifiable_alias_after_successful_auth_is_degraded() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        if "/ai/models/search" in url:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": [
                        {
                            "id": "9b9c87c6-d4b7-494c-b177-87feab5904db",
                            "name": "@cf/meta/llama-3.1-8b-instruct-fp8",
                        }
                    ],
                },
            )
        if "/ai/models/schema" in url:
            return httpx.Response(
                400,
                json={
                    "success": False,
                    "errors": [{"code": 5007, "message": "No such model"}],
                    "result": None,
                },
            )
        raise ValueError(f"Unexpected url: {url}")

    client = DummyAsyncClient(mock_handler)
    probe = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.DEGRADED
    assert health.status.is_available is True
    assert "API reachability and authentication established" in health.reason
    assert "could not be verified" in health.reason
    assert "runtime inference may still succeed" in health.reason
    assert "No such model" in health.reason


@pytest.mark.anyio
async def test_cloudflare_health_check_schema_probe_rate_limit_and_quota() -> None:
    # 1. Rate limited on schema probe
    def rate_limit_handler(method, url, headers, params=None, json_data=None):
        if "/ai/models/search" in url:
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"success": False, "errors": [{"message": "Rate limit exceeded"}]},
        )

    probe_rl = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=DummyAsyncClient(rate_limit_handler),
    )
    h_rl = await probe_rl.check_health()
    assert h_rl.status == InferenceHealthStatus.RATE_LIMITED
    assert h_rl.retry_after_seconds == 30.0

    # 2. Quota exhausted on schema probe
    def quota_handler(method, url, headers, params=None, json_data=None):
        if "/ai/models/search" in url:
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(
            429,
            json={
                "success": False,
                "errors": [{"code": 4006, "message": "AiError: neuron allocation exceeded"}],
            },
        )

    probe_quota = CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        model="@cf/meta/llama-3.1-8b-instruct",
        client=DummyAsyncClient(quota_handler),
    )
    h_quota = await probe_quota.check_health()
    assert h_quota.status == InferenceHealthStatus.QUOTA_EXHAUSTED
    assert "neuron allocation exceeded" in h_quota.reason


@pytest.mark.anyio
async def test_cloudflare_health_check_schema_probe_auth_and_transport_failures() -> None:
    # 1. 401 on schema probe
    def auth_handler(method, url, headers, params=None, json_data=None):
        if "/ai/models/search" in url:
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(401, json={"success": False, "errors": [{"message": "Invalid token"}]})

    h_auth = await CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=DummyAsyncClient(auth_handler),
    ).check_health()
    assert h_auth.status == InferenceHealthStatus.UNAVAILABLE
    assert "authentication failed (HTTP 401)" in h_auth.reason

    # 2. 403 on schema probe
    def authz_handler(method, url, headers, params=None, json_data=None):
        if "/ai/models/search" in url:
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(403, json={"success": False, "errors": [{"message": "Forbidden"}]})

    h_authz = await CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=DummyAsyncClient(authz_handler),
    ).check_health()
    assert h_authz.status == InferenceHealthStatus.UNAVAILABLE
    assert "authorization failed (HTTP 403)" in h_authz.reason

    # 3. Timeout on schema probe
    class SchemaTimeoutClient:
        async def get(self, url, headers=None, params=None):
            if "/ai/models/search" in url:
                return httpx.Response(200, json={"success": True, "result": []})
            raise httpx.ReadTimeout("Schema probe read timeout")

    h_timeout = await CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=SchemaTimeoutClient(),
    ).check_health()
    assert h_timeout.status == InferenceHealthStatus.UNAVAILABLE
    assert "timed out" in h_timeout.reason

    # 4. RequestError with token redaction on schema probe
    class SchemaErrorClient:
        async def get(self, url, headers=None, params=None):
            if "/ai/models/search" in url:
                return httpx.Response(200, json={"success": True, "result": []})
            raise httpx.ConnectError("Failed schema with token secret-token-xyz")

    h_err = await CloudflareHealthProbe(
        account_id="acc123",
        api_token="secret-token-xyz",
        client=SchemaErrorClient(),
    ).check_health()
    assert h_err.status == InferenceHealthStatus.UNAVAILABLE
    assert "secret-token-xyz" not in h_err.reason
    assert "[REDACTED]" in h_err.reason


@pytest.mark.anyio
async def test_cloudflare_health_check_unverifiable_model_absent_from_catalog_is_degraded() -> None:
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
    assert health.status == InferenceHealthStatus.DEGRADED
    assert health.status.is_available is True
    assert "API reachability and authentication established" in health.reason
    assert "could not be verified" in health.reason
    assert "runtime inference may still succeed" in health.reason


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


def test_gemini_client_key_validation_and_defense_in_depth() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        GeminiClient(api_key="")

    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        GeminiClient(api_key="   ")

    client = GeminiClient(api_key="secret-key-12345", model="gemini-2.5-flash")
    assert repr(client) == "GeminiClient(model='gemini-2.5-flash', api_key='[REDACTED]')"
    assert "secret-key-12345" not in repr(client)
    assert not hasattr(client, "api_key")  # Private _api_key only
    assert client.redact("Failed with secret-key-12345 token") == "Failed with [REDACTED] token"

    probe = GeminiHealthProbe(gemini_client=client)
    assert probe.provider_name == "gemini"


@pytest.mark.anyio
async def test_gemini_health_check_successful_metadata_probe() -> None:
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
            json={
                "name": "models/gemini-2.5-flash",
                "version": "001",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(
        api_key="secret-gemini-key",
        model="gemini-2.5-flash",
        client=client,
    )

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.HEALTHY
    assert health.provider_name == "gemini"
    assert health.model == "gemini-2.5-flash"
    assert "verified model 'gemini-2.5-flash' is accessible" in health.reason
    assert health.latency_ms is not None and health.latency_ms >= 0.0

    # Ensure non-generating metadata verification: GET /models/{model} was called, NOT generateContent
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash"
    assert call["headers"]["x-goog-api-key"] == "secret-gemini-key"
    assert call["json"] is None


@pytest.mark.anyio
async def test_gemini_health_check_model_absent_404() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            404,
            json={"error": {"code": 404, "message": "models/nonexistent not found", "status": "NOT_FOUND"}},
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(api_key="secret-key", model="nonexistent", client=client)

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "absent from Google Gemini catalog (HTTP 404)" in health.reason


@pytest.mark.anyio
async def test_gemini_health_check_missing_generate_content_capability() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "name": "models/text-embedding-004",
                "supportedGenerationMethods": ["embedContent"],
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(api_key="secret-key", model="text-embedding-004", client=client)

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "does not advertise generateContent capability" in health.reason


@pytest.mark.anyio
async def test_gemini_health_check_invalid_or_missing_supported_generation_methods() -> None:
    # 1. Missing supportedGenerationMethods field
    probe_missing = GeminiHealthProbe(
        api_key="secret-key",
        client=DummyAsyncClient(
            lambda *a, **kw: httpx.Response(200, json={"name": "models/gemini-2.5-flash"})
        ),
    )
    h_missing = await probe_missing.check_health()
    assert h_missing.status == InferenceHealthStatus.UNAVAILABLE
    assert "missing or invalid 'supportedGenerationMethods' list" in h_missing.reason

    # 2. Null (None) supportedGenerationMethods
    probe_null = GeminiHealthProbe(
        api_key="secret-key",
        client=DummyAsyncClient(
            lambda *a, **kw: httpx.Response(
                200,
                json={"name": "models/gemini-2.5-flash", "supportedGenerationMethods": None},
            )
        ),
    )
    h_null = await probe_null.check_health()
    assert h_null.status == InferenceHealthStatus.UNAVAILABLE
    assert "missing or invalid 'supportedGenerationMethods' list" in h_null.reason

    # 3. String-valued supportedGenerationMethods
    probe_str = GeminiHealthProbe(
        api_key="secret-key",
        client=DummyAsyncClient(
            lambda *a, **kw: httpx.Response(
                200,
                json={
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": "generateContent",
                },
            )
        ),
    )
    h_str = await probe_str.check_health()
    assert h_str.status == InferenceHealthStatus.UNAVAILABLE
    assert "missing or invalid 'supportedGenerationMethods' list" in h_str.reason

    # 4. Object-valued (dict) supportedGenerationMethods
    probe_obj = GeminiHealthProbe(
        api_key="secret-key",
        client=DummyAsyncClient(
            lambda *a, **kw: httpx.Response(
                200,
                json={
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": {"generateContent": True},
                },
            )
        ),
    )
    h_obj = await probe_obj.check_health()
    assert h_obj.status == InferenceHealthStatus.UNAVAILABLE
    assert "missing or invalid 'supportedGenerationMethods' list" in h_obj.reason


@pytest.mark.anyio
async def test_gemini_health_check_valid_supported_generation_methods() -> None:
    probe = GeminiHealthProbe(
        api_key="secret-key",
        client=DummyAsyncClient(
            lambda *a, **kw: httpx.Response(
                200,
                json={
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
            )
        ),
    )
    h = await probe.check_health()
    assert h.status == InferenceHealthStatus.HEALTHY
    assert "verified model 'gemini-2.5-flash' is accessible" in h.reason


@pytest.mark.anyio
async def test_gemini_health_check_auth_failure_401() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            401,
            json={"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}},
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(api_key="invalid-key", client=client)

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "authentication failed (HTTP 401)" in health.reason
    assert health.status.is_connected is False


@pytest.mark.anyio
async def test_gemini_health_check_authorization_failure_403() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            403,
            json={"error": {"code": 403, "message": "Generative Language API has not been used", "status": "PERMISSION_DENIED"}},
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(api_key="secret-key", client=client)

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.UNAVAILABLE
    assert "authorization failed (HTTP 403)" in health.reason
    assert health.status.is_connected is False


@pytest.mark.anyio
async def test_gemini_health_check_rate_limited_conservative() -> None:
    # Conservative guardrail: RESOURCE_EXHAUSTED without explicit quota details is RATE_LIMITED
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted (e.g. check quota)",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(api_key="secret-key", client=client)

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.RATE_LIMITED
    assert health.retry_after_seconds == 30.0
    assert health.status.is_constrained is True
    assert health.status.is_available is False


@pytest.mark.anyio
async def test_gemini_health_check_quota_exhausted_explicit() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "Quota exceeded for quota metric 'GenerateContent requests' per day",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"reason": "QUOTA_EXCEEDED"}],
                }
            },
        )

    client = DummyAsyncClient(mock_handler)
    probe = GeminiHealthProbe(api_key="secret-key", client=client)

    health = await probe.check_health()
    assert health.status == InferenceHealthStatus.QUOTA_EXHAUSTED
    assert "Quota exceeded" in health.reason
    assert health.status.is_constrained is True
    assert health.status.is_available is False


@pytest.mark.anyio
async def test_gemini_health_check_transport_errors_and_credential_sanitization() -> None:
    class TimeoutClient:
        async def get(self, url, headers=None, params=None):
            raise httpx.ReadTimeout("Timed out reading from Google API")

    t_probe = GeminiHealthProbe(api_key="secret-key-xyz", client=TimeoutClient())
    t_health = await t_probe.check_health()
    assert t_health.status == InferenceHealthStatus.UNAVAILABLE
    assert "timed out" in t_health.reason

    class ConnectFailClient:
        async def get(self, url, headers=None, params=None):
            raise httpx.ConnectError("Failed to connect to generativelanguage.googleapis.com with key secret-key-xyz")

    c_probe = GeminiHealthProbe(api_key="secret-key-xyz", client=ConnectFailClient())
    c_health = await c_probe.check_health()
    assert c_health.status == InferenceHealthStatus.UNAVAILABLE
    assert "secret-key-xyz" not in c_health.reason
    assert "[REDACTED]" in c_health.reason

    class BuggyClient:
        async def get(self, url, headers=None, params=None):
            raise TypeError("programming defect")

    b_probe = GeminiHealthProbe(api_key="secret-key-xyz", client=BuggyClient())
    with pytest.raises(TypeError, match="programming defect"):
        await b_probe.check_health()


@pytest.mark.anyio
async def test_service_adapters_gemini_health_delegation() -> None:
    def mock_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
        )

    client = DummyAsyncClient(mock_handler)
    worker_provider = WorkerGeminiProvider(
        api_key="secret-gemini-key",
        client=client,
    )
    research_provider = ResearchGeminiProvider(
        api_key="secret-gemini-key",
        client=client,
    )

    w_health = await worker_provider.check_health()
    assert w_health.status == InferenceHealthStatus.HEALTHY
    assert w_health.provider_name == "gemini"

    r_health = await research_provider.check_health()
    assert r_health.status == InferenceHealthStatus.HEALTHY
    assert r_health.provider_name == "gemini"


@pytest.mark.anyio
async def test_gemini_client_invalid_json_without_swallowing_programming_defects() -> None:
    # 1. Invalid JSON raises ValueError in response.json() -> wrapped as RuntimeError
    class InvalidJsonResponse:
        status_code = 200

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class InvalidJsonClient:
        async def post(self, url, headers=None, json=None):
            return InvalidJsonResponse()

    client1 = GeminiClient(api_key="test-key", client=InvalidJsonClient())
    with pytest.raises(RuntimeError, match="Gemini API returned invalid JSON"):
        await client1.generate_content({"contents": []})

    # 2. Programming defect in response.json() raises TypeError -> NOT swallowed, propagates directly
    class DefectiveResponse:
        status_code = 200

        def json(self):
            raise TypeError("unhashable type: 'slice'")

    class DefectiveClient:
        async def post(self, url, headers=None, json=None):
            return DefectiveResponse()

    client2 = GeminiClient(api_key="test-key", client=DefectiveClient())
    with pytest.raises(TypeError, match="unhashable type: 'slice'"):
        await client2.generate_content({"contents": []})


@pytest.mark.anyio
async def test_gemini_client_candidate_text_validation() -> None:
    # Multiple text parts combined
    def multi_part_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "First part. "},
                                {"text": "Second part."},
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    client_multi = GeminiClient(api_key="test-key", client=DummyAsyncClient(multi_part_handler))
    res = await client_multi.generate_content({"contents": []})
    assert res.text == "First part. Second part."

    # Empty parts array -> raises RuntimeError
    def empty_parts_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]},
        )

    client_empty = GeminiClient(api_key="test-key", client=DummyAsyncClient(empty_parts_handler))
    with pytest.raises(RuntimeError, match="missing or invalid 'parts' list"):
        await client_empty.generate_content({"contents": []})

    # Whitespace-only parts -> raises RuntimeError
    def whitespace_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "   \n\t"}]}, "finishReason": "STOP"}]},
        )

    client_ws = GeminiClient(api_key="test-key", client=DummyAsyncClient(whitespace_handler))
    with pytest.raises(RuntimeError, match="candidate produced empty or whitespace-only text"):
        await client_ws.generate_content({"contents": []})

    # Non-string and null text fields are ignored without converting to strings
    def mixed_parts_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": None},
                                {"text": 12345},
                                {"text": {"nested": "obj"}},
                                {"text": "Valid text part"},
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    client_mixed = GeminiClient(api_key="test-key", client=DummyAsyncClient(mixed_parts_handler))
    res_mixed = await client_mixed.generate_content({"contents": []})
    assert res_mixed.text == "Valid text part"
    assert "None" not in res_mixed.text
    assert "12345" not in res_mixed.text

    # Candidate with only non-string text parts -> raises RuntimeError
    def only_invalid_parts_handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": None}, {"text": 42}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    client_invalid = GeminiClient(api_key="test-key", client=DummyAsyncClient(only_invalid_parts_handler))
    with pytest.raises(RuntimeError, match="candidate produced empty or whitespace-only text"):
        await client_invalid.generate_content({"contents": []})


@pytest.mark.anyio
async def test_gemini_client_usage_metadata_boolean_exclusion() -> None:
    def handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "response text"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 30,
                    "totalTokenCount": 45,
                    "isCached": True,
                    "cacheActive": False,
                },
            },
        )

    client = GeminiClient(api_key="test-key", client=DummyAsyncClient(handler))
    res = await client.generate_content({"contents": []})
    assert res.usage_metadata == {
        "promptTokenCount": 15,
        "candidatesTokenCount": 30,
        "totalTokenCount": 45,
    }
    assert "isCached" not in (res.usage_metadata or {})
    assert "cacheActive" not in (res.usage_metadata or {})


@pytest.mark.anyio
async def test_gemini_client_generate_content_http_402_payment_required() -> None:
    secret_key = "secret-payment-key-test-402"

    def handler_402(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            402,
            json={
                "error": {
                    "code": 402,
                    "message": f"Resource has been exhausted (e.g. check quota). Prepayment credits are depleted. Key: {secret_key}",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )

    client = GeminiClient(api_key=secret_key, client=DummyAsyncClient(handler_402))
    with pytest.raises(FallbackEligibleInferenceError) as exc_info:
        await client.generate_content({"contents": [{"parts": [{"text": "Hello"}]}]})

    err_msg = str(exc_info.value)
    assert exc_info.value.provider_name == "gemini"
    assert exc_info.value.reason == "payment_required"
    assert exc_info.value.status_code == 402
    assert "Gemini API error (status 402)" in err_msg
    assert "Prepayment credits are depleted" in err_msg
    # Ensure credential redaction
    assert secret_key not in err_msg
    assert "[REDACTED]" in err_msg


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "expected_exc_type", "expected_reason"),
    [
        (400, NonFallbackInferenceError, "client_error_400"),
        (401, NonFallbackInferenceError, "client_error_401"),
        (403, NonFallbackInferenceError, "client_error_403"),
        (404, NonFallbackInferenceError, "client_error_404"),
        (429, FallbackEligibleInferenceError, "rate_limited"),
        (500, FallbackEligibleInferenceError, "server_error_500"),
        (503, FallbackEligibleInferenceError, "server_error_503"),
    ],
)
async def test_gemini_client_generate_content_status_code_classification(
    status_code: int,
    expected_exc_type: type[Exception],
    expected_reason: str,
) -> None:
    def handler(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            status_code,
            json={"error": {"code": status_code, "message": f"API error with code {status_code}"}},
        )

    client = GeminiClient(api_key="test-key", client=DummyAsyncClient(handler))
    with pytest.raises(expected_exc_type) as exc_info:
        await client.generate_content({"contents": []})

    assert exc_info.value.provider_name == "gemini"
    assert exc_info.value.reason == expected_reason
    assert exc_info.value.status_code == status_code


@pytest.mark.anyio
async def test_gemini_health_check_exact_canonical_model_matching() -> None:
    # 1. Probe expecting 'gemini-2.5-flash', catalog returns 'models/gemini-2.5-flash' -> HEALTHY
    def handler1(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
        )

    probe1 = GeminiHealthProbe(api_key="test-key", model="gemini-2.5-flash", client=DummyAsyncClient(handler1))
    health1 = await probe1.check_health()
    assert health1.status == InferenceHealthStatus.HEALTHY
    assert health1.model == "gemini-2.5-flash"

    # 2. Probe initialized with 'models/gemini-2.5-flash', catalog returns 'gemini-2.5-flash' -> HEALTHY
    def handler2(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "name": "gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
        )

    probe2 = GeminiHealthProbe(api_key="test-key", model="models/gemini-2.5-flash", client=DummyAsyncClient(handler2))
    health2 = await probe2.check_health()
    assert health2.status == InferenceHealthStatus.HEALTHY


@pytest.mark.anyio
async def test_gemini_health_check_mismatched_suffix_like_model_names() -> None:
    # Catalog returning 'models/other-gemini-2.5-flash' must NOT match 'gemini-2.5-flash'
    def handler_prefix(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "name": "models/other-gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
        )

    probe_prefix = GeminiHealthProbe(api_key="test-key", model="gemini-2.5-flash", client=DummyAsyncClient(handler_prefix))
    health_prefix = await probe_prefix.check_health()
    assert health_prefix.status == InferenceHealthStatus.UNAVAILABLE
    assert "does not match configured model" in health_prefix.reason

    # Catalog returning 'models/gemini-2.5-flash-preview' must NOT match 'gemini-2.5-flash'
    def handler_suffix(method, url, headers, params=None, json_data=None):
        return httpx.Response(
            200,
            json={
                "name": "models/gemini-2.5-flash-preview",
                "supportedGenerationMethods": ["generateContent"],
            },
        )

    probe_suffix = GeminiHealthProbe(api_key="test-key", model="gemini-2.5-flash", client=DummyAsyncClient(handler_suffix))
    health_suffix = await probe_suffix.check_health()
    assert health_suffix.status == InferenceHealthStatus.UNAVAILABLE
    assert "does not match configured model" in health_suffix.reason


def test_gemini_model_helpers_and_empty_validation() -> None:
    assert normalize_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash"
    assert normalize_gemini_model("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert canonical_gemini_model("gemini-2.5-flash") == "models/gemini-2.5-flash"
    assert canonical_gemini_model("models/gemini-2.5-flash") == "models/gemini-2.5-flash"

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        normalize_gemini_model("")

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        normalize_gemini_model("   ")

    with pytest.raises(ValueError, match="model must be a non-empty string after normalization"):
        normalize_gemini_model("models/")

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        GeminiClient(api_key="test-key", model="")

    with pytest.raises(ValueError, match="model must be a non-empty string after normalization"):
        GeminiClient(api_key="test-key", model="models/")
