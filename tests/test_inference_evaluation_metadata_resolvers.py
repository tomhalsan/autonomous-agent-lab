from __future__ import annotations

import json
from dataclasses import asdict

import httpx
import pytest
from agent_infrastructure.gemini_client import GeminiClient

from evaluation.inference.candidates import CANDIDATES
from evaluation.inference.models import Candidate, ProviderModelMetadataSnapshot
from evaluation.inference.stage1a.metadata_resolvers import (
    CloudflareMetadataResolver,
    GeminiMetadataResolver,
    MetadataResolutionError,
    resolve_unique_metadata,
)

_CLOUDFLARE_BASE_URL = "https://api.cloudflare.test/client/v4/accounts"
_GEMINI_BASE_URL = "https://gemini.test/v1beta"
_CLOUDFLARE_CONTEXT_WINDOWS = {
    "@cf/google/gemma-4-26b-a4b-it": "256000",
    "@cf/openai/gpt-oss-20b": "128000",
    "@cf/qwen/qwen3.8-27b": "262144",
}
_GEMINI_RESPONSE = {
    "name": "models/gemini-3.5-flash-lite",
    "supportedGenerationMethods": ["generateContent"],
    "inputTokenLimit": 1048576,
    "outputTokenLimit": 65536,
    "temperature": 1,
    "maxTemperature": 2,
    "thinking": True,
}


def _candidate(candidate_id: str) -> Candidate:
    return next(candidate for candidate in CANDIDATES if candidate.id == candidate_id)


def _catalog_fixture(model: str, reasoning: object = "true") -> dict[str, object]:
    context_window = _CLOUDFLARE_CONTEXT_WINDOWS.get(model)
    properties: list[dict[str, object]] = []
    if context_window is not None:
        properties.append({"property_id": "context_window", "value": context_window})
    if reasoning is not None:
        properties.append({"property_id": "reasoning", "value": reasoning})
    return {
        "success": True,
        "result": [
            {
                "name": model,
                "task": {"name": "Text Generation"},
                "properties": properties,
            }
        ],
    }


def _schema_fixture(
    *,
    temperature_default: float = 1,
    minimum: float = 0,
    maximum: float = 2,
    output_default: int | None = None,
) -> dict[str, object]:
    max_tokens: dict[str, object] = {"type": "integer"}
    if output_default is not None:
        max_tokens["default"] = output_default
    return {
        "success": True,
        "result": {
            "input": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "messages": {"type": "array"},
                            "max_tokens": max_tokens,
                            "temperature": {
                                "default": temperature_default,
                                "anyOf": [
                                    {
                                        "type": "number",
                                        "minimum": minimum,
                                        "maximum": maximum,
                                    },
                                    {"type": "null"},
                                ],
                            },
                        },
                        "required": ["messages"],
                    }
                ]
            }
        },
    }


def _cloudflare_transport(
    candidate: Candidate,
    *,
    reasoning: object = "true",
    temperature_default: float = 1,
    minimum: float = 0,
    maximum: float = 2,
    output_default: int | None = None,
    search_result: dict[str, object] | None = None,
    search_status: int = 200,
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method != "GET":
            return httpx.Response(405, request=request)
        if request.url.path.endswith("/ai/models/search"):
            return httpx.Response(
                search_status,
                json=search_result or _catalog_fixture(candidate.model, reasoning),
                request=request,
            )
        if request.url.path.endswith("/ai/models/schema"):
            return httpx.Response(
                200,
                json=_schema_fixture(
                    temperature_default=temperature_default,
                    minimum=minimum,
                    maximum=maximum,
                    output_default=output_default,
                ),
                request=request,
            )
        return httpx.Response(404, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return client, requests


def _gemini_transport(
    payload: dict[str, object] | None = None,
    *,
    status: int = 200,
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            json=payload or _GEMINI_RESPONSE,
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return client, requests


def _gemini_resolver(
    client: httpx.AsyncClient,
    *,
    api_key: str = "test-gemini-secret",
) -> GeminiMetadataResolver:
    return GeminiMetadataResolver(
        GeminiClient(
            api_key=api_key,
            model="gemini-3.5-flash-lite",
            base_url=_GEMINI_BASE_URL,
            client=client,
        )
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("candidate_id", "temperature", "minimum", "maximum", "output_default"),
    [
        ("general-cloudflare-gemma-4-26b-a4b-it", 1, 0, 2, None),
        ("general-cloudflare-gpt-oss-20b", 0.6, 0, 5, 256),
        ("research-cloudflare-qwen3.8-27b", 1, 0, 2, None),
    ],
)
async def test_cloudflare_maps_only_catalog_and_schema_facts(
    candidate_id: str,
    temperature: float,
    minimum: float,
    maximum: float,
    output_default: int | None,
) -> None:
    candidate = _candidate(candidate_id)
    client, requests = _cloudflare_transport(
        candidate,
        temperature_default=temperature,
        minimum=minimum,
        maximum=maximum,
        output_default=output_default,
    )
    resolver = CloudflareMetadataResolver(
        "test-account-secret",
        "test-cloudflare-secret",
        client=client,
        base_url=_CLOUDFLARE_BASE_URL,
    )
    async with client:
        snapshot = await resolver.resolve(candidate)

    assert snapshot.provider == "cloudflare"
    assert snapshot.model == candidate.model
    assert snapshot.supports_text_generation is True
    assert snapshot.reasoning_supported is True
    assert snapshot.reasoning_default is None
    assert snapshot.temperature_supported is True
    assert snapshot.default_temperature == temperature
    assert snapshot.min_temperature == minimum
    assert snapshot.max_temperature == maximum
    assert snapshot.default_output_tokens == output_default
    assert snapshot.context_limit_tokens is None
    assert snapshot.input_limit_tokens is None
    assert snapshot.max_output_tokens is None
    assert len(requests) == 2
    assert all(request.method == "GET" for request in requests)
    assert {request.url.path.rsplit("/", maxsplit=1)[-1] for request in requests} == {
        "search",
        "schema",
    }
    assert all("/ai/run/" not in request.url.path for request in requests)
    assert "test-cloudflare-secret" not in repr(resolver)
    assert "test-cloudflare-secret" not in json.dumps(asdict(snapshot))
    assert "test-account-secret" not in json.dumps(asdict(snapshot))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [("true", True), ("false", False), ("TRUE", None), (" true", None), ("yes", None)],
)
async def test_cloudflare_reasoning_string_normalization_is_strict(
    reasoning: object,
    expected: bool | None,
) -> None:
    candidate = _candidate("general-cloudflare-gemma-4-26b-a4b-it")
    client, _ = _cloudflare_transport(candidate, reasoning=reasoning)
    async with client:
        snapshot = await CloudflareMetadataResolver(
            "account", "token", client=client, base_url=_CLOUDFLARE_BASE_URL
        ).resolve(candidate)
    assert snapshot.reasoning_supported is expected


@pytest.mark.anyio
async def test_cloudflare_requires_exact_catalog_identity() -> None:
    candidate = _candidate("general-cloudflare-gemma-4-26b-a4b-it")
    client, requests = _cloudflare_transport(
        candidate,
        search_result=_catalog_fixture(f"{candidate.model}-variant"),
    )
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await CloudflareMetadataResolver(
                "account", "token", client=client, base_url=_CLOUDFLARE_BASE_URL
            ).resolve(candidate)

    assert raised.value.category == "model_not_found"
    assert raised.value.retrieved_at is not None
    assert len(requests) == 1


@pytest.mark.anyio
async def test_cloudflare_http_failure_is_sanitized() -> None:
    candidate = _candidate("general-cloudflare-gemma-4-26b-a4b-it")
    secret = "cloudflare-http-secret"
    client, _ = _cloudflare_transport(
        candidate,
        search_result={"errors": [{"message": secret}]},
        search_status=503,
    )
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await CloudflareMetadataResolver(
                "account", secret, client=client, base_url=_CLOUDFLARE_BASE_URL
            ).resolve(candidate)

    assert raised.value.category == "http_error"
    assert raised.value.http_status == 503
    assert raised.value.retrieved_at is not None
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value.__dict__)


@pytest.mark.anyio
async def test_cloudflare_transport_failure_is_sanitized() -> None:
    candidate = _candidate("general-cloudflare-gemma-4-26b-a4b-it")
    secret = "cloudflare-transport-secret"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(secret, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await CloudflareMetadataResolver(
                "account", secret, client=client, base_url=_CLOUDFLARE_BASE_URL
            ).resolve(candidate)

    assert raised.value.category == "transport_error"
    assert raised.value.retrieved_at is None
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value.__dict__)
    assert raised.value.__context__ is None


@pytest.mark.anyio
async def test_gemini_maps_model_get_metadata_without_inference() -> None:
    candidate = _candidate("general-gemini-3.5-flash-lite")
    client, requests = _gemini_transport()
    resolver = _gemini_resolver(client)
    async with client:
        snapshot = await resolver.resolve(candidate)

    assert snapshot.model == "gemini-3.5-flash-lite"
    assert snapshot.supports_text_generation is True
    assert snapshot.input_limit_tokens == 1048576
    assert snapshot.max_output_tokens == 65536
    assert snapshot.default_output_tokens is None
    assert snapshot.temperature_supported is True
    assert snapshot.default_temperature == 1
    assert snapshot.min_temperature is None
    assert snapshot.max_temperature == 2
    assert snapshot.reasoning_supported is True
    assert snapshot.reasoning_default is None
    assert snapshot.context_limit_tokens is None
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path.endswith("/models/gemini-3.5-flash-lite")
    assert "generateContent" not in requests[0].url.path
    assert "test-gemini-secret" not in repr(resolver)
    assert "test-gemini-secret" not in json.dumps(asdict(snapshot))


@pytest.mark.anyio
async def test_gemini_identity_mismatch_fails_resolution() -> None:
    candidate = _candidate("general-gemini-3.5-flash-lite")
    payload = {**_GEMINI_RESPONSE, "name": "models/another-model"}
    client, requests = _gemini_transport(payload)
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await _gemini_resolver(client).resolve(candidate)

    assert raised.value.category == "candidate_mismatch"
    assert raised.value.retrieved_at is not None
    assert len(requests) == 1


@pytest.mark.anyio
async def test_gemini_requires_generate_content_method() -> None:
    candidate = _candidate("general-gemini-3.5-flash-lite")
    payload = {**_GEMINI_RESPONSE, "supportedGenerationMethods": ["countTokens"]}
    client, _ = _gemini_transport(payload)
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await _gemini_resolver(client).resolve(candidate)

    assert raised.value.category == "generation_capability_unconfirmed"


@pytest.mark.anyio
async def test_gemini_http_failure_is_sanitized() -> None:
    candidate = _candidate("general-gemini-3.5-flash-lite")
    secret = "gemini-http-secret"
    client, _ = _gemini_transport(
        {"error": {"message": secret}},
        status=403,
    )
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await _gemini_resolver(client, api_key=secret).resolve(candidate)

    assert raised.value.category == "http_error"
    assert raised.value.http_status == 403
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value.__dict__)


@pytest.mark.anyio
async def test_gemini_transport_failure_is_sanitized() -> None:
    candidate = _candidate("general-gemini-3.5-flash-lite")
    secret = "gemini-transport-secret"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(secret, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    async with client:
        with pytest.raises(MetadataResolutionError) as raised:
            await _gemini_resolver(client, api_key=secret).resolve(candidate)

    assert raised.value.category == "transport_error"
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value.__dict__)
    assert raised.value.__context__ is None


@pytest.mark.anyio
async def test_campaign_resolves_each_unique_model_once() -> None:
    cloudflare_calls: list[httpx.Request] = []
    gemini_calls: list[httpx.Request] = []
    cloudflare_facts = {
        candidate.model: (
            0.6 if "gpt-oss" in candidate.model else 1,
            5 if "gpt-oss" in candidate.model else 2,
            256 if "gpt-oss" in candidate.model else None,
        )
        for candidate in CANDIDATES
        if candidate.provider == "cloudflare"
    }

    def cloudflare_handle(request: httpx.Request) -> httpx.Response:
        cloudflare_calls.append(request)
        model = request.url.params.get("search") or request.url.params.get("model")
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json=_catalog_fixture(model or ""), request=request)
        if request.url.path.endswith("/schema"):
            temperature, maximum, output_default = cloudflare_facts[model or ""]
            return httpx.Response(
                200,
                json=_schema_fixture(
                    temperature_default=temperature,
                    maximum=maximum,
                    output_default=output_default,
                ),
                request=request,
            )
        return httpx.Response(404, request=request)

    cf_client = httpx.AsyncClient(transport=httpx.MockTransport(cloudflare_handle))
    gemini_http_client, gemini_calls = _gemini_transport()
    gemini_resolver = _gemini_resolver(gemini_http_client)
    async with cf_client, gemini_http_client:
        snapshots = await resolve_unique_metadata(
            CANDIDATES,
            cloudflare=CloudflareMetadataResolver(
                "account", "token", client=cf_client, base_url=_CLOUDFLARE_BASE_URL
            ),
            gemini=gemini_resolver,
        )

    assert len(snapshots) == 4
    assert len(cloudflare_calls) == 6
    assert len(gemini_calls) == 1
    assert all(request.method == "GET" for request in cloudflare_calls + gemini_calls)
    assert all("/ai/run/" not in request.url.path for request in cloudflare_calls)
    assert all("generateContent" not in request.url.path for request in gemini_calls)
    assert isinstance(
        snapshots[("gemini", "gemini-3.5-flash-lite")],
        ProviderModelMetadataSnapshot,
    )


def test_metadata_failure_contains_only_safe_candidate_and_status_fields() -> None:
    candidate = _candidate("general-cloudflare-gemma-4-26b-a4b-it")
    secret = "never-persist-this"
    error = MetadataResolutionError(
        candidate, "http_error", retrieved_at="2025-01-01T00:00:00+00:00", http_status=403
    )
    serialized = json.dumps(error.__dict__)

    assert error.candidate_id == candidate.id
    assert error.provider == candidate.provider
    assert error.model == candidate.model
    assert error.category == "http_error"
    assert error.http_status == 403
    assert secret not in str(error)
    assert secret not in serialized
    assert "raw_response" not in error.__dict__
    assert "exception_message" not in error.__dict__
    assert "credentials" not in error.__dict__
