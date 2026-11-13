"""Bounded provider metadata resolution for inference evaluation Stage 1A."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol

import httpx

from evaluation.inference.models import Candidate, ProviderModelMetadataSnapshot

MetadataFailureCategory = Literal[
    "candidate_mismatch",
    "generation_capability_unconfirmed",
    "http_error",
    "invalid_response",
    "model_not_found",
    "timeout",
    "transport_error",
    "unsupported_provider",
]

MetadataKey = tuple[str, str]

_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
_CLOUDFLARE_TEXT_GENERATION_TASK = "Text Generation"
_GEMINI_MODEL_PREFIX = "models/"


class MetadataResolutionError(Exception):
    """Sanitized failure to resolve provider metadata for an evaluation candidate."""

    def __init__(
        self,
        candidate: Candidate,
        category: MetadataFailureCategory,
        *,
        retrieved_at: str | None = None,
        http_status: int | None = None,
    ) -> None:
        self.candidate_id = candidate.id
        self.provider = candidate.provider
        self.model = candidate.model
        self.category = category
        self.retrieved_at = retrieved_at
        self.http_status = http_status
        status = f" (HTTP {http_status})" if http_status is not None else ""
        super().__init__(
            f"Metadata resolution failed for {candidate.provider}/{candidate.model}: "
            f"{category}{status}"
        )


class _MetadataResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class _GeminiMetadataClient(Protocol):
    @property
    def normalized_model(self) -> str: ...

    async def get_model_metadata(self) -> _MetadataResponse: ...


class _CandidateResolver(Protocol):
    async def resolve(self, candidate: Candidate) -> ProviderModelMetadataSnapshot: ...


def _retrieved_at() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _http_status(response: _MetadataResponse) -> int | None:
    status = response.status_code
    return status if type(status) is int else None


def _response_object(
    candidate: Candidate,
    response: _MetadataResponse,
) -> tuple[dict[str, object], str]:
    retrieved_at = _retrieved_at()
    status = _http_status(response)
    if status != 200:
        raise MetadataResolutionError(
            candidate,
            "http_error",
            retrieved_at=retrieved_at,
            http_status=status,
        )
    payload: object = None
    invalid_json = False
    try:
        payload = response.json()
    except ValueError:
        invalid_json = True
    if invalid_json or not isinstance(payload, dict):
        raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
    return payload, retrieved_at


def _consistent_value(values: Sequence[object]) -> object | None:
    if not values or any(value != values[0] for value in values[1:]):
        return None
    return values[0]


def _catalog_property(item: dict[str, object], property_id: str) -> object | None:
    properties = item.get("properties")
    if not isinstance(properties, list):
        return None
    values = [
        prop.get("value")
        for prop in properties
        if isinstance(prop, dict) and prop.get("property_id") == property_id
    ]
    return _consistent_value(values)


def _strict_boolean_string(value: object) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _message_input_properties(schema: object) -> list[dict[str, object]]:
    message_properties: list[dict[str, object]] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if (
                isinstance(properties, dict)
                and "messages" in properties
                and isinstance(value.get("required"), list)
                and "messages" in value["required"]
            ):
                message_properties.append(properties)
            for key in ("allOf", "anyOf", "oneOf"):
                children = value.get(key)
                if isinstance(children, list):
                    for child in children:
                        visit(child)

    visit(schema)
    return message_properties


def _property_specs(
    properties: list[dict[str, object]],
    property_name: str,
) -> list[dict[str, object]]:
    specs = [item[property_name] for item in properties if property_name in item]
    return [spec for spec in specs if isinstance(spec, dict)]


def _numeric_schemas(schema: dict[str, object]) -> list[dict[str, object]]:
    numeric_schemas: list[dict[str, object]] = []
    schema_type = schema.get("type")
    if schema_type in ("number", "integer") or (
        isinstance(schema_type, list) and any(item in ("number", "integer") for item in schema_type)
    ):
        numeric_schemas.append(schema)
    for key in ("allOf", "anyOf", "oneOf"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    numeric_schemas.extend(_numeric_schemas(child))
    return numeric_schemas


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _common_schema_bound(
    specs: list[dict[str, object]],
    bound: Literal["minimum", "maximum"],
) -> float | None:
    values: list[float] = []
    for spec in specs:
        numeric_schemas = _numeric_schemas(spec)
        bounds = [_finite_number(schema.get(bound)) for schema in numeric_schemas]
        if not bounds or any(value is None for value in bounds):
            return None
        common = _consistent_value(bounds)
        if not isinstance(common, float):
            return None
        values.append(common)
    common = _consistent_value(values)
    return common if isinstance(common, float) else None


def _common_numeric_default(specs: list[dict[str, object]]) -> float | None:
    defaults = [_finite_number(spec.get("default")) for spec in specs]
    if not defaults or any(value is None for value in defaults):
        return None
    common = _consistent_value(defaults)
    return common if isinstance(common, float) else None


def _common_integer_default(specs: list[dict[str, object]]) -> int | None:
    defaults = [spec.get("default") for spec in specs]
    if (
        not defaults
        or any(type(value) is not int or value < 0 for value in defaults)
        or any(value != defaults[0] for value in defaults[1:])
    ):
        return None
    value = defaults[0]
    return value if type(value) is int else None


def _catalog_model(
    candidate: Candidate,
    payload: dict[str, object],
    retrieved_at: str,
) -> dict[str, object]:
    if payload.get("success") is not True:
        raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
    result = payload.get("result")
    if not isinstance(result, list):
        raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
    matches = [
        item for item in result if isinstance(item, dict) and item.get("name") == candidate.model
    ]
    if len(matches) != 1:
        raise MetadataResolutionError(candidate, "model_not_found", retrieved_at=retrieved_at)
    model = matches[0]
    task = model.get("task")
    task_name = task.get("name") if isinstance(task, dict) else None
    if task_name != _CLOUDFLARE_TEXT_GENERATION_TASK:
        raise MetadataResolutionError(
            candidate,
            "generation_capability_unconfirmed",
            retrieved_at=retrieved_at,
        )
    return model


def _cloudflare_snapshot(
    candidate: Candidate,
    catalog_model: dict[str, object],
    schema_payload: dict[str, object],
    retrieved_at: str,
) -> ProviderModelMetadataSnapshot:
    schema_result = schema_payload.get("result")
    input_schema = schema_result.get("input") if isinstance(schema_result, dict) else None
    message_properties = _message_input_properties(input_schema)
    if not message_properties:
        raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)

    temperature_specs = _property_specs(message_properties, "temperature")
    if len(temperature_specs) != len(message_properties):
        temperature_specs = []
    numeric_temperature_specs = [spec for spec in temperature_specs if _numeric_schemas(spec)]
    temperature_supported = (
        True
        if len(temperature_specs) == len(message_properties)
        and len(numeric_temperature_specs) == len(message_properties)
        else None
    )
    output_specs = _property_specs(message_properties, "max_tokens")
    if len(output_specs) != len(message_properties):
        output_specs = []
    reasoning = _strict_boolean_string(_catalog_property(catalog_model, "reasoning"))

    return ProviderModelMetadataSnapshot(
        provider=candidate.provider,
        model=candidate.model,
        retrieved_at=retrieved_at,
        metadata_source="account_catalog",
        metadata_method="account_catalog",
        supports_text_generation=True,
        context_limit_tokens=None,
        input_limit_tokens=None,
        max_output_tokens=None,
        default_output_tokens=_common_integer_default(output_specs),
        temperature_supported=temperature_supported,
        min_temperature=_common_schema_bound(temperature_specs, "minimum"),
        default_temperature=_common_numeric_default(temperature_specs),
        max_temperature=_common_schema_bound(temperature_specs, "maximum"),
        reasoning_supported=reasoning,
        reasoning_default=None,
    )


class CloudflareMetadataResolver:
    """Resolve catalog and schema facts without invoking a Cloudflare model."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = _CLOUDFLARE_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._account_id = account_id.strip()
        self._api_token = api_token.strip()
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        if not self._account_id:
            raise ValueError("account_id is required")
        if not self._api_token:
            raise ValueError("api_token is required")
        if not self._base_url:
            raise ValueError("base_url is required")

    def __repr__(self) -> str:
        return "CloudflareMetadataResolver()"

    async def _get(
        self,
        candidate: Candidate,
        client: httpx.AsyncClient,
        endpoint: str,
        params: dict[str, str],
    ) -> _MetadataResponse:
        response: _MetadataResponse | None = None
        failure: Literal["timeout", "transport_error"] | None = None
        try:
            response = await client.get(
                endpoint,
                headers={"Authorization": f"Bearer {self._api_token}"},
                params=params,
            )
        except httpx.TimeoutException:
            failure = "timeout"
        except httpx.RequestError:
            failure = "transport_error"
        if failure is not None:
            raise MetadataResolutionError(candidate, failure)
        if response is None:
            raise MetadataResolutionError(candidate, "invalid_response")
        return response

    async def resolve(self, candidate: Candidate) -> ProviderModelMetadataSnapshot:
        if candidate.provider != "cloudflare":
            raise MetadataResolutionError(candidate, "candidate_mismatch")
        endpoint = f"{self._base_url}/{self._account_id}/ai/models"
        search_endpoint = f"{endpoint}/search"
        schema_endpoint = f"{endpoint}/schema"

        if self._client is not None:
            return await self._resolve_with_client(
                candidate, self._client, search_endpoint, schema_endpoint
            )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._resolve_with_client(
                candidate, client, search_endpoint, schema_endpoint
            )

    async def _resolve_with_client(
        self,
        candidate: Candidate,
        client: httpx.AsyncClient,
        search_endpoint: str,
        schema_endpoint: str,
    ) -> ProviderModelMetadataSnapshot:
        model = candidate.model.lstrip("/")
        catalog_response = await self._get(candidate, client, search_endpoint, {"search": model})
        catalog_payload, catalog_retrieved_at = _response_object(candidate, catalog_response)
        catalog_model = _catalog_model(candidate, catalog_payload, catalog_retrieved_at)

        schema_response = await self._get(candidate, client, schema_endpoint, {"model": model})
        schema_payload, retrieved_at = _response_object(candidate, schema_response)
        if schema_payload.get("success") is not True:
            raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
        snapshot: ProviderModelMetadataSnapshot | None = None
        try:
            snapshot = _cloudflare_snapshot(candidate, catalog_model, schema_payload, retrieved_at)
        except MetadataResolutionError:
            raise
        except (TypeError, ValueError):
            pass
        if snapshot is None:
            raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
        return snapshot


def _normalize_gemini_model(model: str) -> str:
    return model.removeprefix(_GEMINI_MODEL_PREFIX)


def _optional_integer(
    candidate: Candidate,
    payload: dict[str, object],
    field: str,
    retrieved_at: str,
) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
    return value


def _optional_temperature(
    candidate: Candidate,
    payload: dict[str, object],
    field: str,
    retrieved_at: str,
) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    number = _finite_number(value)
    if number is None or number < 0:
        raise MetadataResolutionError(candidate, "invalid_response", retrieved_at=retrieved_at)
    return number


def _optional_boolean(value: object) -> bool | None:
    return value if type(value) is bool else None


class GeminiMetadataResolver:
    """Normalize metadata from the existing non-generating Gemini models.get path."""

    def __init__(self, client: _GeminiMetadataClient) -> None:
        self._client = client

    def __repr__(self) -> str:
        return "GeminiMetadataResolver()"

    async def resolve(self, candidate: Candidate) -> ProviderModelMetadataSnapshot:
        if candidate.provider != "gemini" or _normalize_gemini_model(
            self._client.normalized_model
        ) != _normalize_gemini_model(candidate.model):
            raise MetadataResolutionError(candidate, "candidate_mismatch")
        response: _MetadataResponse | None = None
        failure: Literal["timeout", "transport_error"] | None = None
        try:
            response = await self._client.get_model_metadata()
        except httpx.TimeoutException:
            failure = "timeout"
        except httpx.RequestError:
            failure = "transport_error"
        if failure is not None:
            raise MetadataResolutionError(candidate, failure)
        if response is None:
            raise MetadataResolutionError(candidate, "invalid_response")

        payload, retrieved_at = _response_object(candidate, response)
        name = payload.get("name")
        if not isinstance(name, str) or _normalize_gemini_model(name) != _normalize_gemini_model(
            candidate.model
        ):
            raise MetadataResolutionError(
                candidate, "candidate_mismatch", retrieved_at=retrieved_at
            )
        methods = payload.get("supportedGenerationMethods")
        if not isinstance(methods, list) or "generateContent" not in methods:
            raise MetadataResolutionError(
                candidate,
                "generation_capability_unconfirmed",
                retrieved_at=retrieved_at,
            )

        default_temperature = _optional_temperature(candidate, payload, "temperature", retrieved_at)
        return ProviderModelMetadataSnapshot(
            provider=candidate.provider,
            model=_normalize_gemini_model(name),
            retrieved_at=retrieved_at,
            metadata_source="provider_api",
            metadata_method="model_get",
            supports_text_generation=True,
            context_limit_tokens=None,
            input_limit_tokens=_optional_integer(
                candidate, payload, "inputTokenLimit", retrieved_at
            ),
            max_output_tokens=_optional_integer(
                candidate, payload, "outputTokenLimit", retrieved_at
            ),
            default_output_tokens=None,
            temperature_supported=True if default_temperature is not None else None,
            min_temperature=_optional_temperature(
                candidate, payload, "minTemperature", retrieved_at
            ),
            default_temperature=default_temperature,
            max_temperature=_optional_temperature(
                candidate, payload, "maxTemperature", retrieved_at
            ),
            reasoning_supported=_optional_boolean(payload.get("thinking")),
            reasoning_default=None,
        )


async def resolve_unique_metadata(
    candidates: Iterable[Candidate],
    *,
    cloudflare: _CandidateResolver,
    gemini: _CandidateResolver,
) -> dict[MetadataKey, ProviderModelMetadataSnapshot]:
    """Resolve one snapshot per provider/model identity in an evaluation campaign."""
    snapshots: dict[MetadataKey, ProviderModelMetadataSnapshot] = {}
    for candidate in candidates:
        model = (
            _normalize_gemini_model(candidate.model)
            if candidate.provider == "gemini"
            else candidate.model
        )
        key = (candidate.provider, model)
        if key in snapshots:
            continue
        if candidate.provider == "cloudflare":
            resolver = cloudflare
        elif candidate.provider == "gemini":
            resolver = gemini
        else:
            raise MetadataResolutionError(candidate, "unsupported_provider")
        snapshots[key] = await resolver.resolve(candidate)
    return snapshots
