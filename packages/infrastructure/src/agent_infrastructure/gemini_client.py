import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from agent_contracts import FallbackEligibleInferenceError, NonFallbackInferenceError

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def normalize_gemini_model(model: str) -> str:
    """Normalize model identifier to simple form without leading 'models/' prefix."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    clean = model.strip().removeprefix("models/").lstrip("/")
    if not clean:
        raise ValueError("model must be a non-empty string after normalization")
    return clean


def canonical_gemini_model(model: str) -> str:
    """Normalize model identifier to canonical 'models/{model_id}' form."""
    return f"models/{normalize_gemini_model(model)}"


@dataclass(frozen=True)
class GeminiResponse:
    """Normalized response envelope returned by GeminiClient."""

    text: str
    finish_reason: str | None = None
    usage_metadata: dict[str, int] | None = None


class GeminiClient:
    """Narrowly focused Gemini REST transport client.

    Handles secure headers, endpoint construction, HTTP execution,
    provider-error parsing, candidate-text extraction, finish-reason
    extraction, usage extraction, and credential sanitization.

    Does NOT contain prompt policies, synthesis domain logic, or fallback logic.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        raw_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self._api_key = raw_key.strip() if raw_key else ""
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini provider"
            )

        if model is not None:
            model_val = model
        else:
            env_model = os.getenv("GEMINI_MODEL")
            model_val = env_model if env_model is not None else DEFAULT_GEMINI_MODEL
        self._normalized_model = normalize_gemini_model(model_val)
        self._canonical_model = canonical_gemini_model(model_val)
        self.model = self._normalized_model

        self.base_url = (
            base_url or os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self.client = client

    @property
    def normalized_model(self) -> str:
        """Normalized model identifier (without 'models/' prefix)."""
        return self._normalized_model

    @property
    def canonical_model(self) -> str:
        """Canonical model identifier (with 'models/' prefix)."""
        return self._canonical_model

    def redact(self, text: str) -> str:
        """Redact API key from text or error strings (defense-in-depth)."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "[REDACTED]")
        return text

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r}, api_key='[REDACTED]')"

    def __str__(self) -> str:
        return f"GeminiClient(model={self.model!r})"

    async def generate_content(self, payload: dict[str, Any]) -> GeminiResponse:
        """Execute a content generation POST request against the Gemini REST API."""
        endpoint = f"{self.base_url}/models/{self.normalized_model}:generateContent"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            if self.client is not None:
                response = await self.client.post(endpoint, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            logger.error("Gemini API generateContent request timed out")
            raise FallbackEligibleInferenceError(
                "Gemini API request timed out",
                provider_name="gemini",
                reason="timeout",
            ) from exc
        except httpx.RequestError as exc:
            err_msg = self.redact(str(exc))
            logger.error("Gemini API HTTP request failed: %s", err_msg)
            raise FallbackEligibleInferenceError(
                f"Gemini API request failed: {err_msg}",
                provider_name="gemini",
                reason="connection_error",
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error(
                "Gemini API returned unparseable JSON (status %d)",
                response.status_code,
            )
            raise FallbackEligibleInferenceError(
                f"Gemini API returned invalid JSON (status {response.status_code})",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            ) from exc

        if not isinstance(data, dict):
            raise FallbackEligibleInferenceError(
                "Malformed Gemini API response: top-level JSON must be an object",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            )

        if response.status_code >= 400 or "error" in data:
            error_obj = data.get("error", {})
            err_msg = f"HTTP {response.status_code}"
            if isinstance(error_obj, dict):
                err_msg = error_obj.get("message") or err_msg
            elif isinstance(error_obj, str):
                err_msg = error_obj
            clean_err = self.redact(str(err_msg))
            logger.error("Gemini API error (status %d): %s", response.status_code, clean_err)
            full_msg = f"Gemini API error (status {response.status_code}): {clean_err}"
            if response.status_code == 402:
                raise FallbackEligibleInferenceError(
                    full_msg,
                    provider_name="gemini",
                    reason="payment_required",
                    status_code=402,
                )
            if response.status_code == 429 or (500 <= response.status_code <= 599):
                raise FallbackEligibleInferenceError(
                    full_msg,
                    provider_name="gemini",
                    reason="rate_limited" if response.status_code == 429 else f"server_error_{response.status_code}",
                    status_code=response.status_code,
                )
            if 400 <= response.status_code < 500:
                raise NonFallbackInferenceError(
                    full_msg,
                    provider_name="gemini",
                    reason=f"client_error_{response.status_code}",
                    status_code=response.status_code,
                )
            raise NonFallbackInferenceError(
                full_msg,
                provider_name="gemini",
                reason="unclassified_api_error",
                status_code=response.status_code,
            )

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
                block_reason = prompt_feedback["blockReason"]
                clean_block = self.redact(str(block_reason))
                raise NonFallbackInferenceError(
                    f"Gemini API generation blocked: {clean_block}",
                    provider_name="gemini",
                    reason=f"blocked_{clean_block.lower()}",
                    status_code=response.status_code,
                )
            raise FallbackEligibleInferenceError(
                "Malformed Gemini API response: missing or empty 'candidates' list",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            )

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise FallbackEligibleInferenceError(
                "Malformed Gemini API response: candidate must be an object",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            )

        finish_reason = candidate.get("finishReason")
        if finish_reason and str(finish_reason).upper() in (
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
        ):
            clean_finish = self.redact(str(finish_reason))
            raise NonFallbackInferenceError(
                f"Gemini API generation blocked: {clean_finish}",
                provider_name="gemini",
                reason=f"blocked_{clean_finish.lower()}",
                status_code=response.status_code,
            )

        content = candidate.get("content")
        if not isinstance(content, dict):
            raise FallbackEligibleInferenceError(
                "Malformed Gemini API response: missing or invalid 'content' object",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            )

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise FallbackEligibleInferenceError(
                "Malformed Gemini API response: missing or invalid 'parts' list",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            )

        text_parts: list[str] = []
        for p in parts:
            if isinstance(p, dict):
                val = p.get("text")
                if isinstance(val, str):
                    text_parts.append(val)

        response_text = "".join(text_parts)
        if not response_text.strip():
            raise FallbackEligibleInferenceError(
                "Malformed Gemini API response: candidate produced empty or whitespace-only text",
                provider_name="gemini",
                reason="malformed_response",
                status_code=response.status_code,
            )

        finish_reason = candidate.get("finishReason")

        # Exclude booleans from numeric usage-metadata extraction
        raw_usage = data.get("usageMetadata")
        usage_metadata: dict[str, int] | None = None
        if isinstance(raw_usage, dict):
            extracted = {}
            for k, v in raw_usage.items():
                if isinstance(v, int) and type(v) is not bool:
                    extracted[k] = v
            if extracted:
                usage_metadata = extracted

        return GeminiResponse(
            text=response_text,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage_metadata=usage_metadata,
        )

    async def get_model_metadata(self) -> httpx.Response:
        """Execute a non-generating GET request against the model metadata endpoint."""
        endpoint = f"{self.base_url}/models/{self.normalized_model}"
        headers = {
            "x-goog-api-key": self._api_key,
        }

        if self.client is not None:
            return await self.client.get(endpoint, headers=headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(endpoint, headers=headers)
