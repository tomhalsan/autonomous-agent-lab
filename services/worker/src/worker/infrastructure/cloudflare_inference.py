import logging
import os
from typing import Any

import httpx
from agent_contracts import (
    FallbackEligibleInferenceError,
    InferenceProviderHealth,
    NonFallbackInferenceError,
)
from agent_infrastructure import CloudflareHealthProbe

logger = logging.getLogger(__name__)

DEFAULT_CLOUDFLARE_AI_MODEL = "@cf/meta/llama-3.1-8b-instruct"
DEFAULT_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_CLOUDFLARE_AI_MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a helpful and versatile AI assistant. "
    "Answer the user's prompt directly, clearly, and accurately."
)


class CloudflareInferenceProvider:
    """Infrastructure adapter that executes LLM inference via the Cloudflare Workers AI REST API."""

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_tokens: int | None = None,
    ) -> None:
        self._account_id = (account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")).strip()
        self._api_token = (api_token or os.getenv("CLOUDFLARE_AI_API_TOKEN", "")).strip()
        self.provider_name = "cloudflare"
        raw_model = model or os.getenv("CLOUDFLARE_AI_MODEL", DEFAULT_CLOUDFLARE_AI_MODEL)
        self.model = raw_model.strip() if raw_model else DEFAULT_CLOUDFLARE_AI_MODEL
        self._base_url = (base_url or DEFAULT_CLOUDFLARE_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._client = client

        env_max_tokens = os.getenv("CLOUDFLARE_AI_MAX_TOKENS")
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif env_max_tokens and env_max_tokens.strip().isdigit():
            self.max_tokens = int(env_max_tokens.strip())
        else:
            self.max_tokens = DEFAULT_CLOUDFLARE_AI_MAX_TOKENS

        if not self._account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required for CloudflareInferenceProvider"
            )
        if not self._api_token:
            raise ValueError(
                "CLOUDFLARE_AI_API_TOKEN environment variable is required for CloudflareInferenceProvider"
            )

        self._health_probe = CloudflareHealthProbe(
            account_id=self._account_id,
            api_token=self._api_token,
            model=self.model,
            client=self._client,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def _redact(self, text: str) -> str:
        if self._api_token and self._api_token in text:
            return text.replace(self._api_token, "[REDACTED]")
        return text

    def __repr__(self) -> str:
        return (
            f"CloudflareInferenceProvider(account_id={self._account_id!r}, "
            f"model={self.model!r}, api_token='[REDACTED]')"
        )

    def __str__(self) -> str:
        return f"CloudflareInferenceProvider(model={self.model!r})"

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
        }

    async def check_health(self) -> InferenceProviderHealth:
        """Probe Cloudflare Workers AI availability via the shared infrastructure health probe."""
        return await self._health_probe.check_health()

    async def generate(self, prompt: str) -> str:
        logger.info("Executing Cloudflare Workers AI generation with model %s", self.model)
        endpoint = f"{self._base_url}/{self._account_id}/ai/run/{self.model.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt)

        try:
            if self._client is not None:
                response = await self._client.post(endpoint, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            err_msg = self._redact(str(exc))
            logger.error("Cloudflare Workers AI generate request timed out: %s", err_msg)
            raise FallbackEligibleInferenceError(
                f"Cloudflare Workers AI request failed: {err_msg}",
                provider_name=self.provider_name,
                reason="timeout",
            ) from exc
        except httpx.RequestError as exc:
            err_msg = self._redact(str(exc))
            logger.error("Cloudflare Workers AI HTTP request failed: %s", err_msg)
            raise FallbackEligibleInferenceError(
                f"Cloudflare Workers AI request failed: {err_msg}",
                provider_name=self.provider_name,
                reason="connection_error",
            ) from exc

        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            logger.error(
                "Cloudflare Workers AI returned unparseable JSON (status %d)",
                response.status_code,
            )
            raise FallbackEligibleInferenceError(
                f"Cloudflare Workers AI returned invalid JSON (status {response.status_code})",
                provider_name=self.provider_name,
                reason="malformed_response",
                status_code=response.status_code,
            ) from exc

        if not isinstance(data, dict):
            raise FallbackEligibleInferenceError(
                "Malformed Cloudflare Workers AI response: top-level JSON must be an object",
                provider_name=self.provider_name,
                reason="malformed_response",
                status_code=response.status_code,
            )

        if response.status_code >= 400 or not data.get("success", True):
            errors = data.get("errors") or []
            err_details = "; ".join(
                str(err.get("message", err) if isinstance(err, dict) else err)
                for err in errors
            ) or f"HTTP status {response.status_code}"
            err_msg = self._redact(err_details)
            logger.error(
                "Cloudflare Workers AI API error (status %d): %s",
                response.status_code,
                err_msg,
            )
            full_msg = f"Cloudflare Workers AI API error (status {response.status_code}): {err_msg}"
            if response.status_code == 429 or (500 <= response.status_code <= 599):
                raise FallbackEligibleInferenceError(
                    full_msg,
                    provider_name=self.provider_name,
                    reason="rate_limited" if response.status_code == 429 else f"server_error_{response.status_code}",
                    status_code=response.status_code,
                )
            if 400 <= response.status_code < 500:
                raise NonFallbackInferenceError(
                    full_msg,
                    provider_name=self.provider_name,
                    reason=f"client_error_{response.status_code}",
                    status_code=response.status_code,
                )
            raise NonFallbackInferenceError(
                full_msg,
                provider_name=self.provider_name,
                reason="unclassified_api_error",
                status_code=response.status_code,
            )

        result_payload = data.get("result")
        if not isinstance(result_payload, dict):
            raise FallbackEligibleInferenceError(
                "Malformed Cloudflare Workers AI response: missing or invalid 'result' object",
                provider_name=self.provider_name,
                reason="malformed_response",
                status_code=response.status_code,
            )

        response_text = result_payload.get("response")
        if response_text is None or not isinstance(response_text, str):
            raise FallbackEligibleInferenceError(
                "Malformed Cloudflare Workers AI response: missing 'result.response' string",
                provider_name=self.provider_name,
                reason="malformed_response",
                status_code=response.status_code,
            )

        finish_reason = (
            result_payload.get("finish_reason")
            or data.get("finish_reason")
            or (
                data.get("choices")
                and isinstance(data["choices"], list)
                and data["choices"][0].get("finish_reason")
            )
        )
        if finish_reason in ("length", "max_tokens"):
            logger.warning(
                "Cloudflare generation output token limited (model=%s, finish_reason=%s)",
                self.model,
                finish_reason,
            )

        return response_text.strip()
