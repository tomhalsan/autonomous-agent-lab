import logging
import os
from typing import Any

import httpx
from agent_contracts import (
    FallbackEligibleInferenceError,
    InferenceProviderHealth,
    InferenceResponse,
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

    async def generate(self, prompt: str) -> InferenceResponse:
        logger.info("Executing Cloudflare Workers AI generation with model %s", self.model)
        endpoint = f"{self._base_url}/{self._account_id}/ai/run/{self.model.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt)

        response: Any = None
        request_failure: FallbackEligibleInferenceError | None = None
        try:
            if self._client is not None:
                response = await self._client.post(endpoint, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            error_type = type(exc).__name__
            logger.error(
                "Cloudflare Workers AI generate request timed out (error_type=%s)",
                error_type,
            )
            request_failure = FallbackEligibleInferenceError(
                f"Cloudflare Workers AI request failed (error_type={error_type})",
                provider_name=self.provider_name,
                reason="timeout",
            )
        except httpx.RequestError as exc:
            error_type = type(exc).__name__
            logger.error(
                "Cloudflare Workers AI HTTP request failed (error_type=%s)",
                error_type,
            )
            request_failure = FallbackEligibleInferenceError(
                f"Cloudflare Workers AI request failed (error_type={error_type})",
                provider_name=self.provider_name,
                reason="connection_error",
            )

        if request_failure is not None:
            raise request_failure from None
        if response is None:
            raise FallbackEligibleInferenceError(
                "Cloudflare Workers AI request returned no response",
                provider_name=self.provider_name,
                reason="connection_error",
            )

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

        response_text: str | None = None
        finish_reason: str | None = None

        raw_response = result_payload.get("response")
        if isinstance(raw_response, str) and raw_response.strip():
            response_text = raw_response
            raw_fr = result_payload.get("finish_reason") or data.get("finish_reason")
            if isinstance(raw_fr, str):
                finish_reason = raw_fr
        else:
            choices = result_payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            response_text = content
                            raw_fr = first_choice.get("finish_reason")
                            if isinstance(raw_fr, str):
                                finish_reason = raw_fr

        if response_text is None:
            if "choices" in result_payload:
                err = "Malformed Cloudflare Workers AI response: missing or invalid 'result.choices[0].message.content' string"
            else:
                err = "Malformed Cloudflare Workers AI response: missing 'result.response' string"
            raise FallbackEligibleInferenceError(
                err,
                provider_name=self.provider_name,
                reason="malformed_response",
                status_code=response.status_code,
            )

        if finish_reason in ("length", "max_tokens"):
            logger.warning(
                "Cloudflare generation output token limited (model=%s, finish_reason=%s)",
                self.model,
                finish_reason,
            )

        raw_usage = result_payload.get("usage") or data.get("usage")
        input_tokens = None
        output_tokens = None
        total_tokens = None
        if isinstance(raw_usage, dict):
            pt = raw_usage.get("prompt_tokens")
            if isinstance(pt, int) and type(pt) is not bool and pt >= 0:
                input_tokens = pt
            ct = raw_usage.get("completion_tokens")
            if isinstance(ct, int) and type(ct) is not bool and ct >= 0:
                output_tokens = ct
            tt = raw_usage.get("total_tokens")
            if isinstance(tt, int) and type(tt) is not bool and tt >= 0:
                total_tokens = tt

        return InferenceResponse(
            text=response_text.strip(),
            provider_name=self.provider_name,
            model_name=self.model,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
