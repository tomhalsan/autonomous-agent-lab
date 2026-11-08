import json
import logging
import os
import time
from typing import Any

import httpx
from agent_contracts import InferenceHealthStatus, InferenceProviderHealth

logger = logging.getLogger(__name__)

DEFAULT_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_CLOUDFLARE_AI_MODEL = "@cf/meta/llama-3.1-8b-instruct"


class CloudflareHealthProbe:
    """Infrastructure probe for Cloudflare Workers AI reachability and catalog status.

    Note on scope and limitations:
    This probe issues a zero-generation GET request against the Workers AI model-search
    catalog endpoint (/ai/models/search). It establishes account authentication,
    API reachability, and catalog accessibility for the configured model.

    Crucially, because this probe does not generate tokens or invoke model inference,
    it cannot and does not guarantee that inference compute capacity remains available
    or that the account has not exhausted its inference allocation, unless Cloudflare
    explicitly returns an active rate limit or quota exhaustion response on the catalog
    request itself.
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.account_id = (account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")).strip()
        self.api_token = (api_token or os.getenv("CLOUDFLARE_AI_API_TOKEN", "")).strip()
        raw_model = model or os.getenv("CLOUDFLARE_AI_MODEL", DEFAULT_CLOUDFLARE_AI_MODEL)
        self.model = raw_model.strip() if raw_model else DEFAULT_CLOUDFLARE_AI_MODEL
        self.base_url = (base_url or DEFAULT_CLOUDFLARE_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.client = client
        self.provider_name = "cloudflare"

        if not self.account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID is required for CloudflareHealthProbe"
            )
        if not self.api_token:
            raise ValueError(
                "CLOUDFLARE_AI_API_TOKEN is required for CloudflareHealthProbe"
            )

    def redact(self, text: str) -> str:
        if self.api_token and self.api_token in text:
            return text.replace(self.api_token, "[REDACTED]")
        return text

    async def check_health(self) -> InferenceProviderHealth:
        """Execute a non-generating catalog probe for the configured model."""
        endpoint = f"{self.base_url}/{self.account_id}/ai/models/search"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
        }
        params = {"search": self.model.lstrip("/")}

        start_time = time.perf_counter()
        try:
            if self.client is not None:
                response = await self.client.get(endpoint, headers=headers, params=params)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(endpoint, headers=headers, params=params)
        except httpx.TimeoutException:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.warning("Cloudflare health check timed out after %.2f ms", latency_ms)
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Cloudflare Workers AI request timed out",
            )
        except httpx.RequestError as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = self.redact(str(exc))
            logger.warning("Cloudflare health check transport failure: %s", err_msg)
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI transport failure: {err_msg}",
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        retry_after: float | None = None
        raw_retry_after = response.headers.get("Retry-After")
        if raw_retry_after:
            try:
                retry_after = float(raw_retry_after)
            except ValueError:
                pass

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError):
            data = None

        # 1. Rate limiting & Quota exhaustion (HTTP 429)
        if response.status_code == 429:
            is_quota = False
            err_details = ""
            if isinstance(data, dict):
                errors = data.get("errors") or []
                for err in errors:
                    if isinstance(err, dict):
                        code = err.get("code")
                        msg = str(err.get("message", "")).lower()
                        if code == 4006 or "neuron" in msg or "allocation" in msg or "quota" in msg:
                            is_quota = True
                        if msg:
                            err_details = self.redact(str(err.get("message", "")))
                    elif isinstance(err, str):
                        msg = err.lower()
                        if "neuron" in msg or "allocation" in msg or "quota" in msg:
                            is_quota = True
                        err_details = self.redact(err)

            if is_quota:
                status = InferenceHealthStatus.QUOTA_EXHAUSTED
                reason = err_details or "Cloudflare Workers AI quota exhausted (HTTP 429)"
            else:
                status = InferenceHealthStatus.RATE_LIMITED
                reason = err_details or "Cloudflare Workers AI rate limit exceeded (HTTP 429)"

            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=status,
                model=self.model,
                latency_ms=latency_ms,
                retry_after_seconds=retry_after,
                reason=reason,
            )

        # 2. Authentication (401) vs Authorization (403) failure
        if response.status_code == 401:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Cloudflare Workers AI authentication failed (HTTP 401): invalid or expired API token",
            )
        if response.status_code == 403:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Cloudflare Workers AI authorization failed (HTTP 403): insufficient permissions for account or resource",
            )

        # 3. Server errors (HTTP 5xx)
        if response.status_code >= 500:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI server error (HTTP {response.status_code})",
            )

        # 4. Client errors (HTTP 4xx)
        if response.status_code >= 400:
            err_msg = f"HTTP {response.status_code}"
            if isinstance(data, dict):
                errors = data.get("errors") or []
                if errors:
                    err_msg = self.redact(
                        "; ".join(
                            str(err.get("message", err) if isinstance(err, dict) else err)
                            for err in errors
                        )
                    )
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI API error: {err_msg}",
            )

        # 5. Unsuccessful response payload (success: False)
        # Note: Do not classify as DEGRADED unless affirmative evidence indicates usability.
        if isinstance(data, dict) and not data.get("success", True):
            errors = data.get("errors") or []
            err_msg = (
                "; ".join(
                    str(err.get("message", err) if isinstance(err, dict) else err)
                    for err in errors
                )
                if errors
                else "response payload indicated success=false"
            )
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI catalog search unsuccessful: {self.redact(err_msg)}",
            )

        # 6. Model verification in catalog response
        # Inspect a successful model-search response sufficiently to avoid reporting HEALTHY when the configured model is absent.
        if isinstance(data, dict):
            result = data.get("result")
            if isinstance(result, list):
                model_found = False
                for item in result:
                    if isinstance(item, dict):
                        item_id = item.get("id") or item.get("name")
                        if item_id and (
                            item_id == self.model
                            or item_id == self.model.lstrip("/")
                            or str(item_id).endswith(self.model.lstrip("/"))
                        ):
                            model_found = True
                            break
                    elif isinstance(item, str) and item == self.model:
                        model_found = True
                        break

                if not model_found:
                    return InferenceProviderHealth(
                        provider_name=self.provider_name,
                        status=InferenceHealthStatus.UNAVAILABLE,
                        model=self.model,
                        latency_ms=latency_ms,
                        reason=f"Configured model {self.model!r} absent from Cloudflare Workers AI catalog response",
                    )
            else:
                # If result is not a recognizable list of models:
                return InferenceProviderHealth(
                    provider_name=self.provider_name,
                    status=InferenceHealthStatus.UNAVAILABLE,
                    model=self.model,
                    latency_ms=latency_ms,
                    reason=(
                        f"Cloudflare Workers AI catalog response schema unrecognized "
                        f"(result field type {type(result).__name__}); model presence unverifiable"
                    ),
                )
        else:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Cloudflare Workers AI returned non-object JSON payload; model presence unverifiable",
            )

        return InferenceProviderHealth(
            provider_name=self.provider_name,
            status=InferenceHealthStatus.HEALTHY,
            model=self.model,
            latency_ms=latency_ms,
            reason=f"Cloudflare Workers AI catalog probe verified model {self.model!r} is accessible",
        )
