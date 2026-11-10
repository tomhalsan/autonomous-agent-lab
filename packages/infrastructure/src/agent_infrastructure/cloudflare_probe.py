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


def _matches_exact_model(candidate: Any, target_model: str) -> bool:
    """Determine whether a candidate string exactly matches the target model identifier.

    Enforces that related variants (e.g. '-fp8') are never treated as confirmation of the
    configured model itself.
    """
    if not candidate or not isinstance(candidate, str):
        return False
    cand = candidate.strip()
    target = target_model.strip()
    target_no_slash = target.lstrip("/")

    if cand == target or cand == target_no_slash:
        return True
    if cand.startswith("@cf/") and cand[4:] == target_no_slash:
        return True
    return cand.endswith("/" + target_no_slash)


class CloudflareHealthProbe:
    """Infrastructure probe for Cloudflare Workers AI reachability and catalog status.

    Note on scope and limitations:
    This probe issues zero-generation GET requests against Cloudflare Workers AI metadata
    endpoints to establish account authentication, API reachability, and catalog accessibility
    for the configured model.

    Verification workflow:
    1. Issues a zero-generation GET request against the Workers AI model-search catalog
       endpoint (/ai/models/search).
    2. If the catalog search confirms the exact configured model, it is verified as HEALTHY.
    3. If the catalog search succeeds but returns related variants (e.g., '-fp8') rather than
       the exact configured identifier, the probe does NOT accept the variant as confirmation.
       Instead, it issues a direct zero-generation schema lookup (/ai/models/schema?model=...)
       for the exact configured model to attempt affirmative metadata confirmation.
    4. If exact metadata cannot be confirmed in catalog or schema, but API reachability and
       authentication are established, the probe returns DEGRADED (reflecting an unverified
       executable alias that may still succeed at runtime, preserving execution eligibility).
    5. If authentication, authorization, rate limit, quota, server, or transport errors occur,
       they are classified explicitly as UNAVAILABLE, RATE_LIMITED, or QUOTA_EXHAUSTED.

    Crucially, because this probe does not generate tokens or invoke model inference,
    it cannot and does not guarantee that inference compute capacity remains available
    or that the account has not exhausted its inference allocation, unless Cloudflare
    explicitly returns an active rate limit or quota exhaustion response.
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

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        raw_retry_after = response.headers.get("Retry-After")
        if raw_retry_after:
            try:
                return float(raw_retry_after)
            except ValueError:
                pass
        return None

    def _parse_json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError):
            return None

    def _classify_common_errors(
        self,
        response: httpx.Response,
        data: Any,
        latency_ms: float,
    ) -> InferenceProviderHealth | None:
        """Classify 429, 401, 403, and 5xx responses common to all endpoints."""
        if response.status_code == 429:
            retry_after = self._parse_retry_after(response)
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

        if response.status_code >= 500:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI server error (HTTP {response.status_code})",
            )

        return None

    async def _probe_model_schema(
        self,
        client: Any,
        start_time: float,
    ) -> InferenceProviderHealth:
        """Probe the dedicated zero-generation model schema endpoint.

        Invoked when catalog search does not return an exact match (e.g. returning
        related variants like '-fp8' or unranked matches), attempting affirmative
        schema verification without generating tokens. If metadata cannot confirm
        the exact model, returns DEGRADED rather than falsely excluding an executable alias.
        """
        endpoint = f"{self.base_url}/{self.account_id}/ai/models/schema"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
        }
        params = {"model": self.model.lstrip("/")}

        try:
            response = await client.get(endpoint, headers=headers, params=params)
        except httpx.TimeoutException:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.warning("Cloudflare schema probe timed out after %.2f ms", latency_ms)
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
            logger.warning("Cloudflare schema probe transport failure: %s", err_msg)
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI transport failure: {err_msg}",
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        data = self._parse_json(response)

        common_err = self._classify_common_errors(response, data, latency_ms)
        if common_err is not None:
            return common_err

        # Affirmative schema confirmation
        if response.status_code == 200 and isinstance(data, dict) and data.get("success", True):
            result = data.get("result")
            if isinstance(result, dict) and bool(result):
                return InferenceProviderHealth(
                    provider_name=self.provider_name,
                    status=InferenceHealthStatus.HEALTHY,
                    model=self.model,
                    latency_ms=latency_ms,
                    reason=f"Cloudflare Workers AI schema probe verified model {self.model!r} is accessible",
                )

        err_details = ""
        if isinstance(data, dict):
            errors = data.get("errors") or []
            if errors:
                err_details = self.redact(
                    "; ".join(
                        str(err.get("message", err) if isinstance(err, dict) else err)
                        for err in errors
                    )
                )

        # Authenticated API reachability was established via catalog search, but exact model
        # metadata could not be verified in catalog or schema (e.g. executable alias like
        # @cf/meta/llama-3.1-8b-instruct). Return DEGRADED to preserve eligibility for
        # runtime execution while truthfully reflecting unverified metadata status.
        err_suffix = f" (schema lookup: {err_details})" if err_details else ""
        return InferenceProviderHealth(
            provider_name=self.provider_name,
            status=InferenceHealthStatus.DEGRADED,
            model=self.model,
            latency_ms=latency_ms,
            reason=(
                f"Cloudflare Workers AI API reachability and authentication established; "
                f"exact configured model {self.model!r} could not be verified in catalog search or schema metadata "
                f"(may be an executable alias{err_suffix}); runtime inference may still succeed"
            ),
        )

    async def _check_health_with_client(
        self,
        client: Any,
        start_time: float,
    ) -> InferenceProviderHealth:
        """Execute health-check workflow with the provided HTTP client."""
        endpoint = f"{self.base_url}/{self.account_id}/ai/models/search"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
        }
        params = {"search": self.model.lstrip("/")}

        try:
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
        data = self._parse_json(response)

        common_err = self._classify_common_errors(response, data, latency_ms)
        if common_err is not None:
            return common_err

        # Client errors (HTTP 4xx other than 401/403/429)
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

        # Unsuccessful response payload (success: False)
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

        # Model verification in catalog response
        model_found = False
        if isinstance(data, dict):
            result = data.get("result")
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        for candidate in (item.get("name"), item.get("id")):
                            if _matches_exact_model(candidate, self.model):
                                model_found = True
                                break
                        if model_found:
                            break
                    elif isinstance(item, str) and _matches_exact_model(item, self.model):
                        model_found = True
                        break
            else:
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

        if model_found:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.HEALTHY,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Cloudflare Workers AI catalog probe verified model {self.model!r} is accessible",
            )

        # If catalog search did not affirmatively confirm the model, verify via direct schema lookup
        return await self._probe_model_schema(client, start_time)

    async def check_health(self) -> InferenceProviderHealth:
        """Execute a non-generating catalog probe for the configured model."""
        start_time = time.perf_counter()
        if self.client is not None:
            return await self._check_health_with_client(self.client, start_time)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._check_health_with_client(client, start_time)
