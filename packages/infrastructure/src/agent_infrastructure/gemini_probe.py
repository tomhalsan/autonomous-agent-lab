import json
import logging
import time
from typing import Any

import httpx
from agent_contracts import InferenceHealthStatus, InferenceProviderHealth

from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class GeminiHealthProbe:
    """Infrastructure probe for Google Gemini reachability and model availability.

    Note on scope and limitations:
    This probe executes a non-generating metadata GET request against the Gemini
    model endpoint (GET /models/{model}). It verifies direct API reachability,
    credential acceptance, model visibility, and advertised generateContent support.

    Crucially, because this is a non-generating metadata probe:
    - It does NOT prove remaining generation quota.
    - It does NOT guarantee current real-time inference capacity.
    - It does NOT guarantee success of the next generation request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        gemini_client: GeminiClient | None = None,
    ) -> None:
        if gemini_client is not None:
            self._client = gemini_client
        else:
            self._client = GeminiClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                client=client,
            )
        self.model = self._client.model
        self.provider_name = "gemini"

    def redact(self, text: str) -> str:
        return self._client.redact(text)

    async def check_health(self) -> InferenceProviderHealth:
        """Probe Gemini availability using a non-generating model metadata inspection request."""
        start_time = time.perf_counter()
        try:
            response = await self._client.get_model_metadata()
        except httpx.TimeoutException:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.warning("Gemini health check timed out after %.2f ms", latency_ms)
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Google Gemini request timed out",
            )
        except httpx.RequestError as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = self.redact(str(exc))
            logger.warning("Gemini health check transport failure: %s", err_msg)
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Google Gemini transport failure: {err_msg}",
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

        # 1. Rate limiting vs Quota exhaustion (HTTP 429)
        # Conservative classification: RESOURCE_EXHAUSTED alone can represent rate limiting.
        # Classify QUOTA_EXHAUSTED only when structured provider details or an unmistakable
        # provider message explicitly establish quota exhaustion.
        if response.status_code == 429:
            is_quota = False
            err_details = ""
            if isinstance(data, dict):
                error_obj = data.get("error", {})
                if isinstance(error_obj, dict):
                    msg = str(error_obj.get("message", "")).lower()
                    status_str = str(error_obj.get("status", "")).lower()
                    details = error_obj.get("details", [])

                    # Unmistakable explicit quota exhaustion indicators
                    if "quota exceeded" in msg or "daily limit" in msg or "quota limit" in msg:
                        is_quota = True

                    # Inspect structured details
                    if isinstance(details, list):
                        for detail in details:
                            if isinstance(detail, dict):
                                reason = str(detail.get("reason", "")).upper()
                                if reason in ("QUOTA_EXCEEDED", "DAILY_LIMIT_EXCEEDED"):
                                    is_quota = True

                    err_details = self.redact(str(error_obj.get("message", ""))) or status_str

            if is_quota:
                status = InferenceHealthStatus.QUOTA_EXHAUSTED
                reason = err_details or "Google Gemini quota exhausted (HTTP 429)"
            else:
                status = InferenceHealthStatus.RATE_LIMITED
                reason = err_details or "Google Gemini rate limit exceeded (HTTP 429)"

            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=status,
                model=self.model,
                latency_ms=latency_ms,
                retry_after_seconds=retry_after,
                reason=reason,
            )

        # 2. Authentication failure (HTTP 401)
        if response.status_code == 401:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Google Gemini authentication failed (HTTP 401): invalid or expired API key",
            )

        # 3. Authorization failure (HTTP 403)
        if response.status_code == 403:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=(
                    "Google Gemini authorization failed (HTTP 403): "
                    "API key lacks permissions or Generative Language API is disabled"
                ),
            )

        # 4. Model absent / Not Found (HTTP 404)
        if response.status_code == 404:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Configured model {self.model!r} absent from Google Gemini catalog (HTTP 404)",
            )

        # 5. Server errors (HTTP 5xx)
        if response.status_code >= 500:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Google Gemini server error (HTTP {response.status_code})",
            )

        # 6. Other client errors (HTTP 4xx)
        if response.status_code >= 400:
            err_msg = f"HTTP {response.status_code}"
            if isinstance(data, dict):
                error_obj = data.get("error", {})
                if isinstance(error_obj, dict) and error_obj.get("message"):
                    err_msg = self.redact(str(error_obj["message"]))
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason=f"Google Gemini API error: {err_msg}",
            )

        # 7. Model verification in metadata response
        if isinstance(data, dict):
            # Check model name and supported generation methods
            model_name = str(data.get("name", ""))
            clean_target = self._client.normalized_model
            canonical_target = self._client.canonical_model
            # Compare canonical model identifiers exactly (e.g. 'models/gemini-2.5-flash' or 'gemini-2.5-flash')
            if model_name not in (clean_target, canonical_target):
                return InferenceProviderHealth(
                    provider_name=self.provider_name,
                    status=InferenceHealthStatus.UNAVAILABLE,
                    model=self.model,
                    latency_ms=latency_ms,
                    reason=(
                        f"Google Gemini metadata response model {model_name!r} "
                        f"does not match configured model {self.model!r}"
                    ),
                )

            supported_methods = data.get("supportedGenerationMethods")
            if not isinstance(supported_methods, list):
                return InferenceProviderHealth(
                    provider_name=self.provider_name,
                    status=InferenceHealthStatus.UNAVAILABLE,
                    model=self.model,
                    latency_ms=latency_ms,
                    reason=(
                        f"Google Gemini metadata response missing or invalid 'supportedGenerationMethods' list "
                        f"for model {self.model!r}"
                    ),
                )

            if "generateContent" not in supported_methods:
                return InferenceProviderHealth(
                    provider_name=self.provider_name,
                    status=InferenceHealthStatus.UNAVAILABLE,
                    model=self.model,
                    latency_ms=latency_ms,
                    reason=f"Configured model {self.model!r} does not advertise generateContent capability",
                )
        else:
            return InferenceProviderHealth(
                provider_name=self.provider_name,
                status=InferenceHealthStatus.UNAVAILABLE,
                model=self.model,
                latency_ms=latency_ms,
                reason="Google Gemini returned non-object JSON metadata payload",
            )

        return InferenceProviderHealth(
            provider_name=self.provider_name,
            status=InferenceHealthStatus.HEALTHY,
            model=self.model,
            latency_ms=latency_ms,
            reason=f"Google Gemini metadata probe verified model {self.model!r} is accessible",
        )
