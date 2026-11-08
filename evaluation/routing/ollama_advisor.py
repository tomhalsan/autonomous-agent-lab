"""Evaluation-only Ollama-backed implementation of the RoutingAdvisor port."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.parse import urlsplit

import httpx
from orchestrator.domain.advisor import (
    MAX_ADVISOR_REASON_LENGTH,
    AdvisorExecutionError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
)
from orchestrator.domain.assessment import RoutingAssessment
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.models import OllamaProviderTiming, parse_provider_duration_ns

OLLAMA_ROUTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "capability": {
            "type": "string",
            "enum": ["general", "research"],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["capability", "reason"],
    "additionalProperties": False,
}
EXACT_RESPONSE_KEYS: frozenset[str] = frozenset({"capability", "reason"})
SUPPORTED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def format_evaluation_prompt(prompt: str) -> str:
    """Format a task prompt for prompt-only capability classification.

    Deliberately excludes deterministic provisional capability, signal types,
    reasons, and expected ground-truth labels to prevent model anchoring.
    """
    return (
        "You are a task classification advisor. Analyze the user prompt and classify it into exactly one capability:\n\n"
        "GENERAL:\n"
        "- Established conceptual explanations\n"
        "- Reasoning, computation, or analysis on user-supplied information\n"
        "- Coding, debugging, or transformation tasks using supplied code or text\n"
        "- Tasks answerable without live external retrieval\n\n"
        "RESEARCH:\n"
        "- Current or rapidly changing information (news, recent releases, current pricing, etc.)\n"
        "- Explicit external retrieval or live search requirements\n"
        "- Inquiries targeting external evidence, citations, or literature\n"
        "- Tasks whose accurate completion depends on obtaining outside information\n\n"
        f"User prompt:\n{prompt.strip()}\n\n"
        'Respond with valid JSON containing "capability" ("general" or "research") and a non-empty "reason".'
    )


class OllamaRoutingAdvisor:
    """Evaluation-only Ollama-backed implementation of RoutingAdvisor application port.

    This adapter acts strictly as an experimental evaluation fixture.
    It performs HTTP calls to an Ollama /api/generate endpoint, enforces strict
    schema-validated JSON parsing on untrusted model output, and maps failures
    consistently into the domain AdvisorError hierarchy.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        keep_alive: str | int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")

        parsed_url = urlsplit(base_url.strip())
        if parsed_url.scheme not in SUPPORTED_URL_SCHEMES:
            raise ValueError(
                f"base_url scheme must be 'http' or 'https', got {parsed_url.scheme!r}"
            )
        if not parsed_url.netloc and not parsed_url.hostname:
            raise ValueError("base_url must contain a valid host")

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a numeric value")  # noqa: TRY004
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")

        if keep_alive is not None:
            if isinstance(keep_alive, bool) or not isinstance(keep_alive, (int, str)):
                raise ValueError(
                    "keep_alive must be an integer (seconds), duration string, or None"
                )
            if isinstance(keep_alive, int) and keep_alive < 0:
                raise ValueError("keep_alive integer value must be non-negative")
            if isinstance(keep_alive, str) and not keep_alive.strip():
                raise ValueError("keep_alive string value cannot be empty")

        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = float(timeout)
        self.keep_alive = keep_alive
        self._client = client
        self.last_provider_timing: OllamaProviderTiming | None = None

    async def unload(self) -> bool:
        """Explicitly request model unloading from Ollama memory via keep_alive=0.

        Returns:
            bool: True if the unload request was acknowledged with HTTP 200, False otherwise.
        """
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "keep_alive": 0}
        try:
            if self._client is not None:
                resp = await self._client.post(url, json=payload, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
            return resp.status_code == 200
        except (httpx.HTTPError, TimeoutError):
            return False

    def build_request_payload(self, prompt: str) -> dict[str, Any]:
        """Construct the request payload for the Ollama /api/generate endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": format_evaluation_prompt(prompt),
            "stream": False,
            "format": OLLAMA_ROUTING_SCHEMA,
            "options": {
                "temperature": 0.0,
            },
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    async def advise(
        self,
        prompt: str,
        assessment: RoutingAssessment,
    ) -> AdvisorRecommendation:
        """Consult Ollama for a capability recommendation.

        Note: assessment parameter is preserved to satisfy the RoutingAdvisor protocol,
        but assessment details are deliberately omitted from the prompt payload in
        prompt-only evaluation mode.
        """
        self.last_provider_timing = None
        payload = self.build_request_payload(prompt)
        url = f"{self.base_url}/api/generate"

        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise AdvisorUnavailableError("Ollama service unavailable or connection failed") from exc
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise AdvisorTimeoutError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise AdvisorExecutionError("Ollama HTTP transport failure") from exc

        # HTTP status code checks
        if response.status_code in {502, 503, 504}:
            raise AdvisorUnavailableError(
                f"Ollama service unavailable (HTTP status {response.status_code})"
            )
        if response.status_code != 200:
            raise AdvisorExecutionError(
                f"Ollama request failed with HTTP status {response.status_code}"
            )

        # Parse outer Ollama JSON response (suppress exception chaining to avoid leaking raw response)
        try:
            outer_data = response.json()
        except (json.JSONDecodeError, ValueError):
            raise AdvisorExecutionError("Malformed JSON in Ollama outer response") from None

        if not isinstance(outer_data, dict):
            raise AdvisorExecutionError("Ollama outer response is not a JSON object")

        # Parse provider timing into a local value; publish to self.last_provider_timing
        # only after the recommendation is successfully validated and constructed.
        local_timing = OllamaProviderTiming(
            total_duration_ns=parse_provider_duration_ns(outer_data.get("total_duration")),
            load_duration_ns=parse_provider_duration_ns(outer_data.get("load_duration")),
            prompt_eval_duration_ns=parse_provider_duration_ns(
                outer_data.get("prompt_eval_duration")
            ),
            eval_duration_ns=parse_provider_duration_ns(outer_data.get("eval_duration")),
        )

        if "response" not in outer_data:
            raise AdvisorExecutionError("Ollama outer response missing required 'response' field")

        raw_content = outer_data["response"]
        if not isinstance(raw_content, str):
            raise AdvisorExecutionError("Ollama 'response' field is not a string")

        recommendation = self._validate_and_parse_model_output(raw_content)
        self.last_provider_timing = local_timing
        return recommendation

    def _validate_and_parse_model_output(self, raw_content: str) -> AdvisorRecommendation:
        """Validate and parse raw model response string.

        Applies strict validation: rejects markdown fences, conversational prose,
        invalid JSON, missing fields, unexpected keys, or unrecognized capabilities
        without echoing arbitrary raw model output in exception messages.
        """
        # Strict validation: reject markdown code fences or conversational prose wrappers
        if "```" in raw_content:
            raise AdvisorExecutionError(
                "Model response contains markdown code fences; strict raw JSON required"
            )

        # Parse model JSON output (suppress exception chaining to avoid leaking raw model output)
        try:
            content_data = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError):
            raise AdvisorExecutionError("Model response is not valid JSON") from None

        if not isinstance(content_data, dict):
            raise AdvisorExecutionError("Parsed model response content is not a JSON object")

        if "capability" not in content_data:
            raise AdvisorExecutionError("Model response missing required 'capability' key")

        if "reason" not in content_data:
            raise AdvisorExecutionError("Model response missing required 'reason' key")

        if set(content_data.keys()) != EXACT_RESPONSE_KEYS:
            raise AdvisorExecutionError(
                "Model response contains unexpected fields; exactly 'capability' and 'reason' are allowed"
            )

        cap_raw = content_data["capability"]
        if not isinstance(cap_raw, str):
            raise AdvisorExecutionError("Model 'capability' value must be a string")

        norm_cap = cap_raw.strip().lower()
        if norm_cap == "general":
            capability = AgentCapability.GENERAL
        elif norm_cap == "research":
            capability = AgentCapability.RESEARCH
        else:
            raise AdvisorExecutionError("Unsupported capability value returned by model")

        reason_raw = content_data["reason"]
        if not isinstance(reason_raw, str) or not reason_raw.strip():
            raise AdvisorExecutionError("Model 'reason' value must be a non-empty string")

        clean_reason = reason_raw.strip()
        if len(clean_reason) > MAX_ADVISOR_REASON_LENGTH:
            raise AdvisorExecutionError(
                f"Model reason exceeds maximum allowed length of {MAX_ADVISOR_REASON_LENGTH} characters"
            )

        try:
            return AdvisorRecommendation(
                recommended_capability=capability,
                reason=clean_reason,
            )
        except (TypeError, ValueError):
            raise AdvisorExecutionError("AdvisorRecommendation domain validation failed") from None

