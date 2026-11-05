import logging
import os
from typing import Any

import httpx

from research_agent.domain.models import SynthesisResult

logger = logging.getLogger(__name__)

DEFAULT_CLOUDFLARE_AI_MODEL = "@cf/meta/llama-3.1-8b-instruct"
DEFAULT_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"

SYSTEM_PROMPT = (
    "You are an AI research synthesis assistant. Your goal is to synthesize grounded research evidence "
    "into a concise, factual, and well-structured answer.\n\n"
    "Strict instructions:\n"
    "1. Answer the research question directly and concisely.\n"
    "2. Rely exclusively on the provided grounded evidence. Do not invent, extrapolate, or assume unsupported facts.\n"
    "3. Clearly identify key findings.\n"
    "4. Preserve and source relevant URLs from the evidence where available.\n"
    "5. If the provided evidence is insufficient, contradictory, or empty, explicitly state that the evidence "
    "is insufficient to fully answer the question."
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
    ) -> None:
        self._account_id = (account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")).strip()
        self._api_token = (api_token or os.getenv("CLOUDFLARE_AI_API_TOKEN", "")).strip()
        raw_model = model or os.getenv("CLOUDFLARE_AI_MODEL", DEFAULT_CLOUDFLARE_AI_MODEL)
        self.model = raw_model.strip() if raw_model else DEFAULT_CLOUDFLARE_AI_MODEL
        self._base_url = (base_url or DEFAULT_CLOUDFLARE_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._client = client

        if not self._account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required for CloudflareInferenceProvider"
            )
        if not self._api_token:
            raise ValueError(
                "CLOUDFLARE_AI_API_TOKEN environment variable is required for CloudflareInferenceProvider"
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

    def _build_payload(self, question: str, evidence: str) -> dict[str, Any]:
        stripped_evidence = evidence.strip() if evidence else ""
        formatted_evidence = (
            stripped_evidence if stripped_evidence else "[No grounded evidence provided]"
        )
        user_prompt = (
            f"Research Question:\n{question}\n\n"
            f"Grounded Evidence:\n{formatted_evidence}"
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        }

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        logger.info("Executing Cloudflare Workers AI synthesis with model %s", self.model)
        endpoint = f"{self._base_url}/{self._account_id}/ai/run/{self.model.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(question, evidence)

        try:
            if self._client is not None:
                response = await self._client.post(
                    endpoint, headers=headers, json=payload
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        endpoint, headers=headers, json=payload
                    )
        except Exception as exc:
            err_msg = self._redact(str(exc))
            logger.error("Cloudflare Workers AI HTTP request failed: %s", err_msg)
            raise RuntimeError(
                f"Cloudflare Workers AI request failed: {err_msg}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            logger.error(
                "Cloudflare Workers AI returned unparseable JSON (status %d)",
                response.status_code,
            )
            raise RuntimeError(
                f"Cloudflare Workers AI returned invalid JSON (status {response.status_code})"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(  # noqa: TRY004
                "Malformed Cloudflare Workers AI response: top-level JSON must be an object"
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
            raise RuntimeError(
                f"Cloudflare Workers AI API error (status {response.status_code}): {err_msg}"
            )

        result_payload = data.get("result")
        if not isinstance(result_payload, dict):
            raise RuntimeError(  # noqa: TRY004
                "Malformed Cloudflare Workers AI response: missing or invalid 'result' object"
            )

        response_text = result_payload.get("response")
        if response_text is None or not isinstance(response_text, str):
            raise RuntimeError(
                "Malformed Cloudflare Workers AI response: missing 'result.response' string"
            )

        return SynthesisResult(
            question=question,
            content=response_text.strip(),
            evidence=evidence,
            provider="cloudflare",
            model=self.model,
        )
