import logging
import os
import re
from typing import Any

import httpx

from research_agent.domain.models import SynthesisResult

logger = logging.getLogger(__name__)

DEFAULT_CLOUDFLARE_AI_MODEL = "@cf/meta/llama-3.1-8b-instruct"
DEFAULT_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_CLOUDFLARE_AI_MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are an AI research synthesis assistant. Your goal is to synthesize grounded research evidence "
    "into a concise, factual, and well-structured answer.\n\n"
    "Strict instructions:\n"
    "1. Answer the research question directly and concisely.\n"
    "2. Rely exclusively on the provided grounded evidence. Do not invent, extrapolate, or assume unsupported facts. "
    "Consider the source-quality indicators (e.g. authority tier, cross-domain keyword overlap) when evaluating evidence strength, "
    "without assuming high-ranked sources are infallible. Treat keyword overlap across domains as a retrieval signal rather than factual proof or consensus agreement.\n"
    "3. Answer only the requested number of findings. Clearly identify key findings: include one concise explanation per finding, and include a short key finding per item only if space permits. Avoid adding extra findings beyond the requested count, and do not add an unrequested 'Additionally' section.\n"
    "4. Preserve and source relevant URLs from the evidence where available, finishing with a concise Sources section.\n"
    "5. If the provided evidence is insufficient, contradictory, or empty, explicitly state that the evidence "
    "is insufficient to fully answer the question, or represent competing perspectives rather than forcing consensus."
)


def _is_ending_mid_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    # Check if the text concludes with a URL (e.g., in a Sources section)
    last_word = stripped.split()[-1]
    if last_word.startswith(("http://", "https://")):
        return False

    terminal_markers = (
        ".",
        "!",
        "?",
        ":",
        '."',
        ".'",
        ".)",
        ".]",
        '*"',
        "*)",
        "```",
        ")",
        "]",
        "}",
        ">",
    )
    return not any(stripped.endswith(m) for m in terminal_markers)


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
            f"Grounded Evidence:\n{formatted_evidence}\n\n"
            "Instructions:\n"
            "- Answer only the requested number of findings.\n"
            "- Include one concise explanation per finding (and a short key finding per item only if space permits).\n"
            "- Avoid adding extra findings beyond the requested count, and do not add an unrequested 'Additionally' section.\n"
            "- Finish with a concise Sources section citing URLs from the evidence.\n"
            "- If evidence presents conflicting perspectives, articulate the disagreement rather than forcing consensus.\n"
            "- Treat cross-domain keyword overlap as a retrieval signal; preserve conflicting claims rather than assuming consensus."
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
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

        # Extract safe non-secret completion metadata where available
        finish_reason = (
            result_payload.get("finish_reason")
            or data.get("finish_reason")
            or (
                data.get("choices")
                and isinstance(data["choices"], list)
                and data["choices"][0].get("finish_reason")
            )
        )

        raw_usage = result_payload.get("usage") or data.get("usage") or {}
        usage_dict: dict[str, int] = {}
        if isinstance(raw_usage, dict):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = raw_usage.get(k)
                if isinstance(val, int):
                    usage_dict[k] = val

        is_token_limited = finish_reason in ("length", "max_tokens")
        ends_mid_sentence = _is_ending_mid_sentence(response_text)
        is_truncated = is_token_limited or ends_mid_sentence

        if is_truncated:
            logger.warning(
                "Cloudflare synthesis output detected as truncated (model=%s, finish_reason=%s, ends_mid_sentence=%s, prompt_tokens=%s, completion_tokens=%s, total_tokens=%s)",
                self.model,
                finish_reason or "unknown",
                ends_mid_sentence,
                usage_dict.get("prompt_tokens", "n/a"),
                usage_dict.get("completion_tokens", "n/a"),
                usage_dict.get("total_tokens", "n/a"),
            )
            notice = "[Response truncated: token limit reached]"
            trimmed = response_text.strip()
            if notice not in trimmed:
                if "sources:" not in trimmed.lower() and ("http://" in evidence or "https://" in evidence):
                    urls = re.findall(r'https?://[^\s)\]">]+', evidence)
                    unique_urls = list(dict.fromkeys(urls))
                    if unique_urls:
                        sources_block = "\n\nSources:\n" + "\n".join(f"- {u}" for u in unique_urls)
                        final_content = f"{trimmed}\n\n{notice}{sources_block}"
                    else:
                        final_content = f"{trimmed}\n\n{notice}"
                else:
                    final_content = f"{trimmed}\n\n{notice}"
            else:
                final_content = trimmed
        else:
            logger.info(
                "Cloudflare synthesis completed (model=%s, finish_reason=%s, prompt_tokens=%s, completion_tokens=%s, total_tokens=%s)",
                self.model,
                finish_reason or "unknown",
                usage_dict.get("prompt_tokens", "n/a"),
                usage_dict.get("completion_tokens", "n/a"),
                usage_dict.get("total_tokens", "n/a"),
            )
            final_content = response_text.strip()

        return SynthesisResult(
            question=question,
            content=final_content,
            evidence=evidence,
            provider="cloudflare",
            model=self.model,
            truncated=is_truncated,
            finish_reason=finish_reason,
            usage=usage_dict if usage_dict else None,
        )
