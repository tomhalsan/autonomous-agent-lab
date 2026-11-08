import logging
import os
from typing import Any

from agent_contracts import InferenceProviderHealth
from agent_infrastructure import GeminiClient, GeminiHealthProbe

from research_agent.domain.models import SynthesisResult

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_AI_MAX_TOKENS = 1024

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


class GeminiInferenceProvider:
    """Infrastructure adapter that executes LLM synthesis via the Google Gemini REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_tokens: int | None = None,
    ) -> None:
        self._client = GeminiClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            client=client,
        )
        self.provider_name = "gemini"
        self.model = self._client.model
        self._health_probe = GeminiHealthProbe(gemini_client=self._client)

        env_max_tokens = os.getenv("GEMINI_MAX_TOKENS")
        if max_tokens is not None:
            if type(max_tokens) is bool or not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
            self.max_tokens = max_tokens
        elif env_max_tokens is not None and env_max_tokens.strip():
            stripped_env = env_max_tokens.strip()
            if not stripped_env.isdigit() or int(stripped_env) <= 0:
                raise ValueError(
                    f"GEMINI_MAX_TOKENS must be a positive integer, got {env_max_tokens!r}"
                )
            self.max_tokens = int(stripped_env)
        else:
            self.max_tokens = DEFAULT_GEMINI_AI_MAX_TOKENS

    def __repr__(self) -> str:
        return f"GeminiInferenceProvider(model={self.model!r}, api_key='[REDACTED]')"

    def __str__(self) -> str:
        return f"GeminiInferenceProvider(model={self.model!r})"

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
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": 0.0,
            },
        }

    async def check_health(self) -> InferenceProviderHealth:
        """Probe Gemini availability via the shared infrastructure health probe."""
        return await self._health_probe.check_health()

    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        """Synthesize research evidence into structured findings and cited sources."""
        logger.info("Executing Gemini research synthesis with model %s", self.model)
        payload = self._build_payload(question, evidence)
        response = await self._client.generate_content(payload)

        is_token_limited = (
            response.finish_reason is not None
            and response.finish_reason.upper() in ("MAX_TOKENS", "LENGTH", "TOKEN_LIMIT")
        )
        ends_mid_sentence = _is_ending_mid_sentence(response.text)
        is_truncated = is_token_limited or ends_mid_sentence

        trimmed = response.text.strip()
        if is_token_limited:
            notice = "[Response truncated: token limit reached]"
            logger.warning(
                "Gemini synthesis output detected as truncated by token limit (model=%s, finish_reason=%s, ends_mid_sentence=%s)",
                self.model,
                response.finish_reason,
                ends_mid_sentence,
            )
            final_content = f"{trimmed}\n\n{notice}" if notice not in trimmed else trimmed
        elif ends_mid_sentence:
            notice = "[Response may be incomplete]"
            logger.warning(
                "Gemini synthesis output detected as incomplete mid-sentence (model=%s, finish_reason=%s, ends_mid_sentence=True)",
                self.model,
                response.finish_reason or "unknown",
            )
            final_content = f"{trimmed}\n\n{notice}" if notice not in trimmed else trimmed
        else:
            final_content = trimmed

        return SynthesisResult(
            question=question,
            content=final_content,
            evidence=evidence,
            provider=self.provider_name,
            model=self.model,
            truncated=is_truncated,
            finish_reason=response.finish_reason,
            usage=response.usage_metadata,
        )
