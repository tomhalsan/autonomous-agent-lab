import logging
import os
from typing import Any

from agent_contracts import InferenceProviderHealth, InferenceResponse
from agent_infrastructure import GeminiClient, GeminiHealthProbe

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_AI_MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a helpful and versatile AI assistant. "
    "Answer the user's prompt directly, clearly, and accurately."
)


class GeminiInferenceProvider:
    """Infrastructure adapter that executes LLM inference via the Google Gemini REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_tokens: int | None = None,
        temperature: float | None = None,
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

        if temperature is not None:
            if (
                type(temperature) is bool
                or not isinstance(temperature, (int, float))
                or not (0.0 <= temperature <= 2.0)
            ):
                raise ValueError(
                    f"temperature must be a number between 0.0 and 2.0, got {temperature!r}"
                )
            self.temperature = float(temperature)
        else:
            self.temperature = None

    def __repr__(self) -> str:
        return f"GeminiInferenceProvider(model={self.model!r}, api_key='[REDACTED]')"

    def __str__(self) -> str:
        return f"GeminiInferenceProvider(model={self.model!r})"

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
            },
        }
        if self.temperature is not None:
            payload["generationConfig"]["temperature"] = self.temperature
        return payload

    async def check_health(self) -> InferenceProviderHealth:
        """Probe Gemini availability via the shared infrastructure health probe."""
        return await self._health_probe.check_health()

    async def generate(self, prompt: str) -> InferenceResponse:
        """Generate response for a general worker task prompt."""
        logger.info("Executing Gemini generation with model %s", self.model)
        payload = self._build_payload(prompt)
        response = await self._client.generate_content(payload)

        input_tokens = None
        output_tokens = None
        total_tokens = None
        if response.usage_metadata is not None:
            pt = response.usage_metadata.get("promptTokenCount")
            if isinstance(pt, int) and type(pt) is not bool and pt >= 0:
                input_tokens = pt
            ct = response.usage_metadata.get("candidatesTokenCount")
            if isinstance(ct, int) and type(ct) is not bool and ct >= 0:
                output_tokens = ct
            tt = response.usage_metadata.get("totalTokenCount")
            if isinstance(tt, int) and type(tt) is not bool and tt >= 0:
                total_tokens = tt

        return InferenceResponse(
            text=response.text.strip(),
            provider_name=self.provider_name,
            model_name=self.model,
            finish_reason=response.finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
